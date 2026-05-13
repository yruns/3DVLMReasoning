"""Per-frame visibility for 3D bbox proposals."""
from __future__ import annotations

import numpy as np


def _bbox_corners(bbox_9dof: list[float]) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, *_ = bbox_9dof
    half = np.array([dx, dy, dz]) / 2.0
    base = np.array([cx, cy, cz])
    signs = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    ])
    return base + signs * half  # shape (8, 3)


def _bbox_surface_samples(bbox_9dof: list[float], steps: int = 7) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, *_ = bbox_9dof
    half = np.array([dx, dy, dz], dtype=float) / 2.0
    base = np.array([cx, cy, cz], dtype=float)
    axes = [
        np.linspace(base[axis] - half[axis], base[axis] + half[axis], steps)
        for axis in range(3)
    ]
    samples = []
    for fixed_axis in range(3):
        other_axes = [axis for axis in range(3) if axis != fixed_axis]
        for fixed_value in (
            base[fixed_axis] - half[fixed_axis],
            base[fixed_axis] + half[fixed_axis],
        ):
            for value_a in axes[other_axes[0]]:
                for value_b in axes[other_axes[1]]:
                    point = np.empty(3, dtype=float)
                    point[fixed_axis] = fixed_value
                    point[other_axes[0]] = value_a
                    point[other_axes[1]] = value_b
                    samples.append(point)
    return np.unique(np.asarray(samples, dtype=float), axis=0)


def project_bbox_3d_to_2d(
    bbox_9dof: list[float],
    intrinsic: np.ndarray,
    extrinsic_world_to_cam: np.ndarray,
    image_size: tuple[int, int],
    depth_max: float = 10.0,
) -> tuple[int, int, int, int] | None:
    """Project an axis-aligned 3D bbox to a clamped pixel rectangle.

    Returns None when no bbox corner is in front of the camera and inside
    the image bounds. The Euler angles in ``bbox_9dof`` are ignored to match
    the existing axis-aligned ``_bbox_corners`` helper.
    """
    bbox_arr = np.asarray(bbox_9dof, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    extrinsic_world_to_cam = np.asarray(extrinsic_world_to_cam, dtype=float)
    if bbox_arr.ndim != 1 or bbox_arr.shape[0] < 6:
        raise ValueError("bbox_9dof must be a 1D sequence with at least 6 values")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape (3, 3)")
    if extrinsic_world_to_cam.shape != (4, 4):
        raise ValueError("extrinsic_world_to_cam must have shape (4, 4)")
    if len(image_size) != 2:
        raise ValueError("image_size must be a (width, height) tuple")
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError("image_size dimensions must be positive")
    if depth_max <= 0:
        raise ValueError("depth_max must be positive")

    samples_world = _bbox_surface_samples(bbox_arr.tolist())
    samples_h = np.hstack([samples_world, np.ones((len(samples_world), 1))])
    cam = (extrinsic_world_to_cam @ samples_h.T).T[:, :3]
    valid = cam[(cam[:, 2] > 0) & (cam[:, 2] < depth_max)]
    if len(valid) == 0:
        return None

    px = (intrinsic @ valid.T).T
    px = px[:, :2] / px[:, 2:3]
    min_x = float(px[:, 0].min())
    max_x = float(px[:, 0].max())
    min_y = float(px[:, 1].min())
    max_y = float(px[:, 1].max())
    overlaps_image = max_x >= 0 and min_x < w and max_y >= 0 and min_y < h
    if not overlaps_image:
        return None

    x1 = int(np.clip(np.floor(min_x), 0, w - 1))
    y1 = int(np.clip(np.floor(min_y), 0, h - 1))
    x2 = int(np.clip(np.ceil(max_x), 0, w - 1))
    y2 = int(np.clip(np.ceil(max_y), 0, h - 1))
    return x1, y1, x2, y2


def project_visible_bbox_3d_to_2d(
    bbox_9dof: list[float],
    intrinsic: np.ndarray,
    extrinsic_world_to_cam: np.ndarray,
    image_size: tuple[int, int],
    depth_max: float = 10.0,
    min_in_bounds_samples: int = 1,
) -> tuple[int, int, int, int] | None:
    """Project only the in-image portion of a 3D bbox surface.

    ``project_bbox_3d_to_2d`` returns the full clamped rectangle whenever the
    projected 3D box overlaps the image. That is useful for broad geometric
    feasibility, but noisy for set-of-marks frames: a near-plane or mostly
    off-screen object can become a full-image rectangle even when no sampled
    bbox surface point lands inside the image. This helper requires actual
    in-image surface samples and builds the mark from those samples only.
    """
    bbox_arr = np.asarray(bbox_9dof, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    extrinsic_world_to_cam = np.asarray(extrinsic_world_to_cam, dtype=float)
    if bbox_arr.ndim != 1 or bbox_arr.shape[0] < 6:
        raise ValueError("bbox_9dof must be a 1D sequence with at least 6 values")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape (3, 3)")
    if extrinsic_world_to_cam.shape != (4, 4):
        raise ValueError("extrinsic_world_to_cam must have shape (4, 4)")
    if len(image_size) != 2:
        raise ValueError("image_size must be a (width, height) tuple")
    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError("image_size dimensions must be positive")
    if depth_max <= 0:
        raise ValueError("depth_max must be positive")
    if min_in_bounds_samples <= 0:
        raise ValueError("min_in_bounds_samples must be positive")

    samples_world = _bbox_surface_samples(bbox_arr.tolist())
    samples_h = np.hstack([samples_world, np.ones((len(samples_world), 1))])
    cam = (extrinsic_world_to_cam @ samples_h.T).T[:, :3]
    valid_depth = (cam[:, 2] > 0) & (cam[:, 2] < depth_max)
    if not np.any(valid_depth):
        return None

    cam = cam[valid_depth]
    px = (intrinsic @ cam.T).T
    uv = px[:, :2] / px[:, 2:3]
    in_bounds = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    if int(np.count_nonzero(in_bounds)) < min_in_bounds_samples:
        return None

    visible_uv = uv[in_bounds]
    x1 = int(np.clip(np.floor(visible_uv[:, 0].min()), 0, w - 1))
    y1 = int(np.clip(np.floor(visible_uv[:, 1].min()), 0, h - 1))
    x2 = int(np.clip(np.ceil(visible_uv[:, 0].max()), 0, w - 1))
    y2 = int(np.clip(np.ceil(visible_uv[:, 1].max()), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def project_depth_visible_points_to_2d(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_world_to_cam: np.ndarray,
    depth_map: np.ndarray,
    image_size: tuple[int, int],
    *,
    depth_scale: float = 1000.0,
    depth_tolerance: float = 0.15,
    depth_max: float = 20.0,
    min_visible_points: int = 5,
) -> tuple[int, int, int, int] | None:
    """Project only depth-visible object points to a 2D rectangle.

    This is stricter than rendering a projected 3D bbox. It mirrors the
    visibility-index occlusion test: a point is drawn only if it is in front of
    the camera, in the RGB image, has valid depth, and is not behind the depth
    surface beyond ``depth_tolerance``.
    """
    points = np.asarray(points_3d, dtype=float)
    intrinsic = np.asarray(intrinsic, dtype=float)
    extrinsic_world_to_cam = np.asarray(extrinsic_world_to_cam, dtype=float)
    depth_arr = np.asarray(depth_map)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_3d must have shape (N, >=3)")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape (3, 3)")
    if extrinsic_world_to_cam.shape != (4, 4):
        raise ValueError("extrinsic_world_to_cam must have shape (4, 4)")
    if depth_arr.ndim < 2:
        raise ValueError("depth_map must be at least 2D")
    if len(image_size) != 2:
        raise ValueError("image_size must be a (width, height) tuple")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    if depth_tolerance < 0:
        raise ValueError("depth_tolerance must be non-negative")
    if depth_max <= 0:
        raise ValueError("depth_max must be positive")
    if min_visible_points <= 0:
        raise ValueError("min_visible_points must be positive")

    w, h = image_size
    if w <= 0 or h <= 0:
        raise ValueError("image_size dimensions must be positive")
    points = points[:, :3]
    if len(points) == 0:
        return None

    points_h = np.hstack([points, np.ones((len(points), 1))])
    cam = (extrinsic_world_to_cam @ points_h.T).T[:, :3]
    z = cam[:, 2]
    depth_ok = (z > 0.05) & (z < depth_max)
    if int(np.count_nonzero(depth_ok)) < min_visible_points:
        return None

    cam = cam[depth_ok]
    z = z[depth_ok]
    uv_h = (intrinsic @ cam.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    in_bounds = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    if int(np.count_nonzero(in_bounds)) < min_visible_points:
        return None

    uv_in = uv[in_bounds]
    z_in = z[in_bounds]
    depth_m = depth_arr.astype(np.float32) / depth_scale
    dh, dw = depth_m.shape[:2]
    scale_u = dw / w
    scale_v = dh / h
    u_d = np.clip((uv_in[:, 0] * scale_u).astype(int), 0, dw - 1)
    v_d = np.clip((uv_in[:, 1] * scale_v).astype(int), 0, dh - 1)
    measured = depth_m[v_d, u_d]
    visible_mask = (measured > 0.1) & (measured >= (z_in - depth_tolerance))
    if int(np.count_nonzero(visible_mask)) < min_visible_points:
        return None

    visible_uv = uv_in[visible_mask]
    x1 = int(np.clip(np.floor(visible_uv[:, 0].min()), 0, w - 1))
    y1 = int(np.clip(np.floor(visible_uv[:, 1].min()), 0, h - 1))
    x2 = int(np.clip(np.ceil(visible_uv[:, 0].max()), 0, w - 1))
    y2 = int(np.clip(np.ceil(visible_uv[:, 1].max()), 0, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def bbox_visible_in_frustum(
    bbox_9dof: list[float],
    intrinsic: np.ndarray,
    extrinsic_world_to_cam: np.ndarray,
    image_size: tuple[int, int],
    depth_max: float = 10.0,
) -> bool:
    """Return True if any corner of the bbox falls inside the image
    frustum at positive depth and within depth_max."""
    corners_world = _bbox_corners(bbox_9dof)
    # world -> camera
    corners_h = np.hstack([corners_world, np.ones((8, 1))])
    cam = (extrinsic_world_to_cam @ corners_h.T).T[:, :3]
    if (cam[:, 2] <= 0).all() or (cam[:, 2] >= depth_max).all():
        return False
    # project to pixel
    valid = cam[(cam[:, 2] > 0) & (cam[:, 2] < depth_max)]
    if len(valid) == 0:
        return False
    px = (intrinsic @ valid.T).T
    px = px[:, :2] / px[:, 2:3]
    w, h = image_size
    in_image = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    return bool(in_image.any())


def build_frame_visibility(
    *,
    proposals: list[dict],
    intrinsic: np.ndarray,
    extrinsics_per_frame: dict[int, np.ndarray],
    image_size: tuple[int, int],
    depth_max: float = 10.0,
    proposal_ids: list[int] | None = None,
) -> dict[int, list[int]]:
    """Compute per-frame visibility for a list of 3D bbox proposals.

    The output values are proposal ids: positional indices into ``proposals``
    by default, or the entries of ``proposal_ids`` when supplied (e.g. the
    EmbodiedScan ``bbox_id`` for the GT pool). When ``proposal_ids`` is given
    it must have the same length as ``proposals``.
    """
    if proposal_ids is not None and len(proposal_ids) != len(proposals):
        raise ValueError(
            "proposal_ids length must match proposals; got "
            f"{len(proposal_ids)} vs {len(proposals)}"
        )
    out: dict[int, list[int]] = {}
    for fid, extr in extrinsics_per_frame.items():
        visible = []
        for idx, p in enumerate(proposals):
            bbox = p.get("bbox_3d_9dof") or p.get("bbox_3d")
            if bbox and bbox_visible_in_frustum(bbox, intrinsic, extr, image_size, depth_max):
                visible.append(int(proposal_ids[idx]) if proposal_ids is not None else idx)
        out[int(fid)] = visible
    return out


__all__ = [
    "bbox_visible_in_frustum",
    "build_frame_visibility",
    "project_bbox_3d_to_2d",
    "project_depth_visible_points_to_2d",
    "project_visible_bbox_3d_to_2d",
]
