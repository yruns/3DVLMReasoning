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
]
