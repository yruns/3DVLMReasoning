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

    corners_world = _bbox_corners(bbox_arr.tolist())
    corners_h = np.hstack([corners_world, np.ones((8, 1))])
    cam = (extrinsic_world_to_cam @ corners_h.T).T[:, :3]
    valid = cam[(cam[:, 2] > 0) & (cam[:, 2] < depth_max)]
    if len(valid) == 0:
        return None

    px = (intrinsic @ valid.T).T
    px = px[:, :2] / px[:, 2:3]
    in_image = (
        (px[:, 0] >= 0)
        & (px[:, 0] < w)
        & (px[:, 1] >= 0)
        & (px[:, 1] < h)
    )
    if not in_image.any():
        return None

    x1 = int(np.clip(np.floor(px[:, 0].min()), 0, w - 1))
    y1 = int(np.clip(np.floor(px[:, 1].min()), 0, h - 1))
    x2 = int(np.clip(np.ceil(px[:, 0].max()), 0, w - 1))
    y2 = int(np.clip(np.ceil(px[:, 1].max()), 0, h - 1))
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
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for fid, extr in extrinsics_per_frame.items():
        visible = []
        for idx, p in enumerate(proposals):
            bbox = p.get("bbox_3d_9dof") or p.get("bbox_3d")
            if bbox and bbox_visible_in_frustum(bbox, intrinsic, extr, image_size, depth_max):
                visible.append(idx)
        out[int(fid)] = visible
    return out


__all__ = [
    "bbox_visible_in_frustum",
    "build_frame_visibility",
    "project_bbox_3d_to_2d",
]
