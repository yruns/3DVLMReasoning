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


__all__ = ["bbox_visible_in_frustum", "build_frame_visibility"]
