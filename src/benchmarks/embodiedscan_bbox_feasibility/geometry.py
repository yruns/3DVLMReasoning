from __future__ import annotations

import numpy as np


def is_non_degenerate_bbox(bbox_3d: list[float], *, min_extent: float = 1e-4) -> bool:
    if len(bbox_3d) < 6:
        return False
    values = np.asarray(bbox_3d[:6], dtype=np.float64)
    if not np.isfinite(values).all():
        return False
    return bool(np.all(values[3:6] > min_extent))


def aabb_from_points(points: np.ndarray) -> list[float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        raise ValueError("points must have shape (N, >=3) with N > 0")
    xyz = pts[:, :3]
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = (lo + hi) / 2.0
    extent = hi - lo
    return [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(extent[0]),
        float(extent[1]),
        float(extent[2]),
        0.0,
        0.0,
        0.0,
    ]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    mat = np.asarray(transform, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    if mat.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts[:, :3], ones])
    return (mat @ pts_h.T).T[:, :3]


def backproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    depth_scale: float = 1.0,
    min_depth: float = 1e-6,
) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float64) / float(depth_scale)
    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape[0] < 3 or k.shape[1] < 3:
        raise ValueError("intrinsic must be at least 3x3")
    valid = depth_arr > min_depth
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(valid)
    z = depth_arr[ys, xs]
    x = (xs.astype(np.float64) - k[0, 2]) * z / k[0, 0]
    y = (ys.astype(np.float64) - k[1, 2]) * z / k[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)
