import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.geometry import (
    aabb_from_points,
    backproject_depth,
    is_non_degenerate_bbox,
    transform_points,
)


def test_aabb_from_points_returns_embodiedscan_9dof() -> None:
    pts = np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32)
    bbox = aabb_from_points(pts)
    assert bbox == [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0]
    assert is_non_degenerate_bbox(bbox)


def test_backproject_depth_uses_intrinsics_and_mask() -> None:
    depth = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]])
    intrinsic = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
    pts = backproject_depth(depth, intrinsic, mask=mask)
    assert pts.shape == (2, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 1.0])
    assert np.allclose(pts[1], [4.0, 4.0, 4.0])


def test_transform_points_applies_homogeneous_transform() -> None:
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = [10.0, 20.0, 30.0]
    out = transform_points(pts, mat)
    assert np.allclose(out, [[11.0, 22.0, 33.0]])
