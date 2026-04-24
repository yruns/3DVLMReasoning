import numpy as np
import pytest

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


def test_aabb_from_points_rejects_non_finite_points() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [np.inf, 1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="points must contain only finite values"):
        aabb_from_points(pts)


def test_backproject_depth_uses_intrinsics_and_mask() -> None:
    depth = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [False, True]])
    intrinsic = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
    pts = backproject_depth(depth, intrinsic, mask=mask)
    assert pts.shape == (2, 3)
    assert np.allclose(pts[0], [0.0, 0.0, 1.0])
    assert np.allclose(pts[1], [4.0, 4.0, 4.0])


def test_backproject_depth_returns_empty_array_for_all_invalid_depth() -> None:
    depth = np.array([[0.0, -1.0], [np.nan, 1e-8]], dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    pts = backproject_depth(depth, intrinsic)
    assert pts.shape == (0, 3)
    assert pts.dtype == np.float32


def test_backproject_depth_filters_infinite_depth() -> None:
    depth = np.array([[np.inf, 2.0]], dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    pts = backproject_depth(depth, intrinsic)
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [2.0, 0.0, 2.0])


@pytest.mark.parametrize("depth_scale", [0.0, -1.0, np.nan, np.inf])
def test_backproject_depth_rejects_invalid_depth_scale(depth_scale: float) -> None:
    depth = np.array([[1.0]], dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="depth_scale must be finite and positive"):
        backproject_depth(depth, intrinsic, depth_scale=depth_scale)


@pytest.mark.parametrize("min_depth", [-1.0, np.nan, np.inf])
def test_backproject_depth_rejects_invalid_min_depth(min_depth: float) -> None:
    depth = np.array([[1.0]], dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="min_depth must be finite and non-negative"):
        backproject_depth(depth, intrinsic, min_depth=min_depth)


@pytest.mark.parametrize(
    "depth",
    [
        np.array([1.0, 2.0], dtype=np.float32),
        np.ones((2, 2, 1), dtype=np.float32),
    ],
)
def test_backproject_depth_rejects_non_2d_depth(depth: np.ndarray) -> None:
    intrinsic = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="depth must have shape \\(H, W\\)"):
        backproject_depth(depth, intrinsic)


@pytest.mark.parametrize(
    "intrinsic",
    [
        np.ones((3,), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    ],
)
def test_backproject_depth_rejects_invalid_intrinsics(intrinsic: np.ndarray) -> None:
    depth = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        backproject_depth(depth, intrinsic)


def test_backproject_depth_rejects_mask_shape_mismatch() -> None:
    depth = np.ones((2, 2), dtype=np.float32)
    mask = np.ones((4,), dtype=bool)
    intrinsic = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="mask must have the same shape as depth"):
        backproject_depth(depth, intrinsic, mask=mask)


def test_transform_points_applies_homogeneous_transform() -> None:
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = [10.0, 20.0, 30.0]
    out = transform_points(pts, mat)
    assert np.allclose(out, [[11.0, 22.0, 33.0]])
