from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.build_visibility_index import build_visibility_index


def _single_point_objects(z: float = 3.0) -> list[dict[str, np.ndarray]]:
    return [{"pcd_np": np.array([[0.0, 0.0, z]], dtype=np.float64)}]


def test_build_visibility_index_requires_depth_paths_when_depth_enabled() -> None:
    with pytest.raises(ValueError, match="use_depth=True requires depth_paths"):
        build_visibility_index(
            objects=_single_point_objects(),
            poses=[np.eye(4, dtype=np.float64)],
            depth_paths=None,
            intrinsics=np.eye(3, dtype=np.float64),
            use_depth=True,
            img_w=10,
            img_h=10,
            max_distance=10.0,
            min_visible_ratio=0.0,
            min_visible_points=1,
        )


def test_build_visibility_index_fails_when_depth_image_cannot_be_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Failed to read depth map"):
        build_visibility_index(
            objects=_single_point_objects(),
            poses=[np.eye(4, dtype=np.float64)],
            depth_paths=[tmp_path / "missing-depth.png"],
            intrinsics=np.eye(3, dtype=np.float64),
            use_depth=True,
            img_w=10,
            img_h=10,
            max_distance=10.0,
            min_visible_ratio=0.0,
            min_visible_points=1,
        )


def test_build_visibility_index_removes_depth_occluded_points(
    tmp_path: Path,
) -> None:
    depth_path = tmp_path / "depth.png"
    Image.fromarray(np.full((10, 10), 1000, dtype=np.uint16)).save(depth_path)

    object_to_views, view_to_objects = build_visibility_index(
        objects=_single_point_objects(z=3.0),
        poses=[np.eye(4, dtype=np.float64)],
        depth_paths=[depth_path],
        intrinsics=np.eye(3, dtype=np.float64),
        use_depth=True,
        img_w=10,
        img_h=10,
        max_distance=10.0,
        min_visible_ratio=0.0,
        min_visible_points=1,
    )

    assert object_to_views[0] == []
    assert view_to_objects == {}
