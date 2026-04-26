"""Visibility: project 3D bbox into camera frustum + depth check."""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    bbox_visible_in_frustum,
    build_frame_visibility,
)


def test_bbox_visible_when_in_frustum() -> None:
    # camera at origin, looking down +Z, fov ~60deg
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)  # camera at origin, no rotation
    bbox_9dof = [0, 0, 5, 1, 1, 1, 0, 0, 0]   # 5m in front
    assert bbox_visible_in_frustum(
        bbox_9dof, intrinsic, extrinsic, image_size=(640, 480), depth_max=10.0
    )


def test_bbox_invisible_behind_camera() -> None:
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)
    bbox_9dof = [0, 0, -5, 1, 1, 1, 0, 0, 0]   # 5m behind
    assert not bbox_visible_in_frustum(
        bbox_9dof, intrinsic, extrinsic, image_size=(640, 480), depth_max=10.0
    )


def test_build_frame_visibility_dispatches_per_frame() -> None:
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)
    frames = {10: extrinsic, 11: extrinsic}
    proposals = [
        {"bbox_3d_9dof": [0,0,5,1,1,1,0,0,0]},
        {"bbox_3d_9dof": [0,0,-5,1,1,1,0,0,0]},
    ]
    visibility = build_frame_visibility(
        proposals=proposals,
        intrinsic=intrinsic,
        extrinsics_per_frame=frames,
        image_size=(640, 480),
        depth_max=10.0,
    )
    assert visibility[10] == [0]
    assert visibility[11] == [0]
