"""Visibility: project 3D bbox into camera frustum + depth check."""
from __future__ import annotations

import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    bbox_visible_in_frustum,
    build_frame_visibility,
    project_bbox_3d_to_2d,
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


def test_project_bbox_3d_to_2d_returns_visible_rect() -> None:
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)
    bbox = [0, 0, 5, 1, 1, 1, 0, 0, 0]  # 1m cube at 5m in front
    rect = project_bbox_3d_to_2d(
        bbox, intrinsic, extrinsic, (640, 480), depth_max=10.0
    )
    assert rect is not None
    x1, y1, x2, y2 = rect
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 480
    # Centered on principal point => rect should straddle (320, 240)
    assert x1 < 320 < x2
    assert y1 < 240 < y2


def test_project_bbox_3d_to_2d_returns_none_when_behind_camera() -> None:
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)
    bbox = [0, 0, -5, 1, 1, 1, 0, 0, 0]  # behind camera
    assert (
        project_bbox_3d_to_2d(
            bbox, intrinsic, extrinsic, (640, 480), depth_max=10.0
        )
        is None
    )
