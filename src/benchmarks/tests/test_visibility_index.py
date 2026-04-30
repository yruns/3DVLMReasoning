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


def test_project_bbox_3d_to_2d_returns_clamped_rect_when_bbox_spans_image() -> None:
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    extrinsic = np.eye(4)
    bbox = [0, 0, 5, 10, 8, 1, 0, 0, 0]

    rect = project_bbox_3d_to_2d(
        bbox, intrinsic, extrinsic, (640, 480), depth_max=10.0
    )

    assert rect == (0, 0, 639, 479)


def test_project_bbox_3d_to_2d_handles_near_plane_box_with_no_visible_corners() -> None:
    intrinsic = np.array(
        [
            [1170.187988, 0.0, 647.75],
            [0.0, 1170.187988, 483.75],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    extrinsic = np.array(
        [
            [-0.0737540786, 0.9963085861, 0.0439334179, -1.194721683],
            [0.4724581939, 0.0737026557, -0.8782660832, 0.3076867497],
            [-0.878261485, -0.0440191357, -0.4761496028, 2.1504956031],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    bbox = [
        0.8108325681,
        0.0893528946,
        1.5604110486,
        0.1376120578,
        3.7822901726,
        3.6012418872,
        0.0,
        0.0,
        0.0,
    ]

    rect = project_bbox_3d_to_2d(
        bbox, intrinsic, extrinsic, (1296, 968), depth_max=20.0
    )

    assert rect is not None
    x1, y1, x2, y2 = rect
    assert 0 <= x1 < x2 <= 1295
    assert 0 <= y1 < y2 <= 967
