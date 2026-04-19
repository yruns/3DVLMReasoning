from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from query_scene.frustum import (
    frustum_overlap_l1,
    frustum_overlap_l2,
    load_scene_intrinsic,
)


SCENE_RAW_DIR = Path("data/OpenEQA/scannet/002-scannet-scene0709_00/raw")
POSE_PATH = SCENE_RAW_DIR / "000000.txt"
DEPTH_PATH = SCENE_RAW_DIR / "000000-depth.png"


def _load_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path, dtype=np.float64)
    assert pose.shape == (4, 4)
    return pose


def _load_depth(path: Path) -> np.ndarray:
    depth = np.array(Image.open(path), dtype=np.float32) / 1000.0
    assert depth.ndim == 2
    return depth


def _rotate_local_y(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _perturb_pose(
    pose: np.ndarray,
    *,
    yaw_deg: float = 0.0,
    right_shift_m: float = 0.0,
) -> np.ndarray:
    updated = pose.copy()
    updated[:3, :3] = pose[:3, :3] @ _rotate_local_y(yaw_deg)
    updated[:3, 3] = pose[:3, 3] + pose[:3, 0] * right_shift_m
    return updated


def test_load_intrinsic_from_real_scene() -> None:
    k, img_wh = load_scene_intrinsic(SCENE_RAW_DIR)

    assert img_wh == (1296, 968)
    assert k.shape == (3, 3)
    assert k[0, 0] == np.float64(1165.723022)
    assert k[1, 1] == np.float64(1165.738037)


def test_l1_identity() -> None:
    k, img_wh = load_scene_intrinsic(SCENE_RAW_DIR)
    pose = _load_pose(POSE_PATH)

    overlap = frustum_overlap_l1(pose, pose, k, img_wh)

    assert overlap == np.float64(1.0)


def test_l1_opposite() -> None:
    k, img_wh = load_scene_intrinsic(SCENE_RAW_DIR)
    pose_a = _load_pose(POSE_PATH)
    pose_b = _perturb_pose(pose_a, yaw_deg=180.0)

    overlap = frustum_overlap_l1(pose_a, pose_b, k, img_wh)

    assert overlap == np.float64(0.0)


def test_l1_perpendicular() -> None:
    k, img_wh = load_scene_intrinsic(SCENE_RAW_DIR)
    pose_a = _load_pose(POSE_PATH)
    pose_b = _perturb_pose(pose_a, yaw_deg=90.0)

    overlap = frustum_overlap_l1(pose_a, pose_b, k, img_wh)

    assert 0.3 < overlap < 0.7


def test_l1_translation_only() -> None:
    k, img_wh = load_scene_intrinsic(SCENE_RAW_DIR)
    pose_a = _load_pose(POSE_PATH)
    pose_b = _perturb_pose(pose_a, right_shift_m=2.0)

    overlap = frustum_overlap_l1(pose_a, pose_b, k, img_wh)

    assert overlap == np.float64(0.0)


def test_l1_small_delta() -> None:
    k, img_wh = load_scene_intrinsic(SCENE_RAW_DIR)
    pose_a = _load_pose(POSE_PATH)
    pose_b = _perturb_pose(pose_a, yaw_deg=5.0, right_shift_m=0.1)

    overlap = frustum_overlap_l1(pose_a, pose_b, k, img_wh)

    assert overlap > 0.9


def test_l2_identity() -> None:
    k, img_wh = load_scene_intrinsic(SCENE_RAW_DIR)
    pose = _load_pose(POSE_PATH)
    depth = _load_depth(DEPTH_PATH)

    overlap = frustum_overlap_l2(depth, pose, pose, k, img_wh)

    assert np.isclose(overlap, 1.0, atol=1e-6)


def test_l2_matches_l1_in_small_delta() -> None:
    k, img_wh = load_scene_intrinsic(SCENE_RAW_DIR)
    pose_a = _load_pose(POSE_PATH)
    pose_b = _perturb_pose(pose_a, yaw_deg=10.0, right_shift_m=0.2)
    depth = _load_depth(DEPTH_PATH)

    overlap_l1 = frustum_overlap_l1(pose_a, pose_b, k, img_wh)
    overlap_l2 = frustum_overlap_l2(depth, pose_a, pose_b, k, img_wh)

    assert abs(overlap_l1 - overlap_l2) < 0.15
