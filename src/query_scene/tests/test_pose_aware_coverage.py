from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
import pytest

from query_scene.keyframe_selector import KeyframeSelector


def _make_pose(x: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    theta = np.deg2rad(yaw_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    pose[0, 3] = x
    return pose


def _make_selector(
    tmp_path: Path,
    *,
    view_scores: dict[int, dict[int, float]],
    poses: list[np.ndarray],
    dwell_score: list[float] | None = None,
) -> KeyframeSelector:
    selector = KeyframeSelector.__new__(KeyframeSelector)
    selector.scene_path = tmp_path / "scene" / "conceptgraph"
    selector.scene_path.mkdir(parents=True, exist_ok=True)
    selector.stride = 1
    selector.objects = []
    selector.object_features = None
    selector.camera_poses = poses
    selector.image_paths = []
    selector.depth_paths = []
    selector.scene_categories = []
    selector.object_to_views = {}
    selector.view_to_objects = {}
    selector._clip_model = None
    selector._clip_tokenizer = None
    selector._query_parser = None
    selector._query_executor = None
    selector._relation_checker = None
    selector._K = np.array(
        [
            [1165.723022, 0.0, 648.0],
            [0.0, 1165.738037, 484.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    selector._img_wh = (1296, 968)
    selector._depth_cache = {}
    selector.pose_velocities = np.zeros(len(poses), dtype=np.float64)
    selector.pose_turn_rates = np.zeros(len(poses), dtype=np.float64)
    selector.turn_score = np.zeros(len(poses), dtype=np.float64)
    selector.dwell_score = np.asarray(
        dwell_score if dwell_score is not None else np.zeros(len(poses)),
        dtype=np.float64,
    )

    for view_id, obj_scores in view_scores.items():
        selector.view_to_objects[view_id] = sorted(
            obj_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for obj_id, score in obj_scores.items():
            selector.object_to_views.setdefault(obj_id, []).append((view_id, score))

    for obj_id, scores in selector.object_to_views.items():
        selector.object_to_views[obj_id] = sorted(
            scores,
            key=lambda item: item[1],
            reverse=True,
        )

    return selector


def _v14_joint_coverage(
    selector: KeyframeSelector,
    object_ids: list[int],
    max_views: int,
) -> list[int]:
    candidate_views: set[int] = set()
    for obj_id in object_ids:
        for view_id, _ in selector.object_to_views.get(obj_id, []):
            candidate_views.add(view_id)

    view_scores: dict[int, dict[int, float]] = {}
    for obj_id in object_ids:
        for view_id, score in selector.object_to_views.get(obj_id, []):
            view_scores.setdefault(view_id, {})[obj_id] = score

    selected: list[int] = []
    covered_quality = dict.fromkeys(object_ids, 0.0)

    for _ in range(max_views):
        best_view, best_gain = None, 0.0
        for view_id in candidate_views - set(selected):
            gain = 0.0
            for obj_id in object_ids:
                obj_score = view_scores.get(view_id, {}).get(obj_id, 0.0)
                if obj_score > covered_quality[obj_id]:
                    gain += obj_score - covered_quality[obj_id]

            if gain > best_gain:
                best_gain, best_view = gain, view_id

        if best_view is None:
            break

        selected.append(best_view)
        for obj_id in object_ids:
            obj_score = view_scores.get(best_view, {}).get(obj_id, 0.0)
            covered_quality[obj_id] = max(covered_quality[obj_id], obj_score)

    return selected


def test_pose_aware_false_is_bitfor_bit_v14(tmp_path: Path) -> None:
    selector = _make_selector(
        tmp_path,
        view_scores={
            0: {1: 0.95, 2: 0.10},
            1: {1: 0.20, 2: 0.85},
            2: {1: 0.60, 2: 0.60},
            3: {1: 0.50, 2: 0.40},
            4: {1: 0.10, 2: 0.30},
        },
        poses=[_make_pose(float(i)) for i in range(5)],
    )

    expected = _v14_joint_coverage(selector, object_ids=[1, 2], max_views=3)
    actual = selector.get_joint_coverage_views(
        [1, 2],
        max_views=3,
        pose_aware=False,
    )

    assert actual == expected


def test_pose_aware_drops_near_duplicate(tmp_path: Path) -> None:
    selector = _make_selector(
        tmp_path,
        view_scores={
            0: {1: 1.00, 2: 1.00, 3: 0.85},
            1: {1: 0.95, 2: 0.95, 3: 0.90},
            2: {1: 0.20, 2: 0.20, 3: 0.87},
        },
        poses=[_make_pose(0.0), _make_pose(0.0), _make_pose(3.0)],
    )

    assert selector.get_joint_coverage_views([1, 2, 3], max_views=2) == [0, 1]
    assert selector.get_joint_coverage_views(
        [1, 2, 3],
        max_views=2,
        pose_aware=True,
    ) == [0, 2]


def test_dwell_bonus_promotes_dwell_frame(tmp_path: Path) -> None:
    selector = _make_selector(
        tmp_path,
        view_scores={
            0: {1: 1.00},
            1: {1: 0.96},
        },
        poses=[_make_pose(0.0), _make_pose(2.0)],
        dwell_score=[0.0, 1.0],
    )

    selected = selector.get_joint_coverage_views(
        [1],
        max_views=1,
        pose_aware=True,
    )

    assert selected == [1]


def test_pose_aware_fails_without_frustum_info(tmp_path: Path) -> None:
    selector = _make_selector(
        tmp_path,
        view_scores={
            0: {1: 1.00, 2: 1.00, 3: 0.85},
            1: {1: 0.95, 2: 0.95, 3: 0.90},
            2: {1: 0.20, 2: 0.20, 3: 0.87},
        },
        poses=[_make_pose(0.0), _make_pose(0.0), _make_pose(3.0)],
    )
    selector._K = None
    selector._img_wh = None

    with pytest.raises(FileNotFoundError, match="raw dir not found"):
        selector.get_joint_coverage_views(
            [1, 2, 3],
            max_views=2,
            pose_aware=True,
        )


def test_frustum_method_l2_uses_depth(tmp_path: Path) -> None:
    selector = _make_selector(
        tmp_path,
        view_scores={
            0: {1: 1.00, 2: 1.00, 3: 0.85},
            1: {1: 0.95, 2: 0.95, 3: 0.90},
            2: {1: 0.20, 2: 0.20, 3: 0.87},
        },
        poses=[_make_pose(0.0), _make_pose(0.0), _make_pose(3.0)],
    )

    depth_calls: list[int] = []

    def _load_depth_for_view(self: KeyframeSelector, view_id: int) -> np.ndarray:
        depth_calls.append(view_id)
        return np.full((8, 8), 2.0, dtype=np.float32)

    selector._load_depth_for_view = MethodType(_load_depth_for_view, selector)

    selector.get_joint_coverage_views(
        [1, 2, 3],
        max_views=2,
        pose_aware=True,
        frustum_method="l2",
    )

    assert depth_calls
