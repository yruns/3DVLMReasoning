from __future__ import annotations

from pathlib import Path

import numpy as np

from agents import build_stage2_evidence_bundle
from query_scene.keyframe_selector import KeyframeResult, SceneObject


def _make_pose(yaw_deg: float = 0.0) -> np.ndarray:
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
    return pose


def _make_keyframe_result(status: str = "direct_grounded") -> KeyframeResult:
    return KeyframeResult(
        query="find the pillow",
        target_term="pillow",
        anchor_term="sofa",
        keyframe_indices=[0, 1],
        keyframe_paths=[Path("/tmp/frame0.jpg"), Path("/tmp/frame1.jpg")],
        target_objects=[
            SceneObject(obj_id=1, category="pillow", object_tag="pillow")
        ],
        anchor_objects=[
            SceneObject(obj_id=2, category="sofa", object_tag="sofa")
        ],
        metadata={
            "status": status,
            "selected_hypothesis_kind": "direct",
            "selected_hypothesis_rank": 1,
            "frame_mappings": [
                {
                    "requested_view_id": 0,
                    "requested_frame_id": 0,
                    "resolved_view_id": 0,
                    "resolved_frame_id": 0,
                },
                {
                    "requested_view_id": 2,
                    "requested_frame_id": 10,
                    "resolved_view_id": 2,
                    "resolved_frame_id": 10,
                },
            ],
            "hypothesis_output": {
                "parse_mode": "single",
                "hypotheses": [
                    {
                        "rank": 1,
                        "kind": "direct",
                        "grounding_query": {
                            "root": {
                                "category": ["pillow"],
                                "spatial_constraints": [
                                    {"anchors": [{"category": ["sofa"]}]}
                                ],
                            }
                        },
                    }
                ],
            },
        },
    )


def test_temporal_note_generation_when_pose_aware() -> None:
    selector = type("Selector", (), {})()
    selector.pose_aware_enabled = True
    selector.dwell_score = np.array([0.8, 0.5, 0.2, 0.9], dtype=np.float64)
    selector.camera_poses = [
        _make_pose(0.0),
        _make_pose(15.0),
        _make_pose(30.0),
        _make_pose(45.0),
    ]

    bundle = build_stage2_evidence_bundle(
        _make_keyframe_result(),
        scene_id="room0",
        selector=selector,
    )

    assert bundle.keyframes[0].note == "order=1/2 dwell neighbors=[1, 2]"
    assert "order=2/2" in bundle.keyframes[1].note
    assert "traverse" in bundle.keyframes[1].note
    assert "heading=+30°" in bundle.keyframes[1].note
    assert "neighbors=[1, 3]" in bundle.keyframes[1].note


def test_temporal_note_fallback_to_v14_when_pose_aware_off() -> None:
    selector = type("Selector", (), {})()
    selector.pose_aware_enabled = False
    selector.dwell_score = np.zeros(4, dtype=np.float64)
    selector.camera_poses = [_make_pose() for _ in range(4)]

    bundle = build_stage2_evidence_bundle(
        _make_keyframe_result(status="proxy_grounded"),
        scene_id="room0",
        selector=selector,
    )

    assert [keyframe.note for keyframe in bundle.keyframes] == [
        "proxy_grounded",
        "proxy_grounded",
    ]
