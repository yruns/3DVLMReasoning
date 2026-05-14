"""Shared test fixtures for the six v9 selectors."""

from __future__ import annotations

from typing import Any

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState

PRIMARY_SKILL = "scene-exploration-playbook"


def make_catalog() -> SceneCatalog:
    return SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    10: FrameView(frame_id=10, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                    20: FrameView(frame_id=20, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                },
            ),
            SceneProposal(
                proposal_id=1,
                category="chair",
                position_3d=(2.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    30: FrameView(frame_id=30, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                },
            ),
            SceneProposal(
                proposal_id=2,
                category="table",
                position_3d=(1.0, 1.0, 0.0),
                source="mask3d",
                frame_views={
                    10: FrameView(frame_id=10, bbox_2d=(20, 20, 60, 60), raw_rgb_path="r.png"),
                    40: FrameView(frame_id=40, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                },
            ),
            SceneProposal(
                proposal_id=3,
                category="lamp",
                position_3d=(-3.0, -3.0, 0.0),
                source="mask3d",
                frame_views={
                    50: FrameView(frame_id=50, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"),
                },
            ),
        ],
        total_frames=5,
        frame_id_range=(10, 50),
        valid_frame_ids=[10, 20, 30, 40, 50],
        bev_image_path="bev.png",
    )


def make_runtime(extra: dict | None = None) -> Stage2RuntimeState:
    catalog = make_catalog()
    base_meta: dict[str, Any] = {"scene_catalog": catalog.model_dump()}
    base_meta["camera_trajectory_xy_yaw"] = {
        10: [0.0, 0.0, 0.0],
        20: [0.5, 0.0, 0.5],
        30: [1.0, 0.0, 1.0],
        40: [1.5, 0.5, 0.0],
        50: [-2.0, -2.0, 3.0],
    }
    if extra:
        base_meta.update(extra)
    bundle = Stage2EvidenceBundle(extra_metadata=base_meta)
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.skills_loaded.add(PRIMARY_SKILL)
    return rs
