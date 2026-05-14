"""Tests for v9 catalog-first build_user_message + collect_image_paths.

Per spec docs/superpowers/specs/2026-05-14-v9-catalog-first-scene-exploration-design.md §D.
"""

from pathlib import Path

from PIL import Image

from agents.catalog import SceneCatalog, SceneProposal
from agents.core.agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2TaskType,
)
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime


def _catalog(scene_id: str = "scannet/scene0123_45") -> SceneCatalog:
    return SceneCatalog(
        scene_id=scene_id,
        scene_category="kitchen-living-room",
        proposals=[
            SceneProposal(
                proposal_id=4, category="chair", position_3d=(0, 0, 0), source="mask3d"
            ),
            SceneProposal(
                proposal_id=5, category="chair", position_3d=(1, 0, 0), source="mask3d"
            ),
            SceneProposal(
                proposal_id=8, category="table", position_3d=(2, 0, 0), source="mask3d"
            ),
            SceneProposal(
                proposal_id=11, category="lamp", position_3d=(-1, 0, 0), source="mask3d"
            ),
        ],
        total_frames=187,
        frame_id_range=(0, 1860),
        valid_frame_ids=[i * 10 for i in range(187)],
        bev_image_path="bev.png",
    )


def _bundle(tmp_path: Path) -> Stage2EvidenceBundle:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (200, 200), (10, 10, 10)).save(bev)
    cat = _catalog()
    cat.bev_image_path = str(bev)
    return Stage2EvidenceBundle(
        scene_id=cat.scene_id,
        extra_metadata={"scene_catalog": cat.model_dump()},
        bev_image_path=str(bev),
    )


def test_build_user_message_no_first_person_seed_only_bev(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    bundle = _bundle(tmp_path)
    rs = Stage2RuntimeState(bundle=bundle)
    task = Stage2TaskSpec(
        user_query="this is a brown wooden chair next to the kitchen counter",
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=10,
    )
    msg = rt.build_user_message(task, rs)
    parts = msg.content
    text_parts = [p for p in parts if p.get("type") == "text"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert len(text_parts) == 1
    assert len(image_parts) == 1
    text = text_parts[0]["text"]
    assert "Current keyframes:" not in text
    assert "Stage-1 hypothesis summary" not in text
    assert "## Scene" in text
    assert "## BEV image (attached above)" in text


def test_build_user_message_cat_b_inventory(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    rs = Stage2RuntimeState(bundle=_bundle(tmp_path))
    task = Stage2TaskSpec(
        user_query="a chair",
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=4,
    )
    text = rt.build_user_message(task, rs).content[0]["text"]
    assert "Proposals by category:" in text
    assert "chair" in text and "#4" in text and "#5" in text
    assert "table" in text and "#8" in text
    assert "lamp" in text and "#11" in text


def test_build_user_message_qa_uses_rgb_note(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    rs = Stage2RuntimeState(bundle=_bundle(tmp_path))
    task = Stage2TaskSpec(
        user_query="how many chairs are there?",
        task_type=Stage2TaskType.QA,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=4,
    )
    text = rt.build_user_message(task, rs).content[0]["text"]
    assert "view_keyframe(mode='rgb')" in text or 'mode="rgb"' in text


def test_build_user_message_zero_keyframes_viewed_line(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    rs = Stage2RuntimeState(bundle=_bundle(tmp_path))
    task = Stage2TaskSpec(
        user_query="a chair",
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=2,
    )
    text = rt.build_user_message(task, rs).content[0]["text"]
    assert "viewed 0 keyframes out of 187" in text


def test_collect_image_paths_returns_bev_only(tmp_path: Path):
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    bundle = _bundle(tmp_path)
    paths = rt.collect_image_paths(bundle)
    assert paths == [str(tmp_path / "bev.png")]
