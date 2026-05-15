"""End-to-end-ish QA flow without invoking a real VLM.

Mirrors the VG integration test but for the QA pack. The chassis (and
therefore submit_final) is opt-in for QA; the v9 catalog-first selector +
mark_frame_with_bbox still wire up because the bundle carries a SceneCatalog.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import importlib

import agents.packs.qa_default
import agents.packs.qa_default.registration  # noqa: F401
import agents.packs.vg_embodiedscan
import agents.packs.vg_embodiedscan.registration  # noqa: F401
from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2TaskType,
)
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime
from agents.skills import PACKS


def _ensure_packs_registered() -> None:
    if Stage2TaskType.VISUAL_GROUNDING not in PACKS:
        importlib.reload(agents.packs.vg_embodiedscan)
    if Stage2TaskType.QA not in PACKS:
        importlib.reload(agents.packs.qa_default)


def _bundle(tmp_path: Path) -> Stage2EvidenceBundle:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (256, 256), (10, 10, 10)).save(bev)
    rgb_dir = tmp_path / "raw"
    rgb_dir.mkdir()
    Image.new("RGB", (320, 240), (255, 255, 255)).save(rgb_dir / "000020-rgb.png")
    catalog = SceneCatalog(
        scene_id="002-scannet-scene0709_00",
        scene_category="bedroom",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="bed",
                position_3d=(0.0, 0.0, 0.3),
                bbox_3d_9dof=(0, 0, 0.3, 2.0, 1.5, 0.5, 0, 0, 0),
                frame_views={
                    20: FrameView(
                        frame_id=20,
                        bbox_2d=(40, 30, 280, 200),
                        raw_rgb_path=str(rgb_dir / "000020-rgb.png"),
                    ),
                },
                source="conceptgraph",
            ),
        ],
        total_frames=30,
        frame_id_range=(0, 290),
        valid_frame_ids=[20],
        bev_image_path=str(bev),
    )
    return Stage2EvidenceBundle(
        scene_id=catalog.scene_id,
        extra_metadata={"scene_catalog": catalog.model_dump()},
        bev_image_path=str(bev),
    )


def test_qa_v9_tool_surface_end_to_end(tmp_path: Path):
    """Sanity check: the v9 QA tool surface is wired and a scripted call
    sequence produces a clean tool_trace including select_by_proposal +
    mark_frame_with_bbox."""
    _ensure_packs_registered()
    bundle = _bundle(tmp_path)
    task = Stage2TaskSpec(
        user_query="what colour is the bed?",
        task_type=Stage2TaskType.QA,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=8,
    )

    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    state = Stage2RuntimeState(bundle=bundle, task_type=task.task_type)
    tools = {t.name: t for t in rt.build_runtime_tools(state)}

    for required in (
        "select_by_proposal",
        "select_by_text",
        "mark_frame_with_bbox",
        "view_bev",
        "list_scene_proposals",
        "inspect_proposal",
    ):
        assert required in tools, f"required v9 tool {required!r} not wired"

    # User message renders with the QA-specific view-mode hint.
    message = rt.build_user_message(task, state)
    text = next(p for p in message.content if p.get("type") == "text")["text"]
    assert "view_keyframe(mode='rgb')" in text or 'mode="rgb"' in text
    assert "bed" in text and "#0" in text

    # Selector + list tools are gated on the scene-exploration-playbook skill.
    # Simulate that the agent loaded it first.
    state.skills_loaded.add("scene-exploration-playbook")

    sel = json.loads(
        tools["select_by_proposal"].invoke(
            {"proposal_ids": [0], "require_all": False, "k": 4}
        )
    )
    frame_ids = {f["frame_id"] for f in sel["frames"]}
    assert 20 in frame_ids

    inv = json.loads(
        tools["list_scene_proposals"].invoke({"category": "bed", "limit": 5})
    )
    assert [p["proposal_id"] for p in inv["proposals"]] == [0]

    # No callback-era tools must be in the QA surface.
    for dead in (
        "request_more_views",
        "switch_or_expand_hypothesis",
        "inspect_stage1_metadata",
        "view_keyframe_marked",
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
    ):
        assert dead not in tools
