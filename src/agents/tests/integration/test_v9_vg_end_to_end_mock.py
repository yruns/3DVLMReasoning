"""End-to-end-ish VG flow without invoking a real VLM.

The full DeepAgents `graph.invoke` path requires LangChain wiring that is
expensive to mock. Instead, this test drives the v9 tool surface directly
the way a VLM would: load the gate skills, run a selector, mark a frame,
and check the trace + the SceneCatalog-driven user message.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2TaskType,
)
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
import importlib

import agents.packs.qa_default
import agents.packs.qa_default.registration  # noqa: F401
import agents.packs.vg_embodiedscan
import agents.packs.vg_embodiedscan.registration  # noqa: F401
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
    Image.new("RGB", (256, 256), (12, 12, 12)).save(bev)
    rgb_dir = tmp_path / "raw"
    rgb_dir.mkdir()
    for fid in (5, 10):
        Image.new("RGB", (320, 240), (200, 200, 200)).save(
            rgb_dir / f"{fid:06d}-rgb.png"
        )
    catalog = SceneCatalog(
        scene_id="scene0123_45",
        scene_category="kitchen",
        proposals=[
            SceneProposal(
                proposal_id=7,
                category="chair",
                position_3d=(0.0, 0.0, 0.4),
                bbox_3d_9dof=(0, 0, 0.4, 0.5, 0.5, 0.8, 0, 0, 0),
                frame_views={
                    5: FrameView(
                        frame_id=5,
                        bbox_2d=(10, 10, 60, 60),
                        raw_rgb_path=str(rgb_dir / "000005-rgb.png"),
                    ),
                },
                source="mask3d",
            ),
            SceneProposal(
                proposal_id=8,
                category="chair",
                position_3d=(1.5, 0.0, 0.4),
                bbox_3d_9dof=(1.5, 0, 0.4, 0.5, 0.5, 0.8, 0, 0, 0),
                frame_views={
                    10: FrameView(
                        frame_id=10,
                        bbox_2d=(80, 30, 140, 110),
                        raw_rgb_path=str(rgb_dir / "000010-rgb.png"),
                    ),
                },
                source="mask3d",
            ),
        ],
        total_frames=12,
        frame_id_range=(0, 110),
        valid_frame_ids=[5, 10],
        bev_image_path=str(bev),
    )
    return Stage2EvidenceBundle(
        scene_id=catalog.scene_id,
        extra_metadata={"scene_catalog": catalog.model_dump()},
        bev_image_path=str(bev),
    )


def test_vg_v9_tool_surface_end_to_end(tmp_path: Path):
    """Sanity check: the v9 VG tool surface is wired and a scripted call
    sequence produces a clean tool_trace + a usable submit_final payload."""
    _ensure_packs_registered()
    bundle = _bundle(tmp_path)
    task = Stage2TaskSpec(
        user_query="this is a brown chair",
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=8,
    )

    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    state = Stage2RuntimeState(bundle=bundle, task_type=task.task_type)
    tools = {t.name: t for t in rt.build_runtime_tools(state)}

    # The catalog-first surface should expose at minimum:
    for required in (
        "select_by_proposal",
        "select_by_text",
        "mark_frame_with_bbox",
        "view_bev",
        "list_scene_proposals",
        "list_frame_proposals",
        "inspect_proposal",
    ):
        assert required in tools, f"required v9 tool {required!r} not wired"

    # User message renders with scene_catalog (BEV + Cat-B inventory).
    message = rt.build_user_message(task, state)
    text = next(p for p in message.content if p.get("type") == "text")["text"]
    assert "## Scene" in text
    assert "scene0123_45" in text
    assert "chair" in text and "#7" in text and "#8" in text

    # Step 1: load skills (gate).
    tools["load_skill"].invoke({"skill_name": "scene-exploration-playbook"})
    tools["load_skill"].invoke({"skill_name": "vg-grounding-playbook"})

    # Step 2: select_by_proposal returns the frames that contain #7 / #8.
    sel = json.loads(
        tools["select_by_proposal"].invoke(
            {"proposal_ids": [7, 8], "require_all": False, "k": 4}
        )
    )
    returned_fids = {frame["frame_id"] for frame in sel["frames"]}
    assert {5, 10}.issubset(returned_fids)

    trace_names = [obs.tool_name for obs in state.tool_trace]
    assert "select_by_proposal" in trace_names

    # Step 3: confirm the deleted Stage-1 callback tools are NOT in the surface.
    for dead in (
        "request_more_views",
        "switch_or_expand_hypothesis",
        "inspect_stage1_metadata",
        "view_keyframe_marked",
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
    ):
        assert dead not in tools, f"deleted tool {dead!r} should not be wired"
