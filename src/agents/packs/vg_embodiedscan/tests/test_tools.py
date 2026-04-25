"""VG pack tools: per-tool tests + FAIL-LOUD gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import KeyframeEvidence, Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.packs.vg_embodiedscan.ctx import (
    Proposal,
    VgEmbodiedScanCtx,
)
from agents.packs.vg_embodiedscan.tools import build_vg_tools


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    annotated = tmp_path / "ann"
    annotated.mkdir()
    bundle = Stage2EvidenceBundle(
        keyframes=[
            KeyframeEvidence(keyframe_idx=0, image_path="a.png", frame_id=10),
            KeyframeEvidence(keyframe_idx=1, image_path="b.png", frame_id=11),
        ]
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="vdetr",
        proposals=[
            Proposal(id=0, bbox_3d_9dof=[0]*9, category="chair", score=0.9),
            Proposal(id=1, bbox_3d_9dof=[1]*9, category="desk", score=0.8),
            Proposal(id=2, bbox_3d_9dof=[2]*9, category="chair", score=0.7),
        ],
        frame_index={10: [0, 1], 11: [1, 2]},
        proposal_index={0: [10], 1: [10, 11], 2: [11]},
        annotated_image_dir=annotated,
    )
    return rs


def test_list_keyframes_with_proposals_gates_on_skill(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    tool = next(t for t in build_vg_tools(rs) if t.name == "list_keyframes_with_proposals")
    response = tool.invoke({})
    assert response.startswith("ERROR")
    assert "vg-grounding-playbook" in response


def test_list_keyframes_with_proposals_returns_structured(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "list_keyframes_with_proposals")
    payload = json.loads(tool.invoke({}))
    assert len(payload) == 2
    assert payload[0]["frame_id"] == 10
    assert payload[0]["visible_proposal_ids"] == [0, 1]
    assert payload[0]["annotated_image"].endswith("/ann/frame_10.png")


def test_view_keyframe_marked_returns_image_content(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # create a fake marked image
    marked = rs.task_ctx.annotated_image_dir / "frame_10.png"
    marked.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 10})
    assert "frame_10.png" in response
    assert "visible_proposals" in response
    assert "[0, 1]" in response or "0, 1" in response


def test_view_keyframe_marked_unknown_frame_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "view_keyframe_marked")
    response = tool.invoke({"frame_id": 999})
    assert response.startswith("ERROR")


def test_inspect_proposal_returns_metadata_and_frames(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "inspect_proposal")
    payload = json.loads(tool.invoke({"proposal_id": 1}))
    assert payload["proposal_id"] == 1
    assert payload["category"] == "desk"
    assert payload["score"] == 0.8
    assert payload["frames_appeared"] == [10, 11]
    assert payload["bbox_3d_9dof"] == [1]*9


def test_inspect_proposal_unknown_id_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "inspect_proposal")
    response = tool.invoke({"proposal_id": 99})
    assert response.startswith("ERROR")


def test_find_proposals_by_category_lists_ids(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category")
    payload = json.loads(tool.invoke({"category": "chair"}))
    assert payload["proposal_ids"] == [0, 2]


def test_find_proposals_by_category_unknown_returns_empty(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "find_proposals_by_category")
    payload = json.loads(tool.invoke({"category": "spaceship"}))
    assert payload["proposal_ids"] == []
    assert "available_categories" in payload


def test_compare_proposals_spatial_closest_to(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    # set proposal centers far apart so order is deterministic
    rs.task_ctx.proposals = [
        Proposal(id=0, bbox_3d_9dof=[0,0,0,1,1,1,0,0,0], category="chair", score=0.9),
        Proposal(id=1, bbox_3d_9dof=[5,5,5,1,1,1,0,0,0], category="chair", score=0.7),
        Proposal(id=2, bbox_3d_9dof=[10,10,10,1,1,1,0,0,0], category="desk", score=0.8),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(tool.invoke({
        "candidate_ids": [0, 1],
        "anchor_id": 2,
        "relation": "closest_to",
    }))
    assert payload["ranked_ids"] == [1, 0]


def test_compare_proposals_spatial_farthest_from(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    rs.task_ctx.proposals = [
        Proposal(id=0, bbox_3d_9dof=[0,0,0,1,1,1,0,0,0], category="chair", score=0.9),
        Proposal(id=1, bbox_3d_9dof=[5,5,5,1,1,1,0,0,0], category="chair", score=0.7),
        Proposal(id=2, bbox_3d_9dof=[10,10,10,1,1,1,0,0,0], category="desk", score=0.8),
    ]
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    payload = json.loads(tool.invoke({
        "candidate_ids": [0, 1],
        "anchor_id": 2,
        "relation": "farthest_from",
    }))
    assert payload["ranked_ids"] == [0, 1]


def test_compare_proposals_spatial_unknown_relation_errors(tmp_path: Path) -> None:
    rs = _runtime(tmp_path)
    rs.skills_loaded.add("vg-grounding-playbook")
    tool = next(t for t in build_vg_tools(rs) if t.name == "compare_proposals_spatial")
    response = tool.invoke({
        "candidate_ids": [0, 1],
        "anchor_id": 2,
        "relation": "left_of",
    })
    assert response.startswith("ERROR")
