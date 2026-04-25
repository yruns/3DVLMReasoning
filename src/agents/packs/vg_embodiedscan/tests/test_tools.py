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
