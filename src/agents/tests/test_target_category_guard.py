from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.packs.vg_embodiedscan.ctx import Proposal, VgEmbodiedScanCtx
from agents.runtime.base import Stage2RuntimeState
from agents.skills.target_category_guard import evaluate_target_category_guard


def _runtime(query: str) -> Stage2RuntimeState:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle(stage1_query=query))
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="unit",
        proposals=[
            Proposal(id=8, category="bed", score=0.9, bbox_3d_9dof=[0] * 9),
            Proposal(id=46, category="pillow", score=0.8, bbox_3d_9dof=[1] * 9),
            Proposal(id=13, category="whiteboard", score=0.7, bbox_3d_9dof=[2] * 9),
            Proposal(id=6, category="chair", score=0.6, bbox_3d_9dof=[3] * 9),
        ],
        frame_index={},
        proposal_index={},
        annotated_image_dir=Path("/tmp"),
    )
    return rs


def test_target_category_guard_blocks_context_object_submission() -> None:
    rs = _runtime("Choose the pillow on the back right bed.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is True
    assert decision.expected_category == "pillow"
    assert decision.submitted_category == "bed"
    assert "TARGET_CATEGORY_GUARD" in decision.message


def test_target_category_guard_explicit_target_overrides_later_context() -> None:
    rs = _runtime(
        "Staring at both beds from their foot, you want the bed on the right. "
        "The pillow is the back right option."
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is False
    assert decision.expected_category == "bed"
    assert decision.submitted_category == "bed"


def test_target_category_guard_allows_matching_head_category() -> None:
    rs = _runtime("The pillow is the back right option.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 46})

    assert decision.blocked is False
    assert decision.expected_category == "pillow"
    assert decision.submitted_category == "pillow"


def test_target_category_guard_passes_ambiguous_head() -> None:
    rs = _runtime("It is the one on the left.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is False
    assert decision.expected_category is None


def test_target_category_guard_passes_conflicting_explicit_targets() -> None:
    rs = _runtime("Choose the pillow on the bed, then pick the bed on the right.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is False
    assert decision.expected_category is None


def test_target_category_guard_passes_conflicting_generic_heads() -> None:
    rs = _runtime("The pillow is on the left. The bed is on the right.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is False
    assert decision.expected_category is None


def test_target_category_guard_reads_current_list_scene_proposals_shape() -> None:
    rs = _runtime("The whiteboard that has a blue note on top.")
    rs.tool_trace.append(
        SimpleNamespace(
            tool_name="list_scene_proposals",
            response_text='{"count":1,"proposals":[{"proposal_id":13,"category":"whiteboard"}]}',
        )
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert decision.blocked is True
    assert decision.expected_category == "whiteboard"
    assert decision.submitted_category == "chair"
