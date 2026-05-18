from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.packs.vg_embodiedscan.ctx import Proposal, VgEmbodiedScanCtx
from agents.runtime.base import Stage2RuntimeState
from agents.skills import FinalizerSpec, SkillSpec, TaskPack, register_pack
from agents.skills.chassis_tools import build_chassis_tools
from agents.skills.registry import PACKS
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


def _runtime_with_categories(query: str, categories: list[tuple[int, str]]) -> Stage2RuntimeState:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle(stage1_query=query))
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.task_ctx = VgEmbodiedScanCtx(
        proposal_pool_source="unit",
        proposals=[
            Proposal(id=proposal_id, category=category, score=0.9, bbox_3d_9dof=[0] * 9)
            for proposal_id, category in categories
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


def test_target_category_guard_blocks_book_case_alias_with_relation_tail() -> None:
    rs = _runtime_with_categories(
        "Choose the book case on the left.",
        [(21, "bookshelf"), (6, "chair")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert decision.blocked is True
    assert decision.expected_category == "bookshelf"
    assert decision.submitted_category == "chair"


def test_target_category_guard_blocks_bookcase_alias_with_relation_tail() -> None:
    rs = _runtime_with_categories(
        "Choose the bookcase on the left.",
        [(21, "bookshelf"), (6, "chair")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert decision.blocked is True
    assert decision.expected_category == "bookshelf"
    assert decision.submitted_category == "chair"


def test_target_category_guard_blocks_bare_trashcan_alias_head() -> None:
    rs = _runtime_with_categories(
        "trashcan in a corner",
        [(31, "trash can"), (6, "chair")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert decision.blocked is True
    assert decision.expected_category == "trash can"
    assert decision.submitted_category == "chair"


def test_target_category_guard_blocks_pronoun_trashcan_alias_head() -> None:
    rs = _runtime_with_categories(
        "it is the trashcan in a corner",
        [(31, "trash can"), (6, "chair")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert decision.blocked is True
    assert decision.expected_category == "trash can"
    assert decision.submitted_category == "chair"


def test_target_category_guard_blocks_white_board_alias_head() -> None:
    rs = _runtime_with_categories(
        "the white board on the wall",
        [(13, "whiteboard"), (6, "chair")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert decision.blocked is True
    assert decision.expected_category == "whiteboard"
    assert decision.submitted_category == "chair"


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


def test_target_category_guard_unwraps_nested_payload_for_matching_category() -> None:
    rs = _runtime("The pillow is the back right option.")

    decision = evaluate_target_category_guard(rs, {"payload": {"proposal_id": 46}})

    assert decision.blocked is False
    assert decision.expected_category == "pillow"
    assert decision.submitted_category == "pillow"


def test_target_category_guard_passes_missing_proposal_id() -> None:
    rs = _runtime("Choose the pillow.")

    decision = evaluate_target_category_guard(rs, {})

    assert decision.blocked is False
    assert decision.submitted_pid is None
    assert decision.expected_category is None
    assert decision.submitted_category is None


def test_target_category_guard_passes_non_int_proposal_id() -> None:
    rs = _runtime("Choose the pillow.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": "46"})

    assert decision.blocked is False
    assert decision.submitted_pid is None
    assert decision.expected_category is None
    assert decision.submitted_category is None


def test_target_category_guard_passes_negative_one_proposal_id() -> None:
    rs = _runtime("Choose the pillow.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": -1})

    assert decision.blocked is False
    assert decision.submitted_pid is None
    assert decision.expected_category is None
    assert decision.submitted_category is None


def test_target_category_guard_passes_unavailable_submitted_proposal_id() -> None:
    rs = _runtime("Choose the pillow.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 999})

    assert decision.blocked is False
    assert decision.submitted_pid == 999
    assert decision.expected_category is None
    assert decision.submitted_category is None


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


def test_submit_final_already_submitted_records_neutral_target_category_fields(
    tmp_path: Path,
) -> None:
    previous_packs = dict(PACKS)
    PACKS.clear()
    try:
        body = tmp_path / "vg_grounding_playbook.md"
        body.write_text("# VG Grounding Playbook\n", encoding="utf-8")
        register_pack(
            TaskPack(
                task_type=Stage2TaskType.VISUAL_GROUNDING,
                tool_builder=lambda runtime: [],
                skills=[
                    SkillSpec(
                        name="vg-grounding-playbook",
                        description="VG main loop.",
                        body_path=body,
                        task_types={Stage2TaskType.VISUAL_GROUNDING},
                    )
                ],
                finalizer=FinalizerSpec(
                    payload_model=dict,
                    validator=lambda payload, runtime: payload,
                    adapter=lambda payload, runtime: {"answer": payload},
                ),
                required_primary_skill="vg-grounding-playbook",
                required_extra_metadata=[],
                ctx_factory=lambda bundle: object(),
            )
        )

        rs = _runtime("Choose the pillow.")
        _, _, submit_final = build_chassis_tools(rs)
        submit_final.invoke(
            {
                "payload": {"proposal_id": 46},
                "rationale": "first",
                "evidence_refs": [],
            }
        )
        submit_final.invoke(
            {
                "payload": {"proposal_id": 46},
                "rationale": "second",
                "evidence_refs": [],
            }
        )

        submit_record = rs.tool_trace[-1]
        assert submit_record.response_text.startswith("ALREADY_SUBMITTED")
        assert submit_record.tool_input["target_category_guard_blocked"] is False
        assert submit_record.tool_input["target_category_guard_submitted_pid"] is None
        assert (
            submit_record.tool_input["target_category_guard_expected_category"] is None
        )
        assert (
            submit_record.tool_input["target_category_guard_submitted_category"]
            is None
        )
        assert submit_record.tool_input["target_category_guard_message"] is None
    finally:
        PACKS.clear()
        PACKS.update(previous_packs)
