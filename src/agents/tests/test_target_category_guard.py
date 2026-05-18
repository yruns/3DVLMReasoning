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


def test_target_category_guard_allows_wall_painting_as_picture_target() -> None:
    rs = _runtime_with_categories(
        "A red, yellow wall painting to the left of the white bedroom door. "
        "It's hung on the wall.",
        [(14, "picture"), (30, "wall"), (2, "door")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 14})

    assert decision.blocked is False
    assert decision.expected_category == "picture"
    assert decision.submitted_category == "picture"


def test_target_category_guard_blocks_wall_for_wall_painting_target() -> None:
    rs = _runtime_with_categories(
        "A red yellow wall painting to the left of the white bedroom door.",
        [(14, "picture"), (30, "wall"), (2, "door")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 30})

    assert decision.blocked is True
    assert decision.expected_category == "picture"
    assert decision.submitted_category == "wall"


def test_target_category_guard_preserves_wall_target() -> None:
    rs = _runtime_with_categories(
        "The wall to the left of the clock.",
        [(30, "wall"), (9, "clock"), (14, "picture")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 30})

    assert decision.blocked is False
    assert decision.expected_category == "wall"
    assert decision.submitted_category == "wall"


def test_target_category_guard_allows_shelf_after_context_wall_phrase() -> None:
    rs = _runtime_with_categories(
        "On the wall opposite to the clock face - the shelf on the left when facing these 3.",
        [(18, "bookshelf"), (31, "wall"), (56, "clock")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 18})

    assert decision.blocked is False
    assert decision.expected_category == "bookshelf"
    assert decision.submitted_category == "bookshelf"


def test_target_category_guard_blocks_wall_for_shelf_after_context_wall_phrase() -> None:
    rs = _runtime_with_categories(
        "On the wall opposite to the clock face - the shelf on the left when facing these 3.",
        [(18, "bookshelf"), (31, "wall"), (56, "clock")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 31})

    assert decision.blocked is True
    assert decision.expected_category == "bookshelf"
    assert decision.submitted_category == "wall"


def test_target_category_guard_prefers_head_before_against_wall_anchor() -> None:
    rs = _runtime_with_categories(
        "The box is against the wall on top of the stack.",
        [(30, "box"), (31, "box"), (10, "wall")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 30})

    assert decision.blocked is False
    assert decision.expected_category == "box"
    assert decision.submitted_category == "box"


def test_target_category_guard_prefers_head_before_closest_to_anchor() -> None:
    rs = _runtime_with_categories(
        "The chair closest to the mirror in the nook.",
        [(30, "chair"), (31, "chair"), (15, "mirror")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 30})

    assert decision.blocked is False
    assert decision.expected_category == "chair"
    assert decision.submitted_category == "chair"


def test_target_category_guard_ignores_relative_that_has_anchor() -> None:
    rs = _runtime_with_categories(
        "Select the shelf that has a plant on top of it.",
        [(24, "shelf"), (9, "plant")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 24})

    assert decision.blocked is False
    assert decision.expected_category == "bookshelf"
    assert decision.submitted_category == "bookshelf"


def test_target_category_guard_prefers_monitor_before_closest_bookshelf_anchor() -> None:
    rs = _runtime_with_categories(
        "The monitor closest to the bookshelf",
        [(2, "monitor"), (3, "monitor"), (4, "bookshelf")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 2})

    assert decision.blocked is False
    assert decision.expected_category == "monitor"
    assert decision.submitted_category == "monitor"


def test_target_category_guard_uses_looking_at_set_with_pronoun_target() -> None:
    rs = _runtime_with_categories(
        "When looking at the three storage bins, it is the highest, "
        "on top of the other two, and closest to the clock on the wall.",
        [(30, "storage bin"), (31, "storage bin"), (32, "storage bin"), (9, "clock")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 32})

    assert decision.blocked is False
    assert decision.expected_category == "storage bin"
    assert decision.submitted_category == "storage bin"


def test_target_category_guard_preserves_demonstrative_target_before_context_set() -> None:
    rs = _runtime_with_categories(
        "If facing this white board the chairs at the table will be in this "
        "order from left to right. The one on the left will be pushed in and "
        "the other two chairs will be pushed out close to the white board.",
        [(13, "whiteboard"), (16, "chair"), (4, "chair"), (7, "chair")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 13})

    assert decision.blocked is False
    assert decision.expected_category == "whiteboard"
    assert decision.submitted_category == "whiteboard"

    wrong_category = evaluate_target_category_guard(rs, {"proposal_id": 16})

    assert wrong_category.blocked is True
    assert wrong_category.expected_category == "whiteboard"
    assert wrong_category.submitted_category == "chair"


def test_target_category_guard_skips_generic_object_for_is_a_category() -> None:
    rs = _runtime_with_categories(
        "The object is a fully closed door.",
        [(5, "object"), (2, "door")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 2})

    assert decision.blocked is False
    assert decision.expected_category == "door"
    assert decision.submitted_category == "door"


def test_target_category_guard_skips_generic_object_you_are_looking_for() -> None:
    rs = _runtime_with_categories(
        "The object you are looking for is a kitchen cabinet. The cabinet "
        "you are looking for is directly over the stove and contains a white "
        "microwave oven.",
        [(8, "kitchen cabinets"), (13, "microwave")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 13})

    assert decision.blocked is True
    assert decision.expected_category == "kitchen cabinet"
    assert decision.submitted_category == "microwave"


def test_target_category_guard_reads_cabinet_head_with_positional_prefix() -> None:
    rs = _runtime_with_categories(
        "the top left of the cabinets near the fridge",
        [(14, "kitchen cabinet"), (16, "refrigerator")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 16})

    assert decision.blocked is True
    assert decision.expected_category == "kitchen cabinet"
    assert decision.submitted_category == "refrigerator"


def test_target_category_guard_ignores_orientation_wall_when_target_head_absent() -> None:
    rs = _runtime_with_categories(
        "When facing the wall of windows, the furthest on the right.",
        [(29, "window"), (23, "wall")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 29})

    assert decision.blocked is False
    assert decision.expected_category is None
    assert decision.submitted_category == "window"


def test_target_category_guard_prefers_this_cart_over_context_wall() -> None:
    rs = _runtime_with_categories(
        "There is a green wall right above this cart",
        [(23, "cart"), (10, "wall")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 23})

    assert decision.blocked is False
    assert decision.expected_category == "cart"
    assert decision.submitted_category == "cart"


def test_target_category_guard_blocks_wall_for_this_cart_query() -> None:
    rs = _runtime_with_categories(
        "There is a green wall right above this cart",
        [(23, "cart"), (10, "wall")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 10})

    assert decision.blocked is True
    assert decision.expected_category == "cart"
    assert decision.submitted_category == "wall"


def test_target_category_guard_later_option_head_overrides_want_context() -> None:
    rs = _runtime(
        "Staring at both beds from their foot, you want the bed on the right. "
        "The pillow is the back right option."
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is True
    assert decision.expected_category == "pillow"
    assert decision.submitted_category == "bed"


def test_target_category_guard_later_context_description_does_not_override_want() -> None:
    rs = _runtime_with_categories(
        "Staring at both beds from their foot, you want the bed on the right. "
        "The wall behind it is blue.",
        [(8, "bed"), (10, "wall")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 8})

    assert decision.blocked is False
    assert decision.expected_category == "bed"
    assert decision.submitted_category == "bed"


def test_target_category_guard_find_bookshelf_overrides_context_wall() -> None:
    rs = _runtime_with_categories(
        "There is one wall of books that at a unique angle. Go to the end of "
        "that wall that is closest to the two small tables containing books. "
        "Find the bookshelf against the angled wall that is just to the left "
        "of the furthest right bookshelf.",
        [(106, "bookshelf"), (6, "wall")],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 106})

    assert decision.blocked is False
    assert decision.expected_category == "bookshelf"
    assert decision.submitted_category == "bookshelf"


def test_target_category_guard_allows_matching_head_category() -> None:
    rs = _runtime("The pillow is the back right option.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 46})

    assert decision.blocked is False
    assert decision.expected_category == "pillow"
    assert decision.submitted_category == "pillow"


def test_target_category_guard_allows_subtype_for_generic_chair_query() -> None:
    rs = _runtime_with_categories(
        "the chair behind the desk closest to the window",
        [
            (6, "office chair"),
            (33, "chair"),
            (4, "desk"),
            (5, "window"),
        ],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert decision.blocked is False
    assert decision.expected_category == "chair"
    assert decision.submitted_category == "office chair"


def test_target_category_guard_blocks_generic_chair_for_specific_office_chair_query() -> None:
    rs = _runtime_with_categories(
        "the office chair behind the desk closest to the window",
        [
            (6, "office chair"),
            (33, "chair"),
            (4, "desk"),
            (5, "window"),
        ],
    )

    decision = evaluate_target_category_guard(rs, {"proposal_id": 33})

    assert decision.blocked is True
    assert decision.expected_category == "office chair"
    assert decision.submitted_category == "chair"


def test_target_category_guard_treats_desk_chair_as_chair_not_desk() -> None:
    rs = _runtime_with_categories(
        "all black desk chair",
        [
            (3, "office chair"),
            (6, "desk"),
        ],
    )

    chair_decision = evaluate_target_category_guard(rs, {"proposal_id": 3})
    desk_decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

    assert chair_decision.blocked is False
    assert chair_decision.expected_category == "chair"
    assert chair_decision.submitted_category == "office chair"
    assert desk_decision.blocked is True
    assert desk_decision.expected_category == "chair"
    assert desk_decision.submitted_category == "desk"


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

    decision = evaluate_target_category_guard(rs, {"proposal_id": "not-a-pid"})

    assert decision.blocked is False
    assert decision.submitted_pid is None
    assert decision.expected_category is None
    assert decision.submitted_category is None


def test_target_category_guard_blocks_numeric_string_wrong_category() -> None:
    rs = _runtime("Choose the pillow.")

    decision = evaluate_target_category_guard(rs, {"proposal_id": "8"})

    assert decision.blocked is True
    assert decision.submitted_pid == 8
    assert decision.expected_category == "pillow"
    assert decision.submitted_category == "bed"


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


def test_target_category_guard_passes_anchor_only_category_mention() -> None:
    rs = _runtime("the one next to the bed")

    decision = evaluate_target_category_guard(rs, {"proposal_id": 6})

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


def test_submit_final_target_category_block_records_neutral_later_guard_fields() -> None:
    rs = _runtime("Choose the pillow.")
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 8},
            "rationale": "wrong category",
            "evidence_refs": [],
        }
    )

    submit_record = rs.tool_trace[-1]
    assert response.startswith("TARGET_CATEGORY_GUARD")
    assert submit_record.tool_input["target_category_guard_blocked"] is True
    assert submit_record.tool_input["tadg_blocked"] is False
    assert submit_record.tool_input["no_match_guard_blocked"] is False
    assert submit_record.tool_input["evidence_frame_guard_blocked"] is False
