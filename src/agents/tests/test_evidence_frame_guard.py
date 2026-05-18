"""Unit tests for the VG final evidence-frame consistency guard."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.core.agent_config import Stage2TaskType
from agents.core.response_schema import Stage2ToolObservation
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.skills import (
    PACKS,
    FinalizerSpec,
    SkillSpec,
    TaskPack,
    register_pack,
)
from agents.skills.chassis_tools import build_chassis_tools
from agents.skills.evidence_frame_guard import (
    _cited_frame_ids,
    _direction_from_text,
    evaluate_evidence_frame_guard,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    PACKS.clear()
    yield
    PACKS.clear()


def _runtime(*, flag_on: bool = True) -> Stage2RuntimeState:
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="a snack machine to the left of the front entrance"
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = flag_on
    return rs


def _record_view(
    rs: Stage2RuntimeState,
    *,
    frame_id: int,
    visible_ids: list[int],
    categories: list[str],
    left_to_right: list[str] | None = None,
    boxes_2d: dict[int, list[int]] | None = None,
) -> None:
    geometry = ""
    if left_to_right is not None:
        geometry += f"; left_to_right={left_to_right}"
    if boxes_2d is not None:
        geometry += f"; boxes_2d={boxes_2d}"
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="mark_frame_with_bbox",
            tool_input={"frame_id": frame_id},
            response_text=(
                f"frame_id={frame_id} marked image at frame_{frame_id}.png; "
                f"visible_proposals={visible_ids}; categories={categories}"
                f"{geometry}"
            ),
        )
    )


def _record_category_candidates(
    rs: Stage2RuntimeState,
    *,
    category: str,
    proposal_ids: list[int],
) -> None:
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="find_proposals_by_category",
            tool_input={"category": category},
            response_text=json.dumps(
                {
                    "category": category,
                    "proposal_ids": proposal_ids,
                    "available_categories": [category],
                }
            ),
        )
    )


def _record_spatial_compare(
    rs: Stage2RuntimeState,
    *,
    candidate_ids: list[int],
    anchor_id: int,
    relation: str,
    ranked_ids: list[int],
    evidence_id: str | None = None,
) -> str:
    if evidence_id is None:
        compare_index = sum(
            1
            for entry in rs.tool_trace
            if entry.tool_name == "compare_proposals_spatial"
        )
        evidence_id = f"compare_proposals_spatial:{compare_index}"
    response = {
        "evidence_id": evidence_id,
        "candidate_ids": candidate_ids,
        "anchor_id": anchor_id,
        "relation": relation,
        "ranked_ids": ranked_ids,
    }
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="compare_proposals_spatial",
            tool_input={
                "candidate_ids": candidate_ids,
                "anchor_id": anchor_id,
                "relation": relation,
                "evidence_id": evidence_id,
            },
            response_text=json.dumps(response),
        )
    )
    return evidence_id


def _register_vg_stub_pack(tmp_path: Path) -> None:
    body = tmp_path / "vg_grounding_playbook.md"
    body.write_text("# VG Grounding Playbook\n", encoding="utf-8")
    register_pack(
        TaskPack(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            tool_builder=lambda r: [],
            skills=[
                SkillSpec(
                    name="vg-grounding-playbook",
                    description="VG main loop.",
                    body_path=body,
                    task_types={Stage2TaskType.VISUAL_GROUNDING},
                ),
            ],
            finalizer=FinalizerSpec(
                payload_model=dict,
                validator=lambda payload, runtime: payload,
                adapter=lambda payload, runtime: {"answer": payload},
            ),
            required_primary_skill="vg-grounding-playbook",
            required_extra_metadata=[],
            ctx_factory=lambda b: object(),
        )
    )


def test_evidence_frame_guard_disabled_passes() -> None:
    rs = _runtime(flag_on=False)
    _record_view(
        rs,
        frame_id=57,
        visible_ids=[7, 10, 59],
        categories=["refrigerator", "shelf", "cabinet"],
    )

    decision = evaluate_evidence_frame_guard(
        rs,
        {"proposal_id": 26, "confidence": 0.62},
        rationale="Frame 57 clearly shows the snack machine.",
    )

    assert decision.blocked is False


def test_evidence_frame_guard_blocks_submit_not_visible_in_cited_frame(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_view(
        rs,
        frame_id=57,
        visible_ids=[7, 10, 59],
        categories=["refrigerator", "shelf", "cabinet"],
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 26, "confidence": 0.62},
            "rationale": "Frame 57 clearly shows the snack machine on the left.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("EVIDENCE_FRAME_GUARD:")
    assert "26" in response
    assert "10:shelf" in response
    assert rs.final_submission is None
    submit_records = [t for t in rs.tool_trace if t.tool_name == "submit_final"]
    assert submit_records[0].tool_input["evidence_frame_guard_blocked"] is True
    assert submit_records[0].tool_input["evidence_frame_guard_cited_frame_ids"] == [57]


def test_evidence_frame_guard_allows_submit_visible_in_cited_frame(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_view(
        rs,
        frame_id=57,
        visible_ids=[7, 10, 59],
        categories=["refrigerator", "shelf", "cabinet"],
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 10, "confidence": 0.72},
            "rationale": "Frame 57 clearly shows the snack machine on the left.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
    assert rs.final_submission == {"answer": {"proposal_id": 10, "confidence": 0.72}}

    relation_evidence = {
        "relation": "left_of",
        "anchor_id": 7,
        "candidate_ids": [10],
        "ranked_ids": [10],
    }
    duplicate_response = submit_final.invoke(
        {
            "payload": {"proposal_id": 10, "confidence": 0.72},
            "rationale": "Frame 57 still shows proposal 10.",
            "evidence_refs": [],
            "relation_evidence": relation_evidence,
        }
    )

    assert duplicate_response.startswith("ALREADY_SUBMITTED:")
    submit_records = [t for t in rs.tool_trace if t.tool_name == "submit_final"]
    duplicate_record = submit_records[-1].tool_input
    assert duplicate_record["relation_evidence"] == relation_evidence
    assert duplicate_record["tadg_blocked"] is False
    assert duplicate_record["no_match_guard_blocked"] is False
    assert duplicate_record["evidence_frame_guard_blocked"] is False


def test_evidence_frame_guard_blocks_spatial_rationale_when_anchor_missing_from_cited_frame(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    rs.bundle.stage1_query = "this is a brown cabinet. it is above a refrigerator."
    _record_view(
        rs,
        frame_id=34,
        visible_ids=[25, 59, 29],
        categories=["kitchen cabinet", "kitchen cabinet", "dishwasher"],
    )
    _record_spatial_compare(
        rs,
        candidate_ids=[25, 59, 41, 46],
        anchor_id=43,
        relation="above",
        ranked_ids=[41, 25, 59, 46],
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 25, "confidence": 0.84},
            "rationale": (
                "Frame 34 clearly shows proposal 25 as the brown cabinet "
                "above the refrigerator."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("EVIDENCE_FRAME_GUARD:")
    assert "anchor proposal 43" in response
    assert "34" in response
    assert rs.final_submission is None


def test_evidence_frame_guard_allows_nested_same_category_anchor_frame() -> None:
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query=(
                "it is the office chair, next to the one in front of the "
                "computer monitor."
            )
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_spatial_compare(
        rs,
        candidate_ids=[40, 41],
        anchor_id=4,
        relation="next_to",
        ranked_ids=[40, 41],
    )
    _record_view(
        rs,
        frame_id=25,
        visible_ids=[40, 41],
        categories=["office chair", "office chair"],
    )

    decision = evaluate_evidence_frame_guard(
        rs,
        {"proposal_id": 41, "confidence": 0.93},
        rationale=(
            "Frame 25 shows proposal #41 next to proposal #40. Proposal #40 "
            "is the one in front of the computer monitor, so #41 is the office "
            "chair next to the one in front of the monitor."
        ),
        evidence_refs=[],
    )

    assert decision.blocked is False
    assert decision.submitted_pid == 41
    assert decision.cited_frame_ids == (25,)


def test_evidence_frame_guard_prefers_bound_relation_evidence_over_latest_compare() -> None:
    rs = _runtime()
    evidence_id = _record_spatial_compare(
        rs,
        candidate_ids=[8, 11],
        anchor_id=7,
        relation="left_of",
        ranked_ids=[8, 11],
    )
    _record_spatial_compare(
        rs,
        candidate_ids=[8],
        anchor_id=43,
        relation="left_of",
        ranked_ids=[8],
    )
    _record_view(
        rs,
        frame_id=57,
        visible_ids=[8, 7],
        categories=["dresser", "bed"],
    )

    decision = evaluate_evidence_frame_guard(
        rs,
        {"proposal_id": 8, "confidence": 0.8},
        rationale="proposal 8 is left of the bed in frame 57",
        evidence_refs=[{"frame_id": 57}],
        relation_evidence={"evidence_id": evidence_id},
    )

    assert decision.blocked is False
    assert decision.submitted_pid == 8


def test_evidence_frame_guard_ignores_malformed_bound_relation_evidence() -> None:
    rs = _runtime()
    _record_spatial_compare(
        rs,
        candidate_ids=[8],
        anchor_id=43,
        relation="left_of",
        ranked_ids=[8],
    )
    _record_view(
        rs,
        frame_id=57,
        visible_ids=[8, 7],
        categories=["dresser", "bed"],
    )

    decision = evaluate_evidence_frame_guard(
        rs,
        {"proposal_id": 8, "confidence": 0.8},
        rationale="proposal 8 is left of the bed in frame 57",
        evidence_refs=[{"frame_id": 57}],
        relation_evidence={
            "relation": "left_of",
            "anchor_id": 7,
            "candidate_ids": [8],
            "ranked_ids": ["8"],
        },
    )

    assert decision.blocked is True
    assert "anchor proposal 43" in decision.message


def test_evidence_frame_guard_ignores_unknown_bound_evidence_id() -> None:
    rs = _runtime()
    _record_spatial_compare(
        rs,
        candidate_ids=[8],
        anchor_id=43,
        relation="left_of",
        ranked_ids=[8],
    )
    _record_view(
        rs,
        frame_id=57,
        visible_ids=[8, 7],
        categories=["dresser", "bed"],
    )

    decision = evaluate_evidence_frame_guard(
        rs,
        {"proposal_id": 8, "confidence": 0.8},
        rationale="proposal 8 is left of the bed in frame 57",
        evidence_refs=[{"frame_id": 57}],
        relation_evidence={"evidence_id": "compare_proposals_spatial:missing"},
    )

    assert decision.blocked is True
    assert "anchor proposal 43" in decision.message


def test_evidence_frame_guard_allows_near_relation_when_spatial_rank_supports_target(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query=(
                "this is a dresser in the bedroom. it is near the right foot "
                "of the bed."
            )
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_category_candidates(rs, category="dresser", proposal_ids=[8, 11])
    _record_spatial_compare(
        rs,
        candidate_ids=[8, 11],
        anchor_id=7,
        relation="near",
        ranked_ids=[8, 11],
    )
    _record_view(
        rs,
        frame_id=145,
        visible_ids=[8, 23, 27, 35, 42, 4],
        categories=["dresser", "bottle", "book", "purse", "book", "mirror"],
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 8, "confidence": 0.69},
            "rationale": (
                "Proposal 8 is the dresser visible in frame 145 and is near "
                "the right foot of the bed according to the latest spatial "
                "comparison."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
    assert rs.final_submission == {"answer": {"proposal_id": 8, "confidence": 0.69}}


def test_evidence_frame_guard_blocks_left_query_when_comparable_left_mark_exists(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_view(
        rs,
        frame_id=53,
        visible_ids=[35, 51, 7, 54, 10, 60, 0, 53, 59, 22, 31],
        categories=[
            "board",
            "object",
            "refrigerator",
            "doorframe",
            "shelf",
            "tv",
            "trash can",
            "doorframe",
            "cabinet",
            "bulletin board",
            "mailbox",
        ],
        left_to_right=[
            "0:trash can@x=377.0",
            "10:shelf@x=612.0",
            "7:refrigerator@x=1006.0",
            "59:cabinet@x=1006.0",
        ],
        boxes_2d={
            0: [202, 684, 552, 967],
            10: [422, 0, 802, 599],
            7: [717, 0, 1295, 675],
            59: [717, 0, 1295, 675],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 7, "confidence": 0.18},
            "rationale": (
                "Frame 53 shows proposal 7 as the right-hand machine-like object."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("EVIDENCE_FRAME_GUARD:")
    assert "left/right" in response
    assert "10:shelf" in response
    assert rs.final_submission is None


def test_evidence_frame_guard_ignores_small_left_mark_for_left_query(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_view(
        rs,
        frame_id=53,
        visible_ids=[0, 7],
        categories=["trash can", "refrigerator"],
        left_to_right=["0:trash can@x=377.0", "7:refrigerator@x=1006.0"],
        boxes_2d={
            0: [202, 684, 552, 967],
            7: [717, 0, 1295, 675],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 7, "confidence": 0.18},
            "rationale": "Frame 53 shows proposal 7 as the best visible candidate.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_ignores_non_target_left_marks_when_target_candidate_is_submitted(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query=(
                "a modern light wood color table placed against the wall. "
                "it is to the left of the tv on the wall."
            )
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_category_candidates(rs, category="table", proposal_ids=[1, 14, 15])
    _record_view(
        rs,
        frame_id=4,
        visible_ids=[1, 16, 35, 29],
        categories=["table", "coffee table", "end table", "tv"],
        left_to_right=[
            "16:coffee table@x=202.0",
            "35:end table@x=202.0",
            "1:table@x=350.0",
        ],
        boxes_2d={
            16: [0, 313, 404, 794],
            35: [0, 313, 404, 794],
            1: [0, 442, 700, 967],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 1, "confidence": 0.87},
            "rationale": (
                "Proposal 1 is the large light-wood table clearly visible in "
                "marked frame 4, positioned directly left of the wall-mounted "
                "TV proposal 29."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
    assert rs.final_submission == {"answer": {"proposal_id": 1, "confidence": 0.87}}


def test_evidence_frame_guard_does_not_replace_target_with_anchor_on_requested_side(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query=(
                "a chair cushion with a high back resting on the floor. it is "
                "underneath a white board and to the right of a book case."
            )
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_view(
        rs,
        frame_id=151,
        visible_ids=[6, 18, 19],
        categories=["chair", "bookshelf", "whiteboard"],
        left_to_right=[
            "6:chair@x=931.0",
            "18:bookshelf@x=1037.5",
            "19:whiteboard@x=1167.5",
        ],
        boxes_2d={
            6: [567, 141, 1295, 967],
            18: [780, 0, 1295, 638],
            19: [1040, 0, 1295, 967],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 6, "confidence": 0.68},
            "rationale": (
                "Frame 151 shows proposal 6 as the chair under the whiteboard "
                "and to the right of the bookcase."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_does_not_replace_target_with_non_target_category(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="Find the table on the far right of the bench seat."
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_view(
        rs,
        frame_id=6,
        visible_ids=[5, 8],
        categories=["table", "couch"],
        left_to_right=[
            "5:table@x=760.0",
            "8:couch@x=940.0",
        ],
        boxes_2d={
            5: [650, 420, 870, 760],
            8: [820, 360, 1060, 780],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 5, "confidence": 0.7},
            "rationale": (
                "Frame 6 shows proposal 5 as the table on the far right of "
                "the bench seat."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
    assert rs.final_submission == {"answer": {"proposal_id": 5, "confidence": 0.7}}


def test_evidence_frame_guard_filters_anchor_from_cabinet_left_alternatives(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="the top left of the cabinets near the fridge"
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_category_candidates(rs, category="kitchen cabinet", proposal_ids=[15, 14, 11])
    _record_view(
        rs,
        frame_id=81,
        visible_ids=[16, 15, 14, 11],
        categories=["refrigerator", "kitchen cabinets", "kitchen cabinet", "kitchen cabinet"],
        left_to_right=[
            "16:refrigerator@x=120.0",
            "15:kitchen cabinets@x=260.0",
            "14:kitchen cabinet@x=420.0",
            "11:kitchen cabinet@x=580.0",
        ],
        boxes_2d={
            16: [20, 100, 220, 700],
            15: [230, 80, 330, 300],
            14: [360, 80, 480, 300],
            11: [520, 80, 640, 300],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 15, "confidence": 0.83},
            "rationale": (
                "Frame 81 shows proposal 15 as the top-left cabinet in the "
                "cabinet group near the fridge."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_does_not_force_target_leftmost_for_anchor_left_of_it(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query=(
                "the object is a black table chair with a red table chair to "
                "the left of it. it is located directly in front of the white "
                "writing board on the right end of the table."
            )
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_category_candidates(rs, category="chair", proposal_ids=[3, 5, 12])
    _record_view(
        rs,
        frame_id=18,
        visible_ids=[12, 1, 3, 5],
        categories=["chair", "whiteboard", "chair", "chair"],
        left_to_right=[
            "5:chair@x=160.0",
            "3:chair@x=250.0",
            "12:chair@x=340.0",
            "1:whiteboard@x=390.0",
        ],
        boxes_2d={
            5: [80, 460, 240, 760],
            3: [170, 450, 330, 760],
            12: [260, 450, 420, 760],
            1: [310, 80, 680, 420],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 3, "confidence": 0.66},
            "rationale": (
                "Frame 18 shows proposal 3 as the black chair in front of the "
                "whiteboard, with proposal 5 as the red chair to the left of it."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
    assert rs.final_submission == {"answer": {"proposal_id": 3, "confidence": 0.66}}


def test_evidence_frame_guard_rejects_plain_rgb_only_evidence(tmp_path: Path) -> None:
    """A selector that injected an RGB frame containing the submitted proposal does NOT
    satisfy the VG evidence requirement — only `mark_frame_with_bbox` counts.
    """
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="select_by_text",
            tool_input={"query": "cabinet", "k": 3, "hidden_categories": []},
            response_text=json.dumps(
                {
                    "hypothesis_summary": "",
                    "frames": [
                        {
                            "frame_id": 42,
                            "visible_proposal_ids": [4],
                            "selected_because": "select_by_text",
                            "image_path": "/tmp/fake.png",
                            "already_seen": False,
                        }
                    ],
                }
            ),
        )
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 4, "confidence": 0.55},
            "rationale": "Frame 42 shows proposal 4 from the selector output.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("EVIDENCE_FRAME_GUARD:")
    assert rs.final_submission is None
    submit_records = [t for t in rs.tool_trace if t.tool_name == "submit_final"]
    assert submit_records[0].tool_input["evidence_frame_guard_blocked"] is True


def test_evidence_frame_guard_query_right_one_beats_contrastive_rationale_left(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="When facing the two tables choose the one on the right."
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_view(
        rs,
        frame_id=8,
        visible_ids=[16, 17],
        categories=["table", "table"],
        left_to_right=["16:table@x=200.0", "17:table@x=500.0"],
        boxes_2d={16: [100, 200, 300, 650], 17: [400, 200, 620, 650]},
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 17, "confidence": 0.7},
            "rationale": (
                "Frame 8 shows proposal 17 as the right table; proposal 16 is "
                "the left-hand alternative."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_query_right_option_beats_left_alternative(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="When facing the two tables choose the right option."
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_view(
        rs,
        frame_id=8,
        visible_ids=[16, 17],
        categories=["table", "table"],
        left_to_right=["16:table@x=200.0", "17:table@x=500.0"],
        boxes_2d={16: [100, 200, 300, 650], 17: [400, 200, 620, 650]},
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 17, "confidence": 0.7},
            "rationale": (
                "Frame 8 shows proposal 17 as the right table; proposal 16 is "
                "the left-hand alternative."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_ignores_rationale_only_side_when_query_has_no_side(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(stage1_query="Choose the table by itself.")
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_category_candidates(rs, category="table", proposal_ids=[10, 25])
    _record_view(
        rs,
        frame_id=45,
        visible_ids=[10, 25],
        categories=["table", "table"],
        left_to_right=["10:table@x=200.0", "25:table@x=520.0"],
        boxes_2d={10: [100, 200, 300, 650], 25: [420, 200, 620, 650]},
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 25, "confidence": 0.7},
            "rationale": (
                "Frame 45 shows proposal 25 by itself. The other table cluster "
                "is in the left wall area, so proposal 25 is isolated."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_does_not_override_closest_to_with_rationale_side(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(stage1_query="The correct chair is closer to the TV.")
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    evidence_id = _record_spatial_compare(
        rs,
        candidate_ids=[2, 3],
        anchor_id=9,
        relation="closest_to",
        ranked_ids=[2, 3],
    )
    _record_category_candidates(rs, category="chair", proposal_ids=[2, 3])
    _record_view(
        rs,
        frame_id=18,
        visible_ids=[2, 3, 9],
        categories=["chair", "chair", "tv"],
        left_to_right=[
            "2:chair@x=220.0",
            "9:tv@x=500.0",
            "3:chair@x=760.0",
        ],
        boxes_2d={
            2: [120, 250, 320, 700],
            9: [420, 120, 580, 360],
            3: [650, 240, 870, 700],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 2, "confidence": 0.84},
            "rationale": (
                "Frame 18 shows proposal 2 as the closer chair to the TV. "
                "Proposal 3 is the right-side/deeper alternative."
            ),
            "evidence_refs": [],
            "relation_evidence": {"evidence_id": evidence_id},
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_does_not_apply_anchor_left_phrase_before_target_head(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query=(
                "on the larger desk to the left, the monitor furthest from the door"
            )
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_category_candidates(rs, category="monitor", proposal_ids=[2, 26])
    evidence_id = _record_spatial_compare(
        rs,
        candidate_ids=[2, 26],
        anchor_id=7,
        relation="farthest_from",
        ranked_ids=[26, 2],
    )
    _record_view(
        rs,
        frame_id=0,
        visible_ids=[2, 26, 10],
        categories=["monitor", "monitor", "desk"],
        left_to_right=[
            "2:monitor@x=180.0",
            "26:monitor@x=360.0",
            "10:desk@x=520.0",
        ],
        boxes_2d={
            2: [100, 220, 260, 540],
            26: [290, 220, 430, 540],
            10: [460, 320, 620, 700],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 26, "confidence": 0.74},
            "rationale": (
                "Frame 0 shows proposal 26 as the monitor on the larger left "
                "desk cluster, and the spatial comparison ranks it furthest "
                "from the door."
            ),
            "evidence_refs": [],
            "relation_evidence": {"evidence_id": evidence_id},
        }
    )

    assert response.startswith("submitted;")
    assert rs.final_submission == {"answer": {"proposal_id": 26, "confidence": 0.74}}


def test_evidence_frame_guard_does_not_treat_room_side_as_image_left(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query=(
                "The monitor you want is the one that is NOT closer to the "
                "window, on the left side of the room. It is facing a pair "
                "of desk chairs."
            )
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    _record_category_candidates(rs, category="monitor", proposal_ids=[22, 23])
    _record_view(
        rs,
        frame_id=12,
        visible_ids=[23, 22, 10],
        categories=["monitor", "monitor", "window"],
        left_to_right=[
            "23:monitor@x=240.0",
            "22:monitor@x=430.0",
            "10:window@x=620.0",
        ],
        boxes_2d={
            23: [170, 230, 310, 520],
            22: [360, 230, 500, 520],
            10: [560, 60, 700, 400],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 22, "confidence": 0.71},
            "rationale": (
                "Frame 12 shows proposal 22 in the left-side room monitor "
                "cluster facing the desk chairs; proposal 23 is the other "
                "monitor in the marked frame."
            ),
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
    assert rs.final_submission == {"answer": {"proposal_id": 22, "confidence": 0.71}}


def test_direction_from_text_covers_plan_target_side_vocabulary() -> None:
    right_phrases = [
        "choose the right option",
        "choose the right side",
        "choose the one in the right corner",
        "pick the item in the right",
        "move toward the right",
        "move towards the right",
        "choose the right one",
        "choose the one on the right",
        "choose the upper right object",
        "choose the back right table",
    ]
    left_phrases = [
        "choose the left option",
        "choose the left side",
        "choose the one in the left corner",
        "pick the item in the left",
        "move toward the left",
        "move towards the left",
        "choose the left one",
        "choose the one on the left",
        "choose the lower left object",
        "choose the front left table",
    ]

    assert [_direction_from_text(text) for text in right_phrases] == ["right"] * len(
        right_phrases
    )
    assert [_direction_from_text(text) for text in left_phrases] == ["left"] * len(
        left_phrases
    )


def test_evidence_frame_guard_uses_current_list_scene_proposals_shape(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(stage1_query="Choose the table on the right.")
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="list_scene_proposals",
            tool_input={"category": "table"},
            response_text=json.dumps(
                {
                    "count": 2,
                    "proposals": [
                        {"proposal_id": 16, "category": "table"},
                        {"proposal_id": 17, "category": "table"},
                    ],
                }
            ),
        )
    )
    _record_view(
        rs,
        frame_id=8,
        visible_ids=[16, 17, 99],
        categories=["table", "table", "table lamp"],
        left_to_right=[
            "16:table@x=200.0",
            "17:table@x=500.0",
            "99:table lamp@x=700.0",
        ],
        boxes_2d={
            16: [100, 200, 300, 650],
            17: [400, 200, 620, 650],
            99: [640, 180, 820, 650],
        },
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 17, "confidence": 0.7},
            "rationale": "Frame 8 shows proposal 17 as the table on the right.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_parses_plural_frame_citations(tmp_path: Path) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    rs.use_evidence_frame_guard = True
    _record_view(rs, frame_id=0, visible_ids=[23], categories=["trash can"])
    _record_view(rs, frame_id=1, visible_ids=[27], categories=["trash can"])
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 27, "confidence": 0.7},
            "rationale": "Frames 0, 1, and 2 show proposal 27 by the outlet.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")
    submit_records = [t for t in rs.tool_trace if t.tool_name == "submit_final"]
    assert submit_records[0].tool_input["evidence_frame_guard_cited_frame_ids"] == [
        0,
        1,
    ]


def test_evidence_frame_guard_blocks_side_answer_with_unmarked_same_category_candidate(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="If you are facing the windows, it is the one on the right."
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    rs.task_ctx = SimpleNamespace(
        proposals=[
            SimpleNamespace(id=23, category="window"),
            SimpleNamespace(id=24, category="window"),
        ]
    )
    _record_view(
        rs,
        frame_id=2,
        visible_ids=[24],
        categories=["window"],
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 24, "confidence": 0.7},
            "rationale": "Frame 2 shows proposal 24 as the window on the right.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("EVIDENCE_FRAME_GUARD:")
    assert "unmarked same-category candidate" in response
    assert "23:window" in response
    assert rs.final_submission is None


def test_evidence_frame_guard_allows_side_answer_after_candidates_are_marked(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="If you are facing the windows, it is the one on the right."
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    rs.task_ctx = SimpleNamespace(
        proposals=[
            SimpleNamespace(id=23, category="window"),
            SimpleNamespace(id=24, category="window"),
        ]
    )
    _record_view(rs, frame_id=2, visible_ids=[24], categories=["window"])
    _record_view(rs, frame_id=3, visible_ids=[23], categories=["window"])
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 24, "confidence": 0.7},
            "rationale": "Frame 2 shows proposal 24 as the window on the right.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("submitted;")


def test_evidence_frame_guard_merges_singular_plural_candidate_categories(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = Stage2RuntimeState(
        bundle=Stage2EvidenceBundle(
            stage1_query="The door closest to the other doors."
        )
    )
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_evidence_frame_guard = True
    rs.task_ctx = SimpleNamespace(
        proposals=[
            SimpleNamespace(id=8, category="doors"),
            SimpleNamespace(id=9, category="door"),
            SimpleNamespace(id=10, category="door"),
            SimpleNamespace(id=32, category="doors"),
            SimpleNamespace(id=35, category="doorframe"),
        ]
    )
    _record_view(rs, frame_id=36, visible_ids=[9, 10], categories=["door", "door"])
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 9, "confidence": 0.7},
            "rationale": "Frame 36 shows proposal 9 as the door closest to the other doors.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("EVIDENCE_FRAME_GUARD:")
    assert "8:doors" in response
    assert "32:doors" in response
    assert "35:doorframe" not in response


def test_cited_frame_ids_parses_oxford_comma_lists_and_dedupes_refs() -> None:
    assert _cited_frame_ids("Frames 0, 1, and 2 show the target.", []) == [0, 1, 2]
    assert _cited_frame_ids(
        "Frames 0, 1, and 2 show the target.",
        [{"frame_id": 1}, {"frame": 3}],
    ) == [0, 1, 2, 3]


def test_cited_frame_ids_preserves_mixed_singular_citation_order() -> None:
    assert _cited_frame_ids("frame_0 frame-id 1 frame 2", []) == [0, 1, 2]


def test_evidence_frame_guard_parses_frame_range_without_marked_evidence(
    tmp_path: Path,
) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    rs.use_evidence_frame_guard = True
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 27, "confidence": 0.7},
            "rationale": "Frames 0-2 show the final object.",
            "evidence_refs": [],
        }
    )

    assert response.startswith("EVIDENCE_FRAME_GUARD:")
    assert "0, 1, 2" in response
