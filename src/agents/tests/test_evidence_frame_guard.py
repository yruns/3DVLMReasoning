"""Unit tests for the VG final evidence-frame consistency guard."""

from __future__ import annotations

import json
from pathlib import Path

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
from agents.skills.evidence_frame_guard import evaluate_evidence_frame_guard


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
            tool_name="view_keyframe_marked",
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
) -> None:
    response = {
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
            },
            response_text=json.dumps(response),
        )
    )


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
