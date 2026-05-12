"""Unit tests for the ScanRefer/VG no-match candidate guard."""

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
from agents.skills.no_match_guard import evaluate_no_match_guard


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
    rs.use_no_match_candidate_guard = flag_on
    return rs


def _record_find(
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
                    "available_categories": ["chair", "door", "shelf"],
                }
            ),
        )
    )


def _record_view(
    rs: Stage2RuntimeState,
    *,
    frame_id: int,
    visible_ids: list[int],
    categories: list[str],
) -> None:
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="view_keyframe_marked",
            tool_input={"frame_id": frame_id},
            response_text=(
                f"frame_id={frame_id} marked image at frame_{frame_id}.png; "
                f"visible_proposals={visible_ids}; categories={categories}"
            ),
        )
    )


def _record_inspect(rs: Stage2RuntimeState, proposal_id: int) -> None:
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="inspect_proposal",
            tool_input={"proposal_id": proposal_id},
            response_text=json.dumps(
                {
                    "proposal_id": proposal_id,
                    "category": "chair",
                    "frames_appeared": [10],
                }
            ),
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


def test_no_match_guard_disabled_passes() -> None:
    rs = _runtime(flag_on=False)
    _record_find(rs, category="chair", proposal_ids=[6, 22])

    decision = evaluate_no_match_guard(rs, {"proposal_id": -1, "confidence": 0.0})

    assert decision.blocked is False


def test_no_match_guard_ignores_non_failed_submission() -> None:
    rs = _runtime()
    _record_find(rs, category="chair", proposal_ids=[6, 22])

    decision = evaluate_no_match_guard(rs, {"proposal_id": 6, "confidence": 0.7})

    assert decision.blocked is False


def test_no_match_guard_blocks_failed_submission_when_category_candidates_exist() -> (
    None
):
    rs = _runtime()
    _record_find(rs, category="chair", proposal_ids=[6, 22])

    decision = evaluate_no_match_guard(rs, {"proposal_id": -1, "confidence": 0.0})

    assert decision.blocked is True
    assert decision.category_candidate_ids == (6, 22)
    assert "chair" in decision.message
    assert "proposal_id=-1" in decision.message


def test_no_match_guard_blocks_failed_submission_with_uninspected_marked_candidates() -> (
    None
):
    rs = _runtime()
    _record_view(
        rs,
        frame_id=51,
        visible_ids=[7, 5, 10],
        categories=["refrigerator", "door", "shelf"],
    )
    _record_inspect(rs, 7)

    decision = evaluate_no_match_guard(rs, {"proposal_id": -1, "confidence": 0.0})

    assert decision.blocked is True
    assert decision.viewed_uninspected_ids == (5, 10)
    assert "frame 51" in decision.message
    assert "shelf" in decision.message
    assert "labels are weak priors" in decision.message


def test_no_match_guard_prioritizes_recent_marked_frames_under_cap() -> None:
    rs = _runtime()
    rs.no_match_guard_max_viewed = 2
    _record_view(
        rs,
        frame_id=121,
        visible_ids=[47, 14, 48],
        categories=["doorframe", "box", "light switch"],
    )
    _record_view(
        rs,
        frame_id=51,
        visible_ids=[7, 5, 10],
        categories=["refrigerator", "door", "shelf"],
    )

    decision = evaluate_no_match_guard(rs, {"proposal_id": -1, "confidence": 0.0})

    assert decision.blocked is True
    assert decision.viewed_uninspected_ids == (7, 5)
    assert "frame 51" in decision.message
    assert "doorframe" not in decision.message


def test_submit_final_blocks_no_match_and_does_not_finalize(tmp_path: Path) -> None:
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_find(rs, category="chair", proposal_ids=[6, 22])
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {"proposal_id": -1, "confidence": 0.0},
            "rationale": "GT not in proposal pool",
            "evidence_refs": [],
        }
    )

    assert response.startswith("NO_MATCH_GUARD:")
    assert rs.final_submission is None
    submit_records = [t for t in rs.tool_trace if t.tool_name == "submit_final"]
    assert len(submit_records) == 1
    assert submit_records[0].tool_input["no_match_guard_blocked"] is True
    assert submit_records[0].tool_input["no_match_guard_category_candidate_ids"] == [
        6,
        22,
    ]


def test_no_match_guard_keeps_blocking_when_category_candidates_remain() -> None:
    rs = _runtime()
    rs.no_match_guard_max_repeats = 2
    _record_find(rs, category="chair", proposal_ids=[6])

    first = evaluate_no_match_guard(rs, {"proposal_id": -1, "confidence": 0.0})
    second = evaluate_no_match_guard(rs, {"proposal_id": -1, "confidence": 0.0})

    assert first.blocked is True
    assert second.blocked is True
    assert second.force_passed is False
    assert "no-match is not an accepted final answer" in second.message
    assert "proposal_id from the candidate set instead: 6" in second.message


def test_no_match_guard_does_not_force_pass_with_uninspected_marked_candidates() -> (
    None
):
    rs = _runtime()
    rs.no_match_guard_max_repeats = 1
    _record_view(
        rs,
        frame_id=51,
        visible_ids=[7, 10],
        categories=["refrigerator", "shelf"],
    )

    decision = evaluate_no_match_guard(rs, {"proposal_id": -1, "confidence": 0.0})

    assert decision.blocked is True
    assert decision.force_passed is False
    assert decision.viewed_uninspected_ids == (7, 10)
