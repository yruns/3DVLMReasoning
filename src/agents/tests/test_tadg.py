"""Unit tests for src/agents/skills/tadg.py — Tool-Answer Disagreement Gate.

Spec: tmp/tadg_spec.md.
"""

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
from agents.skills.tadg import (
    _SUPPORTED_TOOL_RELATIONS,
    _TOOL_RELATION_ALIASES,
    TADGDecision,
    evaluate_tadg,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    PACKS.clear()
    yield
    PACKS.clear()


def _bundle_with_query(query: str) -> Stage2EvidenceBundle:
    """Build a minimal Stage2EvidenceBundle with the raw query in the
    canonical bundle field for keyword scanning."""
    return Stage2EvidenceBundle(stage1_query=query, extra_metadata={})


def _runtime(
    *,
    flag_on: bool = True,
    bundle: Stage2EvidenceBundle | None = None,
) -> Stage2RuntimeState:
    if bundle is None:
        bundle = _bundle_with_query("the cabinet next to the stove")
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.use_tool_answer_disagreement_gate = flag_on
    # Leave window/max_repeats/override_min_chars at their dataclass
    # defaults (32/3/6) so tests exercise the production-default values.
    # Tests that need a different bound override explicitly.
    return rs


def _record_compare(
    rs: Stage2RuntimeState,
    *,
    relation: str,
    anchor_id: int,
    candidate_ids: list[int],
    ranked_ids: list[int],
    payload_relation: str | None = None,
    supporting_frame_counts: list[int] | None = None,
    contradicting_frame_counts: list[int] | None = None,
) -> None:
    request = {
        "relation": relation,
        "anchor_id": anchor_id,
        "candidate_ids": candidate_ids,
    }
    payload = {
        "anchor_id": anchor_id,
        "relation": payload_relation or relation,
        "ranked_ids": ranked_ids,
        "distances": [0.1 * (i + 1) for i in range(len(ranked_ids))],
    }
    if supporting_frame_counts is not None:
        payload["supporting_frame_counts"] = supporting_frame_counts
    if contradicting_frame_counts is not None:
        payload["contradicting_frame_counts"] = contradicting_frame_counts
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="compare_proposals_spatial",
            tool_input=request,
            response_text=json.dumps(payload),
        )
    )


def _record_category_lookup(
    rs: Stage2RuntimeState,
    *,
    category: str,
    proposal_ids: list[int],
) -> None:
    payload = {
        "category": category,
        "proposal_ids": proposal_ids,
        "available_categories": [category],
    }
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="find_proposals_by_category",
            tool_input={"category": category},
            response_text=json.dumps(payload),
        )
    )


def _record_view(rs: Stage2RuntimeState, frame_id: int) -> None:
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="mark_frame_with_bbox",
            tool_input={"frame_id": frame_id},
            response_text=f"frame_id={frame_id}",
        )
    )


# ------------------------------------------------------------------
# Direct evaluate_tadg tests (mock runtime)
# ------------------------------------------------------------------


def test_tadg_disabled_passes() -> None:
    rs = _runtime(flag_on=False)
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 19, "confidence": 0.7})
    assert decision.blocked is False
    assert decision.message == ""


def test_tadg_blocks_when_submitted_not_top1() -> None:
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[33, 14, 43],
        ranked_ids=[33, 14, 43],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is True
    assert decision.subcase == "rank_mismatch"
    assert decision.top1_pid == 33
    assert decision.submitted_pid == 14
    assert "ranked proposal 33" in decision.message
    assert rs.tadg_triggered is True


def test_tadg_passes_when_submitted_is_top1() -> None:
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=15,
        candidate_ids=[4, 2, 3],
        ranked_ids=[4, 2, 3],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 4, "confidence": 0.8})
    assert decision.blocked is False
    assert decision.message == ""
    assert rs.tadg_triggered is False


def test_tadg_blocks_left_right_submit_when_anchor_candidates_untested() -> None:
    rs = _runtime(
        bundle=_bundle_with_query("this office chair is left of the keyboard")
    )
    _record_category_lookup(rs, category="office chair", proposal_ids=[39, 50])
    _record_category_lookup(rs, category="keyboard", proposal_ids=[4, 8])
    _record_compare(
        rs,
        relation="left_of",
        anchor_id=4,
        candidate_ids=[39, 50],
        ranked_ids=[50, 39],
        supporting_frame_counts=[61, 0],
        contradicting_frame_counts=[1, 0],
    )

    decision = evaluate_tadg(rs, {"proposal_id": 50, "confidence": 0.89})

    assert decision.blocked is True
    assert "ambiguous anchor" in decision.message
    assert "proposal 8" in decision.message


def test_tadg_blocks_when_submitted_not_in_ranked() -> None:
    """S6 case: agent submits pid 19 not in the candidate set."""
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 12, 14, 27, 33, 36, 43],
        ranked_ids=[33, 14, 43, 27, 36, 12, 3, 2],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 19, "confidence": 0.7})
    assert decision.blocked is True
    assert decision.subcase == "not_in_candidates"
    assert "not in the spatial-tool's candidate set" in decision.message


def test_tadg_blocks_anchor_self_pick() -> None:
    """S39 case: agent submits the anchor itself."""
    rs = _runtime(bundle=_bundle_with_query("a chair next to another same chair"))
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=3,
        candidate_ids=[9, 1, 5],
        ranked_ids=[9, 1, 5],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 3, "confidence": 0.7})
    assert decision.blocked is True
    assert decision.subcase == "anchor_self"
    assert "anchor itself" in decision.message


def test_tadg_uses_bound_relation_evidence_over_stale_latest_compare() -> None:
    rs = _runtime(bundle=_bundle_with_query("the door nearest the small black chair"))
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=13,
        candidate_ids=[5],
        ranked_ids=[5],
    )
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=37,
        candidate_ids=[39, 5],
        ranked_ids=[39, 5],
    )

    decision = evaluate_tadg(
        rs,
        {"proposal_id": 39, "confidence": 0.8},
        relation_evidence={
            "relation": "closest_to",
            "anchor_id": 37,
            "candidate_ids": [39, 5],
            "ranked_ids": [39, 5],
        },
    )

    assert decision.blocked is False


def test_tadg_rejects_bound_anchor_self_without_forcing_rank1() -> None:
    rs = _runtime(bundle=_bundle_with_query("trash can next to two red chairs"))

    decision = evaluate_tadg(
        rs,
        {"proposal_id": 36, "confidence": 0.8},
        relation_evidence={
            "relation": "next_to",
            "anchor_id": 36,
            "candidate_ids": [39, 40, 41],
            "ranked_ids": [39, 40, 41],
        },
    )

    assert decision.blocked is True
    assert decision.subcase == "anchor_self"
    assert "target/anchor role" in decision.message
    assert "Revise to proposal 39" not in decision.message


def test_tadg_anchor_self_block_requests_role_repair_not_forced_top1() -> None:
    rs = _runtime(
        bundle=_bundle_with_query(
            "the cabinet left of the cabinet with two monitors"
        )
    )
    _record_compare(
        rs,
        relation="left_of",
        anchor_id=21,
        candidate_ids=[0, 22, 32],
        ranked_ids=[32, 0, 22],
    )

    decision = evaluate_tadg(rs, {"proposal_id": 21, "confidence": 0.72})

    assert decision.blocked is True
    assert decision.subcase == "anchor_self"
    assert "target/anchor role" in decision.message
    assert "rerun compare_proposals_spatial" in decision.message
    assert "Revise to proposal 32" not in decision.message


def test_tadg_matches_compare_relation_aliases() -> None:
    rs = _runtime(bundle=_bundle_with_query("the chair closest to the desk"))
    _record_compare(
        rs,
        relation="closer_to",
        payload_relation="closest_to",
        anchor_id=2,
        candidate_ids=[0, 1],
        ranked_ids=[1, 0],
    )

    decision = evaluate_tadg(rs, {"proposal_id": 0, "confidence": 0.72})

    assert decision.blocked is True
    assert decision.relation == "closest_to"
    assert decision.top1_pid == 1


@pytest.mark.parametrize(
    ("query", "relation"),
    [
        ("the keyboard closer to the cabinets", "closest_to"),
        ("the monitor furthest from the door", "farthest_from"),
    ],
)
def test_tadg_matches_query_relation_aliases(query: str, relation: str) -> None:
    rs = _runtime(bundle=_bundle_with_query(query))
    _record_compare(
        rs,
        relation=relation,
        anchor_id=2,
        candidate_ids=[0, 1],
        ranked_ids=[1, 0],
    )

    decision = evaluate_tadg(rs, {"proposal_id": 0, "confidence": 0.72})

    assert decision.blocked is True
    assert decision.relation == relation
    assert decision.top1_pid == 1


def test_tadg_silent_when_no_matching_relation() -> None:
    """Query has no spatial relation; the gate should never fire even
    if the agent ran compare_proposals_spatial as a sanity check."""
    bundle = _bundle_with_query("a brown chair")
    rs = _runtime(bundle=bundle)
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3],
        ranked_ids=[33, 14, 43],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is False


def test_tadg_silent_when_compare_outside_window() -> None:
    """Window=8: an old compare at index 0 is ignored if 8+ later
    tool_trace entries arrived after it."""
    rs = _runtime()
    rs.tadg_window = 8
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    # Push 9 view entries to bury the compare past the window.
    for fid in range(100, 109):
        _record_view(rs, fid)
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is False


def test_tadg_override_accepted() -> None:
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    decision = evaluate_tadg(
        rs,
        {"proposal_id": 14, "confidence": 0.7},
        tool_override_reason="rank-1 occluded by a wall",
    )
    assert decision.blocked is False
    assert "TADG_OVERRIDE_ACCEPTED" in decision.message
    assert rs.tadg_triggered is True
    assert rs.tool_override_reason == "rank-1 occluded by a wall"


def test_tadg_rejects_override_for_same_category_anchor_self_query() -> None:
    rs = _runtime(
        bundle=_bundle_with_query(
            "a black leather arm chair next to another same chair"
        )
    )
    _record_compare(
        rs,
        relation="next_to",
        anchor_id=5,
        candidate_ids=[1, 3, 9],
        ranked_ids=[1, 9, 3],
    )
    decision = evaluate_tadg(
        rs,
        {"proposal_id": 5, "confidence": 0.82},
        tool_override_reason="proposal 5 looks tighter in the current frame",
    )
    assert decision.blocked is True
    assert decision.subcase == "anchor_self"
    assert "same-category" in decision.message
    assert "proposal 1" in decision.message


def test_tadg_rejects_override_for_anchor_self_spatial_role_error() -> None:
    rs = _runtime(
        bundle=_bundle_with_query(
            "the black table chair with a red table chair to the left of it"
        )
    )
    _record_compare(
        rs,
        relation="left_of",
        anchor_id=5,
        candidate_ids=[3, 12],
        ranked_ids=[3, 12],
    )
    decision = evaluate_tadg(
        rs,
        {"proposal_id": 5, "confidence": 0.79},
        tool_override_reason=(
            "direct pixel evidence at the whiteboard end makes proposal 5 better"
        ),
    )

    assert decision.blocked is True
    assert decision.subcase == "anchor_self"
    assert "anchor itself" in decision.message
    assert "proposal 3" in decision.message


def test_tadg_treats_in_front_of_as_next_to_relation() -> None:
    rs = _runtime(
        bundle=_bundle_with_query(
            "the black table chair is located directly in front of the white writing board"
        )
    )
    _record_compare(
        rs,
        relation="next_to",
        anchor_id=1,
        candidate_ids=[3, 5, 12],
        ranked_ids=[3, 5, 12],
    )

    decision = evaluate_tadg(rs, {"proposal_id": 12, "confidence": 0.78})

    assert decision.blocked is True
    assert decision.relation == "next_to"
    assert decision.top1_pid == 3
    assert decision.subcase == "rank_mismatch"


def test_tadg_matches_inverse_left_of_it_relation() -> None:
    rs = _runtime(
        bundle=_bundle_with_query(
            "this is a brown chair. there is a table to the left of it with a lamp on it."
        )
    )
    _record_compare(
        rs,
        relation="right_of",
        anchor_id=3,
        candidate_ids=[7, 8],
        ranked_ids=[8, 7],
    )

    decision = evaluate_tadg(rs, {"proposal_id": 7, "confidence": 0.61})

    assert decision.blocked is True
    assert decision.relation == "right_of"
    assert decision.top1_pid == 8
    assert decision.subcase == "rank_mismatch"


def test_tadg_blocks_override_when_compare_omits_same_category_candidates() -> None:
    rs = _runtime(
        bundle=_bundle_with_query(
            "the black table chair with a red table chair to the left of it"
        )
    )
    _record_category_lookup(rs, category="chair", proposal_ids=[3, 5, 6, 9, 11, 12])
    _record_compare(
        rs,
        relation="left_of",
        anchor_id=1,
        candidate_ids=[6, 9, 11, 12],
        ranked_ids=[6, 9, 12, 11],
    )

    decision = evaluate_tadg(
        rs,
        {"proposal_id": 12, "confidence": 0.77},
        tool_override_reason="frame 18 looks more convincing than the tool rank",
    )

    assert decision.blocked is True
    assert decision.subcase == "candidate_coverage_gap"
    assert "proposal 3" in decision.message
    assert "proposal 5" in decision.message


def test_tadg_same_category_coverage_allows_anchor_excluded_from_candidates() -> None:
    rs = _runtime(bundle=_bundle_with_query("a black chair next to another same chair"))
    _record_category_lookup(rs, category="chair", proposal_ids=[0, 1, 2, 3, 5, 9])
    _record_compare(
        rs,
        relation="next_to",
        anchor_id=3,
        candidate_ids=[0, 1, 2, 5, 9],
        ranked_ids=[9, 1, 5, 0, 2],
    )

    decision = evaluate_tadg(rs, {"proposal_id": 9, "confidence": 0.7})

    assert decision.blocked is False
    assert decision.message == ""


def test_tadg_rejects_override_when_left_right_top1_clearly_outsupports_submitted() -> None:
    rs = _runtime(
        bundle=_bundle_with_query(
            "the black table chair with a red table chair to the left of it"
        )
    )
    _record_compare(
        rs,
        relation="left_of",
        anchor_id=1,
        candidate_ids=[3, 12],
        ranked_ids=[3, 12],
        supporting_frame_counts=[18, 5],
        contradicting_frame_counts=[0, 6],
    )

    decision = evaluate_tadg(
        rs,
        {"proposal_id": 12, "confidence": 0.72},
        tool_override_reason="raw frame 18 looks visually better for proposal 12",
    )

    assert decision.blocked is True
    assert decision.subcase == "rank_mismatch"
    assert "strong 2D shared-frame support" in decision.message


def test_tadg_rejects_override_when_left_right_top1_has_strong_2d_support() -> None:
    rs = _runtime(
        bundle=_bundle_with_query("this office chair is left of the keyboard")
    )
    _record_compare(
        rs,
        relation="left_of",
        anchor_id=8,
        candidate_ids=[39, 50],
        ranked_ids=[39, 50],
        supporting_frame_counts=[66, 0],
        contradicting_frame_counts=[3, 0],
    )
    decision = evaluate_tadg(
        rs,
        {"proposal_id": 50, "confidence": 0.88},
        tool_override_reason="proposal 50 is left of a different visible keyboard",
    )
    assert decision.blocked is True
    assert decision.subcase == "rank_mismatch"
    assert "strong 2D" in decision.message
    assert decision.top1_pid == 39


def test_tadg_override_too_short_still_blocks() -> None:
    rs = _runtime()
    rs.tadg_override_min_chars = 6
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    decision = evaluate_tadg(
        rs,
        {"proposal_id": 14, "confidence": 0.7},
        tool_override_reason="x",
    )
    assert decision.blocked is True
    assert rs.tool_override_reason is None


def test_tadg_repeat_force_pass() -> None:
    """Three identical no-override blocks → force-PASS on attempt 3."""
    rs = _runtime()
    rs.tadg_max_repeats = 3
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    d1 = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    d2 = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    d3 = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert d1.blocked is True
    assert d2.blocked is True
    assert d3.blocked is False
    assert d3.force_passed is True
    assert "TADG_FORCE_PASS" in d3.message
    # Distinct pid keeps its own counter.
    d_other = evaluate_tadg(rs, {"proposal_id": 27, "confidence": 0.7})
    assert d_other.blocked is True
    assert d_other.force_passed is False


def test_tadg_relation_no_match() -> None:
    """Query has no spatial relation in the keyword alias set; the gate
    should never fire even with a recent compare call."""
    bundle = _bundle_with_query("a thing somewhere")
    rs = _runtime(bundle=bundle)
    _record_compare(
        rs,
        relation="above",
        anchor_id=10,
        candidate_ids=[33, 14],
        ranked_ids=[33, 14],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is False


def test_tadg_skips_ood_proposal_id() -> None:
    """proposal_id=-1 is the OOD marker; gate must not fire."""
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3],
        ranked_ids=[33, 14],
    )
    decision = evaluate_tadg(rs, {"proposal_id": -1, "confidence": 0.0})
    assert decision.blocked is False


def test_tadg_alias_map_covers_supported_relations() -> None:
    """Sanity: every supported tool relation has an alias entry that
    includes itself; aliases are lowercase and underscore-normalised."""
    for rel in _SUPPORTED_TOOL_RELATIONS:
        assert rel in _TOOL_RELATION_ALIASES
        aliases = _TOOL_RELATION_ALIASES[rel]
        assert rel in aliases
        for token in aliases:
            assert token == token.lower()


def test_tadg_alias_nearest_to_closest_to() -> None:
    """P1.1 (CDX): `nearest` and `nearest_to` must alias to closest_to."""
    assert "nearest" in _TOOL_RELATION_ALIASES["closest_to"]
    assert "nearest_to" in _TOOL_RELATION_ALIASES["closest_to"]
    bundle = _bundle_with_query("the cabinet nearest the window")
    rs = _runtime(bundle=bundle)
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[33, 14],
        ranked_ids=[33, 14],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is True
    assert decision.top1_pid == 33


def test_tadg_alias_between_aliases_to_closest_to() -> None:
    """M5: `between` is in the closest_to family — agent uses closest_to
    as a proxy when the query says BETWEEN. Documents the known semantic
    looseness of the v1 keyword baseline (multi-relation enforcement is
    out of v1 scope, see spec §10)."""
    bundle = _bundle_with_query("the cabinet between the wall and the stove")
    rs = _runtime(bundle=bundle)
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[33, 14],
        ranked_ids=[33, 14],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is True


@pytest.mark.parametrize(
    ("query", "relation"),
    [
        ("the cabinet next to the stove", "next_to"),
        ("the cabinet near the stove", "near"),
        ("the cabinet to the left of the stove", "left_of"),
        ("the cabinet to the right of the stove", "right_of"),
    ],
)
def test_tadg_matches_new_spatial_tool_relations(query: str, relation: str) -> None:
    """When the agent uses the newer relation names accepted by
    compare_proposals_spatial, TADG should still police submit_final
    disagreement."""
    bundle = _bundle_with_query(query)
    rs = _runtime(bundle=bundle)
    _record_compare(
        rs,
        relation=relation,
        anchor_id=10,
        candidate_ids=[33, 14],
        ranked_ids=[33, 14],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is True
    assert decision.relation == relation
    assert decision.top1_pid == 33


def test_tadg_window_default_boundary_at_runtime_default() -> None:
    """Production-default window=32 boundary check."""
    rs = _runtime()
    # Don't override tadg_window — exercise the runtime dataclass default (32).
    assert rs.tadg_window == 32
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    # 33 fillers → compare at index 0 of last-(window+1) = outside window=32.
    for fid in range(100, 133):
        _record_view(rs, fid)
    d_silent = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert (
        d_silent.blocked is False
    ), "compare buried beyond window=32 should not trigger"
    # Reset and try 31 fillers → still inside window.
    rs2 = _runtime()
    assert rs2.tadg_window == 32
    _record_compare(
        rs2,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    for fid in range(100, 131):
        _record_view(rs2, fid)
    d_block = evaluate_tadg(rs2, {"proposal_id": 14, "confidence": 0.7})
    assert d_block.blocked is True, "compare within window=32 should trigger"


# ------------------------------------------------------------------
# End-to-end through chassis_tools.submit_final
# ------------------------------------------------------------------


def _register_vg_stub_pack(tmp_path: Path) -> None:
    body = tmp_path / "vg_grounding_playbook.md"
    body.write_text("# VG Grounding Playbook\n", encoding="utf-8")

    def _adapter(payload: dict, runtime) -> dict:
        return {"answer": payload}

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
                adapter=_adapter,
            ),
            required_primary_skill="vg-grounding-playbook",
            required_extra_metadata=[],
            ctx_factory=lambda b: object(),
        )
    )


def test_submit_final_blocks_when_tadg_disagrees(tmp_path: Path) -> None:
    """End-to-end: chassis submit_final returns a TADG_BLOCK string and
    does NOT set runtime.final_submission."""
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 14, "confidence": 0.7},
            "rationale": "I think 14 is right",
            "evidence_refs": [],
        }
    )
    assert response.startswith("TADG_BLOCK:")
    assert rs.final_submission is None
    assert rs.tadg_triggered is True
    # Block is recorded for audit.
    block_records = [
        t
        for t in rs.tool_trace
        if t.tool_name == "submit_final" and "TADG_BLOCK" in t.response_text
    ]
    assert len(block_records) == 1
    assert block_records[0].tool_input.get("tadg_blocked") is True
    assert block_records[0].tool_input.get("tadg_top1_pid") == 33


def test_submit_final_passes_with_override(tmp_path: Path) -> None:
    """End-to-end: tool_override_reason bypasses the block and proceeds
    to the chassis adapter, setting final_submission."""
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 14, "confidence": 0.7},
            "rationale": "rank-1 looked wrong on inspect_proposal",
            "evidence_refs": [],
            "tool_override_reason": "rank-1 was visually a different category",
        }
    )
    assert "submitted" in response.lower()
    assert rs.final_submission is not None
    assert rs.final_submission["answer"]["proposal_id"] == 14
    assert rs.tool_override_reason == "rank-1 was visually a different category"


def test_submit_final_unblocked_when_flag_off(tmp_path: Path) -> None:
    """When the flag is off, even a clear disagreement passes through."""
    _register_vg_stub_pack(tmp_path)
    rs = _runtime(flag_on=False)
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {
            "payload": {"proposal_id": 14, "confidence": 0.7},
            "rationale": "ok",
            "evidence_refs": [],
        }
    )
    assert "submitted" in response.lower()
    assert rs.tadg_triggered is False
    assert rs.final_submission is not None


def test_submit_final_block_then_override_succeeds(tmp_path: Path) -> None:
    """M4: end-to-end BLOCK-then-OVERRIDE round trip on the same runtime.
    First call (no override) → TADG_BLOCK + no final_submission; second
    call (with override) → success + final_submission + both records
    preserved in the trace."""
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    _, _, submit_final = build_chassis_tools(rs)

    response_block = submit_final.invoke(
        {
            "payload": {"proposal_id": 14, "confidence": 0.7},
            "rationale": "first attempt",
            "evidence_refs": [],
        }
    )
    assert response_block.startswith("TADG_BLOCK:")
    assert rs.final_submission is None

    response_ok = submit_final.invoke(
        {
            "payload": {"proposal_id": 14, "confidence": 0.85},
            "rationale": "second attempt with override",
            "evidence_refs": [],
            "tool_override_reason": "rank-1 was visually a different category",
        }
    )
    assert "submitted" in response_ok.lower()
    assert rs.final_submission is not None
    assert rs.tool_override_reason == "rank-1 was visually a different category"

    # Both records preserved.
    submit_records = [t for t in rs.tool_trace if t.tool_name == "submit_final"]
    assert len(submit_records) == 2
    assert submit_records[0].response_text.startswith("TADG_BLOCK:")
    assert submit_records[1].response_text.startswith("submitted")
    # Unified trace schema (M2): both records carry tadg_blocked field.
    assert submit_records[0].tool_input.get("tadg_blocked") is True
    assert submit_records[1].tool_input.get("tadg_blocked") is False


def test_submit_final_nested_payload_does_not_bypass_tadg(tmp_path: Path) -> None:
    """P1.2 (CDX): nested {payload: {payload: {...}}} payloads must be
    unwrapped BEFORE TADG so the gate sees the same proposal_id the
    chassis validator will consume. Without the chassis-side unwrap,
    `payload.get('proposal_id')` on the outer dict returns None → TADG
    silently allows → bypass."""
    _register_vg_stub_pack(tmp_path)
    rs = _runtime()
    _record_compare(
        rs,
        relation="closest_to",
        anchor_id=10,
        candidate_ids=[2, 3, 33],
        ranked_ids=[33, 14, 43],
    )
    _, _, submit_final = build_chassis_tools(rs)

    response = submit_final.invoke(
        {
            "payload": {
                # Outer wrapper (e.g. Stage2StructuredResponse-shaped).
                "task_type": "visual_grounding",
                "status": "completed",
                # Inner payload — what the chassis validator unwraps to.
                "payload": {"proposal_id": 14, "confidence": 0.7},
            },
            "rationale": "nested payload disagreement",
            "evidence_refs": [],
        }
    )
    assert response.startswith("TADG_BLOCK:"), (
        "Nested payload bypassed TADG — the gate did not unwrap to find "
        "proposal_id=14 in the inner dict."
    )
    assert rs.final_submission is None


__all__ = ["TADGDecision"]
