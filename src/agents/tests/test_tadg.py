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


def _bundle_with_query(query: str, *, parser_relation: str | None = None) -> Stage2EvidenceBundle:
    """Build a minimal Stage2EvidenceBundle that mimics what the
    ScanRefer pack-prep populates: the raw query string for keyword
    fallback, and an optional `hypothesis_output` shaped like
    HypothesisOutputV1.model_dump() so the parser-first path can fire."""
    extra: dict = {}
    if parser_relation is not None:
        extra["hypothesis_output"] = {
            "hypotheses": [
                {
                    "rank": 1,
                    "grounding_query": {
                        "root": {
                            "categories": ["chair"],
                            "spatial_constraints": [
                                {"relation": parser_relation, "anchors": []}
                            ],
                        },
                    },
                }
            ],
            "parse_mode": "test",
        }
    bundle = Stage2EvidenceBundle(extra_metadata=extra)
    # Stage2EvidenceBundle does not always carry a `query` field, but
    # the gate reads it via getattr so we attach via model_copy when
    # the helper does, otherwise we set on extra.
    if hasattr(bundle, "query"):
        bundle = bundle.model_copy(update={"query": query})
    else:
        bundle.extra_metadata["query"] = query
    return bundle


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
    rs.tadg_window = 8
    rs.tadg_max_repeats = 3
    rs.tadg_override_min_chars = 6
    return rs


def _record_compare(
    rs: Stage2RuntimeState,
    *,
    relation: str,
    anchor_id: int,
    candidate_ids: list[int],
    ranked_ids: list[int],
) -> None:
    request = {
        "relation": relation,
        "anchor_id": anchor_id,
        "candidate_ids": candidate_ids,
    }
    payload = {
        "anchor_id": anchor_id,
        "relation": relation,
        "ranked_ids": ranked_ids,
        "distances": [0.1 * (i + 1) for i in range(len(ranked_ids))],
    }
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="compare_proposals_spatial",
            tool_input=request,
            response_text=json.dumps(payload),
        )
    )


def _record_view(rs: Stage2RuntimeState, frame_id: int) -> None:
    rs.tool_trace.append(
        Stage2ToolObservation(
            tool_name="view_keyframe_marked",
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
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3, 33], ranked_ids=[33, 14, 43],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 19, "confidence": 0.7})
    assert decision.blocked is False
    assert decision.message == ""


def test_tadg_blocks_when_submitted_not_top1() -> None:
    rs = _runtime()
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[33, 14, 43], ranked_ids=[33, 14, 43],
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
        rs, relation="closest_to", anchor_id=15,
        candidate_ids=[4, 2, 3], ranked_ids=[4, 2, 3],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 4, "confidence": 0.8})
    assert decision.blocked is False
    assert decision.message == ""
    assert rs.tadg_triggered is False


def test_tadg_blocks_when_submitted_not_in_ranked() -> None:
    """S6 case: agent submits pid 19 not in the candidate set."""
    rs = _runtime()
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
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
        rs, relation="closest_to", anchor_id=3,
        candidate_ids=[9, 1, 5], ranked_ids=[9, 1, 5],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 3, "confidence": 0.7})
    assert decision.blocked is True
    assert decision.subcase == "anchor_self"
    assert "anchor itself" in decision.message


def test_tadg_silent_when_no_matching_relation() -> None:
    """Query has no spatial relation; the gate should never fire even
    if the agent ran compare_proposals_spatial as a sanity check."""
    bundle = _bundle_with_query("a brown chair")
    rs = _runtime(bundle=bundle)
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3], ranked_ids=[33, 14, 43],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is False


def test_tadg_silent_when_compare_outside_window() -> None:
    """Window=8: an old compare at index 0 is ignored if 8+ later
    tool_trace entries arrived after it."""
    rs = _runtime()
    rs.tadg_window = 8
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3, 33], ranked_ids=[33, 14, 43],
    )
    # Push 9 view entries to bury the compare past the window.
    for fid in range(100, 109):
        _record_view(rs, fid)
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is False


def test_tadg_override_accepted() -> None:
    rs = _runtime()
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3, 33], ranked_ids=[33, 14, 43],
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


def test_tadg_override_too_short_still_blocks() -> None:
    rs = _runtime()
    rs.tadg_override_min_chars = 6
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3, 33], ranked_ids=[33, 14, 43],
    )
    decision = evaluate_tadg(
        rs, {"proposal_id": 14, "confidence": 0.7}, tool_override_reason="x",
    )
    assert decision.blocked is True
    assert rs.tool_override_reason is None


def test_tadg_repeat_force_pass() -> None:
    """Three identical no-override blocks → force-PASS on attempt 3."""
    rs = _runtime()
    rs.tadg_max_repeats = 3
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3, 33], ranked_ids=[33, 14, 43],
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


def test_tadg_relation_alias_match_via_parser() -> None:
    """Parser said `next_to`; tool was called with `closest_to`. Alias
    map should bridge them and trigger the gate."""
    bundle = _bundle_with_query(
        "ignored — parser relation drives match", parser_relation="next_to",
    )
    rs = _runtime(bundle=bundle)
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[33, 14], ranked_ids=[33, 14],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is True


def test_tadg_relation_no_match() -> None:
    """Parser says `inside`; tool was called with `above`. Alias map
    has no overlap → gate stays silent."""
    bundle = _bundle_with_query("a thing inside a box", parser_relation="inside")
    rs = _runtime(bundle=bundle)
    # Strip the keyword fallback by removing trigger words from the query.
    bundle.extra_metadata["query"] = "a thing somewhere"
    _record_compare(
        rs, relation="above", anchor_id=10,
        candidate_ids=[33, 14], ranked_ids=[33, 14],
    )
    decision = evaluate_tadg(rs, {"proposal_id": 14, "confidence": 0.7})
    assert decision.blocked is False


def test_tadg_skips_ood_proposal_id() -> None:
    """proposal_id=-1 is the OOD marker; gate must not fire."""
    rs = _runtime()
    _record_compare(
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3], ranked_ids=[33, 14],
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
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3, 33], ranked_ids=[33, 14, 43],
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
        t for t in rs.tool_trace
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
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3, 33], ranked_ids=[33, 14, 43],
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
        rs, relation="closest_to", anchor_id=10,
        candidate_ids=[2, 3, 33], ranked_ids=[33, 14, 43],
    )
    _, _, submit_final = build_chassis_tools(rs)
    response = submit_final.invoke(
        {"payload": {"proposal_id": 14, "confidence": 0.7},
         "rationale": "ok", "evidence_refs": []}
    )
    assert "submitted" in response.lower()
    assert rs.tadg_triggered is False
    assert rs.final_submission is not None


__all__ = ["TADGDecision"]
