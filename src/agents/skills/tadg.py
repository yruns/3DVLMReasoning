"""TADG — Tool-Answer Disagreement Gate (v1).

Soft-blocks `submit_final` when the submitted `proposal_id` disagrees
with the most recent matched-relation `compare_proposals_spatial`
rank-1 result. The agent can either resubmit with a non-empty
`tool_override_reason` or revise its pick.

Spec: `tmp/tadg_spec.md`. Targets meta-bug 3.1 from
`tmp/picking_error_audit_consolidated.md` (S6 = scene0149_00::2::3,
S39 = scene0660_00::4::4).

The gate is opt-in via `Stage2DeepAgentConfig.use_tool_answer_disagreement_gate`
(default off) and `Stage2RuntimeState.use_tool_answer_disagreement_gate`
mirrors the config. Force-PASS triggers at
`tadg_max_repeats` repeated identical no-override submits to avoid
sample-level eval crashes (supersedes handoff §6.2 risk #3 default).

Relation matching is **keyword-only** in v1 (per spec §3 + §12 Q1
revision after M4-TADG H3 review): the parser-first path that walked
`extra_metadata["hypothesis_output"]` was deleted because no production
ScanRefer pack-prep populates that field, and the unit test that
exercised it created a false sense of coverage. The path can be
re-enabled by editing `_query_relation_set` once Stage-1 pack-prep
persists `HypothesisOutputV1` onto the bundle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Tool relations actually accepted by
# vg_embodiedscan/tools.py:compare_proposals_spatial.
_SUPPORTED_TOOL_RELATIONS = (
    "closest_to",
    "near",
    "next_to",
    "farthest_from",
    "above",
    "below",
    "left_of",
    "right_of",
)

# Bidirectional alias map: tool_relation -> set of synonymous query/parser
# relations that the gate considers a "match". Used both to (a) interpret
# parser-extracted relations from `extra_metadata["hypothesis_output"]`
# and (b) keyword-scan the raw query string when parser data is missing.
_TOOL_RELATION_ALIASES: dict[str, frozenset[str]] = {
    "closest_to": frozenset(
        {
            "closest_to",
            "closest",
            "closer_to",
            "closer",
            "near",
            "nearest",
            "nearest_to",
            "next_to",
            "nextto",
            "next-to",
            "beside",
            "adjacent_to",
            "adjacent",
            "by",
            "between",
        }
    ),
    "near": frozenset(
        {
            "near",
            "nearby",
            "near_to",
            "close_to",
            "next_to",
            "beside",
            "adjacent_to",
            "adjacent",
            "by",
        }
    ),
    "next_to": frozenset(
        {
            "next_to",
            "nextto",
            "next-to",
            "in_front_of",
            "front_of",
            "beside",
            "adjacent_to",
            "adjacent",
            "by",
            "near",
        }
    ),
    "farthest_from": frozenset(
        {
            "farthest_from",
            "farthest",
            "furthest_from",
            "furthest",
            "farther_from",
            "farther",
            "further_from",
            "further",
            "far_from",
            "far",
        }
    ),
    "above": frozenset(
        {
            "above",
            "on",
            "on_top_of",
            "ontop",
            "on_top",
            "over",
            "higher_than",
            "higher",
            "atop",
            "upon",
        }
    ),
    "below": frozenset(
        {
            "below",
            "under",
            "beneath",
            "underneath",
            "lower_than",
        }
    ),
    "left_of": frozenset(
        {
            "left_of",
            "to_the_left_of",
            "on_the_left_of",
            "left_hand",
            "left-hand",
        }
    ),
    "right_of": frozenset(
        {
            "right_of",
            "to_the_right_of",
            "on_the_right_of",
            "right_hand",
            "right-hand",
        }
    ),
}

# Patterns scanned in raw query strings as a fallback when the parser
# yields no usable spatial constraint. Compiled once at import time.
_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    tool_rel: re.compile(
        r"\b(" + "|".join(re.escape(a.replace("_", " ")) for a in aliases) + r")\b",
        re.IGNORECASE,
    )
    for tool_rel, aliases in _TOOL_RELATION_ALIASES.items()
}

_SAME_CATEGORY_ADJACENCY_PATTERN = re.compile(
    r"\b(same|same\s+type|of\s+the\s+same\s+type|another\s+same)\b",
    re.IGNORECASE,
)
_ANCHOR_LEFT_OF_TARGET_PATTERN = re.compile(
    r"\b(?:there\s+is|with|has|having|includes?|including|and)\b[^.?!]{0,120}"
    r"\b(?:to\s+the\s+)?left\s+of\s+it\b",
    re.IGNORECASE,
)
_ANCHOR_RIGHT_OF_TARGET_PATTERN = re.compile(
    r"\b(?:there\s+is|with|has|having|includes?|including|and)\b[^.?!]{0,120}"
    r"\b(?:to\s+the\s+)?right\s+of\s+it\b",
    re.IGNORECASE,
)


def _bundle_query_text(runtime: Any) -> str:
    bundle = getattr(runtime, "bundle", None)
    extra = getattr(bundle, "extra_metadata", {}) or {}
    query_text = ""
    if bundle is not None:
        query_text = getattr(bundle, "stage1_query", "") or extra.get("query", "") or ""
    return query_text if isinstance(query_text, str) else ""


@dataclass(frozen=True)
class TADGDecision:
    """Outcome of a `submit_final` gate evaluation."""

    blocked: bool
    message: str = ""  # populated when blocked OR override-accepted
    top1_pid: int | None = None  # most recent compare's rank-1 (when matched)
    submitted_pid: int | None = None
    relation: str | None = None
    anchor_id: int | None = None
    ranked_ids: tuple[int, ...] = ()
    subcase: str = ""  # 'anchor_self', 'not_in_candidates', 'rank_mismatch', ''
    force_passed: bool = False  # set by anti-loop guard


def _norm(token: str) -> str:
    return (token or "").strip().lower().replace(" ", "_")


_COMPARE_RELATION_ALIASES = {
    "closer_to": "closest_to",
    "closest": "closest_to",
    "nearest": "closest_to",
    "nearest_to": "closest_to",
    "nearer_to": "closest_to",
    "farther_from": "farthest_from",
    "further_from": "farthest_from",
    "furthest_from": "farthest_from",
    "furthest": "farthest_from",
    "farthest": "farthest_from",
    "far_from": "farthest_from",
    "near_to": "near",
    "nearby": "near",
    "beside": "next_to",
    "adjacent": "next_to",
    "adjacent_to": "next_to",
}


def _canonical_compare_relation(token: str) -> str:
    norm = _norm(str(token or "").replace("-", "_"))
    return _COMPARE_RELATION_ALIASES.get(norm, norm)


def _query_relation_set(runtime: Any) -> set[str]:
    """Return the set of *tool* relations the user's query asks about.

    Used to filter which `compare_proposals_spatial` calls count as
    "matching" the user's question.

    v1 baseline: keyword scan over `bundle.stage1_query` (canonical,
    populated by `agents.benchmark_adapters.build_stage2_evidence_bundle`
    and by the VG pack-v1 builder via the `query` kwarg). Falls back to
    `extra_metadata["query"]` for legacy test scaffolds.

    Parser-first matching (walking
    `extra_metadata["hypothesis_output"]["hypotheses"][*].grounding_query.root.spatial_constraints`)
    is **deferred** until Stage-1 pack-prep persists `HypothesisOutputV1`
    onto the bundle. The dispatch shape is reserved (the `bundle` is
    inspected for both surfaces below) so that the parser path can be
    re-enabled without touching this function — see `tmp/tadg_spec.md`
    §3 + §12 Q1 for the v1 baseline contract and the deferred
    parser-first plan.
    """
    matched: set[str] = set()

    # Keyword scan over the raw query string.
    query_text = _bundle_query_text(runtime)
    if query_text:
        for tool_rel, pattern in _KEYWORD_PATTERNS.items():
            if pattern.search(query_text):
                matched.add(tool_rel)
        if _ANCHOR_LEFT_OF_TARGET_PATTERN.search(query_text):
            matched.add("right_of")
        if _ANCHOR_RIGHT_OF_TARGET_PATTERN.search(query_text):
            matched.add("left_of")

    return matched


def _proposal_ids_from_list_scene_response(response: dict[str, Any]) -> list[int]:
    raw_ids = response.get("proposal_ids")
    if isinstance(raw_ids, list):
        return [int(pid) for pid in raw_ids if isinstance(pid, int)]
    rows = response.get("proposals")
    if isinstance(rows, list):
        ids: list[int] = []
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("proposal_id"), int):
                ids.append(int(row["proposal_id"]))
        return ids
    return []


def _last_matching_compare(
    runtime: Any,
    relevant_relations: set[str],
) -> dict[str, Any] | None:
    """Walk runtime.tool_trace backward up to `tadg_window` entries;
    return the most recent compare_proposals_spatial response payload
    (parsed JSON) whose relation argument is in `relevant_relations`.

    Returns None when no matching call is found in the window.
    """
    if not relevant_relations:
        return None
    trace = list(getattr(runtime, "tool_trace", []) or [])
    window = max(int(getattr(runtime, "tadg_window", 32)), 1)
    if window > len(trace):
        window = len(trace)
    for entry in reversed(trace[-window:]):
        name = getattr(entry, "tool_name", None)
        if name != "compare_proposals_spatial":
            continue
        tool_input = getattr(entry, "tool_input", {}) or {}
        requested_relation = _canonical_compare_relation(tool_input.get("relation", ""))
        response_text = getattr(entry, "response_text", "") or ""
        try:
            payload = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if "ranked_ids" not in payload:
            # Earlier entries may be error strings (still recorded with
            # tool_name=compare_proposals_spatial); skip them.
            continue
        relation = _canonical_compare_relation(
            payload.get("relation") or requested_relation
        )
        if relation not in relevant_relations:
            continue
        ranked = payload.get("ranked_ids") or []
        if not isinstance(ranked, list) or not ranked:
            continue
        return {
            "relation": relation,
            "anchor_id": payload.get("anchor_id"),
            "candidate_ids": [
                int(p)
                for p in tool_input.get("candidate_ids", [])
                if isinstance(p, int)
            ],
            "ranked_ids": [int(p) for p in ranked if isinstance(p, int)],
            "supporting_frame_counts": payload.get("supporting_frame_counts") or [],
            "contradicting_frame_counts": payload.get("contradicting_frame_counts")
            or [],
        }
    return None


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        ids.append(item)
    return ids


def _recorded_compare_payload(
    runtime: Any,
    evidence_id: str,
) -> dict[str, Any] | None:
    for entry in list(getattr(runtime, "tool_trace", []) or []):
        if getattr(entry, "tool_name", None) != "compare_proposals_spatial":
            continue
        tool_input = getattr(entry, "tool_input", {}) or {}
        if not isinstance(tool_input, dict) or tool_input.get("evidence_id") != evidence_id:
            continue
        response_text = getattr(entry, "response_text", "") or ""
        try:
            payload = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict) or payload.get("evidence_id") != evidence_id:
            return None
        return payload
    return None


def _field_matches_bound_evidence(
    relation_evidence: dict[str, Any],
    compare: dict[str, Any],
    payload: dict[str, Any],
    field: str,
) -> bool:
    if field == "evidence_id":
        return True
    if field not in payload:
        return False
    if field == "relation":
        value = relation_evidence.get(field)
        return isinstance(value, str) and _canonical_compare_relation(value) == compare[field]
    if field in {"candidate_ids", "ranked_ids"}:
        return _int_list(relation_evidence.get(field)) == compare[field]
    return relation_evidence.get(field) == payload.get(field)


def _bound_compare(
    runtime: Any,
    relation_evidence: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(relation_evidence, dict):
        return None
    evidence_id = relation_evidence.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        return None
    payload = _recorded_compare_payload(runtime, evidence_id)
    if payload is None:
        return None
    relation_value = payload.get("relation")
    if not isinstance(relation_value, str):
        return None
    relation = _canonical_compare_relation(relation_value)
    if relation not in _SUPPORTED_TOOL_RELATIONS:
        return None
    anchor_id = payload.get("anchor_id")
    if isinstance(anchor_id, bool) or not isinstance(anchor_id, int):
        return None
    candidate_ids = _int_list(payload.get("candidate_ids"))
    ranked_ids = _int_list(payload.get("ranked_ids"))
    if (
        candidate_ids is None
        or ranked_ids is None
        or not ranked_ids
        or not set(ranked_ids).issubset(set(candidate_ids))
    ):
        return None
    supporting = payload.get("supporting_frame_counts")
    contradicting = payload.get("contradicting_frame_counts")
    compare = {
        "relation": relation,
        "anchor_id": anchor_id,
        "candidate_ids": candidate_ids,
        "ranked_ids": ranked_ids,
        "supporting_frame_counts": supporting if isinstance(supporting, list) else [],
        "contradicting_frame_counts": (
            contradicting if isinstance(contradicting, list) else []
        ),
    }
    for field in relation_evidence:
        if not _field_matches_bound_evidence(
            relation_evidence,
            compare,
            payload,
            field,
        ):
            return None
    return compare


def _rank_value(values: Any, ranked_ids: list[int], proposal_id: int) -> int:
    if proposal_id not in ranked_ids or not isinstance(values, list):
        return 0
    index = ranked_ids.index(proposal_id)
    if index >= len(values):
        return 0
    try:
        return int(values[index])
    except (TypeError, ValueError):
        return 0


def _is_same_category_anchor_self_query(
    runtime: Any, relevant_relations: set[str]
) -> bool:
    """Return True for phrases like "chair next to another same chair"."""
    if "next_to" not in relevant_relations and "near" not in relevant_relations:
        return False
    query_text = _bundle_query_text(runtime)
    return bool(_SAME_CATEGORY_ADJACENCY_PATTERN.search(query_text))


def _strict_override_rejection_message(
    *,
    submitted_pid: int,
    top1_pid: int,
    relation: str,
    anchor_id: int | None,
    ranked_ids: list[int],
    reason: str,
    subcase: str = "",
) -> str:
    anchor_repr = f"proposal {anchor_id}" if anchor_id is not None else "(anchor=?)"
    if subcase == "anchor_self":
        return (
            f"TADG_ROLE_ERROR: override rejected for relation {relation!r} "
            f"against {anchor_repr}; submitted proposal {submitted_pid} is "
            f"the anchor itself. This is a target/anchor role error, not a "
            f"visual override. {reason}\n\n"
            "rerun compare_proposals_spatial with candidate_ids containing "
            "the target-category proposals and anchor_id set to the described "
            "anchor proposal. If the anchor id was wrong, inspect/mark the "
            "real anchor first. Do not revise solely to proposal "
            f"{top1_pid}; only submit it if the corrected role comparison "
            f"and marked evidence support it. ranked_ids={ranked_ids}."
        )
    return (
        f"TADG_STRICT: override rejected for relation {relation!r} against "
        f"{anchor_repr}; proposal {top1_pid} is the relation-ranked target and "
        f"proposal {submitted_pid} should not bypass that result. {reason}\n\n"
        f"Revise to proposal {top1_pid} (or another id from ranked_ids={ranked_ids}) "
        f"after inspecting the cited evidence; do not resubmit proposal "
        f"{submitted_pid} with only a new override reason."
    )


def _ambiguous_anchor_gap(runtime: Any, compare: dict[str, Any]) -> str | None:
    relation = compare["relation"]
    if relation not in {"left_of", "right_of"}:
        return None
    anchor_id = compare.get("anchor_id")
    if not isinstance(anchor_id, int):
        return None
    candidate_ids = set(compare.get("candidate_ids") or [])
    if not candidate_ids:
        return None

    trace = list(getattr(runtime, "tool_trace", []) or [])
    window = max(int(getattr(runtime, "tadg_window", 32)), 1)
    trace = trace[-window:]

    anchor_candidates: list[int] = []
    for entry in reversed(trace):
        if getattr(entry, "tool_name", None) not in (
            "list_scene_proposals",
            "find_proposals_by_category",
        ):
            continue
        response_text = getattr(entry, "response_text", "") or ""
        try:
            payload = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        ids = _proposal_ids_from_list_scene_response(payload)
        if anchor_id not in ids:
            continue
        if len(ids) <= 1:
            return None
        if set(ids) == candidate_ids:
            # This lookup is for the target category, not the anchor category.
            continue
        anchor_candidates = ids
        break

    if len(anchor_candidates) <= 1:
        return None

    tested_anchor_ids: set[int] = set()
    for entry in trace:
        if getattr(entry, "tool_name", None) != "compare_proposals_spatial":
            continue
        tool_input = getattr(entry, "tool_input", {}) or {}
        if _canonical_compare_relation(tool_input.get("relation", "")) != relation:
            continue
        if set(tool_input.get("candidate_ids") or []) != candidate_ids:
            continue
        tested_anchor = tool_input.get("anchor_id")
        if isinstance(tested_anchor, int):
            tested_anchor_ids.add(tested_anchor)

    missing = [pid for pid in anchor_candidates if pid not in tested_anchor_ids]
    if not missing:
        return None
    missing_text = ", ".join(f"proposal {pid}" for pid in missing[:5])
    return (
        f"TADG_ANCHOR_AMBIGUITY: ambiguous anchor category has untested "
        f"candidate(s): {missing_text}. Compare the same target candidate set "
        f"against each anchor candidate before submitting a {relation!r} answer."
    )


def _candidate_coverage_gap(
    runtime: Any,
    compare: dict[str, Any],
    submitted_pid: int,
) -> tuple[str, list[int]] | None:
    """Return omitted same-category candidates for an incomplete compare.

    The agent often narrows a category candidate list before calling
    `compare_proposals_spatial`. That is valid after explicit visual elimination,
    but it is unsafe for a final answer to override the resulting rank when the
    trace still contains untested same-category proposals. This check is
    trace-only: it reads the agent's own category lookups and compare arguments.
    """
    candidate_ids = {
        int(pid) for pid in compare.get("candidate_ids", []) if isinstance(pid, int)
    }
    if not candidate_ids or submitted_pid not in candidate_ids:
        return None

    trace = list(getattr(runtime, "tool_trace", []) or [])
    window = max(int(getattr(runtime, "tadg_window", 32)), 1)
    for entry in reversed(trace[-window:]):
        if getattr(entry, "tool_name", None) not in (
            "list_scene_proposals",
            "find_proposals_by_category",
        ):
            continue
        response_text = getattr(entry, "response_text", "") or ""
        try:
            payload = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        category_ids = _proposal_ids_from_list_scene_response(payload)
        if submitted_pid not in category_ids:
            continue
        if len(category_ids) > 12:
            return None
        anchor_id = compare.get("anchor_id")
        omitted = [
            pid
            for pid in category_ids
            if pid not in candidate_ids and pid != anchor_id
        ]
        if not omitted:
            return None
        omitted_text = ", ".join(f"proposal {pid}" for pid in omitted[:8])
        message = (
            "TADG_CANDIDATE_COVERAGE: latest spatial comparison used an "
            f"incomplete same-category candidate set; omitted {omitted_text}. "
            "Re-run compare_proposals_spatial with the complete category "
            "candidate set, or explicitly inspect and eliminate the omitted "
            "same-category proposals before submitting."
        )
        return message, omitted
    return None


def _should_reject_override(
    *,
    runtime: Any,
    compare: dict[str, Any],
    relevant_relations: set[str],
    submitted_pid: int,
    top1_pid: int,
    subcase: str,
    ranked_ids: list[int],
) -> str | None:
    relation = compare["relation"]
    if subcase == "anchor_self" and _is_same_category_anchor_self_query(
        runtime, relevant_relations
    ):
        return (
            "This is a same-category adjacency query; the submitted proposal is "
            "the anchor-self rather than the relation-ranked target."
        )

    if subcase == "anchor_self":
        return (
            "The submitted proposal is the spatial anchor itself, not the "
            "relation-ranked target candidate. This indicates a target/anchor "
            "role error, not a visual override."
        )

    if relation in {"left_of", "right_of"} and subcase == "rank_mismatch":
        supporting = compare.get("supporting_frame_counts") or []
        contradicting = compare.get("contradicting_frame_counts") or []
        top_support = _rank_value(supporting, ranked_ids, top1_pid)
        submitted_support = _rank_value(supporting, ranked_ids, submitted_pid)
        top_contradict = _rank_value(contradicting, ranked_ids, top1_pid)
        if (
            top_support >= 15
            and submitted_support <= max(1, top_support // 3)
            and top_contradict <= max(5, top_support // 4)
        ):
            return (
                "The top-ranked proposal has strong 2D shared-frame support "
                f"({top_support} supporting vs {top_contradict} contradicting "
                f"frames), while the submitted proposal has only {submitted_support} "
                "supporting "
                "left/right frames for this anchor."
            )

    return None


def _format_block_message(
    *,
    submitted_pid: int,
    top1_pid: int,
    relation: str,
    anchor_id: int | None,
    ranked_ids: list[int],
    subcase: str,
) -> str:
    if subcase == "anchor_self":
        anchor_repr = f"proposal {anchor_id}" if anchor_id is not None else "(anchor=?)"
        return (
            f"TADG_ROLE_ERROR: compare_proposals_spatial ranked proposal "
            f"{top1_pid} as rank-1 for relation {relation!r} against "
            f"{anchor_repr}; submitted proposal {submitted_pid} is the "
            "anchor itself. This is a target/anchor role error, not a "
            "rank override.\n\n"
            "rerun compare_proposals_spatial with candidate_ids containing "
            "the target-category proposals and anchor_id set to the described "
            "anchor proposal. If the anchor id was wrong, inspect/mark the "
            "real anchor first. Do not revise solely to proposal "
            f"{top1_pid}; only submit it if the corrected role comparison "
            f"and marked evidence support it. ranked_ids={ranked_ids}."
        )
    if subcase == "not_in_candidates":
        extra = "not in the spatial-tool's candidate set"
    else:
        extra = "ranked below position 1 in that call"
    anchor_repr = f"proposal {anchor_id}" if anchor_id is not None else "(anchor=?)"
    return (
        f"TADG: compare_proposals_spatial ranked proposal {top1_pid} as rank-1 "
        f"for relation {relation!r} against {anchor_repr}; you are submitting "
        f"proposal {submitted_pid} ({extra}).\n\n"
        f"Either:\n"
        f"  (a) Resubmit with `tool_override_reason='<one-sentence reason>'` "
        f"to record the explicit divergence, OR\n"
        f"  (b) Revise to proposal {top1_pid} (or another id from "
        f"ranked_ids={ranked_ids}) and resubmit.\n\n"
        f"This block is recorded; if you resubmit identically several times "
        f"in a row the gate auto-escalates and accepts the submission."
    )


def tadg_record_fields(decision: TADGDecision) -> dict[str, Any]:
    """Canonical telemetry fields written into `tool_trace` for any
    TADG-touching submit_final invocation.

    Both the BLOCK path and the override / force-pass / silent-allow
    paths in `chassis_tools.submit_final` emit the same shape so a
    downstream funnel-evaluator extension can collate counts uniformly.
    Fields are populated from the decision; absent values are recorded
    as None so consumers can rely on the keys being present.
    """
    return {
        "tadg_blocked": bool(decision.blocked),
        "tadg_force_passed": bool(decision.force_passed),
        "tadg_top1_pid": decision.top1_pid,
        "tadg_submitted_pid": decision.submitted_pid,
        "tadg_relation": decision.relation,
        "tadg_anchor_id": decision.anchor_id,
        "tadg_subcase": decision.subcase or None,
        "tadg_ranked_ids": list(decision.ranked_ids) if decision.ranked_ids else None,
        "tadg_message": decision.message or None,
    }


def evaluate_tadg(
    runtime: Any,
    payload: dict | Any,
    *,
    tool_override_reason: str | None = None,
    relation_evidence: dict[str, Any] | None = None,
) -> TADGDecision:
    """Decide whether to allow or block a `submit_final` payload.

    Args:
        runtime: Stage2RuntimeState (or any object exposing
            `use_tool_answer_disagreement_gate`, `tadg_*` config fields,
            `tool_trace`, `bundle.extra_metadata`, and the sticky run-state
            fields populated by this function).
        payload: the submit_final payload (`{"proposal_id": int, "confidence": float}`
            for VG; other task types are ignored).
        tool_override_reason: optional non-empty string the agent provides
            to bypass a would-be block. Bypass requires
            `len(reason.strip()) >= runtime.tadg_override_min_chars`.

    Returns:
        TADGDecision describing whether to block or allow. The caller
        (`chassis_tools.submit_final`) is responsible for translating
        block decisions into the soft-block message and recording the
        observation in `runtime.tool_trace`.
    """
    if not bool(getattr(runtime, "use_tool_answer_disagreement_gate", False)):
        return TADGDecision(blocked=False)

    payload_dict = payload if isinstance(payload, dict) else {}
    submitted_pid_raw = payload_dict.get("proposal_id")
    if not isinstance(submitted_pid_raw, int) or submitted_pid_raw < 0:
        # OOD (-1) submits and non-VG payloads are out of scope.
        return TADGDecision(blocked=False)
    submitted_pid: int = submitted_pid_raw

    compare = _bound_compare(runtime, relation_evidence)
    if compare is not None:
        relevant_relations = {compare["relation"]}
    else:
        relevant_relations = _query_relation_set(runtime)
        compare = _last_matching_compare(runtime, relevant_relations)
    if compare is None:
        return TADGDecision(blocked=False)

    ranked_ids = compare["ranked_ids"]
    if not ranked_ids:
        return TADGDecision(blocked=False)
    top1_pid = ranked_ids[0]

    ambiguous_anchor_message = _ambiguous_anchor_gap(runtime, compare)
    if ambiguous_anchor_message:
        runtime.tadg_triggered = True
        return TADGDecision(
            blocked=True,
            message=ambiguous_anchor_message,
            top1_pid=top1_pid,
            submitted_pid=submitted_pid,
            relation=compare["relation"],
            anchor_id=(
                compare.get("anchor_id")
                if isinstance(compare.get("anchor_id"), int)
                else None
            ),
            ranked_ids=tuple(ranked_ids),
            subcase="ambiguous_anchor",
        )

    coverage_gap = _candidate_coverage_gap(runtime, compare, submitted_pid)
    if coverage_gap:
        message, _ = coverage_gap
        runtime.tadg_triggered = True
        return TADGDecision(
            blocked=True,
            message=message,
            top1_pid=top1_pid,
            submitted_pid=submitted_pid,
            relation=compare["relation"],
            anchor_id=(
                compare.get("anchor_id")
                if isinstance(compare.get("anchor_id"), int)
                else None
            ),
            ranked_ids=tuple(ranked_ids),
            subcase="candidate_coverage_gap",
        )

    if submitted_pid == top1_pid:
        # Tool agrees with submission; no disagreement.
        return TADGDecision(blocked=False)

    anchor_id = compare.get("anchor_id")
    if (
        anchor_id is not None
        and isinstance(anchor_id, int)
        and submitted_pid == anchor_id
    ):
        subcase = "anchor_self"
    elif submitted_pid not in ranked_ids:
        subcase = "not_in_candidates"
    else:
        subcase = "rank_mismatch"

    # Override path: non-empty reason of sufficient length bypasses the block.
    min_chars = max(int(getattr(runtime, "tadg_override_min_chars", 6)), 1)
    reason = (tool_override_reason or "").strip()
    if reason and len(reason) >= min_chars:
        strict_reason = _should_reject_override(
            runtime=runtime,
            compare=compare,
            relevant_relations=relevant_relations,
            submitted_pid=submitted_pid,
            top1_pid=top1_pid,
            subcase=subcase,
            ranked_ids=ranked_ids,
        )
        if strict_reason:
            runtime.tadg_triggered = True
            block_message = _strict_override_rejection_message(
                submitted_pid=submitted_pid,
                top1_pid=top1_pid,
                relation=compare["relation"],
                anchor_id=anchor_id if isinstance(anchor_id, int) else None,
                ranked_ids=ranked_ids,
                reason=strict_reason,
                subcase=subcase,
            )
            return TADGDecision(
                blocked=True,
                message=block_message,
                top1_pid=top1_pid,
                submitted_pid=submitted_pid,
                relation=compare["relation"],
                anchor_id=anchor_id if isinstance(anchor_id, int) else None,
                ranked_ids=tuple(ranked_ids),
                subcase=subcase,
            )
        runtime.tadg_triggered = True
        runtime.tool_override_reason = reason
        message = (
            f"TADG_OVERRIDE_ACCEPTED: agent submitted {submitted_pid} despite "
            f"compare_proposals_spatial rank-1={top1_pid} (relation={compare['relation']!r}, "
            f"anchor={anchor_id}); override_reason={reason!r}."
        )
        return TADGDecision(
            blocked=False,
            message=message,
            top1_pid=top1_pid,
            submitted_pid=submitted_pid,
            relation=compare["relation"],
            anchor_id=anchor_id if isinstance(anchor_id, int) else None,
            ranked_ids=tuple(ranked_ids),
            subcase=subcase,
        )

    # Anti-loop force-PASS: tally identical no-override submits per pid.
    block_count = dict(getattr(runtime, "tadg_block_count", {}) or {})
    next_count = block_count.get(submitted_pid, 0) + 1
    block_count[submitted_pid] = next_count
    runtime.tadg_block_count = block_count
    max_repeats = max(int(getattr(runtime, "tadg_max_repeats", 3)), 1)
    if next_count >= max_repeats:
        runtime.tadg_triggered = True
        message = (
            f"TADG_FORCE_PASS: identical submission of proposal {submitted_pid} "
            f"reached {next_count} attempts without a tool_override_reason; the "
            f"gate auto-escalates and accepts the submission. "
            f"(rank-1 was {top1_pid} for relation {compare['relation']!r}.)"
        )
        return TADGDecision(
            blocked=False,
            message=message,
            top1_pid=top1_pid,
            submitted_pid=submitted_pid,
            relation=compare["relation"],
            anchor_id=anchor_id if isinstance(anchor_id, int) else None,
            ranked_ids=tuple(ranked_ids),
            subcase=subcase,
            force_passed=True,
        )

    runtime.tadg_triggered = True
    block_message = _format_block_message(
        submitted_pid=submitted_pid,
        top1_pid=top1_pid,
        relation=compare["relation"],
        anchor_id=anchor_id if isinstance(anchor_id, int) else None,
        ranked_ids=ranked_ids,
        subcase=subcase,
    )
    return TADGDecision(
        blocked=True,
        message=block_message,
        top1_pid=top1_pid,
        submitted_pid=submitted_pid,
        relation=compare["relation"],
        anchor_id=anchor_id if isinstance(anchor_id, int) else None,
        ranked_ids=tuple(ranked_ids),
        subcase=subcase,
    )


__all__ = [
    "TADGDecision",
    "evaluate_tadg",
    "tadg_record_fields",
    "_TOOL_RELATION_ALIASES",
    "_SUPPORTED_TOOL_RELATIONS",
]
