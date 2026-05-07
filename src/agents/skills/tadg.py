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
_SUPPORTED_TOOL_RELATIONS = ("closest_to", "farthest_from", "above", "below")

# Bidirectional alias map: tool_relation -> set of synonymous query/parser
# relations that the gate considers a "match". Used both to (a) interpret
# parser-extracted relations from `extra_metadata["hypothesis_output"]`
# and (b) keyword-scan the raw query string when parser data is missing.
_TOOL_RELATION_ALIASES: dict[str, frozenset[str]] = {
    "closest_to": frozenset({
        "closest_to", "closest", "near", "nearest", "nearest_to",
        "next_to", "nextto", "next-to", "beside", "adjacent_to",
        "adjacent", "by", "between",
    }),
    "farthest_from": frozenset({
        "farthest_from", "farthest", "far_from", "far",
    }),
    "above": frozenset({
        "above", "on", "on_top_of", "ontop", "on_top", "over",
        "higher_than", "higher", "atop", "upon",
    }),
    "below": frozenset({
        "below", "under", "beneath", "underneath", "lower_than",
    }),
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


@dataclass(frozen=True)
class TADGDecision:
    """Outcome of a `submit_final` gate evaluation."""

    blocked: bool
    message: str = ""               # populated when blocked OR override-accepted
    top1_pid: int | None = None     # most recent compare's rank-1 (when matched)
    submitted_pid: int | None = None
    relation: str | None = None
    anchor_id: int | None = None
    ranked_ids: tuple[int, ...] = ()
    subcase: str = ""               # 'anchor_self', 'not_in_candidates', 'rank_mismatch', ''
    force_passed: bool = False      # set by anti-loop guard


def _norm(token: str) -> str:
    return (token or "").strip().lower().replace(" ", "_")


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
    bundle = getattr(runtime, "bundle", None)
    extra = getattr(bundle, "extra_metadata", {}) or {}
    matched: set[str] = set()

    # Keyword scan over the raw query string.
    query_text = ""
    if bundle is not None:
        query_text = (
            getattr(bundle, "stage1_query", "")
            or extra.get("query", "")
            or ""
        )
    if not isinstance(query_text, str):
        query_text = ""
    if query_text:
        for tool_rel, pattern in _KEYWORD_PATTERNS.items():
            if pattern.search(query_text):
                matched.add(tool_rel)

    return matched


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
    window = max(int(getattr(runtime, "tadg_window", 16)), 1)
    if window > len(trace):
        window = len(trace)
    for entry in reversed(trace[-window:]):
        name = getattr(entry, "tool_name", None)
        if name != "compare_proposals_spatial":
            continue
        tool_input = getattr(entry, "tool_input", {}) or {}
        relation = _norm(tool_input.get("relation", ""))
        if relation not in relevant_relations:
            continue
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
        ranked = payload.get("ranked_ids") or []
        if not isinstance(ranked, list) or not ranked:
            continue
        return {
            "relation": relation,
            "anchor_id": payload.get("anchor_id"),
            "ranked_ids": [int(p) for p in ranked if isinstance(p, int)],
        }
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
        extra = "the anchor itself — re-evaluate the spatial query"
    elif subcase == "not_in_candidates":
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

    relevant_relations = _query_relation_set(runtime)
    compare = _last_matching_compare(runtime, relevant_relations)
    if compare is None:
        return TADGDecision(blocked=False)

    ranked_ids = compare["ranked_ids"]
    if not ranked_ids:
        return TADGDecision(blocked=False)
    top1_pid = ranked_ids[0]
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
