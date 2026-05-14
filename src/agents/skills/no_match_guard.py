"""No-match candidate guard for VG `submit_final(proposal_id=-1)`.

The guard uses only the agent's own tool trace. It does not inspect GT
targets, GT bboxes, target visibility, or per-case metric data.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoMatchGuardDecision:
    """Outcome of a no-match guard evaluation."""

    blocked: bool
    message: str = ""
    category_candidate_ids: tuple[int, ...] = ()
    category_candidate_labels: tuple[str, ...] = ()
    viewed_uninspected_ids: tuple[int, ...] = ()
    viewed_uninspected_labels: tuple[str, ...] = ()
    force_passed: bool = False


_VISIBLE_RE = re.compile(r"visible_proposals=(\[[^\]]*\])")
_CATEGORIES_RE = re.compile(r"categories=(\[[^\]]*\])")


def _payload_dict(payload: dict | Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("payload")
    if isinstance(inner, dict) and "proposal_id" in inner:
        return inner
    return payload


def _json_dict(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _literal_list(pattern: re.Pattern[str], text: str) -> list[Any]:
    match = pattern.search(text)
    if match is None:
        return []
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _tool_name(entry: Any) -> str:
    return str(getattr(entry, "tool_name", "") or "")


def _tool_input(entry: Any) -> dict[str, Any]:
    value = getattr(entry, "tool_input", {}) or {}
    return value if isinstance(value, dict) else {}


def _response_text(entry: Any) -> str:
    return str(getattr(entry, "response_text", "") or "")


def _candidate_evidence(runtime: Any) -> dict[str, Any]:
    category_ids: list[int] = []
    category_labels: list[str] = []
    inspected_ids: set[int] = set()
    viewed_events: list[list[tuple[int, str, int | None]]] = []

    for entry in list(getattr(runtime, "tool_trace", []) or []):
        name = _tool_name(entry)
        tool_input = _tool_input(entry)
        if name == "inspect_proposal":
            proposal_id = tool_input.get("proposal_id")
            if isinstance(proposal_id, int):
                inspected_ids.add(proposal_id)
            continue

        if name in ("list_scene_proposals", "find_proposals_by_category"):
            payload = _json_dict(_response_text(entry))
            proposal_ids = payload.get("proposal_ids") or []
            if not isinstance(proposal_ids, list):
                continue
            label = str(payload.get("category") or tool_input.get("category") or "")
            for proposal_id in proposal_ids:
                if not isinstance(proposal_id, int):
                    continue
                if proposal_id not in category_ids:
                    category_ids.append(proposal_id)
                    category_labels.append(label)
            continue

        if name == "view_keyframe" and (tool_input.get("mode") or "auto") in (
            "marked",
            "auto",
        ):
            response = _response_text(entry)
            frame_id_raw = tool_input.get("frame_id")
            frame_id = frame_id_raw if isinstance(frame_id_raw, int) else None
            visible_ids = _literal_list(_VISIBLE_RE, response)
            categories = _literal_list(_CATEGORIES_RE, response)
            event: list[tuple[int, str, int | None]] = []
            for idx, proposal_id in enumerate(visible_ids):
                if not isinstance(proposal_id, int):
                    continue
                label = ""
                if idx < len(categories):
                    label = str(categories[idx])
                event.append((proposal_id, label, frame_id))
            if event:
                viewed_events.append(event)

    max_viewed = max(int(getattr(runtime, "no_match_guard_max_viewed", 12)), 1)
    viewed_uninspected: list[int] = []
    viewed_labels: list[str] = []
    viewed_frames: list[int | None] = []
    for event in reversed(viewed_events):
        for proposal_id, label, frame_id in event:
            if proposal_id in inspected_ids or proposal_id in viewed_uninspected:
                continue
            viewed_uninspected.append(proposal_id)
            viewed_labels.append(label)
            viewed_frames.append(frame_id)
            if len(viewed_uninspected) >= max_viewed:
                break
        if len(viewed_uninspected) >= max_viewed:
            break

    return {
        "category_ids": category_ids,
        "category_labels": category_labels,
        "viewed_uninspected_ids": viewed_uninspected,
        "viewed_uninspected_labels": viewed_labels,
        "viewed_uninspected_frames": viewed_frames,
    }


def _format_message(
    evidence: dict[str, Any],
    *,
    block_count: int = 1,
    max_repeats: int = 3,
) -> str:
    chunks: list[str] = []
    category_ids = evidence["category_ids"]
    category_labels = evidence["category_labels"]
    if category_ids:
        pairs = [
            f"{pid}:{label or '?'}"
            for pid, label in zip(category_ids[:12], category_labels[:12], strict=False)
        ]
        chunks.append(
            "category candidates from list_scene_proposals: " + ", ".join(pairs)
        )
    viewed_ids = evidence["viewed_uninspected_ids"]
    viewed_labels = evidence["viewed_uninspected_labels"]
    viewed_frames = evidence["viewed_uninspected_frames"]
    if viewed_ids:
        pairs = [
            f"{pid}:{label or '?'}"
            + (f"@frame {frame_id}" if frame_id is not None else "")
            for pid, label, frame_id in zip(
                viewed_ids[:12],
                viewed_labels[:12],
                viewed_frames[:12],
                strict=False,
            )
        ]
        chunks.append("viewed but uninspected marked proposals: " + ", ".join(pairs))
    base = (
        "NO_MATCH_GUARD: submit_final(proposal_id=-1) is premature because "
        + "; ".join(chunks)
        + ". Inspect or choose the best remaining proposal, or only repeat -1 "
        "after explicitly closing these candidates in the rationale. Proposal "
        "labels are weak priors: if the pixels show the referent, choose the "
        "proposal that covers it even when its Mask3D label mismatches the query."
    )
    if category_ids and block_count >= max_repeats:
        candidate_ids = ", ".join(str(pid) for pid in category_ids[:12])
        return (
            base + " Category candidates are still present, so no-match is not an "
            "accepted final answer for this task. Submit the best available "
            f"proposal_id from the candidate set instead: {candidate_ids}."
        )
    return base


def no_match_guard_record_fields(decision: NoMatchGuardDecision) -> dict[str, Any]:
    """Canonical trace fields for no-match guard submit records."""
    return {
        "no_match_guard_blocked": bool(decision.blocked),
        "no_match_guard_force_passed": bool(decision.force_passed),
        "no_match_guard_category_candidate_ids": (
            list(decision.category_candidate_ids)
            if decision.category_candidate_ids
            else None
        ),
        "no_match_guard_category_candidate_labels": (
            list(decision.category_candidate_labels)
            if decision.category_candidate_labels
            else None
        ),
        "no_match_guard_viewed_uninspected_ids": (
            list(decision.viewed_uninspected_ids)
            if decision.viewed_uninspected_ids
            else None
        ),
        "no_match_guard_viewed_uninspected_labels": (
            list(decision.viewed_uninspected_labels)
            if decision.viewed_uninspected_labels
            else None
        ),
        "no_match_guard_message": decision.message or None,
    }


def evaluate_no_match_guard(runtime: Any, payload: dict | Any) -> NoMatchGuardDecision:
    """Block premature `proposal_id=-1` submissions when trace evidence
    still contains unresolved candidates.
    """
    if not bool(getattr(runtime, "use_no_match_candidate_guard", False)):
        return NoMatchGuardDecision(blocked=False)

    submitted_pid = _payload_dict(payload).get("proposal_id")
    if submitted_pid != -1:
        return NoMatchGuardDecision(blocked=False)

    evidence = _candidate_evidence(runtime)
    category_ids = tuple(evidence["category_ids"])
    category_labels = tuple(evidence["category_labels"])
    viewed_ids = tuple(evidence["viewed_uninspected_ids"])
    viewed_labels = tuple(evidence["viewed_uninspected_labels"])
    if not category_ids and not viewed_ids:
        return NoMatchGuardDecision(blocked=False)

    block_count = int(getattr(runtime, "no_match_guard_block_count", 0) or 0) + 1
    runtime.no_match_guard_block_count = block_count
    runtime.no_match_guard_triggered = True
    max_repeats = max(int(getattr(runtime, "no_match_guard_max_repeats", 3)), 1)
    message = _format_message(
        evidence,
        block_count=block_count,
        max_repeats=max_repeats,
    )
    return NoMatchGuardDecision(
        blocked=True,
        message=message,
        category_candidate_ids=category_ids,
        category_candidate_labels=category_labels,
        viewed_uninspected_ids=viewed_ids,
        viewed_uninspected_labels=viewed_labels,
    )


__all__ = [
    "NoMatchGuardDecision",
    "evaluate_no_match_guard",
    "no_match_guard_record_fields",
]
