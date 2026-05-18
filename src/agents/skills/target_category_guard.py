"""Target-category drift guard for VG `submit_final`.

The guard uses only query text, runtime tool traces, and proposal labels.
It does not inspect GT targets, GT bboxes, target visibility, or metrics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetCategoryDecision:
    """Outcome of a target-category guard evaluation."""

    blocked: bool
    message: str = ""
    submitted_pid: int | None = None
    expected_category: str | None = None
    submitted_category: str | None = None


_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "bookshelf": ("bookshelf", "bookcase", "book case"),
    "whiteboard": ("whiteboard", "white board"),
    "trash can": ("trash can", "trashcan"),
}
_LEADING_HEAD_RE = re.compile(
    r"^\s*(?:(?:it|this|that)\s+is\s+)?(?:the|a|an|this|that)?\s*"
    r"(?P<label>[a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,2})\b",
    re.I,
)
_WANT_HEAD_RE = re.compile(
    r"\b(?:want|select|choose|pick|find|looking\s+for|refer(?:ring)?\s+to)\s+"
    r"(?:the|a|an|this|that)?\s*"
    r"(?P<label>[a-z][a-z0-9]*(?:\s+[a-z][a-z0-9]*){0,2})\b",
    re.I,
)


def _payload_dict(payload: dict | Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("payload")
    if isinstance(inner, dict) and "proposal_id" in inner:
        return inner
    return payload


def _proposal_id(value: Any) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text)
    return None


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _normalize_category(value: str) -> str:
    text = " ".join(str(value).lower().split())
    compact = _compact(text)
    for canonical, aliases in _LABEL_ALIASES.items():
        if compact in {_compact(alias) for alias in aliases}:
            return canonical
    return text


def _category_matches(left: str | None, right: str | None) -> bool:
    left_norm = _normalize_category(left or "")
    right_norm = _normalize_category(right or "")
    if not left_norm or not right_norm:
        return False
    return _compact(left_norm) == _compact(right_norm)


def _bundle_query_text(runtime: Any) -> str:
    bundle = getattr(runtime, "bundle", None)
    if bundle is None:
        return ""
    query = getattr(bundle, "stage1_query", "") or ""
    return query if isinstance(query, str) else ""


def _proposal_category_map(runtime: Any) -> dict[int, str]:
    ctx = getattr(runtime, "task_ctx", None)
    proposals = list(getattr(ctx, "proposals", []) or [])
    category_by_pid: dict[int, str] = {}
    for proposal in proposals:
        proposal_id = getattr(proposal, "id", None)
        category = getattr(proposal, "category", None)
        if isinstance(proposal_id, int) and isinstance(category, str) and category:
            category_by_pid[proposal_id] = _normalize_category(category)
    return category_by_pid


def _json_dict(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _trace_list_scene_categories(runtime: Any) -> list[str]:
    """Return categories from the most recent usable list_scene_proposals call."""
    for entry in reversed(list(getattr(runtime, "tool_trace", []) or [])):
        tool_name = str(getattr(entry, "tool_name", "") or "")
        if tool_name != "list_scene_proposals":
            continue
        payload = _json_dict(str(getattr(entry, "response_text", "") or ""))
        categories: list[str] = []
        proposals = payload.get("proposals")
        if isinstance(proposals, list):
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                category = proposal.get("category")
                if isinstance(category, str) and category:
                    categories.append(_normalize_category(category))
        category = payload.get("category")
        if isinstance(category, str) and category:
            categories.append(_normalize_category(category))
        unique = _unique(categories)
        if unique:
            return unique
    return []


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _category_aliases(category: str) -> tuple[str, ...]:
    category_norm = _normalize_category(category)
    return _LABEL_ALIASES.get(category_norm, (category_norm,))


def _alias_mentioned(text: str, alias: str) -> bool:
    alias_norm = " ".join(str(alias).lower().split())
    if not alias_norm:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(
        re.escape(part) for part in alias_norm.split()
    )
    pattern += r"(?![a-z0-9])"
    return re.search(pattern, str(text).lower()) is not None


def _query_mentions_category(text: str, category: str) -> bool:
    return any(_alias_mentioned(text, alias) for alias in _category_aliases(category))


def _mentioned_categories(text: str, categories: list[str]) -> list[str]:
    return [category for category in categories if _query_mentions_category(text, category)]


def _split_clauses(query: str) -> list[str]:
    return [
        clause.strip()
        for clause in re.split(r"[.?!;]\s*|,\s+", query)
        if clause.strip()
    ]


def _match_label_to_category(label: str, categories: list[str]) -> str | None:
    label_norm = _normalize_category(label)
    for category in categories:
        if _category_matches(label_norm, category):
            return category
    matches = [
        category
        for category in categories
        if any(
            _alias_mentioned(label_norm, alias) for alias in _category_aliases(category)
        )
    ]
    unique = _unique(matches)
    return unique[0] if len(unique) == 1 else None


def _head_category_from_query(query: str, categories: list[str]) -> str | None:
    if not query or not categories:
        return None

    clauses = _split_clauses(query)
    explicit_found = False
    explicit_candidates: list[str] = []
    for clause in clauses:
        for target in _WANT_HEAD_RE.finditer(clause):
            explicit_found = True
            category = _match_label_to_category(target.group("label"), categories)
            if category is not None:
                explicit_candidates.append(category)

    if explicit_found:
        unique_explicit = _unique(explicit_candidates)
        return unique_explicit[0] if len(unique_explicit) == 1 else None

    head_candidates: list[str] = []
    for clause in clauses:
        leading = _LEADING_HEAD_RE.search(clause)
        if leading is not None:
            category = _match_label_to_category(leading.group("label"), categories)
            if category is not None:
                head_candidates.append(category)
    unique_heads = _unique(head_candidates)
    if unique_heads:
        return unique_heads[0] if len(unique_heads) == 1 else None

    return None


def _expected_category(runtime: Any) -> str | None:
    query = _bundle_query_text(runtime)
    proposal_categories = _unique(list(_proposal_category_map(runtime).values()))
    trace_categories = _trace_list_scene_categories(runtime)

    return _head_category_from_query(query, _unique(proposal_categories + trace_categories))


def _is_visual_grounding(runtime: Any) -> bool:
    task_type = getattr(runtime, "task_type", None)
    return str(getattr(task_type, "name", task_type)) == "VISUAL_GROUNDING"


def _format_message(
    *,
    expected_category: str,
    submitted_category: str,
    submitted_pid: int,
) -> str:
    return (
        "TARGET_CATEGORY_GUARD: the query appears to target category "
        f"{expected_category!r}, but submit_final is submitting proposal "
        f"{submitted_pid} with category {submitted_category!r}. Re-check the "
        "target head noun and submit a proposal whose label matches the target "
        "category unless visual evidence clearly justifies an alias-compatible "
        "category."
    )


def evaluate_target_category_guard(
    runtime: Any,
    payload: dict | Any,
) -> TargetCategoryDecision:
    """Block VG final submissions whose label drifts from the target head."""
    if not _is_visual_grounding(runtime):
        return TargetCategoryDecision(blocked=False)

    submitted_pid = _proposal_id(_payload_dict(payload).get("proposal_id"))
    if submitted_pid is None or submitted_pid == -1:
        return TargetCategoryDecision(blocked=False)

    submitted_category = _proposal_category_map(runtime).get(submitted_pid)
    if submitted_category is None:
        return TargetCategoryDecision(blocked=False, submitted_pid=submitted_pid)

    expected_category = _expected_category(runtime)
    if expected_category is None:
        return TargetCategoryDecision(
            blocked=False,
            submitted_pid=submitted_pid,
            submitted_category=submitted_category,
        )

    if _category_matches(expected_category, submitted_category):
        return TargetCategoryDecision(
            blocked=False,
            submitted_pid=submitted_pid,
            expected_category=expected_category,
            submitted_category=submitted_category,
        )

    message = _format_message(
        expected_category=expected_category,
        submitted_category=submitted_category,
        submitted_pid=submitted_pid,
    )
    return TargetCategoryDecision(
        blocked=True,
        message=message,
        submitted_pid=submitted_pid,
        expected_category=expected_category,
        submitted_category=submitted_category,
    )


def target_category_guard_record_fields(
    decision: TargetCategoryDecision,
) -> dict[str, Any]:
    """Canonical trace fields for target-category guard submit records."""
    return {
        "target_category_guard_blocked": bool(decision.blocked),
        "target_category_guard_submitted_pid": decision.submitted_pid,
        "target_category_guard_expected_category": decision.expected_category,
        "target_category_guard_submitted_category": decision.submitted_category,
        "target_category_guard_message": decision.message or None,
    }


__all__ = [
    "TargetCategoryDecision",
    "evaluate_target_category_guard",
    "target_category_guard_record_fields",
]
