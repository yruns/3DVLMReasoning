"""Final evidence-frame consistency guard for VG `submit_final`.

The guard uses only the agent's own tool trace and final rationale. It
does not inspect GT targets, GT bboxes, target visibility, or metrics.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceFrameGuardDecision:
    """Outcome of an evidence-frame guard evaluation."""

    blocked: bool
    message: str = ""
    submitted_pid: int | None = None
    cited_frame_ids: tuple[int, ...] = ()
    visible_proposal_ids: tuple[int, ...] = ()
    visible_proposal_labels: tuple[str, ...] = ()
    relative_position_conflict: bool = False
    relative_position_direction: str | None = None
    relative_position_alternatives: tuple[int, ...] = ()
    relative_position_alternative_labels: tuple[str, ...] = ()


_VISIBLE_RE = re.compile(r"visible_proposals=(\[[^\]]*\])")
_CATEGORIES_RE = re.compile(r"categories=(\[[^\]]*\])")
_LEFT_TO_RIGHT_RE = re.compile(r"left_to_right=(\[[^\]]*\])")
_BOXES_2D_RE = re.compile(r"boxes_2d=({[^}]*})")
_FRAME_CITATION_RE = re.compile(r"\bframe(?:[_\s-]?id)?[_\s-]*(\d+)\b", re.I)
_LEFT_RELATION_RE = re.compile(r"\b(?:to\s+the\s+)?left\s+of\b|\bleft-hand\b", re.I)
_RIGHT_RELATION_RE = re.compile(r"\b(?:to\s+the\s+)?right\s+of\b|\bright-hand\b", re.I)
_ANCHOR_RELATIVE_TO_TARGET_PATTERNS = {
    "left": re.compile(
        r"\b(?:with|has|having|includes?|including|and)\b[^.?!]{0,120}"
        r"\b(?:to\s+the\s+)?left\s+of\s+it\b",
        re.I,
    ),
    "right": re.compile(
        r"\b(?:with|has|having|includes?|including|and)\b[^.?!]{0,120}"
        r"\b(?:to\s+the\s+)?right\s+of\s+it\b",
        re.I,
    ),
}
_RELATION_RATIONALE_PATTERNS = {
    "above": re.compile(r"\b(?:above|over|on\s+top\s+of)\b", re.I),
    "below": re.compile(r"\b(?:below|under|beneath)\b", re.I),
    "left_of": re.compile(r"\b(?:to\s+the\s+)?left\s+of\b|\bleft-hand\b", re.I),
    "right_of": re.compile(r"\b(?:to\s+the\s+)?right\s+of\b|\bright-hand\b", re.I),
    "near": re.compile(r"\b(?:near|nearby)\b", re.I),
    "next_to": re.compile(r"\b(?:next\s+to|beside|adjacent\s+to)\b", re.I),
    "closest_to": re.compile(r"\b(?:closest\s+to|nearest\s+to)\b", re.I),
}
_PROXIMITY_RELATIONS = {"near", "next_to", "closest_to"}
_LABEL_QUERY_ALIASES = {
    "bookshelf": ("bookshelf", "bookcase", "book case"),
    "whiteboard": ("whiteboard", "white board"),
}


def _payload_dict(payload: dict | Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("payload")
    if isinstance(inner, dict) and "proposal_id" in inner:
        return inner
    return payload


def _literal_list(pattern: re.Pattern[str], text: str) -> list[Any]:
    match = pattern.search(text)
    if match is None:
        return []
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _literal_dict(pattern: re.Pattern[str], text: str) -> dict[Any, Any]:
    match = pattern.search(text)
    if match is None:
        return {}
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _tool_name(entry: Any) -> str:
    return str(getattr(entry, "tool_name", "") or "")


def _tool_input(entry: Any) -> dict[str, Any]:
    value = getattr(entry, "tool_input", {}) or {}
    return value if isinstance(value, dict) else {}


def _response_text(entry: Any) -> str:
    return str(getattr(entry, "response_text", "") or "")


def _parse_left_to_right(response: str) -> list[int]:
    order: list[int] = []
    for raw in _literal_list(_LEFT_TO_RIGHT_RE, response):
        if isinstance(raw, int):
            proposal_id = raw
        elif isinstance(raw, str):
            prefix = raw.split(":", 1)[0].strip()
            try:
                proposal_id = int(prefix)
            except ValueError:
                continue
        else:
            continue
        if proposal_id not in order:
            order.append(proposal_id)
    return order


def _parse_boxes_2d(response: str) -> dict[int, tuple[float, float, float, float]]:
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for raw_key, raw_box in _literal_dict(_BOXES_2D_RE, response).items():
        try:
            proposal_id = int(raw_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
        except (TypeError, ValueError):
            continue
        boxes[proposal_id] = (x1, y1, x2, y2)
    return boxes


def _is_marked_view(entry: Any) -> bool:
    """v9.1: only `mark_frame_with_bbox` entries count as marked evidence."""
    return _tool_name(entry) == "mark_frame_with_bbox"


def _viewed_frame_map(runtime: Any) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    for entry in list(getattr(runtime, "tool_trace", []) or []):
        if not _is_marked_view(entry):
            continue
        frame_id = _tool_input(entry).get("frame_id")
        if not isinstance(frame_id, int):
            continue
        response = _response_text(entry)
        visible_ids = _literal_list(_VISIBLE_RE, response)
        categories = _literal_list(_CATEGORIES_RE, response)
        pairs: list[tuple[int, str]] = []
        for idx, proposal_id in enumerate(visible_ids):
            if not isinstance(proposal_id, int):
                continue
            label = str(categories[idx]) if idx < len(categories) else ""
            pairs.append((proposal_id, label))
        if pairs:
            boxes_2d = _parse_boxes_2d(response)
            frames[frame_id] = {
                "pairs": pairs,
                "left_to_right": _parse_left_to_right(response),
                "boxes_2d": boxes_2d,
            }
    return frames


def _cited_frame_ids(rationale: str, evidence_refs: list[dict] | None) -> list[int]:
    frame_ids: list[int] = []
    for match in _FRAME_CITATION_RE.finditer(rationale or ""):
        try:
            frame_id = int(match.group(1))
        except ValueError:
            continue
        if frame_id not in frame_ids:
            frame_ids.append(frame_id)

    for ref in evidence_refs or []:
        if not isinstance(ref, dict):
            continue
        for key in ("frame_id", "frame"):
            value = ref.get(key)
            if isinstance(value, int) and value not in frame_ids:
                frame_ids.append(value)
    return frame_ids


def _format_pairs(pairs: list[tuple[int, str]]) -> str:
    return ", ".join(f"{pid}:{label or '?'}" for pid, label in pairs[:20])


def _desired_relative_direction(runtime: Any, rationale: str) -> str | None:
    bundle = getattr(runtime, "bundle", None)
    query = str(getattr(bundle, "stage1_query", "") or "")
    if _anchor_relative_to_target_direction(query) is not None:
        return None
    if _anchor_relative_to_target_direction(rationale or "") is not None:
        return None
    if _LEFT_RELATION_RE.search(query):
        return "left"
    if _RIGHT_RELATION_RE.search(query):
        return "right"
    if _LEFT_RELATION_RE.search(rationale or ""):
        return "left"
    if _RIGHT_RELATION_RE.search(rationale or ""):
        return "right"
    return None


def _anchor_relative_to_target_direction(text: str) -> str | None:
    for direction, pattern in _ANCHOR_RELATIVE_TO_TARGET_PATTERNS.items():
        if pattern.search(text or ""):
            return direction
    return None


def _compact_label_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _query_mentions_label(query: str, label: str) -> bool:
    label_norm = " ".join(str(label).lower().split())
    if not label_norm:
        return False
    compact_query = _compact_label_text(query)
    aliases = _LABEL_QUERY_ALIASES.get(label_norm, (label_norm,))
    return any(_compact_label_text(alias) in compact_query for alias in aliases)


def _alternatives_are_query_anchors(
    runtime: Any,
    *,
    submitted_label: str,
    alternatives: list[tuple[int, str]],
) -> bool:
    bundle = getattr(runtime, "bundle", None)
    query = str(getattr(bundle, "stage1_query", "") or "")
    if not _query_mentions_label(query, submitted_label):
        return False
    submitted_norm = " ".join(str(submitted_label).lower().split())
    return all(
        label
        and " ".join(str(label).lower().split()) != submitted_norm
        and _query_mentions_label(query, label)
        for _, label in alternatives
    )


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_height(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[3] - box[1])


def _is_comparable_mark(
    candidate_box: tuple[float, float, float, float],
    submitted_box: tuple[float, float, float, float],
) -> bool:
    submitted_height = _box_height(submitted_box)
    submitted_area = _box_area(submitted_box)
    if submitted_height <= 0 or submitted_area <= 0:
        return False
    return (
        _box_height(candidate_box) >= 0.60 * submitted_height
        and _box_area(candidate_box) >= 0.35 * submitted_area
    )


def _relative_position_alternatives(
    frame_data: dict[str, Any],
    *,
    submitted_pid: int,
    direction: str,
) -> list[tuple[int, str]]:
    order = frame_data.get("left_to_right") or []
    boxes = frame_data.get("boxes_2d") or {}
    if submitted_pid not in order or submitted_pid not in boxes:
        return []
    submitted_index = order.index(submitted_pid)
    candidates = (
        order[:submitted_index] if direction == "left" else order[submitted_index + 1 :]
    )
    labels = dict(frame_data.get("pairs", []))
    submitted_box = boxes[submitted_pid]
    alternatives: list[tuple[int, str]] = []
    for proposal_id in candidates:
        candidate_box = boxes.get(proposal_id)
        if candidate_box is None:
            continue
        if not _is_comparable_mark(candidate_box, submitted_box):
            continue
        alternatives.append((proposal_id, labels.get(proposal_id, "")))
    return alternatives


def _candidate_ids_for_submitted_pid(
    runtime: Any,
    submitted_pid: int,
) -> set[int] | None:
    """Return the category-candidate set that contains the submitted proposal.

    `list_scene_proposals` (the v9 replacement for `find_proposals_by_category`)
    is the agent's trace-only record of the target category shortlist. If the
    submitted proposal belongs to one of those shortlists, relative-position
    guard alternatives should come from that same shortlist; otherwise the
    guard can accidentally force the agent to pick an anchor or unrelated
    object that merely lies farther left/right in the image.
    """
    for entry in list(getattr(runtime, "tool_trace", []) or []):
        if _tool_name(entry) not in ("list_scene_proposals", "find_proposals_by_category"):
            continue
        try:
            response = json.loads(_response_text(entry))
        except json.JSONDecodeError:
            continue
        proposal_ids = response.get("proposal_ids")
        if not isinstance(proposal_ids, list):
            continue
        candidate_ids = {
            proposal_id for proposal_id in proposal_ids if isinstance(proposal_id, int)
        }
        if submitted_pid in candidate_ids:
            return candidate_ids
    return None


def _latest_spatial_compare_for_submission(
    runtime: Any,
    submitted_pid: int,
) -> dict[str, Any] | None:
    trace = list(getattr(runtime, "tool_trace", []) or [])
    for index in range(len(trace) - 1, -1, -1):
        entry = trace[index]
        if _tool_name(entry) != "compare_proposals_spatial":
            continue
        tool_input = _tool_input(entry)
        candidate_ids = tool_input.get("candidate_ids")
        if not isinstance(candidate_ids, list) or submitted_pid not in candidate_ids:
            continue
        anchor_id = tool_input.get("anchor_id")
        relation = tool_input.get("relation")
        if not isinstance(anchor_id, int) or not isinstance(relation, str):
            continue
        try:
            response = json.loads(_response_text(entry))
        except json.JSONDecodeError:
            response = {}
        ranked_ids = response.get("ranked_ids")
        return {
            "anchor_id": anchor_id,
            "relation": relation,
            "ranked_ids": ranked_ids if isinstance(ranked_ids, list) else [],
        }
    return None


def _rationale_mentions_relation(rationale: str, relation: str) -> bool:
    pattern = _RELATION_RATIONALE_PATTERNS.get(relation)
    return bool(pattern and pattern.search(rationale or ""))


def _spatial_compare_supports_submission(
    spatial_compare: dict[str, Any] | None,
    submitted_pid: int,
) -> bool:
    if spatial_compare is None:
        return False
    relation = spatial_compare.get("relation")
    if relation not in _PROXIMITY_RELATIONS:
        return False
    ranked_ids = spatial_compare.get("ranked_ids")
    return bool(ranked_ids and ranked_ids[0] == submitted_pid)


def _mark_evidence_frame_guard_triggered(runtime: Any) -> None:
    runtime.evidence_frame_guard_triggered = True
    runtime.evidence_frame_guard_block_count = (
        int(getattr(runtime, "evidence_frame_guard_block_count", 0) or 0) + 1
    )


def evidence_frame_guard_record_fields(
    decision: EvidenceFrameGuardDecision,
) -> dict[str, Any]:
    """Canonical trace fields for evidence-frame guard submit records."""
    return {
        "evidence_frame_guard_blocked": bool(decision.blocked),
        "evidence_frame_guard_submitted_pid": decision.submitted_pid,
        "evidence_frame_guard_cited_frame_ids": (
            list(decision.cited_frame_ids) if decision.cited_frame_ids else None
        ),
        "evidence_frame_guard_visible_proposal_ids": (
            list(decision.visible_proposal_ids)
            if decision.visible_proposal_ids
            else None
        ),
        "evidence_frame_guard_visible_proposal_labels": (
            list(decision.visible_proposal_labels)
            if decision.visible_proposal_labels
            else None
        ),
        "evidence_frame_guard_relative_position_conflict": bool(
            decision.relative_position_conflict
        ),
        "evidence_frame_guard_relative_position_direction": (
            decision.relative_position_direction
        ),
        "evidence_frame_guard_relative_position_alternatives": (
            list(decision.relative_position_alternatives)
            if decision.relative_position_alternatives
            else None
        ),
        "evidence_frame_guard_relative_position_alternative_labels": (
            list(decision.relative_position_alternative_labels)
            if decision.relative_position_alternative_labels
            else None
        ),
        "evidence_frame_guard_message": decision.message or None,
    }


def evaluate_evidence_frame_guard(
    runtime: Any,
    payload: dict | Any,
    *,
    rationale: str,
    evidence_refs: list[dict] | None = None,
) -> EvidenceFrameGuardDecision:
    """Block final VG submissions inconsistent with cited marked frames."""
    if not bool(getattr(runtime, "use_evidence_frame_guard", False)):
        return EvidenceFrameGuardDecision(blocked=False)

    submitted_pid = _payload_dict(payload).get("proposal_id")
    if not isinstance(submitted_pid, int) or submitted_pid == -1:
        return EvidenceFrameGuardDecision(blocked=False)

    cited_frame_ids = _cited_frame_ids(rationale, evidence_refs)
    if not cited_frame_ids:
        return EvidenceFrameGuardDecision(blocked=False, submitted_pid=submitted_pid)

    frame_map = _viewed_frame_map(runtime)
    cited_visible: list[tuple[int, str]] = []
    cited_with_trace: list[int] = []
    submitted_visible = False
    submitted_visible_without_anchor: list[tuple[int, list[tuple[int, str]]]] = []
    submitted_visible_with_anchor = False
    spatial_compare = _latest_spatial_compare_for_submission(runtime, submitted_pid)
    direction = _desired_relative_direction(runtime, rationale)
    for frame_id in cited_frame_ids:
        frame_data = frame_map.get(frame_id)
        if not frame_data:
            continue
        pairs = list(frame_data["pairs"])
        cited_with_trace.append(frame_id)
        cited_visible.extend(pairs)
        if any(pid == submitted_pid for pid, _ in pairs):
            submitted_visible = True
            visible_ids = {pid for pid, _ in pairs}
            if spatial_compare is not None:
                anchor_id = spatial_compare["anchor_id"]
                if anchor_id in visible_ids:
                    submitted_visible_with_anchor = True
                else:
                    submitted_visible_without_anchor.append((frame_id, pairs))
            if direction is None:
                continue
            alternatives = _relative_position_alternatives(
                frame_data,
                submitted_pid=submitted_pid,
                direction=direction,
            )
            candidate_ids = _candidate_ids_for_submitted_pid(runtime, submitted_pid)
            if candidate_ids is not None:
                alternatives = [
                    (proposal_id, label)
                    for proposal_id, label in alternatives
                    if proposal_id in candidate_ids
                ]
            if not alternatives:
                continue
            submitted_label = dict(pairs).get(submitted_pid, "")
            if _alternatives_are_query_anchors(
                runtime,
                submitted_label=submitted_label,
                alternatives=alternatives,
            ):
                continue
            _mark_evidence_frame_guard_triggered(runtime)
            alt_ids = tuple(pid for pid, _ in alternatives)
            alt_labels = tuple(label for _, label in alternatives)
            message = (
                "EVIDENCE_FRAME_GUARD: left/right marked-frame geometry "
                f"conflicts with submitted proposal {submitted_pid}. The query "
                f"or rationale implies the target should be to the {direction} "
                f"in cited frame {frame_id}, but comparable marked proposal(s) "
                f"on that side are: {_format_pairs(alternatives)}. Use "
                "`left_to_right` and the 2D boxes from the cited marked frame "
                "to choose the proposal whose mark covers the requested side; "
                "proposal labels are weak priors."
            )
            return EvidenceFrameGuardDecision(
                blocked=True,
                message=message,
                submitted_pid=submitted_pid,
                cited_frame_ids=tuple(cited_with_trace),
                visible_proposal_ids=tuple(pid for pid, _ in pairs),
                visible_proposal_labels=tuple(label for _, label in pairs),
                relative_position_conflict=True,
                relative_position_direction=direction,
                relative_position_alternatives=alt_ids,
                relative_position_alternative_labels=alt_labels,
            )

    if submitted_visible:
        if (
            spatial_compare is not None
            and submitted_visible_without_anchor
            and not submitted_visible_with_anchor
            and _rationale_mentions_relation(rationale, spatial_compare["relation"])
            and not _spatial_compare_supports_submission(
                spatial_compare,
                submitted_pid,
            )
        ):
            _mark_evidence_frame_guard_triggered(runtime)
            anchor_id = spatial_compare["anchor_id"]
            relation = spatial_compare["relation"]
            frame_list = ", ".join(
                str(frame_id) for frame_id, _ in submitted_visible_without_anchor
            )
            visible_pairs: list[tuple[int, str]] = []
            for _, pairs in submitted_visible_without_anchor:
                visible_pairs.extend(pairs)
            message = (
                "EVIDENCE_FRAME_GUARD: rationale/evidence cites marked frame(s) "
                f"{frame_list} for relation '{relation}', and submitted proposal "
                f"{submitted_pid} is visible there, but anchor proposal {anchor_id} "
                "from the latest spatial comparison is not visible in those cited "
                "frame(s). Cite or inspect a marked frame that shows both the "
                "submitted target and the spatial anchor, or revise the rationale "
                "so target-only appearance evidence is not used as relation "
                "evidence. Visible proposals in cited frame(s): "
                f"{_format_pairs(visible_pairs)}."
            )
            return EvidenceFrameGuardDecision(
                blocked=True,
                message=message,
                submitted_pid=submitted_pid,
                cited_frame_ids=tuple(
                    frame_id for frame_id, _ in submitted_visible_without_anchor
                ),
                visible_proposal_ids=tuple(pid for pid, _ in visible_pairs),
                visible_proposal_labels=tuple(label for _, label in visible_pairs),
            )
        return EvidenceFrameGuardDecision(
            blocked=False,
            submitted_pid=submitted_pid,
            cited_frame_ids=tuple(cited_with_trace),
            relative_position_direction=direction,
        )

    if not cited_visible:
        if cited_frame_ids:
            _mark_evidence_frame_guard_triggered(runtime)
            frame_list = ", ".join(str(fid) for fid in cited_frame_ids)
            message = (
                "EVIDENCE_FRAME_GUARD: rationale cites frame(s) "
                f"{frame_list}, but there is no marked-frame evidence "
                "(`mark_frame_with_bbox`) for those frames in the tool trace. "
                "Selector-returned RGB alone is not sufficient; mark the cited "
                "frame(s) before submitting."
            )
            return EvidenceFrameGuardDecision(
                blocked=True,
                message=message,
                submitted_pid=submitted_pid,
                cited_frame_ids=tuple(cited_frame_ids),
            )
        return EvidenceFrameGuardDecision(blocked=False, submitted_pid=submitted_pid)

    _mark_evidence_frame_guard_triggered(runtime)
    visible_ids = tuple(pid for pid, _ in cited_visible)
    visible_labels = tuple(label for _, label in cited_visible)
    frame_list = ", ".join(str(fid) for fid in cited_with_trace)
    message = (
        "EVIDENCE_FRAME_GUARD: rationale/evidence cites marked frame(s) "
        f"{frame_list}, but submitted proposal {submitted_pid} is not visible "
        f"there. Visible proposals in cited frame(s): {_format_pairs(cited_visible)}. "
        "If a cited marked frame shows the target, submit the proposal id whose "
        "mark directly covers the target in that frame; proposal labels are weak "
        "priors, so a mislabeled proposal covering the target beats a semantically "
        "named proposal from another frame."
    )
    return EvidenceFrameGuardDecision(
        blocked=True,
        message=message,
        submitted_pid=submitted_pid,
        cited_frame_ids=tuple(cited_with_trace),
        visible_proposal_ids=visible_ids,
        visible_proposal_labels=visible_labels,
    )


__all__ = [
    "EvidenceFrameGuardDecision",
    "evaluate_evidence_frame_guard",
    "evidence_frame_guard_record_fields",
]
