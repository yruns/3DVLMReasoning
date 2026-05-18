"""Rationale/payload consistency guard for VG `submit_final`.

The guard uses only the submitted payload and the agent's own final
rationale. It does not inspect GT targets, GT bboxes, target visibility,
or per-case metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RationalePayloadDecision:
    """Outcome of final-rationale vs structured-payload consistency checks."""

    blocked: bool
    message: str = ""
    submitted_pid: int | None = None
    ruled_out_ids: tuple[int, ...] = ()
    final_ids: tuple[int, ...] = ()


_ID_REF_RE = re.compile(r"(?:#\s*(?P<hash>\d+)\b|proposal\s*#?\s*(?P<proposal>\d+)\b)", re.I)
_SPLIT_RE = re.compile(r"[.?!;\n]+")
_QUOTED_SPAN_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_NEGATIVE_BEFORE_ID_TEMPLATE = (
    r"\b(?:"
    r"rules?\s+out|ruled\s+out|"
    r"exclude(?:s|d|ing)?|to\s+exclude|"
    r"reject(?:s|ed|ing)?|eliminate(?:s|d|ing)?|discard(?:s|ed|ing)?|"
    r"not\s+(?:choose|select|pick|submit|use)"
    r")\b[^.?!;\n]{0,100}{id_ref}"
)
_NEGATIVE_AFTER_ID_TEMPLATE = (
    r"{id_ref}[^.?!;\n]{0,120}?\b(?:"
    r"to\s+exclude|should\s+be\s+excluded|must\s+be\s+excluded|"
    r"is\s+(?:the\s+)?(?:one\s+)?to\s+exclude|"
    r"(?:is|was|are|were)\s+not\s+(?:the\s+)?(?:target|answer|match|one|"
    r"candidate|proposal|referent|referenced|correct|chosen)|"
    r"(?:does|do|did)\s+not\s+match|"
    r"not\s+(?:choose|select|pick|submit|use)"
    r")\b"
)
_FINAL_ID_PATTERNS = (
    re.compile(
        r"\b(?:best|correct|final)\s+"
        r"(?:match|answer|candidate|proposal|one|choice)?\s*"
        r"(?:for\b[^.?!;\n]{0,180}?)?\s*"
        r"(?:is|=|:)\s*(?P<id>(?:#\s*\d+\b|proposal\s*#?\s*\d+\b))",
        re.I,
    ),
    re.compile(
        r"(?P<id>(?:#\s*\d+\b|proposal\s*#?\s*\d+\b))\s+"
        r"(?:is|as|being)\s+(?:the\s+)?(?:best|correct|final)\s+"
        r"(?:match|answer|candidate|proposal|one|choice)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:choose|choosing|select|selecting|pick|picking|submit|submitting|use)\s+"
        r"(?P<id>(?:#\s*\d+\b|proposal\s*#?\s*\d+\b))",
        re.I,
    ),
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
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    return None


def _id_ref_pattern(proposal_id: int) -> re.Pattern[str]:
    return re.compile(rf"(?:#\s*{proposal_id}\b|proposal\s*#?\s*{proposal_id}\b)", re.I)


def _ids_from_text(text: str) -> tuple[int, ...]:
    ids: list[int] = []
    for match in _ID_REF_RE.finditer(text):
        raw = match.group("hash") or match.group("proposal")
        if raw is None:
            continue
        proposal_id = int(raw)
        if proposal_id not in ids:
            ids.append(proposal_id)
    return tuple(ids)


def _split_clauses(text: str) -> list[str]:
    return [clause.strip() for clause in _SPLIT_RE.split(text) if clause.strip()]


def _strip_quoted_spans(text: str) -> str:
    return _QUOTED_SPAN_RE.sub(" ", text)


def _next_id_ref_start(clause: str, start: int) -> int:
    match = _ID_REF_RE.search(clause, start)
    return match.start() if match else len(clause)


def _previous_id_ref_end(clause: str, start: int) -> int:
    previous_end = 0
    for match in _ID_REF_RE.finditer(clause):
        if match.start() >= start:
            break
        previous_end = match.end()
    return previous_end


def _submitted_id_spans(clause: str, submitted_pid: int) -> list[tuple[int, int]]:
    submitted_ref = _id_ref_pattern(submitted_pid)
    return [(match.start(), match.end()) for match in submitted_ref.finditer(clause)]


def _ruled_out_submitted_id(rationale: str, submitted_pid: int) -> bool:
    submitted_ref = rf"(?:#\s*{submitted_pid}\b|proposal\s*#?\s*{submitted_pid}\b)"
    before_re = re.compile(
        _NEGATIVE_BEFORE_ID_TEMPLATE.replace("{id_ref}", submitted_ref),
        re.I,
    )
    after_re = re.compile(
        _NEGATIVE_AFTER_ID_TEMPLATE.replace("{id_ref}", submitted_ref),
        re.I,
    )
    for clause in _split_clauses(rationale):
        clause = _strip_quoted_spans(clause)
        spans = _submitted_id_spans(clause, submitted_pid)
        if not spans:
            continue
        for start, end in spans:
            previous_id_end = _previous_id_ref_end(clause, start)
            next_id_start = _next_id_ref_start(clause, end)
            before_segment = clause[previous_id_end:end]
            after_segment = clause[start:next_id_start]
            if before_re.search(before_segment) or after_re.search(after_segment):
                return True
    return False


def _final_ids(rationale: str) -> tuple[int, ...]:
    ids: list[int] = []
    for clause in _split_clauses(rationale):
        for pattern in _FINAL_ID_PATTERNS:
            for match in pattern.finditer(clause):
                for proposal_id in _ids_from_text(match.group("id")):
                    if proposal_id not in ids:
                        ids.append(proposal_id)
    return tuple(ids)


def evaluate_rationale_payload_guard(
    payload: dict | Any,
    *,
    rationale: str,
) -> RationalePayloadDecision:
    """Block stale final payloads contradicted by the final rationale."""
    submitted_pid = _proposal_id(_payload_dict(payload).get("proposal_id"))
    if submitted_pid is None or submitted_pid < 0:
        return RationalePayloadDecision(blocked=False, submitted_pid=submitted_pid)

    text = str(rationale or "")
    final_ids = _final_ids(text)

    if _ruled_out_submitted_id(text, submitted_pid):
        message = (
            f"RATIONALE_PAYLOAD_GUARD: rationale rules out submitted proposal "
            f"#{submitted_pid}; update payload.proposal_id to match the final "
            "rationale, or revise the rationale before submitting."
        )
        return RationalePayloadDecision(
            blocked=True,
            message=message,
            submitted_pid=submitted_pid,
            ruled_out_ids=(submitted_pid,),
            final_ids=final_ids,
        )

    conflicting_final_ids = tuple(pid for pid in final_ids if pid != submitted_pid)
    if conflicting_final_ids and submitted_pid not in final_ids:
        first = conflicting_final_ids[0]
        message = (
            f"RATIONALE_PAYLOAD_GUARD: rationale names final proposal #{first}, "
            f"but payload.proposal_id is #{submitted_pid}; update the payload "
            "or revise the rationale before submitting."
        )
        return RationalePayloadDecision(
            blocked=True,
            message=message,
            submitted_pid=submitted_pid,
            final_ids=final_ids,
        )

    return RationalePayloadDecision(
        blocked=False,
        submitted_pid=submitted_pid,
        final_ids=final_ids,
    )


def rationale_payload_guard_record_fields(
    decision: RationalePayloadDecision,
) -> dict[str, Any]:
    """Canonical trace fields for rationale/payload guard submit records."""
    return {
        "rationale_payload_guard_blocked": bool(decision.blocked),
        "rationale_payload_guard_submitted_pid": decision.submitted_pid,
        "rationale_payload_guard_ruled_out_ids": (
            list(decision.ruled_out_ids) if decision.ruled_out_ids else None
        ),
        "rationale_payload_guard_final_ids": (
            list(decision.final_ids) if decision.final_ids else None
        ),
        "rationale_payload_guard_message": decision.message or None,
    }


__all__ = [
    "RationalePayloadDecision",
    "evaluate_rationale_payload_guard",
    "rationale_payload_guard_record_fields",
]
