"""QA FinalizerSpec: validate answer payloads and adapt to response fields."""
from __future__ import annotations

from pydantic import BaseModel

from agents.skills.finalizer import FinalizerSpec


class QaPayload(BaseModel):
    """submit_final payload schema for QA."""

    answer: str
    supporting_claims: list[str]


def qa_validator(payload: QaPayload, runtime) -> QaPayload:
    answer = payload.answer.strip()
    if not answer:
        raise ValueError("answer must be non-empty")
    supporting_claims = [claim.strip() for claim in payload.supporting_claims]
    return QaPayload(answer=answer, supporting_claims=supporting_claims)


def qa_adapter(payload: QaPayload, runtime) -> dict:
    """Convert validated QA payload to Stage2StructuredResponse payload fields."""
    return {
        "status": "completed",
        "answer": payload.answer,
        "supporting_claims": list(payload.supporting_claims),
    }


QA_FINALIZER = FinalizerSpec(
    payload_model=QaPayload,
    validator=qa_validator,
    adapter=qa_adapter,
)


__all__ = ["QA_FINALIZER", "QaPayload", "qa_adapter", "qa_validator"]
