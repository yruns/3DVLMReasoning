"""QA FinalizerSpec validates answer payloads."""
from __future__ import annotations

import pytest

from agents.packs.qa_default.finalizer import QA_FINALIZER, QaPayload


def test_qa_validator_rejects_empty_answer() -> None:
    payload = QaPayload(answer="   ", supporting_claims=[])

    with pytest.raises(ValueError, match="answer must be non-empty"):
        QA_FINALIZER.validator(payload, runtime=None)


def test_qa_validator_strips_whitespace() -> None:
    payload = QaPayload(
        answer="  It is on the table.  ",
        supporting_claims=["  frame 1 shows the table  "],
    )

    result = QA_FINALIZER.validator(payload, runtime=None)

    assert result.answer == "It is on the table."
    assert result.supporting_claims == ["frame 1 shows the table"]


def test_qa_adapter_emits_supporting_claims() -> None:
    payload = QaPayload(
        answer="It is on the table.",
        supporting_claims=["frame 1 shows the table"],
    )
    validated = QA_FINALIZER.validator(payload, runtime=None)

    out = QA_FINALIZER.adapter(validated, runtime=None)

    assert out == {
        "status": "completed",
        "answer": "It is on the table.",
        "supporting_claims": ["frame 1 shows the table"],
    }
