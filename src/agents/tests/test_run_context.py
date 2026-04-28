"""Tests for agents._run_context (per-question_id contextvar)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from agents._run_context import (
    get_question_id,
    question_id_scope,
    set_question_id,
)


def test_default_is_none() -> None:
    assert get_question_id() is None


def test_set_and_read() -> None:
    token = set_question_id("q-abc")
    try:
        assert get_question_id() == "q-abc"
    finally:
        # Reset so other tests are not affected.
        from agents._run_context import _qid_var  # noqa: PLC2701

        _qid_var.reset(token)
    assert get_question_id() is None


def test_question_id_scope_nests() -> None:
    with question_id_scope("outer"):
        assert get_question_id() == "outer"
        with question_id_scope("inner"):
            assert get_question_id() == "inner"
        assert get_question_id() == "outer"
    assert get_question_id() is None


def test_thread_pool_isolation() -> None:
    """Each worker sets its own qid; reads should not bleed across threads."""

    def worker(qid: str) -> str | None:
        with question_id_scope(qid):
            return get_question_id()

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(worker, f"q-{i}") for i in range(20)]
        results = [f.result() for f in futs]

    assert results == [f"q-{i}" for i in range(20)]
    # Outer thread is unaffected.
    assert get_question_id() is None
