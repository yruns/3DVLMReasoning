"""Per-question run context shared between the pilot and the LLM-call wrapper.

The OpenEQA pilot runs questions in a ThreadPoolExecutor. Each worker thread
processes one question_id at a time. The token-logger wrapper monkey-patches
``BaseChatOpenAI._generate`` and needs to know which question is being
processed at the moment of each LLM call so the per-call usage row can carry
``question_id``.

Use ``set_question_id(qid)`` from inside the worker; ``get_question_id()``
reads it. Implemented with a ContextVar so it's thread-safe and natural to
copy across asyncio boundaries (``contextvars.copy_context()``).

For ``ThreadPoolExecutor``, the executor does NOT propagate context by
default; callers must wrap the worker function with ``copy_context().run``
or set the contextvar on each thread's first call.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

_qid_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "stage2_question_id", default=None
)


def set_question_id(qid: str | None) -> contextvars.Token:
    """Set the current question_id in this context. Returns a reset token."""
    return _qid_var.set(qid)


def get_question_id() -> str | None:
    """Return the current question_id, or None if not set."""
    return _qid_var.get()


@contextmanager
def question_id_scope(qid: str | None):
    """Context manager: bind question_id for the duration of a block."""
    token = _qid_var.set(qid)
    try:
        yield
    finally:
        _qid_var.reset(token)


__all__ = ["set_question_id", "get_question_id", "question_id_scope"]
