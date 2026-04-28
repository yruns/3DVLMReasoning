"""Wrapper that runs the OpenEQA official question pilot while logging
per-LLM-call token usage (including prompt-cache hits) to a JSONL file.

Usage:
    TOKEN_LOG=tmp/<run>/token_usage.jsonl \
    python scripts/run_openeqa_with_token_log.py \
        <same args as agents.examples.openeqa_official_question_pilot>

The patch records one JSONL line per chat-completion response with:
    {ts, model, prompt_tokens, completion_tokens, cached_tokens, session_id}

Roll-up:
    python -c "import json; from pathlib import Path; \
      rows=[json.loads(l) for l in Path('tmp/.../token_usage.jsonl').read_text().splitlines()]; \
      pt=sum(r['prompt_tokens'] for r in rows); ct=sum(r['cached_tokens'] for r in rows); \
      print(f'calls={len(rows)} prompt={pt:,} cached={ct:,} hit={ct/pt*100:.1f}%')"
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

_TOKEN_LOG = Path(os.environ.get("TOKEN_LOG", "tmp/token_usage.jsonl"))
_TOKEN_LOG.parent.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
# Open append-mode; flushed per write so tail -f works.
_FH = _TOKEN_LOG.open("a", buffering=1)


def _emit(row: dict) -> None:
    with _LOCK:
        _FH.write(json.dumps(row, separators=(",", ":")) + "\n")


def _extract_usage(message_or_meta) -> dict:
    """Pull token counts from a langchain AIMessage or response_metadata dict."""
    meta = {}
    if hasattr(message_or_meta, "response_metadata"):
        meta = message_or_meta.response_metadata or {}
    elif isinstance(message_or_meta, dict):
        meta = message_or_meta
    usage = meta.get("token_usage") or meta.get("usage") or {}
    ptd = usage.get("prompt_tokens_details") or {}
    out = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "cached_tokens": int(ptd.get("cached_tokens", 0) or 0),
    }
    # Some providers return usage_metadata on the message itself.
    if hasattr(message_or_meta, "usage_metadata") and message_or_meta.usage_metadata:
        um = message_or_meta.usage_metadata
        if not out["prompt_tokens"]:
            out["prompt_tokens"] = int(um.get("input_tokens", 0) or 0)
        if not out["completion_tokens"]:
            out["completion_tokens"] = int(um.get("output_tokens", 0) or 0)
        details = um.get("input_token_details") or {}
        if not out["cached_tokens"]:
            out["cached_tokens"] = int(details.get("cache_read", 0) or 0)
    return out


def _patch_chat_models() -> None:
    from langchain_openai.chat_models.base import BaseChatOpenAI

    # Lazy import: agents/_run_context only exists once PYTHONPATH=src is set.
    try:
        from agents._run_context import get_question_id
    except Exception:
        def get_question_id() -> str | None:  # type: ignore[misc]
            return None

    orig_generate = BaseChatOpenAI._generate

    def _wrapped_generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = orig_generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
        try:
            for gen in result.generations:
                msg = gen.message
                usage = _extract_usage(msg)
                row = {
                    "ts": time.time(),
                    "model": getattr(self, "model_name", None)
                    or getattr(self, "deployment_name", None)
                    or "?",
                    **usage,
                    "session_id": str(
                        (kwargs.get("extra_body") or {}).get("session_id", "")
                    ),
                    "question_id": get_question_id(),
                }
                _emit(row)
        except Exception as exc:  # never let logging break a run
            _emit({"ts": time.time(), "log_error": repr(exc)})
        return result

    BaseChatOpenAI._generate = _wrapped_generate


def main() -> None:
    _patch_chat_models()
    # Forward all CLI args verbatim to the pilot module.
    runpy.run_module(
        "agents.examples.openeqa_official_question_pilot",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
