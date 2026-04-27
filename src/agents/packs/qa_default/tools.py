"""Default QA pack tools.

QA currently reuses the runtime's shared Stage-2 tools, so the pack contributes
no task-specific tools beyond the chassis trio.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool


def build_qa_tools(runtime: Any) -> list[BaseTool]:
    return []


__all__ = ["build_qa_tools"]
