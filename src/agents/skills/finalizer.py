"""Per-task finalization contract."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FinalizerSpec:
    """How a pack validates + adapts its `submit_final` payload."""

    payload_model: type[Any]
    validator: Callable[[Any, Any], Any]   # (payload, runtime) -> resolved
    adapter: Callable[[Any, Any], dict]    # (payload, runtime) -> dict for Stage2StructuredResponse


__all__ = ["FinalizerSpec"]
