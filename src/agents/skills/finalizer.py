"""Per-task finalization contract."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type


@dataclass(frozen=True)
class FinalizerSpec:
    """How a pack validates + adapts its `submit_final` payload."""

    payload_model: Type[Any]
    validator: Callable[[Any, Any], Any]   # (payload, runtime) -> resolved
    adapter: Callable[[Any, Any], dict]    # (payload, runtime) -> dict for Stage2StructuredResponse


__all__ = ["FinalizerSpec"]
