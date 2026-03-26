"""Retrieval module for query-driven keyframe selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .indices import (
    CLIPIndex,
    PointLevelIndex,
    RegionIndex,
    SceneIndices,
    SpatialIndex,
    VisibilityIndex,
)
from .spatial_checker import RelationResult, SpatialRelationChecker

if TYPE_CHECKING:  # pragma: no cover
    from ..keyframe_selector import KeyframeResult, KeyframeSelector, SceneObject

__all__ = [
    "CLIPIndex",
    "VisibilityIndex",
    "SpatialIndex",
    "RegionIndex",
    "PointLevelIndex",
    "SceneIndices",
    "KeyframeSelector",
    "KeyframeResult",
    "SceneObject",
    "SpatialRelationChecker",
    "RelationResult",
]


def __getattr__(name: str) -> Any:
    """Lazily expose canonical selector types from ``query_scene.keyframe_selector``."""
    if name in {"KeyframeSelector", "KeyframeResult", "SceneObject"}:
        from ..keyframe_selector import KeyframeResult, KeyframeSelector, SceneObject

        mapping = {
            "KeyframeSelector": KeyframeSelector,
            "KeyframeResult": KeyframeResult,
            "SceneObject": SceneObject,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
