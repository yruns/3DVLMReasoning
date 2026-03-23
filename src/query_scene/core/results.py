"""Execution result structures.

This module defines the result types returned by query execution,
including keyframe selection and grounding results.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .query_types import GroundingResult, ObjectNode, ViewScore


@dataclass
class KeyframeResult:
    """Result of keyframe selection for an object.

    Attributes:
        object_id: ID of the target object
        frame_ids: Selected frame indices (sorted by score)
        scores: ViewScore for each selected frame
        total_candidates: Total frames where object is visible
    """

    object_id: int
    frame_ids: list[int] = field(default_factory=list)
    scores: list[ViewScore] = field(default_factory=list)
    total_candidates: int = 0

    @property
    def top_frame(self) -> int | None:
        """Get the top-ranked frame ID."""
        return self.frame_ids[0] if self.frame_ids else None

    @property
    def top_score(self) -> ViewScore | None:
        """Get the top-ranked score."""
        return self.scores[0] if self.scores else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "frame_ids": self.frame_ids,
            "scores": [s.to_dict() for s in self.scores],
            "total_candidates": self.total_candidates,
        }


@dataclass
class ExecutionResult:
    """Result of query execution.

    Attributes:
        success: Whether execution succeeded
        grounding: Grounding result (if successful)
        keyframes: Keyframe selection results
        timing: Execution timing breakdown
        diagnostics: Debug information
    """

    success: bool = False
    grounding: GroundingResult | None = None
    keyframes: KeyframeResult | None = None
    timing: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(
        cls, reason: str, diagnostics: dict[str, Any] | None = None
    ) -> "ExecutionResult":
        """Create a failure result."""
        return cls(
            success=False,
            grounding=GroundingResult.failure(reason),
            diagnostics=diagnostics or {},
        )

    @classmethod
    def from_grounding(
        cls,
        grounding: GroundingResult,
        keyframes: KeyframeResult | None = None,
        timing: dict[str, float] | None = None,
    ) -> "ExecutionResult":
        """Create from a grounding result."""
        return cls(
            success=grounding.success,
            grounding=grounding,
            keyframes=keyframes,
            timing=timing or {},
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "success": self.success,
            "timing": self.timing,
            "diagnostics": self.diagnostics,
        }
        if self.grounding is not None:
            result["grounding"] = self.grounding.to_dict()
        if self.keyframes is not None:
            result["keyframes"] = self.keyframes.to_dict()
        return result


@dataclass
class EvidenceBundle:
    """Bundle of visual evidence for VLM reasoning.

    Attributes:
        query: Original query string
        target_object: Target object (if identified)
        keyframes: Selected keyframe data
        crops: Object crop images
        bev_image: Bird's eye view image
        context: Additional context information
    """

    query: str
    target_object: ObjectNode | None = None
    keyframes: list[dict[str, Any]] = field(default_factory=list)
    crops: list[np.ndarray] = field(default_factory=list)
    bev_image: np.ndarray | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def num_keyframes(self) -> int:
        return len(self.keyframes)

    @property
    def has_crops(self) -> bool:
        return len(self.crops) > 0

    @property
    def has_bev(self) -> bool:
        return self.bev_image is not None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "query": self.query,
            "num_keyframes": self.num_keyframes,
            "has_crops": self.has_crops,
            "has_bev": self.has_bev,
            "context": self.context,
        }
        if self.target_object is not None:
            result["target_object"] = self.target_object.to_dict()
        return result
