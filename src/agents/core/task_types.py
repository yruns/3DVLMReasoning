"""Task types and specifications for Stage-2 agents.

This module defines the task input/output structures.
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .agent_config import Stage2PlanMode, Stage2TaskType

if TYPE_CHECKING:
    from .response_schema import Stage2StructuredResponse, Stage2ToolObservation


class Stage2TaskSpec(BaseModel):
    """Downstream task specification for the Stage-2 agent."""

    task_type: Stage2TaskType = Stage2TaskType.GENERAL
    user_query: str = Field(..., min_length=1)
    output_instruction: str = ""
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    plan_mode: Stage2PlanMode = Stage2PlanMode.BRIEF
    max_reasoning_turns: int = Field(default=6, ge=1, le=12)


class KeyframeEvidence(BaseModel):
    """One visual evidence item produced by Stage 1."""

    keyframe_idx: int = Field(..., ge=0)
    image_path: str
    view_id: int | None = None
    frame_id: int | None = None
    score: float | None = None
    note: str = ""


class Stage1HypothesisSummary(BaseModel):
    """Compact summary of Stage-1 query grounding metadata."""

    status: str = ""
    hypothesis_kind: str = ""
    hypothesis_rank: int | None = None
    parse_mode: str = ""
    raw_query: str = ""
    target_categories: list[str] = Field(default_factory=list)
    anchor_categories: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Stage2EvidenceBundle(BaseModel):
    """Evidence package passed from Stage 1 into the agent."""

    scene_id: str = ""
    stage1_query: str = ""
    keyframes: list[KeyframeEvidence] = Field(default_factory=list)
    bev_image_path: str | None = None
    scene_summary: str = ""
    object_context: dict[str, str] = Field(default_factory=dict)
    hypothesis: Stage1HypothesisSummary | None = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class Stage2AgentResult(BaseModel):
    """End-to-end Stage-2 execution result."""

    task: Stage2TaskSpec
    result: "Stage2StructuredResponse"
    tool_trace: list["Stage2ToolObservation"] = Field(default_factory=list)
    final_bundle: Stage2EvidenceBundle
    raw_state: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Stage2TaskSpec",
    "KeyframeEvidence",
    "Stage1HypothesisSummary",
    "Stage2EvidenceBundle",
    "Stage2AgentResult",
]
