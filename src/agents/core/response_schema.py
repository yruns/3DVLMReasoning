"""Response schemas for Stage-2 agent outputs.

This module defines the structured response and tool result types.
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .agent_config import Stage2Status, Stage2TaskType

if TYPE_CHECKING:
    from .task_types import Stage2EvidenceBundle


class Stage2EvidenceCitation(BaseModel):
    """One evidence-backed claim in the final response."""

    claim: str = ""
    frame_indices: list[int] = Field(default_factory=list)
    object_terms: list[str] = Field(default_factory=list)


class Stage2ToolObservation(BaseModel):
    """Recorded tool usage during a single agent run."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    response_text: str = ""


class Stage2ToolResult(BaseModel):
    """Normalized result returned by Stage-2 evidence tools."""

    response_text: str
    updated_bundle: "Stage2EvidenceBundle | None" = None


class Stage2StructuredResponse(BaseModel):
    """Unified structured output envelope returned by Stage 2."""

    task_type: Stage2TaskType
    status: Stage2Status = Stage2Status.COMPLETED
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainties: list[str] = Field(default_factory=list)
    cited_frame_indices: list[int] = Field(default_factory=list)
    evidence_items: list[Stage2EvidenceCitation] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Stage2EvidenceCitation",
    "Stage2ToolObservation",
    "Stage2ToolResult",
    "Stage2StructuredResponse",
]
