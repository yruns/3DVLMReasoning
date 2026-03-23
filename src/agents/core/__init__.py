"""Core agent types and configurations.

This module exports the fundamental types used by Stage-2 agents.
"""

from .agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2Status,
    Stage2TaskType,
)
from .response_schema import (
    Stage2EvidenceCitation,
    Stage2StructuredResponse,
    Stage2ToolObservation,
    Stage2ToolResult,
)
from .task_types import (
    KeyframeEvidence,
    Stage1HypothesisSummary,
    Stage2AgentResult,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
)

# Resolve Pydantic forward references for cross-module schemas.
Stage2ToolResult.model_rebuild(
    _types_namespace={"Stage2EvidenceBundle": Stage2EvidenceBundle}
)
Stage2AgentResult.model_rebuild(
    _types_namespace={
        "Stage2StructuredResponse": Stage2StructuredResponse,
        "Stage2ToolObservation": Stage2ToolObservation,
        "Stage2EvidenceBundle": Stage2EvidenceBundle,
    }
)

__all__ = [
    # Configuration
    "Stage2TaskType",
    "Stage2PlanMode",
    "Stage2Status",
    "Stage2DeepAgentConfig",
    # Response schema
    "Stage2EvidenceCitation",
    "Stage2StructuredResponse",
    "Stage2ToolObservation",
    "Stage2ToolResult",
    # Task types
    "Stage2TaskSpec",
    "KeyframeEvidence",
    "Stage1HypothesisSummary",
    "Stage2EvidenceBundle",
    "Stage2AgentResult",
]
