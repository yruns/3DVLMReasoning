"""Shared schemas for the Stage-2 research agent.

This module re-exports types from the core subpackage for backward compatibility.
New code should import directly from agents.core or agents.core.*.
"""

from .core import (
    KeyframeEvidence,
    Stage1HypothesisSummary,
    Stage2AgentResult,
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2EvidenceCitation,
    Stage2PlanMode,
    Stage2Status,
    Stage2StructuredResponse,
    Stage2TaskSpec,
    Stage2TaskType,
    Stage2ToolObservation,
    Stage2ToolResult,
)

__all__ = [
    "Stage2TaskType",
    "Stage2PlanMode",
    "Stage2Status",
    "Stage2DeepAgentConfig",
    "Stage2TaskSpec",
    "KeyframeEvidence",
    "Stage1HypothesisSummary",
    "Stage2EvidenceBundle",
    "Stage2EvidenceCitation",
    "Stage2ToolObservation",
    "Stage2ToolResult",
    "Stage2StructuredResponse",
    "Stage2AgentResult",
]
