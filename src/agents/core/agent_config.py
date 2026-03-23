"""Core agent configuration and enums.

This module defines the agent configuration and enumeration types.
"""

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Stage2TaskType(str, Enum):
    """Supported downstream task families."""

    QA = "qa"
    VISUAL_GROUNDING = "visual_grounding"
    NAV_PLAN = "nav_plan"
    MANIPULATION = "manipulation"
    GENERAL = "general"


class Stage2PlanMode(str, Enum):
    """How much explicit planning the DeepAgent should do."""

    OFF = "off"
    BRIEF = "brief"
    FULL = "full"


class Stage2Status(str, Enum):
    """Unified status values for Stage-2 outputs."""

    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    FAILED = "failed"


class Stage2DeepAgentConfig(BaseModel):
    """Runtime configuration for the DeepAgents-backed Stage-2 agent."""

    base_url: str = "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl"
    model_name: str = "gpt-5.2-2025-12-11"
    api_key: str = "Eyt11Oeoj77MfGcMweDRODBsbYnPkWUp"
    api_version: str = "2024-03-01-preview"
    max_tokens: int = Field(default=10000, ge=1)
    temperature: float = 0.1
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=2, ge=0)
    include_thoughts: bool = False
    session_id: str = Field(default_factory=lambda: str(time.time()))
    extra_body: dict[str, Any] = Field(default_factory=dict)
    max_images: int = Field(default=6, ge=1, le=12)
    image_max_size: int = Field(default=900, ge=256, le=2048)
    enable_subagents: bool = True
    # Uncertainty-aware stopping configuration
    confidence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required to complete a task. "
        "Below this threshold, the agent returns 'insufficient_evidence' status.",
    )
    enable_uncertainty_stopping: bool = Field(
        default=True,
        description="When True, agent will stop with 'insufficient_evidence' if confidence "
        "is below threshold and no more evidence can be acquired.",
    )


__all__ = [
    "Stage2TaskType",
    "Stage2PlanMode",
    "Stage2Status",
    "Stage2DeepAgentConfig",
]
