"""Core agent configuration and enums.

This module defines the agent configuration and enumeration types.
"""

import os
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

    base_url: str = "https://aidp-i18ntt-sg.tiktok-row.net"
    model_name: str = "gpt-5.4-2026-03-05"
    api_keys: list[str] = Field(
        default_factory=lambda: _default_modelhub_api_keys(),
        description="ModelHub AKs rotated on retryable quota/rate-limit errors.",
    )
    modelhub_path: str = "/api/modelhub/online/v2/crawl"
    api_version: str = "2024-03-01-preview"
    max_tokens: int = Field(default=10000, ge=1)
    temperature: float = 0.1
    timeout: int = Field(default=120, ge=1)
    max_retries: int = Field(default=2, ge=0)
    include_thoughts: bool = False
    session_id: str = "v15_eval_default"
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
    enable_temporal_fan: bool = Field(
        default=False,
        description="When True, advertise mode='temporal_fan' in the Stage 2 prompt.",
    )

    @property
    def api_key(self) -> str:
        """Backward-compatible single-key accessor."""
        return self.api_keys[0] if self.api_keys else ""


def _default_modelhub_api_keys() -> list[str]:
    env_value = os.environ.get("MODELHUB_AKS") or os.environ.get("AZURE_API_KEYS")
    if env_value:
        return [key.strip() for key in env_value.split(",") if key.strip()]

    return [
        "hnJAK3LscxwLcy5OpZGQqQAzNyQmdx0a_GPT_AK",
        "cjodAcZmk7eIwm8wtizk1MfqyEJ7V8lG_GPT_AK",
    ]


__all__ = [
    "Stage2TaskType",
    "Stage2PlanMode",
    "Stage2Status",
    "Stage2DeepAgentConfig",
]
