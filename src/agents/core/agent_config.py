"""Core agent configuration and enums.

This module defines the agent configuration and enumeration types.
"""

import os
from enum import Enum
from typing import Any, Literal

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
    model_name: str = Field(default_factory=lambda: _default_modelhub_model_name())
    api_keys: list[str] = Field(
        default_factory=lambda: _default_modelhub_api_keys(),
        description="ModelHub AKs rotated on retryable quota/rate-limit errors.",
    )
    api_key_weights: list[float] = Field(
        default_factory=lambda: _default_modelhub_api_key_weights(),
        description="Relative QPS weights for api_keys; same order as api_keys.",
    )
    api_key_initial_offset: int = Field(
        default_factory=lambda: _default_modelhub_api_key_initial_offset(),
        ge=0,
        description="Number of weighted AK selections to skip when initializing the rotator.",
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
    enable_chassis_tools: bool = Field(
        default=False,
        description="Register chassis trio (list_skills, load_skill, submit_final) "
        "for tasks without a registered TaskPack. Default OFF preserves QA byte-stable.",
    )
    vg_backend: Literal["legacy", "pack_v1"] = Field(
        default="pack_v1",
        description="VG backend. Only 'pack_v1' is supported after Plan C.",
    )
    chassis_tools_version: int = Field(
        default=19,
        ge=1,
        description="Bump when chassis tool surface changes; folded into "
        "derive_eval_session_id so prompt-cache invalidates correctly. "
        "v4 added the optional `tool_override_reason` kwarg on submit_final "
        "(TADG bypass path); v5 added the no-match candidate guard; "
        "v6 added the evidence-frame consistency guard; v7 added 2D "
        "left-to-right mark geometry to view_keyframe_marked; v8 added "
        "left/right consistency checks to the evidence-frame guard; v9 "
        "added left/right/near/next_to spatial ranking; v10 rejects "
        "direct VG structured finals while a submit_final turn remains; "
        "v11 widens the TADG lookback window to 32 trace entries; v12 keeps "
        "a higher-confidence deferred submit_final over a weaker direct VG "
        "structured replacement; v13 restricts evidence-frame left/right "
        "alternatives to the submitted proposal's category-candidate set "
        "when that trace set exists; v14 documents reversed pronoun "
        "left/right relations in the VG spatial skill; v15 tightens "
        "ambiguous-anchor and same-category adjacency spatial guidance; "
        "v16 hard-blocks narrow unsafe TADG overrides; v17 blocks "
        "left/right submissions with untested ambiguous anchors; v18 "
        "blocks relation-rationale frame citations missing the spatial anchor; "
        "v19 adds list_frame_proposals and filtered marked-frame rendering.",
    )
    use_clip_visible_aug: bool = Field(
        default=False,
        description="Enable CVRA label-agnostic visible-proposal CLIP augmentation "
        "for VG category lookup.",
    )
    clip_visible_tau: float = Field(
        default=0.18,
        ge=0.13,
        le=1.0,
        description="Minimum CLIP cosine score for CVRA visible-proposal augmentation.",
    )
    clip_visible_k_aug: int = Field(
        default=8,
        ge=1,
        le=10,
        description="Hard cap for standard CVRA augmented visible proposals.",
    )
    clip_visible_backbone: str = Field(
        default="ViT-H-14/laion2b_s32b_b79k",
        description="open_clip backbone/pretrained pair for CVRA crops.",
    )
    clip_visible_cache_dir: str | None = Field(
        default=None,
        description="Optional on-disk cache for CVRA crop embeddings.",
    )
    use_tool_answer_disagreement_gate: bool = Field(
        default=False,
        description="Enable TADG: soft-block submit_final when the submitted "
        "proposal_id disagrees with the most recent matched-relation "
        "compare_proposals_spatial rank-1.",
    )
    tadg_window: int = Field(
        default=32,
        ge=1,
        le=64,
        description="Maximum tool_trace lookback (entries) when TADG searches "
        "for a recent compare_proposals_spatial call. Widened from 16 to 32 "
        "after v3.8 trace audit found high-impact relation disagreements "
        "19-26 tool calls after the relevant compare.",
    )
    tadg_max_repeats: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of identical TADG-blocked submits before the gate "
        "force-passes (anti-loop guard, supersedes handoff §6.2 risk #3 default).",
    )
    tadg_override_min_chars: int = Field(
        default=6,
        ge=1,
        le=200,
        description="Minimum non-whitespace length for tool_override_reason to "
        "bypass a TADG block.",
    )
    use_no_match_candidate_guard: bool = Field(
        default=False,
        description=(
            "Enable the VG no-match candidate guard: soft-block "
            "submit_final(proposal_id=-1) when the agent's own tool trace "
            "still contains unresolved proposal candidates."
        ),
    )
    use_evidence_frame_guard: bool = Field(
        default=False,
        description=(
            "Enable the VG evidence-frame consistency guard: soft-block "
            "submit_final when the rationale cites a marked frame that does "
            "not contain the submitted proposal id."
        ),
    )
    no_match_guard_max_repeats: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Force-pass a repeated -1 after this many no-match guard blocks.",
    )
    no_match_guard_max_viewed: int = Field(
        default=12,
        ge=1,
        le=40,
        description="Maximum viewed-but-uninspected proposal ids listed by the guard.",
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
        "cjodAcZmk7eIwm8wtizk1MfqyEJ7V8lG_GPT_AK",
        "hnJAK3LscxwLcy5OpZGQqQAzNyQmdx0a_GPT_AK",
        "K1Hn1GahMi3dpvLesYH67sS0S2Z1yFYE_GPT_AK",
    ]


def _default_modelhub_model_name() -> str:
    return os.environ.get("MODELHUB_MODEL_NAME") or "gpt-5.4-2026-03-05"


def _default_modelhub_api_key_weights() -> list[float]:
    env_weights = os.environ.get("MODELHUB_AK_WEIGHTS")
    if env_weights:
        return [
            float(weight.strip()) for weight in env_weights.split(",") if weight.strip()
        ]

    env_keys = os.environ.get("MODELHUB_AKS") or os.environ.get("AZURE_API_KEYS")
    if env_keys:
        key_count = len([key.strip() for key in env_keys.split(",") if key.strip()])
        return [1.0] * key_count

    return [5.0, 1.0, 2.5]


def _default_modelhub_api_key_initial_offset() -> int:
    env_offset = os.environ.get("MODELHUB_AK_INITIAL_OFFSET")
    if not env_offset:
        return 0
    return int(env_offset)


__all__ = [
    "Stage2TaskType",
    "Stage2PlanMode",
    "Stage2Status",
    "Stage2DeepAgentConfig",
]
