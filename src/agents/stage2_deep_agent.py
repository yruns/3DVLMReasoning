"""Backward-compatible Stage-2 DeepAgents entrypoint.

This module preserves the historical ``agents.stage2_deep_agent`` API while
its implementation now lives in ``agents.runtime``.
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from loguru import logger

from .models import (
    Stage2AgentResult,
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2Status,
    Stage2StructuredResponse,
    Stage2TaskSpec,
)
from .runtime import (
    DeepAgentsStage2Runtime,
    Stage2RuntimeState,
    ToolChoiceCompatibleAzureChatOpenAI,
)

ToolCallback = Any


class Stage2DeepResearchAgent:
    """Compatibility wrapper around the runtime-based Stage-2 implementation.

    Older tests, examples, and downstream code patch methods such as
    ``_get_llm()``, ``_build_runtime_tools()``, and ``build_agent()`` on this
    class directly. The wrapper therefore re-exposes that surface while routing
    the real work through ``DeepAgentsStage2Runtime``.
    """

    def __init__(
        self,
        config: Stage2DeepAgentConfig | None = None,
        more_views_callback: ToolCallback | None = None,
        crop_callback: ToolCallback | None = None,
        hypothesis_callback: ToolCallback | None = None,
    ) -> None:
        self._runtime = DeepAgentsStage2Runtime(
            config=config,
            more_views_callback=more_views_callback,
            crop_callback=crop_callback,
            hypothesis_callback=hypothesis_callback,
        )

    @property
    def config(self) -> Stage2DeepAgentConfig:
        return self._runtime.config

    @property
    def more_views_callback(self) -> ToolCallback | None:
        return self._runtime.more_views_callback

    @property
    def crop_callback(self) -> ToolCallback | None:
        return self._runtime.crop_callback

    @property
    def hypothesis_callback(self) -> ToolCallback | None:
        return self._runtime.hypothesis_callback

    def _build_extra_body(self) -> dict[str, Any]:
        """Backward-compatible helper for tests."""
        return self._runtime.build_extra_body()

    def _get_llm(self):
        """Return the configured Azure-compatible chat model.

        This intentionally uses symbols imported in this module so older unit
        tests can patch ``ToolChoiceCompatibleAzureChatOpenAI`` here and inspect
        constructor arguments.
        """
        if getattr(self._runtime, "_llm", None) is None:
            self._runtime._llm = ToolChoiceCompatibleAzureChatOpenAI(
                azure_deployment=self.config.model_name,
                model=self.config.model_name,
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url,
                api_version=self.config.api_version,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                extra_body=self._build_extra_body(),
            )
        return self._runtime._llm

    def _build_runtime_tools(self, runtime: Stage2RuntimeState):
        return self._runtime.build_runtime_tools(runtime)

    def _build_system_prompt(
        self, task: Stage2TaskSpec, object_context: dict[str, str] | None = None
    ) -> str:
        return self._runtime.build_system_prompt(task, object_context=object_context)

    def _build_subagents(self, task: Stage2TaskSpec):
        return self._runtime.build_subagents(task)

    def _build_user_message(self, task: Stage2TaskSpec, runtime: Stage2RuntimeState):
        return self._runtime.build_user_message(task, runtime)

    def _build_evidence_update_message(self, runtime: Stage2RuntimeState):
        return self._runtime.build_evidence_update_message(runtime)

    def _normalize_final_response(
        self,
        task: Stage2TaskSpec,
        raw_state: dict[str, Any],
    ) -> Stage2StructuredResponse:
        return self._runtime.normalize_final_response(task, raw_state)

    def _apply_uncertainty_stopping(
        self,
        response: Stage2StructuredResponse,
        can_acquire_more_evidence: bool,
    ) -> Stage2StructuredResponse:
        return self._runtime.apply_uncertainty_stopping(
            response,
            can_acquire_more_evidence,
        )

    def build_agent(self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle):
        """Build the DeepAgents graph and runtime state.

        Delegates to the runtime implementation which handles VG tool
        registration, axis alignment, and other task-type-specific setup.
        """
        return self._runtime.build_agent(task, bundle)

    def run(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
    ) -> Stage2AgentResult:
        """Execute the Stage-2 DeepAgent with iterative evidence refinement.

        Delegates to the runtime implementation which handles VG tool
        state export and other task-type-specific logic.
        """
        return self._runtime.run(task, bundle)


__all__ = [
    "Stage2DeepResearchAgent",
    "Stage2RuntimeState",
    "ToolChoiceCompatibleAzureChatOpenAI",
]
