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
    Stage2TaskType,
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
        crop_callback: ToolCallback | None = None,
    ) -> None:
        self._runtime = DeepAgentsStage2Runtime(
            config=config,
            crop_callback=crop_callback,
        )

    @property
    def config(self) -> Stage2DeepAgentConfig:
        return self._runtime.config

    @property
    def crop_callback(self) -> ToolCallback | None:
        return self._runtime.crop_callback

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
                api_keys=self.config.api_keys,
                api_key_weights=self.config.api_key_weights,
                api_key_initial_offset=self.config.api_key_initial_offset,
                modelhub_path=self.config.modelhub_path,
                session_id=self.config.session_id,
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

    def _build_evidence_nudge(
        self,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState,
    ):
        return self._runtime._build_evidence_nudge(response, runtime)

    def _normalize_final_response(
        self,
        task: Stage2TaskSpec,
        raw_state: dict[str, Any],
        runtime: Stage2RuntimeState | None = None,
    ) -> Stage2StructuredResponse:
        return self._runtime.normalize_final_response(task, raw_state, runtime)

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

        This compatibility wrapper intentionally uses the symbols imported in
        this module so older tests can patch ``create_deep_agent`` and wrapper
        methods such as ``_get_llm()`` / ``_build_runtime_tools()`` directly.
        """
        from agents.packs import ensure_default_packs_registered

        ensure_default_packs_registered()

        runtime = Stage2RuntimeState(bundle=bundle.model_copy(deep=True))
        runtime.task_type = task.task_type

        # Keep the wrapper contract for tests that patch this class directly,
        # while using the same pack-v1 VG setup as the runtime implementation.
        if task.task_type == Stage2TaskType.VISUAL_GROUNDING:
            if self.config.vg_backend != "pack_v1":
                raise ValueError(
                    f"vg_backend={self.config.vg_backend!r} no longer supported; "
                    "legacy branch removed in Plan C. Set vg_backend='pack_v1'."
                )
            from agents.packs.vg_embodiedscan.ctx import build_ctx_from_bundle

            runtime.task_ctx = build_ctx_from_bundle(runtime.bundle)

        from agents.skills.validate import validate_packs

        validate_packs(
            task.task_type,
            bundle,
            require_pack=task.task_type == Stage2TaskType.VISUAL_GROUNDING,
        )

        graph = create_deep_agent(
            model=self._get_llm(),
            tools=self._build_runtime_tools(runtime),
            system_prompt=self._build_system_prompt(
                task, object_context=bundle.object_context
            ),
            subagents=self._build_subagents(task),
            response_format=Stage2StructuredResponse,
            name="query_scene_stage2_agent",
        )
        return graph, runtime

    def run(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
    ) -> Stage2AgentResult:
        """Execute the Stage-2 DeepAgent with iterative evidence refinement.

        Compatibility wrapper around the runtime loop.

        This intentionally calls ``self.build_agent(...)`` so older tests can
        patch the wrapper method and avoid live backend calls.
        """
        # WARNING: this wrapper intentionally mirrors the runtime loop in
        # runtime/deepagents_agent.py so legacy patch-on-class tests keep
        # working. Changes here must stay in sync with the runtime copy.
        graph, runtime = self.build_agent(task, bundle)
        message = self._build_user_message(task, runtime)
        logger.info(
            "[Stage2DeepResearchAgent] task={} plan_mode={} keyframes={} max_turns={}",
            task.task_type.value,
            task.plan_mode.value,
            len(runtime.bundle.keyframes),
            task.max_reasoning_turns,
        )

        messages = [message]
        raw_state: dict[str, Any] = {}
        turns_used = 0

        while turns_used < task.max_reasoning_turns:
            turns_used += 1
            raw_state = graph.invoke({"messages": messages})

            # Pack-v1 terminal: chassis submit_final populates runtime.final_submission.
            if runtime.final_submission is not None:
                if runtime.consume_evidence_update():
                    evidence_message = self._build_evidence_update_message(runtime)
                    if evidence_message is not None:
                        if "messages" in raw_state:
                            messages = raw_state["messages"]
                        messages.append(evidence_message)
                        runtime.final_submission = None
                        logger.info(
                            "[Stage2DeepResearchAgent] turn {}: deferring submit_final "
                            "until newly queued visual evidence is injected",
                            turns_used,
                        )
                        continue
                logger.info(
                    "[Stage2DeepResearchAgent] terminated at turn {} via chassis submit_final",
                    turns_used,
                )
                break

            structured = raw_state.get("structured_response")
            if structured is not None:
                response = Stage2StructuredResponse.model_validate(structured)
                if response.status in (Stage2Status.COMPLETED, Stage2Status.FAILED):
                    if (
                        turns_used < task.max_reasoning_turns
                        and self._runtime.should_continue_after_guarded_no_match(
                            task, response, runtime
                        )
                    ):
                        if "messages" in raw_state:
                            messages = raw_state["messages"]
                        messages.append(
                            self._runtime.build_no_match_guard_nudge(response, runtime)
                        )
                        logger.info(
                            "[Stage2DeepResearchAgent] turn {}: continuing after "
                            "guarded direct no-match response",
                            turns_used,
                        )
                        continue
                    if (
                        turns_used < task.max_reasoning_turns
                        and self._runtime.should_continue_after_guarded_evidence_frame(
                            task, response, runtime
                        )
                    ):
                        if "messages" in raw_state:
                            messages = raw_state["messages"]
                        messages.append(
                            self._runtime.build_evidence_frame_guard_nudge(
                                response, runtime
                            )
                        )
                        logger.info(
                            "[Stage2DeepResearchAgent] turn {}: continuing after "
                            "guarded direct final response",
                            turns_used,
                        )
                        continue
                    if (
                        turns_used < task.max_reasoning_turns
                        and self._runtime.should_continue_after_direct_vg_response(
                            task, response, runtime
                        )
                    ):
                        if "messages" in raw_state:
                            messages = raw_state["messages"]
                        messages.append(
                            self._runtime.build_direct_vg_response_nudge(
                                response, runtime
                            )
                        )
                        logger.info(
                            "[Stage2DeepResearchAgent] turn {}: continuing after "
                            "direct VG structured response",
                            turns_used,
                        )
                        continue
                    logger.info(
                        "[Stage2DeepResearchAgent] completed at turn {} with status={}",
                        turns_used,
                        response.status.value,
                    )
                    break

            if runtime.consume_evidence_update():
                evidence_message = self._build_evidence_update_message(runtime)
                if evidence_message is not None:
                    if "messages" in raw_state:
                        messages = raw_state["messages"]
                    messages.append(evidence_message)
                    logger.info(
                        "[Stage2DeepResearchAgent] turn {}: injecting new evidence, continuing loop",
                        turns_used,
                    )
                    continue

            if structured is not None and turns_used < task.max_reasoning_turns:
                response = Stage2StructuredResponse.model_validate(structured)
                if response.status in (
                    Stage2Status.INSUFFICIENT_EVIDENCE,
                    Stage2Status.NEEDS_MORE_EVIDENCE,
                ):
                    nudge = self._build_evidence_nudge(response, runtime)
                    if "messages" in raw_state:
                        messages = raw_state["messages"]
                    messages.append(nudge)
                    logger.info(
                        "[Stage2DeepResearchAgent] turn {}: agent reported {}, nudging to seek more evidence ({} turns remaining)",
                        turns_used,
                        response.status.value,
                        task.max_reasoning_turns - turns_used,
                    )
                    continue

            break

        logger.info(
            "[Stage2DeepResearchAgent] finished after {} turns, tool_calls={}",
            turns_used,
            len(runtime.tool_trace),
        )

        # v9: the agent can always acquire more evidence via the catalog-first
        # selectors + view tools, regardless of optional callbacks.
        can_acquire_more_evidence = turns_used < task.max_reasoning_turns

        final_response = self._normalize_final_response(task, raw_state, runtime)
        final_response = self._apply_uncertainty_stopping(
            final_response, can_acquire_more_evidence
        )

        result_state = {k: v for k, v in raw_state.items() if k != "messages"}

        return Stage2AgentResult(
            task=task,
            result=final_response,
            tool_trace=runtime.tool_trace,
            final_bundle=runtime.bundle,
            raw_state=result_state,
        )


__all__ = [
    "Stage2DeepResearchAgent",
    "Stage2RuntimeState",
    "ToolChoiceCompatibleAzureChatOpenAI",
]
