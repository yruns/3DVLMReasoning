"""DeepAgents-based runtime implementation for Stage-2 agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from loguru import logger

from ..models import (
    Stage2AgentResult,
    Stage2DeepAgentConfig,
    Stage2EvidenceBundle,
    Stage2PlanMode,
    Stage2Status,
    Stage2StructuredResponse,
    Stage2TaskSpec,
    Stage2TaskType,
)
from .base import (
    BaseStage2Runtime,
    Stage2RuntimeState,
    default_output_instruction,
    default_payload_schema,
)
from .langchain_agent import ToolChoiceCompatibleAzureChatOpenAI


class DeepAgentsStage2Runtime(BaseStage2Runtime):
    """DeepAgents-backed Stage-2 research agent with iterative evidence refinement.

    This runtime uses the DeepAgents framework with LangChain v1 integration to provide:
    - ReAct-style tool use with structured responses
    - Optional subagent decomposition for complex tasks
    - Iterative evidence refinement loop
    - Uncertainty-aware stopping criteria
    """

    def __init__(
        self,
        config: Stage2DeepAgentConfig | None = None,
        more_views_callback=None,
        crop_callback=None,
        hypothesis_callback=None,
    ) -> None:
        """Initialize the DeepAgents runtime.

        Args:
            config: Agent configuration
            more_views_callback: Callback for requesting more views
            crop_callback: Callback for requesting crops
            hypothesis_callback: Callback for hypothesis updates
        """
        super().__init__(
            config, more_views_callback, crop_callback, hypothesis_callback
        )
        self._llm = None

    def get_llm(self):
        """Return a single-key AzureOpenAI-compatible chat model.

        Uses a stable single-key client so the runtime can keep a consistent
        session_id for provider-side prompt caching.
        """
        if self._llm is None:
            self._llm = ToolChoiceCompatibleAzureChatOpenAI(
                azure_deployment=self.config.model_name,
                model=self.config.model_name,
                api_key=self.config.api_key,
                azure_endpoint=self.config.base_url,
                api_version=self.config.api_version,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                extra_body=self.build_extra_body(),
            )
        return self._llm

    def build_runtime_tools(self, runtime: Stage2RuntimeState) -> list[BaseTool]:
        """Create Stage-2 evidence tools bound to one runtime state.

        Args:
            runtime: Runtime state with bundle and tool trace

        Returns:
            List of LangChain tools for the agent
        """

        @tool
        def inspect_stage1_metadata() -> str:
            """Inspect the Stage-1 hypothesis, selector status, and frame mapping metadata."""
            payload = {
                "hypothesis": (
                    runtime.bundle.hypothesis.model_dump()
                    if runtime.bundle.hypothesis
                    else None
                ),
                "extra_metadata": runtime.bundle.extra_metadata,
                "num_keyframes": len(runtime.bundle.keyframes),
            }
            response = json.dumps(payload, indent=2, ensure_ascii=False)
            runtime.record("inspect_stage1_metadata", {}, response)
            return response

        @tool
        def retrieve_object_context(object_terms: list[str] | None = None) -> str:
            """Retrieve scene-level or object-specific context summaries."""
            request = {"object_terms": object_terms or []}
            response = self.select_object_context(runtime.bundle, object_terms)
            runtime.record("retrieve_object_context", request, response)
            return response

        @tool
        def request_more_views(
            request_text: str,
            frame_indices: list[int] | None = None,
            object_terms: list[str] | None = None,
        ) -> str:
            """Request additional keyframes or neighboring views from Stage 1."""
            request = {
                "request_text": request_text,
                "frame_indices": frame_indices or [],
                "object_terms": object_terms or [],
            }
            if self.more_views_callback is None:
                response_obj = self.coerce_callback_result(
                    "request_more_views callback is not configured."
                )
            else:
                response_obj = self.coerce_callback_result(
                    self.more_views_callback(runtime.bundle, request)
                )
                if response_obj.updated_bundle is not None:
                    runtime.bundle = response_obj.updated_bundle
                    runtime.mark_evidence_updated()
            runtime.record("request_more_views", request, response_obj.response_text)
            return response_obj.response_text

        @tool
        def request_crops(
            request_text: str,
            frame_indices: list[int] | None = None,
            object_terms: list[str] | None = None,
        ) -> str:
            """Request object-centric or region-centric crops from the current evidence."""
            request = {
                "request_text": request_text,
                "frame_indices": frame_indices or [],
                "object_terms": object_terms or [],
            }
            if self.crop_callback is None:
                response_obj = self.coerce_callback_result(
                    "request_crops callback is not configured."
                )
            else:
                response_obj = self.coerce_callback_result(
                    self.crop_callback(runtime.bundle, request)
                )
                if response_obj.updated_bundle is not None:
                    runtime.bundle = response_obj.updated_bundle
                    runtime.mark_evidence_updated()
            runtime.record("request_crops", request, response_obj.response_text)
            return response_obj.response_text

        @tool
        def switch_or_expand_hypothesis(
            request_text: str,
            preferred_kind: str | None = None,
        ) -> str:
            """Request Stage-1 hypothesis expansion or direct/proxy/context switching."""
            request = {
                "request_text": request_text,
                "preferred_kind": preferred_kind or "",
            }
            if self.hypothesis_callback is None:
                response_obj = self.coerce_callback_result(
                    "switch_or_expand_hypothesis callback is not configured."
                )
            else:
                response_obj = self.coerce_callback_result(
                    self.hypothesis_callback(runtime.bundle, request)
                )
                if response_obj.updated_bundle is not None:
                    runtime.bundle = response_obj.updated_bundle
                    runtime.mark_evidence_updated()
            runtime.record(
                "switch_or_expand_hypothesis", request, response_obj.response_text
            )
            return response_obj.response_text

        return [
            inspect_stage1_metadata,
            retrieve_object_context,
            request_more_views,
            request_crops,
            switch_or_expand_hypothesis,
        ]

    def build_subagents(self, task: Stage2TaskSpec) -> list[dict[str, Any]]:
        """Build optional DeepAgents subagents for richer decomposition.

        Args:
            task: Task specification

        Returns:
            List of subagent configurations for DeepAgents
        """
        if not self.config.enable_subagents or task.plan_mode != Stage2PlanMode.FULL:
            return []

        return [
            {
                "name": "evidence_scout",
                "description": "Diagnose evidence gaps and decide which view/crop/hypothesis tool to call next.",
                "system_prompt": (
                    "You are the evidence scout. Focus only on whether current keyframes are "
                    "sufficient, which missing views or crops are needed, and what uncertainty "
                    "remains. Do not produce the final user-facing answer."
                ),
            },
            {
                "name": "task_head",
                "description": "Synthesize the final task-specific payload from collected evidence.",
                "system_prompt": (
                    "You are the task head. Use the collected evidence to assemble the final "
                    "task-specific payload. Stay faithful to cited frames and explicit uncertainty."
                ),
            },
        ]

    def build_user_message(
        self,
        task: Stage2TaskSpec,
        runtime: Stage2RuntimeState,
    ) -> HumanMessage:
        """Assemble the multimodal task message for the DeepAgent.

        Args:
            task: Task specification
            runtime: Runtime state

        Returns:
            HumanMessage with text and image content
        """
        bundle = runtime.bundle
        keyframe_lines = []
        for keyframe in bundle.keyframes:
            keyframe_lines.append(
                f"- idx={keyframe.keyframe_idx}, view_id={keyframe.view_id}, "
                f"frame_id={keyframe.frame_id}, note={keyframe.note or 'N/A'}"
            )
        if not keyframe_lines:
            keyframe_lines.append("- no keyframes available")

        hypothesis_text = (
            json.dumps(bundle.hypothesis.model_dump(), indent=2, ensure_ascii=False)
            if bundle.hypothesis
            else "{}"
        )
        payload_schema = task.expected_output_schema or default_payload_schema(
            task.task_type
        )
        instruction = task.output_instruction or default_output_instruction(
            task.task_type
        )

        # Inject VG candidate list for visual grounding tasks
        vg_candidates_section = ""
        if task.task_type == Stage2TaskType.VISUAL_GROUNDING:
            vg_candidates_section = self._format_vg_candidates(
                bundle.extra_metadata or {}
            )

        prompt = (
            f"Task type: {task.task_type.value}\n"
            f"User query: {task.user_query}\n"
            f"Plan mode: {task.plan_mode.value}\n"
            f"Output instruction: {instruction}\n"
            f"Expected payload schema: {json.dumps(payload_schema, indent=2, ensure_ascii=False)}\n\n"
            f"Stage-1 query: {bundle.stage1_query or task.user_query}\n"
            f"Scene id: {bundle.scene_id or 'unknown'}\n\n"
            f"Current keyframes:\n{chr(10).join(keyframe_lines)}\n\n"
            f"Stage-1 hypothesis summary:\n{hypothesis_text}\n\n"
            f"{vg_candidates_section}"
            f"Scene summary:\n{bundle.scene_summary or 'N/A'}\n\n"
            f"Available object context keys:\n"
            f"{sorted(bundle.object_context.keys()) if bundle.object_context else []}\n\n"
            "Use tools when evidence is missing. Return the final answer through the "
            "structured response schema."
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in self.collect_image_paths(bundle):
            runtime.seen_image_paths.add(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self.image_to_data_url(image_path)},
                }
            )
        return HumanMessage(content=content)

    def _build_evidence_nudge(
        self,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState,
    ) -> HumanMessage:
        """Build a follow-up message that nudges the agent to seek evidence.

        Called when the agent returned insufficient_evidence / needs_more_evidence
        but still has turns remaining. Adapts its guidance based on whether
        evidence-acquisition callbacks are configured.
        """
        available_tools: list[str] = []
        has_acquisition_tools = False

        if self.more_views_callback is not None:
            has_acquisition_tools = True
            available_tools.append(
                "request_more_views — ask for additional keyframes showing "
                "specific objects or regions you need to see"
            )
        if self.crop_callback is not None:
            has_acquisition_tools = True
            available_tools.append(
                "request_crops — ask for close-up crops of specific objects "
                "in the current keyframes"
            )
        if self.hypothesis_callback is not None:
            has_acquisition_tools = True
            available_tools.append(
                "switch_or_expand_hypothesis — request a different retrieval "
                "hypothesis from Stage 1"
            )

        # Always-available tools
        available_tools.append(
            "inspect_stage1_metadata — review the Stage-1 hypothesis and "
            "frame mapping to understand what was retrieved and why"
        )
        available_tools.append(
            "retrieve_object_context — get scene-level or object-specific "
            "context summaries for additional clues"
        )

        tools_list = "\n".join(f"  - {t}" for t in available_tools)

        uncertainties_text = ""
        if response.uncertainties:
            uncertainties_text = (
                "Your reported uncertainties:\n"
                + "\n".join(f"  - {u}" for u in response.uncertainties)
                + "\n\n"
            )

        if has_acquisition_tools:
            action_guidance = (
                "Use the evidence-seeking tools to acquire the missing views or crops, "
                "then re-examine the images and produce your final answer."
            )
        else:
            action_guidance = (
                "Look more carefully at the existing keyframes — the answer may be "
                "partially visible even if not obvious at first glance. Use "
                "inspect_stage1_metadata or retrieve_object_context to gather "
                "additional clues. Then provide your best answer with appropriate "
                "confidence, rather than reporting insufficient evidence."
            )

        prompt = (
            "You reported that the current evidence is insufficient to answer the question. "
            "Do NOT give up — try harder before concluding.\n\n"
            f"{uncertainties_text}"
            f"Available tools:\n{tools_list}\n\n"
            f"{action_guidance}"
        )
        return HumanMessage(content=[{"type": "text", "text": prompt}])

    def build_evidence_update_message(
        self,
        runtime: Stage2RuntimeState,
    ) -> HumanMessage | None:
        """Build a follow-up message injecting newly acquired visual evidence.

        Args:
            runtime: Runtime state

        Returns:
            HumanMessage with new images, or None if no new images
        """
        new_images: list[str] = []
        for keyframe in runtime.bundle.keyframes:
            if Path(keyframe.image_path).exists():
                if keyframe.image_path not in runtime.seen_image_paths:
                    new_images.append(keyframe.image_path)

        if (
            runtime.bundle.bev_image_path
            and Path(runtime.bundle.bev_image_path).exists()
            and runtime.bundle.bev_image_path not in runtime.seen_image_paths
        ):
            new_images.append(runtime.bundle.bev_image_path)

        if not new_images:
            return None

        # Limit new images to avoid token explosion
        new_images = new_images[
            : self.config.max_images - len(runtime.seen_image_paths)
        ]
        if not new_images:
            return None

        keyframe_lines = []
        for keyframe in runtime.bundle.keyframes:
            if keyframe.image_path in new_images:
                keyframe_lines.append(
                    f"- idx={keyframe.keyframe_idx}, view_id={keyframe.view_id}, "
                    f"frame_id={keyframe.frame_id}, note={keyframe.note or 'N/A'}"
                )

        prompt = (
            "New visual evidence has been acquired:\n\n"
            f"Newly added keyframes:\n{chr(10).join(keyframe_lines) if keyframe_lines else '- BEV or crop images'}\n\n"
            "Please examine these new images and continue your analysis. "
            "If the evidence is now sufficient, produce your final answer."
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in new_images:
            runtime.seen_image_paths.add(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self.image_to_data_url(image_path)},
                }
            )

        logger.info(
            "[DeepAgentsStage2Runtime] injecting {} new images into context",
            len(new_images),
        )
        return HumanMessage(content=content)

    def build_agent(self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle):
        """Compile a DeepAgent and return it with runtime state.

        Args:
            task: Task specification
            bundle: Evidence bundle

        Returns:
            Tuple of (graph, runtime_state)
        """
        runtime = Stage2RuntimeState(bundle=bundle.model_copy(deep=True))
        tools = self.build_runtime_tools(runtime)
        graph = create_deep_agent(
            model=self.get_llm(),
            tools=tools,
            system_prompt=self.build_system_prompt(task, object_context=bundle.object_context),
            subagents=self.build_subagents(task),
            response_format=Stage2StructuredResponse,
            name="query_scene_stage2_agent",
        )
        return graph, runtime

    def normalize_final_response(
        self,
        task: Stage2TaskSpec,
        raw_state: dict[str, Any],
    ) -> Stage2StructuredResponse:
        """Convert DeepAgents final state into the unified Stage-2 schema.

        Args:
            task: Task specification
            raw_state: Raw state from DeepAgents graph

        Returns:
            Normalized structured response
        """
        structured = raw_state.get("structured_response")
        if structured is not None:
            response = Stage2StructuredResponse.model_validate(structured)
            if response.task_type != task.task_type:
                response.task_type = task.task_type
            return response

        return Stage2StructuredResponse(
            task_type=task.task_type,
            status=Stage2Status.FAILED,
            summary="The agent returned without a structured response.",
            confidence=0.0,
            uncertainties=["Missing structured_response in DeepAgents final state."],
            payload={},
        )

    def run(
        self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle
    ) -> Stage2AgentResult:
        """Execute the Stage-2 DeepAgent with iterative evidence refinement.

        This implementation supports a true evidence-seeking loop:
        1. Initial invocation with all currently available keyframes
        2. If tools acquire new evidence (via callbacks), inject new images
        3. Continue until structured response or max_reasoning_turns reached

        Args:
            task: Task specification
            bundle: Evidence bundle

        Returns:
            AgentResult with response, tool trace, and final bundle
        """
        graph, runtime = self.build_agent(task, bundle)
        message = self.build_user_message(task, runtime)
        logger.info(
            "[DeepAgentsStage2Runtime] task={} plan_mode={} keyframes={} max_turns={}",
            task.task_type.value,
            task.plan_mode.value,
            len(runtime.bundle.keyframes),
            task.max_reasoning_turns,
        )

        # Iterative evidence refinement loop
        messages = [message]
        raw_state: dict[str, Any] = {}
        turns_used = 0

        while turns_used < task.max_reasoning_turns:
            turns_used += 1
            raw_state = graph.invoke({"messages": messages})

            # Check if structured response indicates completion
            structured = raw_state.get("structured_response")
            if structured is not None:
                response = Stage2StructuredResponse.model_validate(structured)
                if response.status in (Stage2Status.COMPLETED, Stage2Status.FAILED):
                    logger.info(
                        "[DeepAgentsStage2Runtime] completed at turn {} with status={}",
                        turns_used,
                        response.status.value,
                    )
                    break

            # Check if new evidence was acquired and needs injection
            if runtime.consume_evidence_update():
                evidence_message = self.build_evidence_update_message(runtime)
                if evidence_message is not None:
                    # Append the agent's response and new evidence to continue
                    if "messages" in raw_state:
                        messages = raw_state["messages"]
                    messages.append(evidence_message)
                    logger.info(
                        "[DeepAgentsStage2Runtime] turn {}: injecting new evidence, continuing loop",
                        turns_used,
                    )
                    continue

            # If agent reported insufficient/needs-more evidence and we have
            # remaining turns, nudge it to actively seek evidence or
            # re-examine the existing frames instead of giving up.
            if (
                structured is not None
                and turns_used < task.max_reasoning_turns
            ):
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
                        "[DeepAgentsStage2Runtime] turn {}: agent reported {}, "
                        "nudging to seek more evidence ({} turns remaining)",
                        turns_used,
                        response.status.value,
                        task.max_reasoning_turns - turns_used,
                    )
                    continue

            # No new evidence and no explicit continuation needed
            break

        logger.info(
            "[DeepAgentsStage2Runtime] finished after {} turns, tool_calls={}",
            turns_used,
            len(runtime.tool_trace),
        )

        # Determine if more evidence can be acquired
        can_acquire_more_evidence = turns_used < task.max_reasoning_turns and (
            self.more_views_callback is not None
            or self.crop_callback is not None
            or self.hypothesis_callback is not None
        )

        final_response = self.normalize_final_response(task, raw_state)
        # Apply uncertainty-aware stopping rules
        final_response = self.apply_uncertainty_stopping(
            final_response, can_acquire_more_evidence
        )
        return Stage2AgentResult(
            task=task,
            result=final_response,
            tool_trace=runtime.tool_trace,
            final_bundle=runtime.bundle,
            raw_state={k: v for k, v in raw_state.items() if k != "messages"},
        )
