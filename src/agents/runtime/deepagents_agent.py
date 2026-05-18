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


def _collect_v9_tools(
    *, runtime: Stage2RuntimeState, task_type: Stage2TaskType | None
) -> list[BaseTool]:
    """Build the v9 catalog-aware tool set for a given task pack.

    Returns selectors, scene perception, and mark_frame_with_bbox. The order matches
    the historical loading order so existing trace HTML colour-coding stays stable.

    `task_type` is currently unused but exposed for future task-pack-specific tool
    gating; do not remove it.
    """
    _ = task_type  # reserved for task-pack-specific tool gating
    from agents.tools.mark_frame_with_bbox import build_mark_frame_with_bbox_tool
    from agents.tools.scene_perception import build_scene_perception_tools
    from agents.tools.selectors import build_selector_tools

    tools: list[BaseTool] = []
    tools.extend(build_selector_tools(runtime))
    tools.extend(build_scene_perception_tools(runtime))
    tools.append(build_mark_frame_with_bbox_tool(runtime))
    return tools


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
        crop_callback=None,
        text_frame_selector: Any | None = None,
    ) -> None:
        """Initialize the DeepAgents runtime.

        v9: only crop_callback is accepted; more_views / hypothesis callbacks were
        deleted with the corresponding tool wrappers.

        v9.1: `text_frame_selector` is forwarded to the runtime state so the
        `select_by_text` tool can run Stage-1 language-to-frame retrieval.
        """
        super().__init__(config, crop_callback, text_frame_selector=text_frame_selector)
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
        self.configure_runtime_state(runtime)

        @tool
        def retrieve_object_context(object_terms: list[str] | None = None) -> str:
            """Retrieve scene-level or object-specific context summaries."""
            request = {"object_terms": object_terms or []}
            response = self.retrieve_object_context_text(runtime.bundle, object_terms)
            runtime.record("retrieve_object_context", request, response)
            return response

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

        tools = [
            retrieve_object_context,
            request_crops,
        ]

        # v9.1 catalog-first scene-perception tools (selectors +
        # mark_frame_with_bbox + view_bev + list_scene_proposals +
        # inspect_proposal). Wired for every task type — VG and QA share the
        # same exploration surface. Only enabled when the bundle carries a
        # SceneCatalog (the runtime is tolerant to legacy bundles for
        # back-compat).
        if (runtime.bundle.extra_metadata or {}).get("scene_catalog") is not None:
            tools.extend(
                _collect_v9_tools(runtime=runtime, task_type=runtime.task_type)
            )

        # Chassis trio attaches when the active task pack opts in
        # (TaskPack.exposes_chassis=True) or when the operator forces it via
        # enable_chassis_tools. QA pack opts out — see qa_default/registration.py.
        from agents.skills import PACKS
        from agents.skills.chassis_tools import build_chassis_tools

        pack = PACKS.get(runtime.task_type)
        pack_exposes_chassis = pack is not None and pack.exposes_chassis
        if pack_exposes_chassis or self.config.enable_chassis_tools:
            tools.extend(build_chassis_tools(runtime))

        # VG-specific tools are pack-v1 only after Plan C.
        if runtime.task_type == Stage2TaskType.VISUAL_GROUNDING:
            if self.config.vg_backend != "pack_v1":
                raise ValueError(
                    f"vg_backend={self.config.vg_backend!r} no longer supported; "
                    "legacy branch removed in Plan C. Set vg_backend='pack_v1'."
                )
            pack = PACKS.get(Stage2TaskType.VISUAL_GROUNDING)
            if pack is None:
                raise RuntimeError(
                    "VG pack not registered; import agents.packs to trigger registration"
                )
            tools.extend(pack.tool_builder(runtime))

        return tools

    def build_subagents(self, task: Stage2TaskSpec) -> list[dict[str, Any]]:
        """Return DeepAgents subagents.

        Plan B demotes the old evidence_scout/task_head subagents into skill
        bodies, so the runtime no longer passes DeepAgents subagents.
        """
        return []

    def build_user_message(
        self,
        task: Stage2TaskSpec,
        runtime: Stage2RuntimeState,
    ) -> HumanMessage:
        """Catalog-first multimodal task message (v9). BEV image + Cat-B text only."""
        from agents.catalog import SceneCatalog

        bundle = runtime.bundle
        extra = bundle.extra_metadata or {}
        catalog_raw = extra.get("scene_catalog")
        if catalog_raw is None:
            raise RuntimeError(
                "v9 build_user_message requires bundle.extra_metadata.scene_catalog; "
                "ensure pack prep wrote scene_catalog.json"
            )
        catalog = SceneCatalog(**catalog_raw)

        payload_schema = task.expected_output_schema or default_payload_schema(
            task.task_type
        )
        instruction = task.output_instruction or default_output_instruction(
            task.task_type
        )

        view_note = (
            "- use view_bev(highlight=[ids]) to declutter\n"
            "- selectors return ≤3 first-person RGB frames per call; use "
            "mark_frame_with_bbox to annotate the one frame worth verifying"
        )

        by_cat = catalog.proposals_by_category()
        cat_lines: list[str] = []
        if by_cat:
            max_cat_len = max(len(c) for c in by_cat)
            for cat in sorted(by_cat):
                ids_str = ", ".join(f"#{pid}" for pid in sorted(by_cat[cat]))
                cat_lines.append(f"  {cat.ljust(max_cat_len)} : [{ids_str}]")
        cat_block = "\n".join(cat_lines) if cat_lines else "  (catalog is empty)"
        source = catalog.proposals[0].source if catalog.proposals else "n/a"

        prompt = (
            "## Task\n"
            f"Task type: {task.task_type.value}\n"
            f"Plan mode: {task.plan_mode.value}\n"
            f'User query: "{task.user_query}"\n'
            f"Output instruction: {instruction}\n\n"
            f"Expected payload schema:\n"
            f"{json.dumps(payload_schema, indent=2, ensure_ascii=False)}\n\n"
            "## Scene\n"
            f"Scene id: {catalog.scene_id}\n"
            f"Scene category: {catalog.scene_category or 'unknown'}\n"
            f"Total frames: {catalog.total_frames} "
            f"(frame_id range: {list(catalog.frame_id_range)})\n"
            f"Proposal pool: {len(catalog.proposals)} items, source={source}\n\n"
            f"Proposals by category:\n{cat_block}\n\n"
            "## BEV image (attached above)\n"
            "- mesh-based top-down render with camera trajectory\n"
            "- each proposal labeled `#id category` at its 3D center\n"
            f"{view_note}\n\n"
            "## Available tools\n"
            "Always load_skill('scene-exploration-playbook') first.\n"
            "Then load_skill('vg-grounding-playbook') (VG) or "
            "load_skill('qa-answering-playbook') (QA).\n\n"
            f"You have viewed 0 first-person frames out of {catalog.total_frames}. "
            "Use selectors to fetch ≤3 RGB frames per call; use "
            "mark_frame_with_bbox to annotate one frame for verification."
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
        but still has turns remaining. v9 nudges towards selectors + view tools.
        """
        text_first = bool(getattr(runtime, "enable_stage1_text_retrieval", True))
        if text_first:
            available_tools = [
                "select_by_text(query, k≤3, hidden_categories) — Stage-1 "
                "language→frame, primary entry; returns ≤3 RGB frames",
                "select_by_proposal / select_by_frame_neighbor / select_by_region / "
                "select_by_coverage — catalog-driven selectors, each returns ≤3 RGB frames",
                "mark_frame_with_bbox(frame_id, labels?, ids?) — high-contrast "
                "annotated zoom on one selected frame",
                "view_bev(highlight=[ids]) — re-inject the BEV (optionally focused)",
                "list_scene_proposals / list_frame_proposals / inspect_proposal — "
                "text-only inventory queries",
                "request_crops — close-up crop on small or ambiguous regions",
                "retrieve_object_context — scene / object context summaries",
            ]
        else:
            available_tools = [
                "select_by_proposal(proposal_ids, require_all, k≤3) — primary "
                "entry; fetch frames containing candidate catalog IDs",
                "select_by_region / select_by_frame_neighbor / select_by_coverage — "
                "catalog-driven selectors, each returns ≤3 RGB frames",
                "mark_frame_with_bbox(frame_id, labels?, ids?) — high-contrast "
                "annotated zoom on one selected frame",
                "view_bev(highlight=[ids]) — re-inject the BEV (optionally focused)",
                "list_scene_proposals / list_frame_proposals / inspect_proposal — "
                "text-only inventory queries",
                "request_crops — close-up crop on small or ambiguous regions",
                "retrieve_object_context — scene / object context summaries",
            ]
        tools_list = "\n".join(f"  - {t}" for t in available_tools)

        uncertainties_text = ""
        if response.uncertainties:
            uncertainties_text = (
                "Your reported uncertainties:\n"
                + "\n".join(f"  - {u}" for u in response.uncertainties)
                + "\n\n"
            )

        prompt = (
            "You reported that the current evidence is insufficient to answer the "
            "question. Do NOT give up — fetch one or two more frames before "
            "concluding.\n\n"
            f"{uncertainties_text}"
            f"Available tools:\n{tools_list}\n\n"
            "Pick the cheapest-first selector that maps onto the missing evidence, "
            "view one or two frames, then produce your final answer."
        )
        return HumanMessage(content=[{"type": "text", "text": prompt}])

    @staticmethod
    def _payload_proposal_id(response: Stage2StructuredResponse) -> int | None:
        value = response.payload.get(
            "proposal_id", response.payload.get("selected_object_id")
        )
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def should_continue_after_guarded_no_match(
        self,
        task: Stage2TaskSpec,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState,
    ) -> bool:
        """Return true when a blocked `submit_final(-1)` was bypassed directly.

        The pack-v1 chassis treats `submit_final` as the terminal path. If the
        no-match guard blocks `proposal_id=-1`, the same DeepAgents invocation
        can still emit a terminal structured response. That response has not
        passed through the chassis, so the outer loop must continue.
        """
        if task.task_type != Stage2TaskType.VISUAL_GROUNDING:
            return False
        if runtime.final_submission is not None:
            return False
        if (
            not runtime.use_no_match_candidate_guard
            or not runtime.no_match_guard_triggered
            or runtime.no_match_guard_block_count <= 0
        ):
            return False
        if response.status not in (Stage2Status.COMPLETED, Stage2Status.FAILED):
            return False
        return (
            self._payload_proposal_id(response) == -1
            or response.status == Stage2Status.FAILED
        )

    def build_no_match_guard_nudge(
        self,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState,
    ) -> HumanMessage:
        latest_guard_message = ""
        for entry in reversed(runtime.tool_trace):
            if entry.tool_name != "submit_final":
                continue
            value = entry.tool_input.get("no_match_guard_message")
            if isinstance(value, str) and value:
                latest_guard_message = value
            elif entry.response_text.startswith("NO_MATCH_GUARD"):
                latest_guard_message = entry.response_text
            break

        checklist = ""
        if latest_guard_message:
            checklist = f"\n\nLatest guard checklist:\n{latest_guard_message}"

        prompt = (
            "Your previous `submit_final(proposal_id=-1)` was blocked by "
            "NO_MATCH_GUARD, so a direct structured no-match response in the "
            "same turn is not accepted. Continue the visual-grounding task. "
            "Inspect or choose among the unresolved candidates, then terminate "
            "through `submit_final` with a concrete proposal_id if any candidate "
            "is plausible. Only submit `proposal_id=-1` after explicitly closing "
            "the guard checklist. Treat Mask3D labels as weak priors: if the "
            "pixels show the referent, submit the proposal that covers it even "
            "when its label mismatches the query."
            f"{checklist}\n\n"
            "Do not finish by returning a direct structured `proposal_id=-1`."
        )
        return HumanMessage(content=[{"type": "text", "text": prompt}])

    def should_continue_after_guarded_evidence_frame(
        self,
        task: Stage2TaskSpec,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState,
    ) -> bool:
        """Return true when an evidence-frame block was bypassed directly."""
        if task.task_type != Stage2TaskType.VISUAL_GROUNDING:
            return False
        if runtime.final_submission is not None:
            return False
        if (
            not runtime.use_evidence_frame_guard
            or not runtime.evidence_frame_guard_triggered
            or runtime.evidence_frame_guard_block_count <= 0
        ):
            return False
        return response.status in (Stage2Status.COMPLETED, Stage2Status.FAILED)

    def build_evidence_frame_guard_nudge(
        self,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState,
    ) -> HumanMessage:
        latest_guard_message = ""
        for entry in reversed(runtime.tool_trace):
            if entry.tool_name != "submit_final":
                continue
            value = entry.tool_input.get("evidence_frame_guard_message")
            if isinstance(value, str) and value:
                latest_guard_message = value
            elif entry.response_text.startswith("EVIDENCE_FRAME_GUARD"):
                latest_guard_message = entry.response_text
            break

        checklist = ""
        if latest_guard_message:
            checklist = f"\n\nLatest guard message:\n{latest_guard_message}"

        prompt = (
            "Your previous `submit_final` was blocked by EVIDENCE_FRAME_GUARD, "
            "so a direct structured final response in the same turn is not "
            "accepted. Continue the visual-grounding task and align the final "
            "proposal id with the marked frame you cite. If the cited marked "
            "frame shows the target, submit the proposal id whose mark directly "
            "covers it, even if that proposal label is a weak or wrong detector "
            "class."
            f"{checklist}\n\n"
            "Terminate through `submit_final` after revising the proposal id or "
            "the cited evidence."
        )
        return HumanMessage(content=[{"type": "text", "text": prompt}])

    def should_continue_after_direct_vg_response(
        self,
        task: Stage2TaskSpec,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState,
    ) -> bool:
        """Return true when a VG answer bypassed the pack finalizer."""
        if task.task_type != Stage2TaskType.VISUAL_GROUNDING:
            return False
        if runtime.final_submission is not None:
            return False
        return response.status in (Stage2Status.COMPLETED, Stage2Status.FAILED)

    def build_direct_vg_response_nudge(
        self,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState,
    ) -> HumanMessage:
        del runtime
        proposal_id = self._payload_proposal_id(response)
        if proposal_id is None:
            proposal_text = (
                "Your direct structured response did not contain a proposal_id."
            )
        else:
            proposal_text = (
                f"Your direct structured response selected proposal_id={proposal_id}."
            )

        prompt = (
            "Direct structured final responses are not accepted for visual grounding. "
            "The VG task must terminate through the `submit_final` tool so the "
            "pack validator, TADG, no-match guard, evidence-frame guard, and bbox "
            "adapter all run on the final proposal.\n\n"
            f"{proposal_text} If that choice is still correct, call "
            "`submit_final` with the same proposal_id, confidence, rationale, "
            "and evidence references. If the new evidence changes your decision, "
            "call `submit_final` with the revised proposal_id. Do not finish with "
            "another direct structured response."
        )
        return HumanMessage(content=[{"type": "text", "text": prompt}])

    @staticmethod
    def _response_from_submission(
        task: Stage2TaskSpec,
        submission: dict[str, Any],
        *,
        summary: str = "Submitted via chassis submit_final.",
    ) -> Stage2StructuredResponse:
        status_str = str(submission.get("status", "completed")).lower()
        try:
            status = Stage2Status(status_str)
        except ValueError:
            status = Stage2Status.COMPLETED
        return Stage2StructuredResponse(
            task_type=task.task_type,
            status=status,
            summary=summary,
            confidence=float(submission.get("confidence", 0.0)),
            payload=dict(submission),
        )

    def _prefer_deferred_submission_over_direct_vg_response(
        self,
        task: Stage2TaskSpec,
        response: Stage2StructuredResponse,
        runtime: Stage2RuntimeState | None,
    ) -> Stage2StructuredResponse | None:
        if task.task_type != Stage2TaskType.VISUAL_GROUNDING or runtime is None:
            return None
        extra = runtime.bundle.extra_metadata or {}
        submission = extra.get("stage2_submission")
        if not isinstance(submission, dict):
            return None
        submitted_pid = submission.get(
            "proposal_id", submission.get("selected_object_id")
        )
        try:
            submitted_pid_int = int(submitted_pid)
        except (TypeError, ValueError):
            return None
        response_pid = self._payload_proposal_id(response)
        if response_pid is None or response_pid == submitted_pid_int:
            return None
        submitted_conf = float(submission.get("confidence", 0.0) or 0.0)
        if submitted_conf <= float(response.confidence):
            return None
        return self._response_from_submission(
            task,
            submission,
            summary=(
                "Kept earlier chassis submit_final because a lower-confidence "
                "direct VG structured response tried to replace it."
            ),
        )

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
        # Drain explicit pending images. First-person frames only arrive here
        # after the agent calls a selector, mark_frame_with_bbox, or
        # request_crops; pack prep and bundle construction never inject RGB
        # frames directly.
        extra = runtime.bundle.extra_metadata or {}
        pending = extra.get("vg_pending_images", [])
        for marked_path in pending:
            if (
                marked_path not in runtime.seen_image_paths
                and Path(marked_path).exists()
            ):
                new_images.append(marked_path)
        if pending:
            # Pop the queue so we don't re-inject on the next turn.
            runtime.bundle = runtime.bundle.model_copy(
                update={"extra_metadata": {**extra, "vg_pending_images": []}}
            )

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

        metadata_by_path: dict[str, Any] = {}
        for item in extra.get("vg_pending_image_metadata", []) or []:
            if isinstance(item, dict) and item.get("image_path"):
                metadata_by_path[str(item["image_path"])] = item

        evidence_lines: list[str] = []
        for image_path in new_images:
            metadata = metadata_by_path.get(image_path)
            if isinstance(metadata, dict):
                frame_id = metadata.get("frame_id", "N/A")
                source = metadata.get("source_tool", metadata.get("source", "tool"))
                reason = metadata.get("selected_because", metadata.get("reason", ""))
                evidence_lines.append(
                    f"- image={Path(image_path).name}, frame_id={frame_id}, "
                    f"source={source}, reason={reason or 'N/A'}"
                )
            else:
                evidence_lines.append(f"- image={Path(image_path).name}")

        prompt = (
            "New visual evidence has been acquired:\n\n"
            f"Newly added visual evidence:\n{chr(10).join(evidence_lines)}\n\n"
            "If you already called submit_final before seeing these newly injected images, "
            "that submission was premature and was not accepted. Re-examine the visual "
            "evidence and submit again only after using it.\n\n"
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
        from agents.packs import ensure_default_packs_registered

        ensure_default_packs_registered()

        runtime = Stage2RuntimeState(bundle=bundle.model_copy(deep=True))
        runtime.task_type = task.task_type
        # Forward the pre-built Stage-1 text-to-frame selector to the runtime
        # state so `select_by_text` can call selector.select_keyframes_v2 at
        # tool-invocation time.
        runtime.text_frame_selector = self.text_frame_selector

        # Populate VG runtime state from bundle extra_metadata.
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

        tools = self.build_runtime_tools(runtime)
        graph = create_deep_agent(
            model=self.get_llm(),
            tools=tools,
            system_prompt=self.build_system_prompt(
                task, object_context=bundle.object_context
            ),
            subagents=self.build_subagents(task),
            response_format=Stage2StructuredResponse,
            name="query_scene_stage2_agent",
        )
        return graph, runtime

    def normalize_final_response(
        self,
        task: Stage2TaskSpec,
        raw_state: dict[str, Any],
        runtime: Stage2RuntimeState | None = None,
    ) -> Stage2StructuredResponse:
        """Convert DeepAgents final state into the unified Stage-2 schema.

        Args:
            task: Task specification
            raw_state: Raw state from DeepAgents graph
            runtime: Runtime state (optional). When provided and the
                pack chassis populated `runtime.final_submission`, the
                response is built from that adapted payload directly.

        Returns:
            Normalized structured response
        """
        if runtime is not None and runtime.final_submission is not None:
            return self._response_from_submission(task, runtime.final_submission)

        structured = raw_state.get("structured_response")
        if structured is not None:
            response = Stage2StructuredResponse.model_validate(structured)
            if response.task_type != task.task_type:
                response.task_type = task.task_type
            preferred = self._prefer_deferred_submission_over_direct_vg_response(
                task, response, runtime
            )
            if preferred is not None:
                return preferred
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
        1. Initial invocation with scene context and BEV only
        2. If tools acquire new evidence, inject new images
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
            "[DeepAgentsStage2Runtime] task={} plan_mode={} max_turns={}",
            task.task_type.value,
            task.plan_mode.value,
            task.max_reasoning_turns,
        )

        # Iterative evidence refinement loop
        messages = [message]
        raw_state: dict[str, Any] = {}
        turns_used = 0

        while turns_used < task.max_reasoning_turns:
            turns_used += 1
            raw_state = graph.invoke({"messages": messages})

            # Pack-v1 terminal: chassis submit_final populates runtime.final_submission.
            if runtime.final_submission is not None:
                if runtime.consume_evidence_update():
                    evidence_message = self.build_evidence_update_message(runtime)
                    if evidence_message is not None:
                        if "messages" in raw_state:
                            messages = raw_state["messages"]
                        messages.append(evidence_message)
                        runtime.final_submission = None
                        logger.info(
                            "[DeepAgentsStage2Runtime] turn {}: deferring submit_final "
                            "until newly queued visual evidence is injected",
                            turns_used,
                        )
                        continue
                logger.info(
                    "[DeepAgentsStage2Runtime] terminated at turn {} via chassis submit_final",
                    turns_used,
                )
                break

            # Check if structured response indicates completion
            structured = raw_state.get("structured_response")
            if structured is not None:
                response = Stage2StructuredResponse.model_validate(structured)
                if response.status in (Stage2Status.COMPLETED, Stage2Status.FAILED):
                    if (
                        turns_used < task.max_reasoning_turns
                        and self.should_continue_after_guarded_no_match(
                            task, response, runtime
                        )
                    ):
                        if "messages" in raw_state:
                            messages = raw_state["messages"]
                        messages.append(
                            self.build_no_match_guard_nudge(response, runtime)
                        )
                        logger.info(
                            "[DeepAgentsStage2Runtime] turn {}: continuing after "
                            "guarded direct no-match response",
                            turns_used,
                        )
                        continue
                    if (
                        turns_used < task.max_reasoning_turns
                        and self.should_continue_after_guarded_evidence_frame(
                            task, response, runtime
                        )
                    ):
                        if "messages" in raw_state:
                            messages = raw_state["messages"]
                        messages.append(
                            self.build_evidence_frame_guard_nudge(response, runtime)
                        )
                        logger.info(
                            "[DeepAgentsStage2Runtime] turn {}: continuing after "
                            "guarded direct final response",
                            turns_used,
                        )
                        continue
                    if (
                        turns_used < task.max_reasoning_turns
                        and self.should_continue_after_direct_vg_response(
                            task, response, runtime
                        )
                    ):
                        if "messages" in raw_state:
                            messages = raw_state["messages"]
                        messages.append(
                            self.build_direct_vg_response_nudge(response, runtime)
                        )
                        logger.info(
                            "[DeepAgentsStage2Runtime] turn {}: continuing after "
                            "direct VG structured response",
                            turns_used,
                        )
                        continue
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

        # v9: the agent can always acquire more evidence via the catalog-first
        # selectors + view tools, regardless of optional callbacks.
        can_acquire_more_evidence = turns_used < task.max_reasoning_turns

        final_response = self.normalize_final_response(task, raw_state, runtime)
        # Apply uncertainty-aware stopping rules
        final_response = self.apply_uncertainty_stopping(
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
