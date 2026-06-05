"""Base runtime class for Stage-2 agents with shared functionality."""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

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
    Stage2ToolObservation,
    Stage2ToolResult,
)

ToolCallback = Callable[[Stage2EvidenceBundle, dict[str, Any]], Any]

@dataclass
class Stage2RuntimeState:
    """Mutable per-run state shared by agent tools."""

    bundle: Stage2EvidenceBundle
    tool_trace: list[Stage2ToolObservation] = field(default_factory=list)
    # Tool calls set this when their trace observation includes image metadata.
    # The runtime then scans `tool_trace`; it does not maintain an image queue.
    evidence_updated: bool = False
    seen_image_paths: set[str] = field(
        default_factory=set
    )  # Track already-injected images

    # v9.1: pre-built Stage-1 text-to-frame selector instance. Populated by
    # `DeepAgentsStage2Runtime.build_agent` from the agent constructor's
    # text_frame_selector argument. The `select_by_text` tool reads this at
    # invocation time.
    text_frame_selector: Any | None = None

    # Codex MCP path: selector construction is expensive and must happen inside
    # the tool process, so `select_by_text` may lazily call this factory on its
    # first invocation and then cache the selector in `text_frame_selector`.
    text_frame_selector_factory: Callable[[], Any] | None = None

    # v9.2: when False, the `select_by_text` tool is omitted from the tool
    # set, the system prompt and playbooks switch to their catalog-first
    # variants. Copied from `Stage2DeepAgentConfig.enable_stage1_text_retrieval`
    # by `BaseStage2Runtime.configure_runtime_state`.
    enable_stage1_text_retrieval: bool = True

    # v9.4 cadence experiment (see
    # docs/benchmark/nr3d/v9_1_fix_vs_v9_3_audit30_20260517.md). When True,
    # `select_by_text` is still registered but short-circuits to ERROR
    # before touching `KeyframeSelector`. Copied from
    # `Stage2DeepAgentConfig.force_stage1_text_retrieval_to_error` by
    # `BaseStage2Runtime.configure_runtime_state`. Only meaningful when
    # `enable_stage1_text_retrieval=True`.
    force_stage1_text_retrieval_to_error: bool = False

    task_type: Stage2TaskType | None = None

    # Task-pack state populated for pack-backed tasks.
    task_ctx: Any | None = None
    skills_loaded: set[str] = field(default_factory=set)
    # Pack-v1 chassis terminal signal (set by submit_final on success).
    final_submission: dict | None = None

    # CVRA visible-proposal CLIP augmentation config. Copied from
    # Stage2DeepAgentConfig before pack tools are built.
    use_clip_visible_aug: bool = False
    clip_visible_tau: float = 0.18
    clip_visible_k_aug: int = 5
    clip_visible_backbone: str = "ViT-H-14/laion2b_s32b_b79k"
    clip_visible_cache_dir: str | None = None
    clip_visible_provider: Any | None = None

    # TADG (Tool-Answer Disagreement Gate) config + sticky run state.
    # The config fields are copied from Stage2DeepAgentConfig before
    # pack tools are built. The run-state fields below are mutated by
    # `evaluate_tadg` during submit_final.
    use_tool_answer_disagreement_gate: bool = False
    tadg_window: int = 32
    tadg_max_repeats: int = 3
    tadg_override_min_chars: int = 6
    tadg_triggered: bool = False  # ≥1 block fired this run
    tool_override_reason: str | None = None  # last accepted override
    tadg_block_count: dict[int, int] = field(default_factory=dict)

    # No-match candidate guard config + sticky state. This prevents VG
    # agents from prematurely finalizing `proposal_id=-1` while their own
    # tool trace still contains unresolved candidates.
    use_no_match_candidate_guard: bool = False
    no_match_guard_max_repeats: int = 3
    no_match_guard_max_viewed: int = 12
    no_match_guard_triggered: bool = False
    no_match_guard_block_count: int = 0

    # Final evidence-frame guard config + sticky state. This catches VG
    # submissions whose rationale cites a marked frame that does not
    # actually contain the submitted proposal id.
    use_evidence_frame_guard: bool = False
    evidence_frame_guard_triggered: bool = False
    evidence_frame_guard_block_count: int = 0

    def __setattr__(self, name: str, value: Any) -> None:
        if ("pending" in name and "image" in name) or (
            "initial" in name and "keyframe" in name
        ):
            raise AttributeError(
                f"{type(self).__name__}.{name} is forbidden; visual evidence "
                "must enter through tool_trace image_metadata, except BEV."
            )
        super().__setattr__(name, value)

    def record(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        response_text: str,
        *,
        image_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record a tool invocation in the trace."""
        self.tool_trace.append(
            Stage2ToolObservation(
                tool_name=tool_name,
                tool_input=tool_input,
                response_text=response_text,
                image_metadata=list(image_metadata or []),
            )
        )
        if image_metadata:
            self.mark_evidence_updated()

    def mark_evidence_updated(self) -> None:
        """Signal that tool-trace evidence should be scanned this turn."""
        self.evidence_updated = True

    def consume_evidence_update(self) -> bool:
        """Check and reset the evidence-updated flag."""
        updated = self.evidence_updated
        self.evidence_updated = False
        return updated


def default_output_instruction(task_type: Stage2TaskType) -> str:
    """Generate default output instruction based on task type."""
    if task_type == Stage2TaskType.QA:
        return "Answer the question and keep the answer grounded in cited frames."
    if task_type == Stage2TaskType.VISUAL_GROUNDING:
        return (
            "Identify and localize the described object in the 3D scene. "
            "Select the best matching proposal from the task proposal pool. "
            "If multiple candidates exist, explain why you chose one over others."
        )
    if task_type == Stage2TaskType.NAV_PLAN:
        return (
            "Produce a navigation plan grounded in visible landmarks and uncertainty."
        )
    if task_type == Stage2TaskType.MANIPULATION:
        return "Produce a manipulation plan with visible preconditions and missing evidence."
    return "Produce an evidence-grounded answer with explicit uncertainty."


def default_payload_schema(task_type: Stage2TaskType) -> dict[str, Any]:
    """Generate default payload schema based on task type."""
    if task_type == Stage2TaskType.QA:
        return {"answer": "str", "supporting_claims": ["str"]}
    if task_type == Stage2TaskType.VISUAL_GROUNDING:
        return {
            "proposal_id": "int from proposal pool; use -1 when target is not in pool",
            "confidence": "float in [0, 1]",
        }
    if task_type == Stage2TaskType.NAV_PLAN:
        return {
            "subgoals": ["str"],
            "landmarks": ["str"],
            "risks": ["str"],
        }
    if task_type == Stage2TaskType.MANIPULATION:
        return {
            "target_object": "str",
            "preconditions": ["str"],
            "action_sequence": ["str"],
            "failure_checks": ["str"],
        }
    return {"result": "str"}


class BaseStage2Runtime(ABC):
    """Abstract base class for Stage-2 agent runtime implementations.

    This class provides shared functionality for image handling, message building,
    tool callbacks, and uncertainty management. Concrete implementations (LangChain,
    DeepAgents) inherit from this and implement framework-specific execution logic.
    """

    def __init__(
        self,
        config: Stage2DeepAgentConfig | None = None,
        crop_callback: ToolCallback | None = None,
        text_frame_selector: Any | None = None,
        text_frame_selector_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize the agent runtime with configuration and (optional) crop callback.

        v9 removed the more_views / hypothesis Stage-1 callbacks; only the crop
        callback is preserved (still used by `request_crops`).

        v9.1 introduces `text_frame_selector`: a pre-built Stage-1 selector
        instance the agent uses when invoking the `select_by_text` tool
        (language → frames). Other selectors do not depend on this attribute.

        v9.3: construction MUST fail loud when the config wants text retrieval
        but neither a selector nor a selector factory is supplied. Previously
        this configuration produced an agent whose ``select_by_text`` tool was
        registered (so the system prompt said it was available) but returned
        ``"ERROR: runtime.text_frame_selector is None; cannot run Stage-1 text
        retrieval"`` at every invocation — a silent regression-trap that bit
        v9.1_fix (see docs/benchmark/nr3d/v9_1_real_stage1_actually_works_20260516.md
        and v9_1_fix_selector_wiring_20260515.md). The contract is now
        symmetric: either pass a selector/factory, or explicitly disable text
        retrieval via
        ``Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)``.

        Raises:
            ValueError: when ``config.enable_stage1_text_retrieval=True`` and
                neither a selector nor a selector factory is supplied.
        """
        self.config = config or Stage2DeepAgentConfig()
        self.crop_callback = crop_callback
        self.text_frame_selector: Any | None = text_frame_selector
        self.text_frame_selector_factory = text_frame_selector_factory
        if (
            self.config.enable_stage1_text_retrieval
            and self.text_frame_selector is None
            and self.text_frame_selector_factory is None
        ):
            raise ValueError(
                "Stage2 runtime constructed with "
                "enable_stage1_text_retrieval=True but no text_frame_selector "
                "or text_frame_selector_factory. Either pass a pre-built "
                "text-frame selector via text_frame_selector=..., pass a lazy "
                "factory via text_frame_selector_factory=..., or disable "
                "Stage-1 text retrieval explicitly via "
                "Stage2DeepAgentConfig(enable_stage1_text_retrieval=False). "
                "Without one of these, select_by_text would be registered "
                "as a tool but fail at every invocation with "
                "'runtime.text_frame_selector is None; cannot run Stage-1 text "
                "retrieval'. See docs/benchmark/nr3d/"
                "v9_1_real_stage1_actually_works_20260516.md for the historical "
                "v9.1_fix regression this guard now prevents."
            )
        self._session_id = self.config.session_id

    def build_extra_body(self) -> dict[str, Any]:
        """Build the provider-specific extra_body payload for prompt caching."""
        extra_body = dict(self.config.extra_body)
        thinking = dict(extra_body.get("thinking", {}))
        if self.config.include_thoughts:
            thinking["include_thoughts"] = True
        if thinking:
            extra_body["thinking"] = thinking
        extra_body["session_id"] = self._session_id
        return extra_body

    def configure_runtime_state(self, runtime: Stage2RuntimeState) -> None:
        """Copy config-backed feature flags onto one mutable runtime state."""
        runtime.use_clip_visible_aug = self.config.use_clip_visible_aug
        runtime.clip_visible_tau = self.config.clip_visible_tau
        runtime.clip_visible_k_aug = self.config.clip_visible_k_aug
        runtime.clip_visible_backbone = self.config.clip_visible_backbone
        runtime.clip_visible_cache_dir = self.config.clip_visible_cache_dir

        if os.environ.get("CVRA_DISABLE") == "1":
            runtime.use_clip_visible_aug = False
        backbone_override = os.environ.get("CVRA_BACKBONE_OVERRIDE")
        if backbone_override:
            runtime.clip_visible_backbone = backbone_override

        runtime.use_tool_answer_disagreement_gate = (
            self.config.use_tool_answer_disagreement_gate
        )
        runtime.tadg_window = self.config.tadg_window
        runtime.tadg_max_repeats = self.config.tadg_max_repeats
        runtime.tadg_override_min_chars = self.config.tadg_override_min_chars
        runtime.use_no_match_candidate_guard = self.config.use_no_match_candidate_guard
        runtime.no_match_guard_max_repeats = self.config.no_match_guard_max_repeats
        runtime.no_match_guard_max_viewed = self.config.no_match_guard_max_viewed
        runtime.use_evidence_frame_guard = self.config.use_evidence_frame_guard
        runtime.enable_stage1_text_retrieval = self.config.enable_stage1_text_retrieval
        runtime.force_stage1_text_retrieval_to_error = (
            self.config.force_stage1_text_retrieval_to_error
        )
        if (
            self.text_frame_selector_factory is not None
            and runtime.text_frame_selector_factory is None
        ):
            runtime.text_frame_selector_factory = self.text_frame_selector_factory

        if os.environ.get("TADG_DISABLE") == "1":
            runtime.use_tool_answer_disagreement_gate = False
        if os.environ.get("NO_MATCH_GUARD_DISABLE") == "1":
            runtime.use_no_match_candidate_guard = False
        if os.environ.get("EVIDENCE_FRAME_GUARD_DISABLE") == "1":
            runtime.use_evidence_frame_guard = False

    def image_to_data_url(self, image_path: str | Path) -> str:
        """Convert an image file into a data URL for multimodal chat models."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for Stage-2 image encoding.") from exc

        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        if max(width, height) > self.config.image_max_size:
            ratio = self.config.image_max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def collect_image_paths(self, bundle: Stage2EvidenceBundle) -> list[str]:
        """v9 catalog-first: only the BEV image is part of the initial HumanMessage."""
        if bundle.bev_image_path and Path(bundle.bev_image_path).exists():
            return [str(bundle.bev_image_path)]
        return []

    def coerce_callback_result(self, result: Any) -> Stage2ToolResult:
        """Normalize external callback payloads for tool responses."""
        if isinstance(result, Stage2ToolResult):
            return result
        if isinstance(result, Stage2EvidenceBundle):
            return Stage2ToolResult(
                response_text="Received updated evidence bundle.",
                updated_bundle=result,
            )
        if isinstance(result, str):
            return Stage2ToolResult(response_text=result)
        if isinstance(result, dict):
            updated_bundle = result.get("updated_bundle")
            if isinstance(updated_bundle, Stage2EvidenceBundle):
                payload = dict(result)
                payload.pop("updated_bundle", None)
                image_metadata = payload.pop("image_metadata", None)
                return Stage2ToolResult(
                    response_text=json.dumps(payload, indent=2, ensure_ascii=False),
                    updated_bundle=updated_bundle,
                    image_metadata=list(image_metadata or []),
                )
            payload = dict(result)
            image_metadata = payload.pop("image_metadata", None)
            return Stage2ToolResult(
                response_text=json.dumps(payload, indent=2, ensure_ascii=False),
                image_metadata=list(image_metadata or []),
            )
        return Stage2ToolResult(response_text=str(result))

    def retrieve_object_context_text(
        self,
        bundle: Stage2EvidenceBundle,
        object_terms: Sequence[str] | None,
    ) -> str:
        """Return the requested subset of object context."""
        if not bundle.object_context:
            return (
                bundle.scene_summary or "No object context or scene summary available."
            )

        if not object_terms:
            return json.dumps(bundle.object_context, indent=2, ensure_ascii=False)

        lowered = [term.lower() for term in object_terms]
        selected: dict[str, str] = {}
        for key, value in bundle.object_context.items():
            key_lower = key.lower()
            if any(term in key_lower or key_lower in term for term in lowered):
                selected[key] = value

        if not selected:
            return "No matching object context found for requested terms."
        return json.dumps(selected, indent=2, ensure_ascii=False)

    @staticmethod
    def _format_vg_candidates(
        extra_metadata: dict[str, Any],
    ) -> str:
        """Format VG candidate objects with 3D positions for the prompt."""
        candidates = extra_metadata.get("vg_candidates", [])
        if not candidates:
            return ""
        lines = ["## Object Candidates for Grounding\n"]
        for c in candidates:
            line = (
                f"- [ID={c['obj_id']}] {c['category']}: "
                f"position=({c['cx']:.2f}, {c['cy']:.2f}, {c['cz']:.2f}), "
                f"size=({c['dx']:.2f}, {c['dy']:.2f}, {c['dz']:.2f})"
            )
            desc = c.get("description", "")
            if desc:
                line += f"\n  Description: {desc[:100]}"
            lines.append(line)
        return "\n".join(lines) + "\n\n"

    def _format_skill_catalog(self, task_type: Stage2TaskType) -> str:
        """Render the skill catalog block for the system prompt.

        Suppressed for packs that do NOT expose chassis (e.g. QA), so the
        prompt stays byte-stable for tasks where the skills are dead weight.
        """
        from agents.skills.registry import PACKS, skills_for

        pack = PACKS.get(task_type)
        if pack is not None and not pack.exposes_chassis:
            return ""
        skills = skills_for(task_type)
        if not skills:
            return ""
        lines = ["Available skills (use list_skills() / load_skill(name) for details):"]
        for s in skills:
            lines.append(f"- {s.name}: {s.description}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _format_scene_inventory(object_context: dict[str, str] | None) -> str:
        """Format object context as a scene inventory for the system prompt."""
        if not object_context:
            return ""
        lines = []
        for name, desc in sorted(object_context.items()):
            short_desc = desc[:120].replace("\n", " ")
            lines.append(f"- {name}: {short_desc}")
        inventory = "\n".join(lines)
        return (
            "Scene object inventory (from 3D scene graph + LLM enrichment):\n"
            "Use this to identify objects that may be in the scene but not immediately "
            "visible in the currently selected first-person frames. Cross-reference when identifying objects.\n"
            f"{inventory}\n\n"
        )

    def build_system_prompt(
        self,
        task: Stage2TaskSpec,
        object_context: dict[str, str] | None = None,
    ) -> str:
        """Build the v9 catalog-first system prompt."""
        plan_instructions = {
            Stage2PlanMode.OFF: (
                "Plan mode is OFF. Only use the todo list if the task is unexpectedly complex."
            ),
            Stage2PlanMode.BRIEF: (
                "Plan mode is BRIEF. Maintain a short todo list (2-4 items) covering evidence "
                "acquisition and answer synthesis."
            ),
            Stage2PlanMode.FULL: (
                "Plan mode is FULL. Maintain an explicit todo list decomposed into evidence "
                "acquisition, verification, and task synthesis."
            ),
        }
        payload_schema = task.expected_output_schema or default_payload_schema(
            task.task_type
        )
        instruction = task.output_instruction or default_output_instruction(
            task.task_type
        )
        text_first = self.config.enable_stage1_text_retrieval
        if text_first:
            workflow_hint = (
                "Workflow: selectors inject candidate RGB frames; use "
                "mark_frame_with_bbox on the one frame you have decided is worth "
                "verifying."
            )
            selector_lines = (
                "1. Selectors (each returns ≤3 first-person RGB frames + metadata):\n"
                "   - select_by_text(query, k≤3, hidden_categories) — Stage-1 "
                "language→frame, primary entry\n"
                "   - select_by_proposal(proposal_ids, require_all, k≤3)\n"
                "   - select_by_frame_neighbor(anchor_frame_id, "
                "mode='temporal'|'viewpoint_diverse', k≤3)\n"
                "   - select_by_region(region, region_type, k≤3)\n"
                "   - select_by_coverage(method='obj_iou'|'pose_depth', k≤3, "
                "seen_frame_ids?)\n"
            )
        else:
            workflow_hint = (
                "Workflow: read the BEV and the SceneCatalog Cat-B inventory, "
                "pick candidate proposal_ids, fetch frames with "
                "select_by_proposal / select_by_region, then mark_frame_with_bbox "
                "on the one frame you have decided is worth verifying."
            )
            selector_lines = (
                "1. Selectors (each returns ≤3 first-person RGB frames + metadata):\n"
                "   - select_by_proposal(proposal_ids, require_all, k≤3) — "
                "primary entry; fetch frames containing candidate catalog IDs\n"
                "   - select_by_region(region, region_type, k≤3)\n"
                "   - select_by_frame_neighbor(anchor_frame_id, "
                "mode='temporal'|'viewpoint_diverse', k≤3)\n"
                "   - select_by_coverage(method='obj_iou'|'pose_depth', k≤3, "
                "seen_frame_ids?)\n"
            )

        crop_tool_line = ""
        if self.crop_callback is not None:
            crop_tool_line = (
                "5. request_crops(request_text, frame_indices=[...], "
                "object_terms=[...]) — optional pixel zoom. Treat it as evidence "
                "only when it returns crop image outputs; if it returns ERROR or "
                "No crops generated, do not cite it.\n"
            )

        return (
            "You are the Stage-2 scene reasoning agent.\n\n"
            "Scene perception model:\n"
            "- You start with a BEV overview image plus a SceneCatalog text "
            "(category -> [#id, ...]).\n"
            "- You have viewed 0 first-person frames at task start.\n"
            "- The BEV labels are a starting point, not first-person evidence; "
            "you must fetch frames.\n\n"
            "Tool families (always `load_skill('scene-exploration-playbook')` "
            "before selectors / view tools):\n"
            f"{selector_lines}"
            "2. mark_frame_with_bbox(frame_id, labels?, ids?) — high-contrast "
            "annotated zoom.\n"
            "   Requires at least one of labels / ids.\n"
            "3. view_bev(highlight=?) — re-inject BEV (full or filtered).\n"
            "4. Catalog: list_scene_proposals, list_frame_proposals, "
            "inspect_proposal.\n"
            f"{crop_tool_line}"
            f"{workflow_hint}\n\n"
            "Skill gate:\n"
            "- Every selector + mark_frame_with_bbox + view_bev + "
            "list_scene_proposals + inspect_proposal + compare_proposals_spatial\n"
            "  refuses to run until you `load_skill('scene-exploration-playbook')`. "
            "After that, load the\n"
            "  task-specific playbook (`vg-grounding-playbook` for VG, "
            "`qa-answering-playbook` for QA).\n\n"
            f"{self._format_skill_catalog(task.task_type)}"
            "Framework constraints:\n"
            "- LangChain v1 + DeepAgents runtime.\n"
            f"- Maximum reasoning budget: {task.max_reasoning_turns} turns.\n\n"
            f"{plan_instructions[task.plan_mode]}\n\n"
            "Unified output contract:\n"
            f"- task_type must be `{task.task_type.value}`.\n"
            "- status must reflect whether the task is complete or evidence-limited.\n"
            "- payload must follow the schema below.\n"
            "- cited_frame_indices must only cite frames you actually viewed.\n\n"
            f"Task-specific instruction: {instruction}\n"
            f"Expected payload schema: "
            f"{json.dumps(payload_schema, indent=2, ensure_ascii=False)}"
        )

    def apply_uncertainty_stopping(
        self,
        response: Stage2StructuredResponse,
        can_acquire_more_evidence: bool,
    ) -> Stage2StructuredResponse:
        """Apply uncertainty-aware stopping rules to the response.

        This implements the "evidence-grounded uncertainty" principle:
        - If confidence is below threshold AND no more evidence can be acquired,
          the agent should stop with INSUFFICIENT_EVIDENCE status
        - If the agent claims completion but confidence is too low, downgrade status
        - Ensures the agent doesn't hallucinate answers when evidence is missing

        Args:
            response: The structured response from the agent
            can_acquire_more_evidence: Whether the loop can continue acquiring evidence

        Returns:
            Potentially modified response with appropriate status
        """
        if not self.config.enable_uncertainty_stopping:
            return response

        threshold = self.config.confidence_threshold

        # Case 1: Agent completed with low confidence and no more evidence available
        if (
            response.status == Stage2Status.COMPLETED
            and response.confidence < threshold
            and not can_acquire_more_evidence
        ):
            logger.info(
                "[BaseStage2Runtime] downgrading COMPLETED to INSUFFICIENT_EVIDENCE: "
                "confidence={:.2f} < threshold={:.2f}, cannot acquire more evidence",
                response.confidence,
                threshold,
            )
            return Stage2StructuredResponse(
                task_type=response.task_type,
                status=Stage2Status.INSUFFICIENT_EVIDENCE,
                summary=f"Low confidence answer ({response.confidence:.2f}): {response.summary}",
                confidence=response.confidence,
                uncertainties=list(response.uncertainties)
                + [
                    f"Confidence {response.confidence:.2f} below threshold {threshold:.2f}. "
                    "The answer may not be reliable due to insufficient visual evidence."
                ],
                cited_frame_indices=response.cited_frame_indices,
                evidence_items=response.evidence_items,
                plan=response.plan,
                payload=response.payload,
            )

        # Case 2: Agent already indicated insufficient evidence - validate
        if response.status == Stage2Status.INSUFFICIENT_EVIDENCE:
            logger.info(
                "[BaseStage2Runtime] agent correctly reported insufficient evidence "
                "with confidence={:.2f}",
                response.confidence,
            )
            return response

        # Case 3: Agent needs more evidence but can't acquire it
        if (
            response.status == Stage2Status.NEEDS_MORE_EVIDENCE
            and not can_acquire_more_evidence
        ):
            logger.info(
                "[BaseStage2Runtime] upgrading NEEDS_MORE_EVIDENCE to INSUFFICIENT_EVIDENCE: "
                "evidence acquisition exhausted"
            )
            return Stage2StructuredResponse(
                task_type=response.task_type,
                status=Stage2Status.INSUFFICIENT_EVIDENCE,
                summary=response.summary,
                confidence=response.confidence,
                uncertainties=list(response.uncertainties)
                + ["Unable to acquire additional evidence to complete the task."],
                cited_frame_indices=response.cited_frame_indices,
                evidence_items=response.evidence_items,
                plan=response.plan,
                payload=response.payload,
            )

        return response

    @abstractmethod
    def run(
        self, task: Stage2TaskSpec, bundle: Stage2EvidenceBundle
    ) -> Stage2AgentResult:
        """Execute the agent with the given task and evidence bundle.

        This method must be implemented by concrete runtime classes to provide
        framework-specific execution logic (LangChain, DeepAgents, etc.).

        Args:
            task: Task specification with query, type, and constraints
            bundle: Evidence bundle with scene context and active visual metadata

        Returns:
            AgentResult with response, tool trace, and final bundle
        """
        raise NotImplementedError("Subclasses must implement run()")
