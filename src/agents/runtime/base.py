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
    evidence_updated: bool = False  # Signals new images need injection
    seen_image_paths: set[str] = field(
        default_factory=set
    )  # Track already-injected images

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

    def record(
        self, tool_name: str, tool_input: dict[str, Any], response_text: str
    ) -> None:
        """Record a tool invocation in the trace."""
        self.tool_trace.append(
            Stage2ToolObservation(
                tool_name=tool_name,
                tool_input=tool_input,
                response_text=response_text,
            )
        )

    def mark_evidence_updated(self) -> None:
        """Signal that the bundle was updated and new images may need injection."""
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
        more_views_callback: ToolCallback | None = None,
        crop_callback: ToolCallback | None = None,
        hypothesis_callback: ToolCallback | None = None,
    ) -> None:
        """Initialize the agent runtime with configuration and callbacks.

        Args:
            config: Agent configuration (model, API, behavior settings)
            more_views_callback: Callback for requesting additional views
            crop_callback: Callback for requesting object/region crops
            hypothesis_callback: Callback for hypothesis switching/expansion
        """
        self.config = config or Stage2DeepAgentConfig()
        self.more_views_callback = more_views_callback
        self.crop_callback = crop_callback
        self.hypothesis_callback = hypothesis_callback
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
        """Collect keyframes and optional BEV images for a run."""
        images: list[str] = []
        for keyframe in bundle.keyframes[: self.config.max_images]:
            if Path(keyframe.image_path).exists():
                images.append(keyframe.image_path)

        if (
            bundle.bev_image_path
            and Path(bundle.bev_image_path).exists()
            and len(images) < self.config.max_images
        ):
            images.append(bundle.bev_image_path)

        return images

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
                return Stage2ToolResult(
                    response_text=json.dumps(payload, indent=2, ensure_ascii=False),
                    updated_bundle=updated_bundle,
                )
            return Stage2ToolResult(
                response_text=json.dumps(result, indent=2, ensure_ascii=False)
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

    def _format_vg_section(self, extra_schema: dict[str, Any]) -> str:
        """Build the VG-specific system prompt section."""
        if self.config.vg_backend != "pack_v1":
            raise ValueError(
                f"vg_backend={self.config.vg_backend!r} no longer supported; "
                "legacy branch removed in Plan C. Set vg_backend='pack_v1'."
            )
        return (
            "## Visual Grounding Protocol (pack v1)\n\n"
            "You are localizing a target object described in natural language.\n\n"
            "The VG pack provides a proposal pool, marked-keyframe renderings, "
            "proposal inspection, spatial-comparison helpers, and the chassis trio.\n\n"
            "Marked-keyframe images show colored 2D boxes for visible candidate "
            "proposals. The text inside each box is `#proposal_id category`; "
            "the `proposal_id` after `#` is the id to use in `inspect_proposal` "
            "and `submit_final`. Treat the category text as a weak detector "
            "label and choose the box that covers the referred object.\n\n"
            "### How to start\n"
            "1. Call `list_skills()` to see what skills are available for VG.\n"
            "2. Load `vg-grounding-playbook` first via "
            "`load_skill('vg-grounding-playbook')`; it explains every VG tool "
            "and the `submit_final` payload schema. The 5 VG tools refuse to run "
            "until that skill is loaded.\n"
            "3. Follow the playbook's decision tree, then call "
            '`submit_final({"proposal_id": int, "confidence": float}, '
            "rationale=...)` to terminate.\n\n"
            "### MANDATORY rules\n"
            "- Do NOT invent a `proposal_id` outside the pool; the chassis "
            "validator will reject it.\n"
            "- If the referent genuinely is not in the pool, submit "
            "`proposal_id=-1, confidence=0.0` with a rationale (see playbook).\n\n"
            "- If the no-match candidate guard is enabled, `proposal_id=-1` "
            "will be soft-blocked when your own tool trace still has unresolved "
            "category candidates or viewed-but-uninspected marked proposals. "
            "Use the block message as a compact candidate checklist, inspect or "
            "choose the best remaining proposal, and only repeat `-1` after "
            "explicitly closing those candidates. Treat proposal labels as weak "
            "priors: if the pixels show the referent, choose the proposal that "
            "covers it even when the Mask3D label mismatches the query.\n\n"
            "- If the evidence-frame guard is enabled, a final answer whose "
            "rationale cites a marked frame must submit a proposal id visible "
            "in that cited frame. When a marked frame shows the target, choose "
            "the mark covering the target; a large mislabeled proposal in that "
            "frame is better evidence than a semantically named proposal from "
            "a different frame.\n\n"
            "- For left/right referring expressions, use the `left_to_right` "
            "ordering returned by `view_keyframe_marked` to align proposal ids "
            "with the marked frame before submitting.\n\n"
        )

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
            "visible in your keyframes. Cross-reference when identifying objects.\n"
            f"{inventory}\n\n"
        )

    def build_system_prompt(
        self,
        task: Stage2TaskSpec,
        object_context: dict[str, str] | None = None,
    ) -> str:
        """Build the agent system prompt."""
        plan_instructions = {
            Stage2PlanMode.OFF: (
                "Plan mode is OFF. Only use the todo list if the task is unexpectedly complex."
            ),
            Stage2PlanMode.BRIEF: (
                "Plan mode is BRIEF. Before major evidence collection, keep a short todo list "
                "with 2-4 items covering evidence acquisition and answer synthesis."
            ),
            Stage2PlanMode.FULL: (
                "Plan mode is FULL. Maintain an explicit todo list throughout execution and "
                "decompose work into evidence acquisition, verification, and task synthesis."
            ),
        }

        payload_schema = task.expected_output_schema or default_payload_schema(
            task.task_type
        )
        instruction = task.output_instruction or default_output_instruction(
            task.task_type
        )

        # Build uncertainty-aware instructions
        uncertainty_instructions = (
            "Uncertainty-aware stopping:\n"
            f"- Minimum confidence threshold for completion: {self.config.confidence_threshold:.2f}\n"
            "- If you cannot find sufficient evidence to answer with confidence above this threshold, "
            "set status to 'insufficient_evidence' rather than guessing.\n"
            "- List all missing evidence or ambiguous observations in the 'uncertainties' field.\n"
            "- It is better to admit uncertainty than to hallucinate answers.\n"
            "- Your confidence score should reflect actual evidence quality, not task difficulty.\n\n"
        )

        vg_section = ""
        if task.task_type == Stage2TaskType.VISUAL_GROUNDING:
            extra = task.expected_output_schema or {}
            vg_section = self._format_vg_section(extra)

        temporal_fan_line = ""
        if self.config.enable_temporal_fan:
            temporal_fan_line = (
                "3. request_more_views(mode='temporal_fan', frame_indices=[X, Y]) — "
                "get views temporally adjacent to X, Y along the scan trajectory, "
                "filtered to overlap < 50% for meaningful visual change. Use when "
                "you want more angles of objects already visible in specific keyframes.\n"
            )

        crops_index = "4" if self.config.enable_temporal_fan else "3"
        hypothesis_index = "5" if self.config.enable_temporal_fan else "4"

        return (
            "You are the Stage-2 research agent for query-scene.\n\n"
            "Research role:\n"
            "- Stage 1 is a high-recall evidence retriever, not ground truth.\n"
            "- Stage 2 must verify, repair, or reject Stage-1 hypotheses using pixels.\n"
            "- Prefer evidence-seeking behavior over one-shot answering.\n"
            "- Use tools when keyframes are insufficient; do not hallucinate missing evidence.\n"
            "- Explicitly surface uncertainty when the necessary evidence is absent.\n\n"
            "CRITICAL - Evidence-seeking protocol:\n"
            "- ALWAYS examine the provided keyframe images FIRST before calling any tools.\n"
            "- If the answer is clearly visible in the current images, answer directly.\n"
            "- If the TARGET OBJECT or QUERIED ATTRIBUTE is NOT visible in ANY keyframe, "
            "you MUST seek more evidence before answering or reporting insufficient_evidence. "
            "Do NOT guess from contextual clues when the target is simply not in frame.\n\n"
            "Tool strategy (use in this order):\n"
            "1. request_more_views(mode='targeted', object_terms=[...]) — get views showing specific objects\n"
            "2. request_more_views(mode='explore') — get views of unseen scene regions; optional frame_indices=[...] pins specific frames you already want, capped at the per-call max, and the selector treats them as the first preferred candidates.\n"
            f"{temporal_fan_line}"
            f"{crops_index}. request_crops(object_terms=[...]) — zoom into small/ambiguous objects with annotated bboxes\n"
            f"{hypothesis_index}. switch_or_expand_hypothesis(new_query='...') — re-run retrieval with a different query (costly, use as last resort)\n"
            "Use multiple tools across turns: explore → crop details → answer.\n\n"
            "MANDATORY tool-usage rules:\n"
            "- If your answer CONTRADICTS the question's premise (e.g., question asks about 'non-black chairs' "
            "but you only see black ones), you MUST call request_more_views before answering.\n"
            "- If the target object is NOT visible in ANY keyframe, you MUST call request_more_views at least once.\n"
            "- For ALL color/attribute questions, you MUST call request_crops on the target object BEFORE answering. No exceptions.\n"
            "- When describing colors, list ALL distinct colors visible on the object, not just the dominant one "
            "(e.g., 'white with green accents' not just 'teal').\n"
            "- For YES/NO state questions (full/empty, clean/dirty, well-lit/dark, organized), "
            "you MUST request_crops or request_more_views before answering, regardless of confidence.\n"
            "- For spatial 'between X and Y' questions, verify BOTH landmarks are visible before answering.\n"
            "- Do NOT report confidence > 0.7 if you used zero tools and the question involves "
            "spatial relations, object attributes, or object identification.\n\n"
            "SELF-CHECK before final answer:\n"
            "- Does your answer contradict the question premise? If so, request more views.\n"
            "- For spatial questions, list at least 2-3 candidate objects before selecting your answer.\n"
            "- WARNING: The correct answer may be a smaller or less prominent object. "
            "Do NOT default to the largest/most obvious object in frame. Consider ALL objects "
            "including small items on surfaces, items on the floor, and partially occluded objects.\n\n"
            f"{uncertainty_instructions}"
            f"{vg_section}"
            f"{self._format_skill_catalog(task.task_type)}"
            f"{self._format_scene_inventory(object_context)}"
            "Framework constraints:\n"
            "- This runtime is built with LangChain v1 and DeepAgents.\n"
            "- Use the built-in todo planning capability according to the selected plan mode.\n"
            "- Skill-based decomposition replaces DeepAgents subagents; use the skill catalog "
            "and load evidence/output skills when decomposition is useful.\n"
            f"- Maximum reasoning budget: {task.max_reasoning_turns} turns.\n\n"
            f"{plan_instructions[task.plan_mode]}\n\n"
            "Unified output contract:\n"
            f"- task_type must be `{task.task_type.value}`.\n"
            "- status must reflect whether the task is complete or evidence-limited.\n"
            "- summary must be concise and evidence-grounded.\n"
            "- confidence must stay calibrated.\n"
            "- uncertainties must list missing or ambiguous evidence.\n"
            "- cited_frame_indices must only cite visible supporting frames.\n"
            "- evidence_items should map concrete claims to frames and objects.\n"
            "- payload should follow the expected task-specific schema below.\n\n"
            f"Task-specific instruction: {instruction}\n"
            f"Expected payload schema: {json.dumps(payload_schema, indent=2, ensure_ascii=False)}"
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
            bundle: Evidence bundle with keyframes and context

        Returns:
            AgentResult with response, tool trace, and final bundle
        """
        raise NotImplementedError("Subclasses must implement run()")
