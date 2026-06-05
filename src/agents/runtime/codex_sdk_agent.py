"""Codex Agent SDK runtime for Stage-2 visual grounding.

This runtime is intentionally separate from the DeepAgents runtime. The Codex
SDK is an agent runtime, not a LangChain chat model replacement, so it cannot
consume the existing ``BaseTool`` list directly. The first supported path is a
catalog-first NR3D/VG entrypoint that asks Codex to choose a proposal from the
existing pack-v1 proposal pool and returns the same Stage2 result envelope used
by the benchmark runner.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.agent_config import Stage2Status, Stage2TaskType
from ..core.response_schema import Stage2StructuredResponse, Stage2ToolObservation
from ..core.task_types import Stage2AgentResult, Stage2EvidenceBundle, Stage2TaskSpec
from ..models import Stage2DeepAgentConfig
from ..packs.vg_embodiedscan.ctx import build_ctx_from_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CODEX_HOME = PROJECT_ROOT / ".codex-home"
DEFAULT_CODEX_MODEL = "gpt-5.5-2026-04-24"
DEFAULT_CODEX_MODEL_PROVIDER = "modelhub_adapter"
DEFAULT_NR3D_SKILL_PATH = (
    PROJECT_ROOT / ".agents" / "skills" / "nr3d-codex-sdk" / "SKILL.md"
)


class CodexVisualGroundingDecision(BaseModel):
    """Minimal JSON contract requested from Codex SDK."""

    proposal_id: int = Field(
        description="Selected proposal id from the provided proposal pool; -1 if absent."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = ""
    uncertainties: list[str] = Field(default_factory=list)
    cited_frame_indices: list[int] = Field(default_factory=list)


@dataclass(frozen=True)
class CodexTurnMetadata:
    id: str | None = None
    status: str | None = None
    duration_ms: int | None = None
    usage: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "usage": self.usage,
        }


class CodexSdkStage2Runtime:
    """Stage-2 runtime backed by the OpenAI Codex Agent SDK."""

    def __init__(
        self,
        config: Stage2DeepAgentConfig | None = None,
        *,
        codex_home: str | Path | None = None,
        model: str | None = None,
        model_provider: str | None = None,
        skill_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.config = config or Stage2DeepAgentConfig(
            enable_stage1_text_retrieval=False
        )
        if self.config.enable_stage1_text_retrieval:
            raise ValueError(
                "CodexSdkStage2Runtime does not expose the DeepAgents "
                "select_by_text tool. Construct it with "
                "Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)."
            )
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.codex_home = Path(
            codex_home
            or os.environ.get("CODEX_HOME")
            or self.project_root / ".codex-home"
        )
        self.model = model or os.environ.get("CODEX_AGENT_MODEL", DEFAULT_CODEX_MODEL)
        self.model_provider = model_provider or os.environ.get(
            "CODEX_AGENT_MODEL_PROVIDER", DEFAULT_CODEX_MODEL_PROVIDER
        )
        self.skill_path = Path(skill_path) if skill_path else DEFAULT_NR3D_SKILL_PATH

    def run(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
    ) -> Stage2AgentResult:
        if task.task_type != Stage2TaskType.VISUAL_GROUNDING:
            raise NotImplementedError(
                "Codex SDK Stage-2 runtime currently supports visual_grounding only"
            )

        ctx = build_ctx_from_bundle(bundle)
        valid_ids = {proposal.id for proposal in ctx.proposals}
        prompt = self.build_decision_prompt(task, bundle)
        image_paths = self.collect_codex_image_paths(bundle)
        response_text, metadata_raw = self._run_codex_turn(prompt, image_paths)
        decision = CodexVisualGroundingDecision.model_validate(
            self._parse_json_object(response_text)
        )
        if decision.proposal_id != -1 and decision.proposal_id not in valid_ids:
            raise ValueError(
                f"Codex SDK returned proposal_id={decision.proposal_id}, "
                f"which is not in the proposal pool"
            )

        status = (
            Stage2Status.FAILED
            if decision.proposal_id == -1
            else Stage2Status.COMPLETED
        )
        payload: dict[str, Any] = {
            "proposal_id": decision.proposal_id,
            "confidence": decision.confidence,
        }
        if decision.proposal_id != -1:
            payload["selected_object_id"] = decision.proposal_id

        structured = Stage2StructuredResponse(
            task_type=task.task_type,
            status=status,
            summary=decision.summary,
            confidence=decision.confidence,
            uncertainties=decision.uncertainties,
            cited_frame_indices=decision.cited_frame_indices,
            payload=payload,
        )
        metadata = self._coerce_metadata(metadata_raw)
        trace = [
            Stage2ToolObservation(
                tool_name="codex_sdk_turn",
                tool_input={
                    "model": self.model,
                    "model_provider": self.model_provider,
                    "codex_home": str(self.codex_home),
                    "skill_path": str(self.skill_path),
                    "attached_images": [str(path) for path in image_paths],
                },
                response_text=response_text,
                image_metadata=[
                    {"image_path": str(path), "kind": "bev"} for path in image_paths
                ],
            )
        ]
        return Stage2AgentResult(
            task=task,
            result=structured,
            tool_trace=trace,
            final_bundle=bundle,
            raw_state={
                "runtime": "codex_sdk",
                "codex_turn": metadata.as_dict(),
            },
        )

    def build_decision_prompt(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
    ) -> str:
        """Build a catalog-first VG prompt without leaking benchmark GT fields."""
        ctx = build_ctx_from_bundle(bundle)
        extra = bundle.extra_metadata or {}
        catalog = extra.get("scene_catalog") if isinstance(extra, dict) else None
        scene_category = ""
        total_frames = ""
        frame_range = ""
        if isinstance(catalog, dict):
            scene_category = str(catalog.get("scene_category") or "")
            total_frames = str(catalog.get("total_frames") or "")
            frame_range = str(catalog.get("frame_id_range") or "")

        proposal_lines = []
        for proposal in sorted(ctx.proposals, key=lambda item: item.id):
            cx, cy, cz, sx, sy, sz, rx, ry, rz = proposal.bbox_3d_9dof
            note = self._truncate(
                proposal.compact_note or proposal.enriched_category or "",
                220,
            )
            enriched = (
                f", enriched={proposal.enriched_category}"
                if proposal.enriched_category
                else ""
            )
            visible = sorted(proposal.frame_views)
            visible_preview = visible[:8]
            proposal_lines.append(
                f"- #{proposal.id}: category={proposal.category}{enriched}; "
                f"center=({cx:.3f},{cy:.3f},{cz:.3f}); "
                f"size=({sx:.3f},{sy:.3f},{sz:.3f}); "
                f"rot=({rx:.3f},{ry:.3f},{rz:.3f}); "
                f"visible_frames={len(visible)} {visible_preview}; "
                f"note={note}"
            )

        by_category: dict[str, list[int]] = {}
        for proposal in ctx.proposals:
            by_category.setdefault(proposal.category, []).append(proposal.id)
        category_lines = [
            f"- {category}: {sorted(ids)}"
            for category, ids in sorted(by_category.items())
        ]
        attached_images = self.collect_codex_image_paths(bundle)
        image_note = (
            "A BEV/top-down scene image is attached to this turn. Use the printed "
            "proposal ids and the BEV labels together."
            if attached_images
            else "No image is attached; rely on the proposal metadata only."
        )

        schema = CodexVisualGroundingDecision.model_json_schema()
        return (
            "You are solving one NR3D visual grounding sample using the Codex "
            "Agent SDK entrypoint for 3DVLMReasoning.\n\n"
            "Rules:\n"
            "- Pick exactly one proposal id from the provided proposal pool.\n"
            "- Use -1 only if the target is absent from the proposal pool.\n"
            "- Do not use benchmark ground truth fields; none are provided here.\n"
            "- Return only JSON matching the schema. Do not write files.\n\n"
            "Task:\n"
            f"- task_type: {task.task_type.value}\n"
            f"- query: {task.user_query}\n"
            f"- scene_id: {bundle.scene_id}\n"
            f"- scene_category: {scene_category or 'unknown'}\n"
            f"- total_frames: {total_frames or 'unknown'}\n"
            f"- frame_id_range: {frame_range or 'unknown'}\n"
            f"- image_note: {image_note}\n\n"
            "Proposals by category:\n"
            + "\n".join(category_lines)
            + "\n\nProposal pool:\n"
            + "\n".join(proposal_lines)
            + "\n\nOutput JSON schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )

    def collect_codex_image_paths(self, bundle: Stage2EvidenceBundle) -> list[Path]:
        extra = bundle.extra_metadata or {}
        raw_paths = [
            bundle.bev_image_path,
            extra.get("bev_image_path") if isinstance(extra, dict) else None,
        ]
        paths: list[Path] = []
        for raw in raw_paths:
            if not raw:
                continue
            path = Path(str(raw))
            if not path.is_absolute():
                path = self.project_root / path
            if not path.exists():
                raise FileNotFoundError(f"Codex SDK BEV image does not exist: {path}")
            if path not in paths:
                paths.append(path)
        return paths

    def _run_codex_turn(
        self,
        prompt: str,
        image_paths: list[Path],
    ) -> tuple[str, dict[str, Any]]:
        try:
            from openai_codex import (
                Codex,
                LocalImageInput,
                Sandbox,
                SkillInput,
                TextInput,
            )
        except ImportError as exc:
            raise ImportError(
                "openai-codex is required for CodexSdkStage2Runtime. "
                "Install with `uv pip install openai-codex` or "
                "`uv pip install -e '.[agents]'` after syncing pyproject.toml."
            ) from exc

        if not self.skill_path.exists():
            raise FileNotFoundError(
                f"Codex SDK skill file is missing: {self.skill_path}"
            )
        if not self.codex_home.exists():
            raise FileNotFoundError(
                f"CODEX_HOME for Codex SDK does not exist: {self.codex_home}"
            )

        os.environ["CODEX_HOME"] = str(self.codex_home)
        turn_input: list[Any] = [
            SkillInput(name="nr3d-codex-sdk", path=str(self.skill_path)),
            TextInput(prompt),
        ]
        turn_input.extend(LocalImageInput(path=str(path)) for path in image_paths)
        output_schema = CodexVisualGroundingDecision.model_json_schema()

        with Codex() as codex:
            thread = codex.thread_start(
                model=self.model,
                model_provider=self.model_provider,
                sandbox=Sandbox.read_only,
                cwd=str(self.project_root),
            )
            result = thread.run(
                turn_input,
                cwd=str(self.project_root),
                output_schema=output_schema,
                sandbox=Sandbox.read_only,
            )
        if result.final_response is None:
            raise RuntimeError(
                f"Codex SDK turn completed without final_response; status={result.status}"
            )
        return result.final_response, {
            "id": result.id,
            "status": str(getattr(result.status, "value", result.status)),
            "duration_ms": result.duration_ms,
            "usage": self._dump_model(result.usage),
        }

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if fence:
            stripped = fence.group(1)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            if start < 0:
                raise
            parsed, _ = json.JSONDecoder().raw_decode(stripped[start:])
        if not isinstance(parsed, dict):
            raise ValueError(f"Codex SDK final response must be a JSON object: {text}")
        return parsed

    @staticmethod
    def _coerce_metadata(raw: dict[str, Any] | CodexTurnMetadata) -> CodexTurnMetadata:
        if isinstance(raw, CodexTurnMetadata):
            return raw
        return CodexTurnMetadata(
            id=raw.get("id"),
            status=raw.get("status"),
            duration_ms=raw.get("duration_ms"),
            usage=raw.get("usage"),
        )

    @staticmethod
    def _dump_model(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            dumped = value.model_dump(mode="json", by_alias=True, exclude_none=True)
            return dict(dumped) if isinstance(dumped, dict) else {"value": dumped}
        if isinstance(value, dict):
            return value
        return {"value": str(value)}

    @staticmethod
    def _truncate(value: str, max_chars: int) -> str:
        normalized = " ".join(str(value).split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3] + "..."


__all__ = [
    "CodexSdkStage2Runtime",
    "CodexVisualGroundingDecision",
]
