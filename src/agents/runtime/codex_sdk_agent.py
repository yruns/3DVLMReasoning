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
import shlex
import sys
import time
import uuid
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
DEFAULT_CODEX_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_CODEX_MODEL_PROVIDER = "modelhub_adapter"
CODEX_MODELHUB_EXTRA_HEADER_ENV = "CODEX_AGENT_MODELHUB_EXTRA_HEADER"
CODEX_MODELHUB_LOGID_ENV = "CODEX_AGENT_MODELHUB_LOGID"
MODELHUB_EXTRA_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)
MODELHUB_EXTRA_SESSION_ID_MAX_LENGTH = 128
DEFAULT_NR3D_SKILL_PATH = (
    PROJECT_ROOT / ".agents" / "skills" / "nr3d-codex-sdk" / "SKILL.md"
)
DEFAULT_NR3D_MCP_SERVER_PATH = (
    PROJECT_ROOT / "src" / "agents" / "mcp" / "nr3d_tools_server.py"
)
DEFAULT_NR3D_TOOLS_CLI_PATH = (
    PROJECT_ROOT / "src" / "agents" / "mcp" / "nr3d_tools_cli.py"
)
DEFAULT_NR3D_MCP_STATE_DIR = PROJECT_ROOT / "tmp" / "codex_sdk_mcp_state"
CODEX_TOOL_ENV_ALLOWLIST: tuple[str, ...] = (
    "STAGE1_TEXT_RETRIEVAL_MAX_CONCURRENCY",
    "STAGE1_TEXT_RETRIEVAL_LOCK_DIR",
)
DEFAULT_CODEX_PLAYBOOK_SKILL_PATHS: tuple[tuple[str, Path], ...] = (
    (
        "scene-exploration-playbook",
        PROJECT_ROOT / ".agents" / "skills" / "scene-exploration-playbook" / "SKILL.md",
    ),
    (
        "vg-grounding-playbook",
        PROJECT_ROOT / ".agents" / "skills" / "vg-grounding-playbook" / "SKILL.md",
    ),
    (
        "vg-spatial-disambiguation",
        PROJECT_ROOT / ".agents" / "skills" / "vg-spatial-disambiguation" / "SKILL.md",
    ),
)


class CodexVisualGroundingDecision(BaseModel):
    """Minimal JSON contract requested from Codex SDK."""

    model_config = {"extra": "forbid"}

    proposal_id: int = Field(
        description="Selected proposal id from the provided proposal pool; -1 if absent."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    uncertainties: list[str]
    cited_frame_indices: list[int]


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
        enable_mcp_tools: bool | None = None,
        mcp_server_path: str | Path | None = None,
        mcp_state_dir: str | Path | None = None,
        mcp_python: str | Path | None = None,
        enable_cli_tools: bool | None = None,
        tools_cli_path: str | Path | None = None,
        enable_prefix_cache: bool | None = None,
        prefix_cache_session_id: str | None = None,
    ) -> None:
        self.config = config or Stage2DeepAgentConfig()
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
        self.enable_mcp_tools = (
            self._env_bool("CODEX_AGENT_ENABLE_MCP_TOOLS", default=False)
            if enable_mcp_tools is None
            else bool(enable_mcp_tools)
        )
        self.mcp_server_path = (
            Path(mcp_server_path) if mcp_server_path else DEFAULT_NR3D_MCP_SERVER_PATH
        )
        self.mcp_state_dir = (
            Path(mcp_state_dir) if mcp_state_dir else DEFAULT_NR3D_MCP_STATE_DIR
        )
        self.mcp_python = str(
            mcp_python or os.environ.get("CODEX_AGENT_MCP_PYTHON") or sys.executable
        )
        self.enable_cli_tools = (
            self._env_bool("CODEX_AGENT_ENABLE_CLI_TOOLS", default=True)
            if enable_cli_tools is None
            else bool(enable_cli_tools)
        )
        self.tools_cli_path = (
            Path(tools_cli_path) if tools_cli_path else DEFAULT_NR3D_TOOLS_CLI_PATH
        )
        self.enable_prefix_cache = (
            self._env_bool("CODEX_AGENT_ENABLE_PREFIX_CACHE", default=True)
            if enable_prefix_cache is None
            else bool(enable_prefix_cache)
        )
        self.prefix_cache_session_id = self._safe_modelhub_session_id(
            prefix_cache_session_id
            or os.environ.get("CODEX_AGENT_PREFIX_CACHE_SESSION_ID")
            or self.config.session_id
        )

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
        image_paths = self.collect_codex_image_paths(bundle)
        mcp_state_path: Path | None = None
        mcp_trace_path: Path | None = None
        if self.enable_mcp_tools or self.enable_cli_tools:
            mcp_state_path, mcp_trace_path = self.write_mcp_state(task, bundle)
        cli_trace_path = (
            self.cli_trace_path_for(mcp_trace_path)
            if self.enable_cli_tools and mcp_trace_path is not None
            else None
        )
        prompt = self.build_decision_prompt(
            task,
            bundle,
            tool_state_path=mcp_state_path,
            tool_trace_path=cli_trace_path or mcp_trace_path,
        )
        codex_mcp_trace_path = mcp_trace_path if self.enable_mcp_tools else None
        response_text, metadata_raw = self._run_codex_turn(
            prompt,
            image_paths,
            mcp_state_path=mcp_state_path,
            mcp_trace_path=codex_mcp_trace_path,
        )
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
        mcp_trace, mcp_trace_meta = self.load_tool_traces(
            mcp_trace_path,
            cli_trace_path,
        )
        trace = [
            *mcp_trace,
            Stage2ToolObservation(
                tool_name="codex_sdk_turn",
                tool_input={
                    "model": self.model,
                    "model_provider": self.model_provider,
                    "codex_home": str(self.codex_home),
                    "skill_path": str(self.skill_path),
                    "skill_paths": [str(path) for _, path in self.codex_skill_specs()],
                    "attached_images": [str(path) for path in image_paths],
                    "mcp_tools_enabled": self.enable_mcp_tools,
                    "mcp_server_path": (
                        str(self.mcp_server_path) if self.enable_mcp_tools else None
                    ),
                    "mcp_state_path": str(mcp_state_path) if mcp_state_path else None,
                    "mcp_trace_path": (
                        str(mcp_trace_path) if self.enable_mcp_tools else None
                    ),
                    "cli_tools_enabled": self.enable_cli_tools,
                    "tools_cli_path": (
                        str(self.tools_cli_path) if self.enable_cli_tools else None
                    ),
                    "cli_trace_path": str(cli_trace_path) if cli_trace_path else None,
                    "prefix_cache_enabled": self.enable_prefix_cache,
                    "prefix_cache_session_id": (
                        self.prefix_cache_session_id
                        if self.enable_prefix_cache
                        else None
                    ),
                },
                response_text=response_text,
                image_metadata=[
                    {"image_path": str(path), "kind": "bev"} for path in image_paths
                ],
            ),
        ]
        return Stage2AgentResult(
            task=task,
            result=structured,
            tool_trace=trace,
            final_bundle=bundle,
            raw_state={
                "runtime": "codex_sdk",
                "codex_turn": metadata.as_dict(),
                "mcp_tools_enabled": self.enable_mcp_tools,
                "cli_tools_enabled": self.enable_cli_tools,
                "mcp_trace": mcp_trace_meta,
                "prefix_cache_enabled": self.enable_prefix_cache,
                "prefix_cache_session_id": (
                    self.prefix_cache_session_id if self.enable_prefix_cache else None
                ),
            },
        )

    def build_decision_prompt(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
        *,
        tool_state_path: Path | None = None,
        tool_trace_path: Path | None = None,
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
        if self.enable_mcp_tools:
            text_tool_note = (
                "`select_by_text` (lazily initializes the Stage-1 "
                "KeyframeSelector on first call), "
                if self.config.enable_stage1_text_retrieval
                else ""
            )
            tool_note = (
                "MCP tools are mounted from the `nr3d_tools` server. The relevant "
                "playbook skills are already attached to this turn, so call evidence "
                "tools directly before finalizing. Useful tools include "
                "`list_scene_proposals`, `inspect_proposal`, "
                "`compare_proposals_spatial`, `list_frame_proposals`, "
                f"{text_tool_note}`select_by_proposal`, `select_by_region`, "
                "and `mark_frame_with_bbox` when visual verification is needed."
            )
        elif self.enable_cli_tools:
            tool_note = "Use CLI evidence tools only for this SDK run."
        else:
            tool_note = "No external evidence tools are enabled for this turn."
        if self.enable_cli_tools:
            if tool_state_path is None or tool_trace_path is None:
                cli_note = (
                    "The CLI evidence tools are enabled but state paths were "
                    "not prepared; do not attempt CLI calls."
                )
            else:
                cli_note = self.build_cli_tool_note(tool_state_path, tool_trace_path)
            tool_note = f"{tool_note}\n- {cli_note}"

        schema = CodexVisualGroundingDecision.model_json_schema()
        return (
            "You are solving one NR3D visual grounding sample using the Codex "
            "Agent SDK entrypoint for 3DVLMReasoning.\n\n"
            "Rules:\n"
            "- Pick exactly one proposal id from the provided proposal pool.\n"
            "- Use -1 only if the target is absent from the proposal pool.\n"
            "- Do not use benchmark ground truth fields; none are provided here.\n"
            "- Return only JSON matching the schema. Do not write files.\n\n"
            "Tool access:\n"
            f"- {tool_note}\n\n"
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
        *,
        mcp_state_path: Path | None = None,
        mcp_trace_path: Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            from openai_codex import (
                Codex,
                CodexConfig,
                LocalImageInput,
                SkillInput,
                TextInput,
            )
        except ImportError as exc:
            raise ImportError(
                "openai-codex is required for CodexSdkStage2Runtime. "
                "Install with `uv pip install openai-codex` or "
                "`uv pip install -e '.[agents]'` after syncing pyproject.toml."
            ) from exc

        skill_specs = self.codex_skill_specs()
        for skill_name, skill_path in skill_specs:
            if not skill_path.exists():
                raise FileNotFoundError(
                    f"Codex SDK skill file is missing: {skill_name} at {skill_path}"
                )
        if not self.codex_home.exists():
            raise FileNotFoundError(
                f"CODEX_HOME for Codex SDK does not exist: {self.codex_home}"
            )
        if self.enable_mcp_tools:
            if mcp_state_path is None or mcp_trace_path is None:
                raise ValueError("MCP tools enabled but state/trace paths are missing")
            config_overrides = self.build_codex_app_server_config_overrides(
                mcp_state_path=mcp_state_path,
                mcp_trace_path=mcp_trace_path,
            )
        else:
            config_overrides = self.build_codex_prefix_cache_config_overrides()

        os.environ["CODEX_HOME"] = str(self.codex_home)
        turn_input: list[Any] = [
            SkillInput(name=skill_name, path=str(skill_path))
            for skill_name, skill_path in skill_specs
        ]
        turn_input.append(TextInput(prompt))
        turn_input.extend(LocalImageInput(path=str(path)) for path in image_paths)
        output_schema = CodexVisualGroundingDecision.model_json_schema()

        with Codex(
            config=CodexConfig(
                config_overrides=config_overrides,
                cwd=str(self.project_root),
                env=self.build_codex_app_server_env(),
            )
        ) as codex:
            thread_kwargs: dict[str, Any] = {
                "model": self.model,
                "model_provider": self.model_provider,
                "sandbox": self.codex_sandbox(),
                "cwd": str(self.project_root),
            }
            thread = codex.thread_start(**thread_kwargs)
            result = thread.run(
                turn_input,
                cwd=str(self.project_root),
                output_schema=output_schema,
                sandbox=self.codex_sandbox(),
            )
            results = [result]
            if not self._is_valid_decision_json(result.final_response or ""):
                result = thread.run(
                    [
                        TextInput(
                            self.build_json_finalization_prompt(
                                previous_response=result.final_response
                            )
                        )
                    ],
                    cwd=str(self.project_root),
                    output_schema=output_schema,
                    sandbox=self.codex_sandbox(),
                )
                results.append(result)
        if result.final_response is None:
            raise RuntimeError(
                f"Codex SDK turn completed without final_response; status={result.status}"
            )
        return result.final_response, {
            "id": result.id,
            "status": str(getattr(result.status, "value", result.status)),
            "duration_ms": result.duration_ms,
            "usage": self._dump_model(result.usage),
            "attempts": [
                {
                    "id": item.id,
                    "status": str(getattr(item.status, "value", item.status)),
                    "duration_ms": item.duration_ms,
                    "has_json_response": self._is_valid_decision_json(
                        item.final_response or ""
                    ),
                }
                for item in results
            ],
        }

    def build_json_finalization_prompt(
        self,
        *,
        previous_response: str | None = None,
    ) -> str:
        preview = self._truncate(previous_response or "", 600)
        return (
            "The previous turn did not return the required structured final "
            "answer. Use the evidence, tool outputs, and images already present "
            "in this thread. Return only one JSON object matching the requested "
            "schema, with fields `proposal_id`, `confidence`, `summary`, "
            "`uncertainties`, and `cited_frame_indices`. Do not include markdown, "
            "status updates, prose outside JSON, or tool commands."
            + (f"\nPrevious non-JSON response: {preview}" if preview else "")
        )

    def _is_valid_decision_json(self, text: str) -> bool:
        try:
            CodexVisualGroundingDecision.model_validate(self._parse_json_object(text))
        except Exception:
            return False
        return True

    def codex_skill_specs(self) -> list[tuple[str, Path]]:
        specs = [("nr3d-codex-sdk", self.skill_path)]
        if self.enable_mcp_tools or self.enable_cli_tools:
            specs.extend(DEFAULT_CODEX_PLAYBOOK_SKILL_PATHS)
        return specs

    def write_mcp_state(
        self,
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
    ) -> tuple[Path, Path]:
        if not self.mcp_server_path.exists():
            raise FileNotFoundError(
                f"NR3D MCP server file is missing: {self.mcp_server_path}"
            )
        self.mcp_state_dir.mkdir(parents=True, exist_ok=True)
        token = f"{self._safe_token(bundle.scene_id)}_{uuid.uuid4().hex}"
        state_path = self.mcp_state_dir / f"{token}.state.json"
        trace_path = self.mcp_state_dir / f"{token}.trace.json"
        payload = {
            "task": task.model_dump(mode="json"),
            "bundle": bundle.model_dump(mode="json"),
            "config": self._mcp_config_payload(),
        }
        tool_env = self._tool_env_payload()
        if tool_env:
            payload["tool_env"] = tool_env
        state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return state_path, trace_path

    @staticmethod
    def _tool_env_payload() -> dict[str, str]:
        return {
            name: value
            for name in CODEX_TOOL_ENV_ALLOWLIST
            if (value := os.environ.get(name))
        }

    @staticmethod
    def cli_trace_path_for(mcp_trace_path: Path) -> Path:
        name = mcp_trace_path.name
        if name.endswith(".trace.json"):
            return mcp_trace_path.with_name(
                f"{name.removesuffix('.trace.json')}.cli.trace.json"
            )
        return mcp_trace_path.with_name(f"{name}.cli.trace.json")

    def build_cli_tool_note(
        self,
        tool_state_path: Path,
        tool_trace_path: Path,
    ) -> str:
        cli = str(self.tools_cli_path)
        python = self.mcp_python
        state = str(tool_state_path)
        trace = str(tool_trace_path)
        return (
            "CLI tools are the evidence interface for this SDK run. They share "
            "the same NR3D tool implementations as the MCP server and write only "
            "the provided trace file. Required CLI evidence policy: inspect the "
            "candidate you plan to select; when there is more than one plausible "
            "same-category candidate, use `select_by_proposal` to fetch frames "
            "for the shortlist and then `mark_frame_with_bbox` on a selector "
            "returned frame before final JSON; for spatial, ordinal, nearest, "
            "left/right, above/below, or anchor-based language, call "
            "`compare_proposals_spatial` or `compare_candidates_to_anchors`; "
            "`select_by_text` lazily initializes the Stage-1 KeyframeSelector "
            "and is required when the text query itself is the best way to form "
            "the shortlist. "
            "Examples:\n"
            f"  {python} {cli} --state {self._shell_quote(state)} "
            f"--trace {self._shell_quote(trace)} list-tools\n"
            f"  {python} {cli} --state {self._shell_quote(state)} "
            f"--trace {self._shell_quote(trace)} call inspect_proposal "
            "'{\"proposal_id\": 6}'\n"
            f"  {python} {cli} --state {self._shell_quote(state)} "
            f"--trace {self._shell_quote(trace)} call select_by_proposal "
            '\'{"proposal_ids": [6], "k": 3}\'\n'
            f"  {python} {cli} --state {self._shell_quote(state)} "
            f"--trace {self._shell_quote(trace)} call mark_frame_with_bbox "
            '\'{"frame_id": 10, "ids": [6]}\'\n'
            f"  {python} {cli} --state {self._shell_quote(state)} "
            f"--trace {self._shell_quote(trace)} call compare_proposals_spatial "
            '\'{"candidate_ids": [6, 7], "anchor_id": 3, "relation": "closest_to"}\'\n'
            f"  {python} {cli} --state {self._shell_quote(state)} "
            f"--trace {self._shell_quote(trace)} call select_by_text "
            '\'{"query": "the target object", "k": 3}\''
        )

    def codex_sandbox(self) -> Any:
        from openai_codex import Sandbox

        requested = os.environ.get("CODEX_AGENT_SANDBOX", "").strip().lower()
        if requested:
            normalized = requested.replace("-", "_")
            if normalized in {"read_only", "readonly"}:
                return Sandbox.read_only
            if normalized in {"workspace_write", "workspace"}:
                return Sandbox.workspace_write
            if normalized in {"full_access", "danger_full_access"}:
                return Sandbox.full_access
            raise ValueError(
                "CODEX_AGENT_SANDBOX must be one of read_only, "
                "workspace_write, or full_access"
            )

        if self.enable_cli_tools or self.enable_mcp_tools:
            return Sandbox.full_access
        return Sandbox.read_only

    def codex_sandbox_value(self) -> str:
        return str(self.codex_sandbox().value)

    def build_codex_app_server_config_overrides(
        self,
        *,
        mcp_state_path: Path,
        mcp_trace_path: Path,
    ) -> tuple[str, ...]:
        server_key = "mcp_servers.nr3d_tools"
        args = [
            str(self.mcp_server_path),
            "--state",
            str(mcp_state_path),
            "--trace",
            str(mcp_trace_path),
        ]
        overrides = [
            *self.build_codex_prefix_cache_config_overrides(),
            f"{server_key}.command={self._toml_literal(self.mcp_python)}",
            f"{server_key}.cwd={self._toml_literal(str(self.project_root))}",
            (
                f"{server_key}.env.PYTHONPATH="
                f"{self._toml_literal(str(self.project_root / 'src'))}"
            ),
            f"{server_key}.args={self._toml_literal(args)}",
            f"{server_key}.startup_timeout_sec=30",
            f"{server_key}.tool_timeout_sec=60",
            f"{server_key}.enabled=true",
            f"{server_key}.required=true",
            f'{server_key}.default_tools_approval_mode="approve"',
        ]
        return tuple(overrides)

    def build_codex_prefix_cache_config_overrides(self) -> tuple[str, ...]:
        if not self.enable_prefix_cache:
            return ()
        provider_key = f"model_providers.{self._toml_key_part(self.model_provider)}"
        return (
            (
                f"{provider_key}.env_http_headers.extra="
                f"{self._toml_literal(CODEX_MODELHUB_EXTRA_HEADER_ENV)}"
            ),
            (
                f"{provider_key}.env_http_headers.X-TT-LOGID="
                f"{self._toml_literal(CODEX_MODELHUB_LOGID_ENV)}"
            ),
        )

    def build_codex_app_server_env(self) -> dict[str, str]:
        if not self.enable_prefix_cache:
            return {}
        extra = {
            "session_id": self.prefix_cache_session_id,
            "source": "codex_agent_sdk",
        }
        return {
            CODEX_MODELHUB_EXTRA_HEADER_ENV: json.dumps(
                extra,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            CODEX_MODELHUB_LOGID_ENV: (
                f"codexsdk_{self.prefix_cache_session_id}_{int(time.time() * 1000)}"
            ),
        }

    def load_tool_traces(
        self,
        mcp_trace_path: Path | None,
        cli_trace_path: Path | None = None,
    ) -> tuple[list[Stage2ToolObservation], dict[str, Any] | None]:
        observations: list[Stage2ToolObservation] = []
        sources: list[dict[str, Any]] = []
        for kind, trace_path in (
            ("mcp", mcp_trace_path),
            ("cli", cli_trace_path),
        ):
            loaded, meta = self.load_mcp_trace(trace_path, kind=kind)
            observations.extend(loaded)
            if meta is not None:
                sources.append(meta)
        if not sources:
            return observations, None
        return observations, {"sources": sources, "tool_count": len(observations)}

    def load_mcp_trace(
        self,
        trace_path: Path | None,
        *,
        kind: str = "mcp",
    ) -> tuple[list[Stage2ToolObservation], dict[str, Any] | None]:
        if trace_path is None or not trace_path.exists():
            return [], None
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"NR3D MCP trace must be a JSON object: {trace_path}")
        observations = [
            Stage2ToolObservation.model_validate(item)
            for item in payload.get("tool_trace", [])
        ]
        return observations, {
            "kind": kind,
            "trace_path": str(trace_path),
            "tool_count": len(observations),
            "skills_loaded": payload.get("skills_loaded", []),
            "final_submission": payload.get("final_submission"),
        }

    def _mcp_config_payload(self) -> dict[str, Any]:
        return {
            "enable_stage1_text_retrieval": self.config.enable_stage1_text_retrieval,
            "force_stage1_text_retrieval_to_error": (
                self.config.force_stage1_text_retrieval_to_error
            ),
            "enable_chassis_tools": self.config.enable_chassis_tools,
            "vg_backend": self.config.vg_backend,
            "use_clip_visible_aug": self.config.use_clip_visible_aug,
            "clip_visible_tau": self.config.clip_visible_tau,
            "clip_visible_k_aug": self.config.clip_visible_k_aug,
            "clip_visible_backbone": self.config.clip_visible_backbone,
            "clip_visible_cache_dir": self.config.clip_visible_cache_dir,
            "use_tool_answer_disagreement_gate": (
                self.config.use_tool_answer_disagreement_gate
            ),
            "tadg_window": self.config.tadg_window,
            "tadg_max_repeats": self.config.tadg_max_repeats,
            "tadg_override_min_chars": self.config.tadg_override_min_chars,
            "use_no_match_candidate_guard": self.config.use_no_match_candidate_guard,
            "no_match_guard_max_repeats": self.config.no_match_guard_max_repeats,
            "no_match_guard_max_viewed": self.config.no_match_guard_max_viewed,
            "use_evidence_frame_guard": self.config.use_evidence_frame_guard,
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

    @staticmethod
    def _safe_token(value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return token or "sample"

    @staticmethod
    def _toml_literal(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _toml_key_part(value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]+", value):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _shell_quote(value: str) -> str:
        return shlex.quote(value)

    @staticmethod
    def _safe_modelhub_session_id(value: Any) -> str:
        raw = str(value or "codex_agent_sdk").strip()
        safe = "".join(ch if ch in MODELHUB_EXTRA_ALLOWED_CHARS else "_" for ch in raw)
        safe = safe.strip("._:-")[:MODELHUB_EXTRA_SESSION_ID_MAX_LENGTH]
        return safe or "codex_agent_sdk"

    @staticmethod
    def _env_bool(name: str, *, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}


__all__ = [
    "CodexSdkStage2Runtime",
    "CodexVisualGroundingDecision",
]
