"""MCP wrapper for the NR3D/VG Stage-2 runtime tools.

The server is intentionally small and stdio-only so Codex Agent SDK can spawn
one isolated tool process per sample. The actual tool bodies remain the
existing LangChain ``BaseTool`` implementations bound to a restored
``Stage2RuntimeState``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from agents.core.agent_config import Stage2TaskType
from agents.core.response_schema import Stage2ToolObservation
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.models import Stage2DeepAgentConfig
from agents.packs import ensure_default_packs_registered
from agents.packs.vg_embodiedscan.ctx import build_ctx_from_bundle
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime

CODEX_PRELOADED_SKILLS: tuple[str, ...] = (
    "scene-exploration-playbook",
    "vg-grounding-playbook",
    "vg-spatial-disambiguation",
)
CODEX_HIDDEN_CHASSIS_TOOLS: frozenset[str] = frozenset(
    {"list_skills", "load_skill", "submit_final"}
)


class Nr3dToolMcpServer:
    """Expose existing NR3D/VG runtime tools through MCP JSON-RPC."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        trace_path: str | Path | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.trace_path = Path(trace_path) if trace_path else None
        self.task, self.bundle, self.config = self._load_state(self.state_path)
        self.runtime = self._build_runtime_state(self.task, self.bundle, self.config)
        self.tools = self._build_tools(self.runtime, self.config)

    def list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": name,
                    "description": tool.description or "",
                    "inputSchema": self._tool_input_schema(tool),
                }
                for name, tool in sorted(self.tools.items())
            ]
        }

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            return self._tool_result(f"unknown tool: {name}", is_error=True)
        try:
            result = tool.invoke(arguments)
        except Exception as exc:  # MCP must report tool errors without killing stdio.
            self._dump_trace()
            return self._tool_result(
                f"ERROR: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        self._dump_trace()
        return self._tool_result(self._stringify_tool_result(result), is_error=False)

    @staticmethod
    def _load_state(
        state_path: Path,
    ) -> tuple[Stage2TaskSpec, Stage2EvidenceBundle, Stage2DeepAgentConfig]:
        if not state_path.exists():
            raise FileNotFoundError(f"NR3D MCP state file missing: {state_path}")
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"NR3D MCP state must be a JSON object: {state_path}")

        task = Stage2TaskSpec.model_validate(payload.get("task"))
        bundle = Stage2EvidenceBundle.model_validate(payload.get("bundle"))
        config_payload = payload.get("config") or {}
        if not isinstance(config_payload, dict):
            raise ValueError("NR3D MCP state field 'config' must be an object")
        config = Stage2DeepAgentConfig.model_validate(config_payload)
        return task, bundle, config

    @staticmethod
    def _build_runtime_state(
        task: Stage2TaskSpec,
        bundle: Stage2EvidenceBundle,
        config: Stage2DeepAgentConfig,
    ) -> Stage2RuntimeState:
        if task.task_type != Stage2TaskType.VISUAL_GROUNDING:
            raise NotImplementedError(
                "NR3D MCP tools currently support visual_grounding only"
            )
        runtime = Stage2RuntimeState(bundle=bundle.model_copy(deep=True))
        runtime.task_type = task.task_type
        runtime.task_ctx = build_ctx_from_bundle(runtime.bundle)
        if config.enable_stage1_text_retrieval:
            runtime.text_frame_selector_factory = (
                Nr3dToolMcpServer._build_text_frame_selector_factory(runtime.bundle)
            )
        runtime.skills_loaded.update(CODEX_PRELOADED_SKILLS)
        return runtime

    @staticmethod
    def _build_text_frame_selector_factory(
        bundle: Stage2EvidenceBundle,
    ) -> Callable[[], Any] | None:
        extra = bundle.extra_metadata or {}
        selector_meta = extra.get("stage1_text_selector")
        if selector_meta is None:
            return None
        if not isinstance(selector_meta, dict):
            raise ValueError(
                "bundle.extra_metadata.stage1_text_selector must be an object"
            )

        scene_id = str(selector_meta.get("scene_id") or bundle.scene_id)
        conceptgraph_raw = selector_meta.get("conceptgraph_root")
        if conceptgraph_raw:
            conceptgraph_root = Path(str(conceptgraph_raw))
        else:
            phase8_raw = selector_meta.get("phase8_data_root")
            if not phase8_raw:
                raise ValueError(
                    "stage1_text_selector requires conceptgraph_root or "
                    "phase8_data_root"
                )
            conceptgraph_root = Path(str(phase8_raw)) / scene_id / "conceptgraph"
        llm_model = str(selector_meta.get("llm_model") or "gemini-2.5-pro")

        def _factory() -> Any:
            enriched = conceptgraph_root / "enriched_objects.json"
            if not enriched.exists():
                raise FileNotFoundError(
                    f"Stage-1 text retrieval requires enriched object metadata: "
                    f"{enriched}"
                )
            from query_scene import KeyframeSelector

            return KeyframeSelector.from_scene_path(
                str(conceptgraph_root),
                stride=1,
                llm_model=llm_model,
                prefer_lightweight_pcd=True,
                ensure_lightweight_pcd=True,
            )

        return _factory

    @staticmethod
    def _build_tools(
        runtime_state: Stage2RuntimeState,
        config: Stage2DeepAgentConfig,
    ) -> dict[str, BaseTool]:
        ensure_default_packs_registered()
        agent_runtime = DeepAgentsStage2Runtime(
            config=config,
            text_frame_selector=None,
            text_frame_selector_factory=runtime_state.text_frame_selector_factory,
        )
        tools = agent_runtime.build_runtime_tools(runtime_state)
        return {
            tool.name: tool
            for tool in tools
            if tool.name not in CODEX_HIDDEN_CHASSIS_TOOLS
        }

    @staticmethod
    def _tool_input_schema(tool: BaseTool) -> dict[str, Any]:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()
            if isinstance(schema, dict):
                schema.setdefault("type", "object")
                return schema
        args = getattr(tool, "args", None)
        if isinstance(args, dict):
            return {
                "type": "object",
                "properties": args,
                "additionalProperties": False,
            }
        return {"type": "object", "properties": {}, "additionalProperties": False}

    def _dump_trace(self) -> None:
        if self.trace_path is None:
            return
        payload = {
            "tool_trace": [
                self._dump_observation(observation)
                for observation in self.runtime.tool_trace
            ],
            "final_submission": self.runtime.final_submission,
            "skills_loaded": sorted(self.runtime.skills_loaded),
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _dump_observation(observation: Stage2ToolObservation) -> dict[str, Any]:
        return observation.model_dump(mode="json", exclude_none=True)

    @staticmethod
    def _stringify_tool_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    @staticmethod
    def _tool_result(text: str, *, is_error: bool) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        }


def _read_message() -> dict[str, Any] | None:
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        stripped = line.strip()
        if not stripped:
            continue
        decoded = stripped.decode("utf-8")
        key, sep, value = decoded.partition(":")
        if sep and key.lower() == "content-length":
            headers: dict[str, str] = {key.lower(): value.strip()}
            while True:
                header_line = sys.stdin.buffer.readline()
                if not header_line:
                    return None
                header_decoded = header_line.decode("utf-8").strip()
                if not header_decoded:
                    break
                header_key, _, header_value = header_decoded.partition(":")
                headers[header_key.lower()] = header_value.strip()

            length = int(headers.get("content-length", "0"))
            if length <= 0:
                return None
            return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))
        return json.loads(decoded)


def _write_message(payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _handle(
    server: Nr3dToolMcpServer,
    message: dict[str, Any],
) -> dict[str, Any] | None:
    method = message.get("method")
    message_id = message.get("id")

    if method == "initialize":
        params = (
            message.get("params") if isinstance(message.get("params"), dict) else {}
        )
        return _result(
            message_id,
            {
                "protocolVersion": params.get("protocolVersion") or "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "nr3d-stage2-tools", "version": "0.1.0"},
            },
        )

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return _result(message_id, server.list_tools())

    if method == "tools/call":
        params = (
            message.get("params") if isinstance(message.get("params"), dict) else {}
        )
        name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(name, str):
            return _error(message_id, -32602, "tools/call requires string field 'name'")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _error(
                message_id,
                -32602,
                "tools/call field 'arguments' must be an object",
            )
        return _result(message_id, server.call_tool(name, arguments))

    if message_id is None:
        return None
    return _error(message_id, -32601, f"unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", required=True, help="Path to per-sample MCP state JSON."
    )
    parser.add_argument("--trace", help="Path to write tool trace JSON after calls.")
    args = parser.parse_args()

    server = Nr3dToolMcpServer(state_path=args.state, trace_path=args.trace)
    while True:
        message = _read_message()
        if message is None:
            break
        response = _handle(server, message)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    main()


__all__ = ["Nr3dToolMcpServer"]
