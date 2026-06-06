"""CLI wrapper for the NR3D/VG Stage-2 runtime tools.

This shares the same state file and tool implementation as
``nr3d_tools_server.py``. It is the default evidence interface for the Codex
Agent SDK path, which avoids depending on SDK-side MCP tool visibility.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from agents.mcp.nr3d_tools_server import Nr3dToolMcpServer
except ModuleNotFoundError:
    # Allow direct execution as ``python src/agents/mcp/nr3d_tools_cli.py``
    # from the project root without requiring callers to set PYTHONPATH.
    project_src = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_src))
    from agents.mcp.nr3d_tools_server import Nr3dToolMcpServer


def _json_arg(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("tool arguments must be a JSON object")
    return parsed


def _content_text(result: dict[str, Any]) -> str:
    parts = []
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(parts)


def _append_trace(
    trace_path: str | None,
    server: Nr3dToolMcpServer,
    start_index: int,
) -> None:
    if not trace_path:
        return
    observations = server.runtime.tool_trace[start_index:]
    if not observations and server.runtime.final_submission is None:
        return

    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {}
            else:
                payload = {}
            existing_trace = payload.get("tool_trace")
            if not isinstance(existing_trace, list):
                existing_trace = []
            existing_trace.extend(
                server._dump_observation(observation) for observation in observations
            )
            existing_skills = payload.get("skills_loaded")
            if not isinstance(existing_skills, list):
                existing_skills = []
            payload = {
                "tool_trace": existing_trace,
                "final_submission": (
                    server.runtime.final_submission
                    if server.runtime.final_submission is not None
                    else payload.get("final_submission")
                ),
                "skills_loaded": sorted(
                    set(existing_skills).union(server.runtime.skills_loaded)
                ),
            }
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            tmp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", required=True, help="Path to per-sample tool state JSON."
    )
    parser.add_argument("--trace", help="Path to write tool trace JSON after calls.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tools", help="Print available tool schemas as JSON.")

    call_parser = subparsers.add_parser(
        "call", help="Call one tool with JSON arguments."
    )
    call_parser.add_argument("tool_name")
    call_parser.add_argument(
        "arguments_json",
        nargs="?",
        default="{}",
        help="Tool arguments JSON object, for example '{\"proposal_id\": 6}'.",
    )

    args = parser.parse_args(argv)
    server = Nr3dToolMcpServer(state_path=args.state, trace_path=None)

    if args.command == "list-tools":
        print(json.dumps(server.list_tools(), ensure_ascii=False))
        return 0

    if args.command == "call":
        arguments = _json_arg(args.arguments_json)
        start_index = len(server.runtime.tool_trace)
        result = server.call_tool(args.tool_name, arguments)
        _append_trace(args.trace, server, start_index)
        print(
            json.dumps(
                {
                    "tool_name": args.tool_name,
                    "arguments": arguments,
                    "is_error": bool(result.get("isError")),
                    "content": result.get("content", []),
                    "content_text": _content_text(result),
                },
                ensure_ascii=False,
            )
        )
        return 1 if result.get("isError") else 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
