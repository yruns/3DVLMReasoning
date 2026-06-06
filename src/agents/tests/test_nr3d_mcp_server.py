from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.mcp.nr3d_tools_server import Nr3dToolMcpServer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _bundle(tmp_path) -> Stage2EvidenceBundle:
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_rgb = raw / "000100-rgb.png"
    Image.new("RGB", (400, 400), color=(220, 220, 220)).save(raw_rgb)
    bev = tmp_path / "bev.png"
    Image.new("RGB", (400, 400), color=(245, 245, 245)).save(bev)
    phase8_root = tmp_path / "phase8"
    conceptgraph_root = phase8_root / "scene0001_00" / "conceptgraph"
    conceptgraph_root.mkdir(parents=True)
    (conceptgraph_root / "enriched_objects.json").write_text("{}", encoding="utf-8")
    return Stage2EvidenceBundle(
        scene_id="scene0001_00",
        stage1_query="the square table by the door",
        extra_metadata={
            "stage1_text_selector": {
                "scene_id": "scene0001_00",
                "phase8_data_root": str(phase8_root),
                "conceptgraph_root": str(conceptgraph_root),
                "llm_model": "fake-model",
            },
            "scene_catalog": {
                "scene_id": "scene0001_00",
                "scene_category": "room",
                "total_frames": 12,
                "frame_id_range": [0, 11],
                "valid_frame_ids": [10],
                "bev_image_path": str(bev),
                "proposals": [
                    {
                        "proposal_id": 6,
                        "category": "table",
                        "position_3d": [1.0, 2.0, 0.4],
                        "bbox_3d_9dof": [1, 2, 0.4, 1.2, 1.1, 0.8, 0, 0, 0],
                        "source": "gt",
                        "enriched_category": "square table",
                        "compact_note": "Square table near the door.",
                        "frame_views": {
                            "10": {
                                "frame_id": 10,
                                "bbox_2d": [100, 120, 300, 320],
                                "raw_rgb_path": str(raw_rgb),
                            }
                        },
                    },
                    {
                        "proposal_id": 7,
                        "category": "table",
                        "position_3d": [3.0, 2.0, 0.4],
                        "bbox_3d_9dof": [3, 2, 0.4, 1.2, 1.1, 0.8, 0, 0, 0],
                        "source": "gt",
                        "enriched_category": "rectangular table",
                        "compact_note": "Another table farther from the door.",
                        "frame_views": {
                            "10": {
                                "frame_id": 10,
                                "bbox_2d": [220, 120, 360, 320],
                                "raw_rgb_path": str(raw_rgb),
                            }
                        },
                    },
                    {
                        "proposal_id": 3,
                        "category": "door",
                        "position_3d": [0.0, 2.0, 0.8],
                        "bbox_3d_9dof": [0, 2, 0.8, 0.5, 1.0, 1.8, 0, 0, 0],
                        "source": "gt",
                        "enriched_category": "door",
                        "compact_note": "Door used as a spatial anchor.",
                        "frame_views": {
                            "10": {
                                "frame_id": 10,
                                "bbox_2d": [20, 100, 80, 340],
                                "raw_rgb_path": str(raw_rgb),
                            }
                        },
                    },
                ],
            },
            "vg_proposal_pool": {
                "source": "gt",
                "annotated_image_dir": str(annotated),
                "frame_index": {"10": [3, 6, 7]},
                "proposal_index": {"3": [10], "6": [10], "7": [10]},
                "proposals": [
                    {
                        "id": 6,
                        "bbox_3d_9dof": [1, 2, 0.4, 1.2, 1.1, 0.8, 0, 0, 0],
                        "category": "table",
                        "score": 1.0,
                        "enriched_category": "square table",
                        "compact_note": "Square table near the door.",
                        "frame_views": {
                            "10": {
                                "frame_id": 10,
                                "bbox_2d": [100, 120, 300, 320],
                                "raw_rgb_path": "raw/000100-rgb.png",
                            }
                        },
                    },
                    {
                        "id": 7,
                        "bbox_3d_9dof": [3, 2, 0.4, 1.2, 1.1, 0.8, 0, 0, 0],
                        "category": "table",
                        "score": 0.9,
                        "enriched_category": "rectangular table",
                        "compact_note": "Another table farther from the door.",
                        "frame_views": {
                            "10": {
                                "frame_id": 10,
                                "bbox_2d": [220, 120, 360, 320],
                                "raw_rgb_path": str(raw_rgb),
                            }
                        },
                    },
                    {
                        "id": 3,
                        "bbox_3d_9dof": [0, 2, 0.8, 0.5, 1.0, 1.8, 0, 0, 0],
                        "category": "door",
                        "score": 1.0,
                        "enriched_category": "door",
                        "compact_note": "Door used as a spatial anchor.",
                        "frame_views": {
                            "10": {
                                "frame_id": 10,
                                "bbox_2d": [20, 100, 80, 340],
                                "raw_rgb_path": str(raw_rgb),
                            }
                        },
                    },
                ],
            },
        },
    )


def _state_path(tmp_path) -> tuple[Stage2TaskSpec, Stage2EvidenceBundle, object]:
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )
    bundle = _bundle(tmp_path)
    state_path = tmp_path / "mcp_state.json"
    state_path.write_text(
        json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "bundle": bundle.model_dump(mode="json"),
                "config": {"enable_stage1_text_retrieval": True},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return task, bundle, state_path


def _run_cli(state_path: Path, trace_path: Path, *args: str) -> dict:
    cli_path = PROJECT_ROOT / "src" / "agents" / "mcp" / "nr3d_tools_cli.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--state",
            str(state_path),
            "--trace",
            str(trace_path),
            *args,
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        cwd=PROJECT_ROOT,
        timeout=20,
    )
    assert proc.returncode == 0, {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    return json.loads(proc.stdout)


def test_nr3d_tools_cli_invokes_core_evidence_tools_and_writes_trace(
    tmp_path,
) -> None:
    _, _, state_path = _state_path(tmp_path)
    trace_path = tmp_path / "cli_trace.json"

    listed = _run_cli(state_path, trace_path, "list-tools")
    tool_names = {tool["name"] for tool in listed["tools"]}
    expected = {
        "view_bev",
        "list_scene_proposals",
        "inspect_proposal",
        "list_frame_proposals",
        "select_by_proposal",
        "select_by_region",
        "select_by_frame_neighbor",
        "select_by_coverage",
        "mark_frame_with_bbox",
        "compare_proposals_spatial",
        "compare_candidates_to_anchors",
    }
    assert expected <= tool_names
    assert "select_by_text" in tool_names
    assert "list_skills" not in tool_names
    assert "load_skill" not in tool_names
    assert "submit_final" not in tool_names

    calls = [
        ("view_bev", {}),
        ("list_scene_proposals", {"category": "table"}),
        ("inspect_proposal", {"proposal_id": 6}),
        ("list_frame_proposals", {"frame_id": 10}),
        ("select_by_proposal", {"proposal_ids": [6, 7], "k": 2}),
        (
            "select_by_region",
            {"region": [0, 1, 0, 4, 3, 2], "region_type": "bbox_3d", "k": 2},
        ),
        (
            "select_by_frame_neighbor",
            {"anchor_frame_id": 10, "mode": "temporal", "k": 1},
        ),
        ("select_by_coverage", {"method": "obj_iou", "k": 1}),
        ("mark_frame_with_bbox", {"frame_id": 10, "ids": [3, 6, 7]}),
        (
            "compare_proposals_spatial",
            {"candidate_ids": [6, 7], "anchor_id": 3, "relation": "closest_to"},
        ),
        (
            "compare_candidates_to_anchors",
            {"candidate_ids": [6, 7], "anchor_ids": [3], "relation": "closest_to"},
        ),
    ]
    for tool_name, arguments in calls:
        result = _run_cli(
            state_path,
            trace_path,
            "call",
            tool_name,
            json.dumps(arguments),
        )
        assert result["tool_name"] == tool_name
        assert result["arguments"] == arguments
        assert result["is_error"] is False
        assert result["content_text"]

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert [item["tool_name"] for item in trace["tool_trace"]] == [
        tool_name for tool_name, _ in calls
    ]
    mark_record = next(
        item
        for item in trace["tool_trace"]
        if item["tool_name"] == "mark_frame_with_bbox"
    )
    assert mark_record["image_metadata"]
    compare_record = next(
        item
        for item in trace["tool_trace"]
        if item["tool_name"] == "compare_proposals_spatial"
    )
    assert '"ranked_ids": [6, 7]' in compare_record["response_text"]


def test_nr3d_tools_cli_invokes_select_by_text_with_lazy_selector(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    _, _, state_path = _state_path(tmp_path)
    trace_path = tmp_path / "cli_text_trace.json"
    built: list[dict[str, object]] = []

    class FakeSelector:
        def select_keyframes_v2(self, **kwargs):
            return SimpleNamespace(
                keyframe_indices=[10],
                metadata={
                    "hypothesis_output": {
                        "hypotheses": [
                            {
                                "grounding_query": {"root": {"category": "table"}},
                                "kind": "direct",
                            }
                        ]
                    }
                },
            )

    class FakeKeyframeSelector:
        @classmethod
        def from_scene_path(cls, scene_path, **kwargs):
            built.append({"scene_path": scene_path, **kwargs})
            return FakeSelector()

    monkeypatch.setitem(
        sys.modules,
        "query_scene",
        SimpleNamespace(KeyframeSelector=FakeKeyframeSelector),
    )
    from agents.mcp import nr3d_tools_cli

    exit_code = nr3d_tools_cli.main(
        [
            "--state",
            str(state_path),
            "--trace",
            str(trace_path),
            "call",
            "select_by_text",
            json.dumps({"query": "square table", "k": 3}),
        ]
    )

    assert exit_code == 0
    response = json.loads(capsys.readouterr().out)
    assert response["is_error"] is False
    assert '"frame_id": 10' in response["content_text"]
    assert len(built) == 1
    assert built[0]["llm_model"] == "fake-model"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["tool_trace"][0]["tool_name"] == "select_by_text"


def test_nr3d_mcp_server_lists_runtime_tools(tmp_path) -> None:
    _, _, state_path = _state_path(tmp_path)
    server = Nr3dToolMcpServer(state_path=state_path)

    names = {tool["name"] for tool in server.list_tools()["tools"]}

    assert {
        "retrieve_object_context",
        "select_by_proposal",
        "select_by_region",
        "select_by_frame_neighbor",
        "select_by_coverage",
        "view_bev",
        "list_scene_proposals",
        "inspect_proposal",
        "mark_frame_with_bbox",
        "list_frame_proposals",
        "compare_proposals_spatial",
        "compare_candidates_to_anchors",
    } <= names
    assert "select_by_text" in names
    assert "list_skills" not in names
    assert "load_skill" not in names
    assert "submit_final" not in names


def test_nr3d_mcp_server_invokes_select_by_text_with_lazy_selector(
    monkeypatch,
    tmp_path,
) -> None:
    _, _, state_path = _state_path(tmp_path)
    built: list[dict[str, object]] = []

    class FakeSelector:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def select_keyframes_v2(self, **kwargs):
            self.calls.append(dict(kwargs))
            return SimpleNamespace(
                keyframe_indices=[10],
                metadata={
                    "hypothesis_output": {
                        "hypotheses": [
                            {
                                "grounding_query": {"root": {"category": "table"}},
                                "kind": "direct",
                            }
                        ]
                    }
                },
            )

    class FakeKeyframeSelector:
        @classmethod
        def from_scene_path(cls, scene_path, **kwargs):
            built.append({"scene_path": scene_path, **kwargs})
            return FakeSelector()

    monkeypatch.setitem(
        sys.modules,
        "query_scene",
        SimpleNamespace(KeyframeSelector=FakeKeyframeSelector),
    )
    server = Nr3dToolMcpServer(state_path=state_path)

    first = server.call_tool("select_by_text", {"query": "square table"})
    second = server.call_tool("select_by_text", {"query": "square table again"})

    assert first["isError"] is False
    assert second["isError"] is False
    assert '"frame_id": 10' in first["content"][0]["text"]
    assert len(built) == 1
    assert built[0]["scene_path"].endswith("scene0001_00/conceptgraph")
    assert built[0]["stride"] == 1
    assert built[0]["llm_model"] == "fake-model"
    assert server.runtime.text_frame_selector is not None


def test_nr3d_mcp_server_invokes_tool_and_writes_trace(tmp_path) -> None:
    _, _, state_path = _state_path(tmp_path)
    trace_path = tmp_path / "mcp_trace.json"
    server = Nr3dToolMcpServer(state_path=state_path, trace_path=trace_path)

    result = server.call_tool("inspect_proposal", {"proposal_id": 6})

    text = result["content"][0]["text"]
    assert "Square table near the door" in text
    assert "proposal_id" in text
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["tool_trace"][0]["tool_name"] == "inspect_proposal"
    assert trace["final_submission"] is None
    assert trace["skills_loaded"] == [
        "scene-exploration-playbook",
        "vg-grounding-playbook",
        "vg-spatial-disambiguation",
    ]


def test_nr3d_mcp_server_speaks_json_lines_stdio(tmp_path) -> None:
    _, _, state_path = _state_path(tmp_path)
    server_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "mcp",
        "nr3d_tools_server.py",
    )
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        },
    }
    list_tools = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    stdin = json.dumps(initialize) + "\n" + json.dumps(list_tools) + "\n"

    proc = subprocess.run(
        [
            sys.executable,
            server_path,
            "--state",
            str(state_path),
            "--trace",
            str(tmp_path / "trace.json"),
        ],
        input=stdin,
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        cwd=PROJECT_ROOT,
        timeout=15,
    )

    responses = [json.loads(line) for line in proc.stdout.splitlines() if line]
    assert [response["id"] for response in responses] == [1, 2]
    tool_names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert "inspect_proposal" in tool_names


def test_nr3d_tools_cli_lists_tools_and_invokes_tool(tmp_path) -> None:
    _, _, state_path = _state_path(tmp_path)
    trace_path = tmp_path / "cli_trace.json"
    cli_path = PROJECT_ROOT / "src" / "agents" / "mcp" / "nr3d_tools_cli.py"

    listed = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--state",
            str(state_path),
            "--trace",
            str(trace_path),
            "list-tools",
        ],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        cwd=PROJECT_ROOT,
        timeout=15,
    )

    list_payload = json.loads(listed.stdout)
    assert "inspect_proposal" in {tool["name"] for tool in list_payload["tools"]}

    called = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--state",
            str(state_path),
            "--trace",
            str(trace_path),
            "call",
            "inspect_proposal",
            '{"proposal_id": 6}',
        ],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        cwd=PROJECT_ROOT,
        timeout=15,
    )

    call_payload = json.loads(called.stdout)
    assert call_payload["tool_name"] == "inspect_proposal"
    assert call_payload["is_error"] is False
    assert "Square table near the door" in call_payload["content_text"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["tool_trace"][0]["tool_name"] == "inspect_proposal"


def test_nr3d_tools_cli_merges_parallel_trace_writes(tmp_path) -> None:
    _, _, state_path = _state_path(tmp_path)
    trace_path = tmp_path / "cli_parallel_trace.json"
    cli_path = PROJECT_ROOT / "src" / "agents" / "mcp" / "nr3d_tools_cli.py"
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    commands = [
        [
            sys.executable,
            str(cli_path),
            "--state",
            str(state_path),
            "--trace",
            str(trace_path),
            "call",
            "inspect_proposal",
            '{"proposal_id": 6}',
        ],
        [
            sys.executable,
            str(cli_path),
            "--state",
            str(state_path),
            "--trace",
            str(trace_path),
            "call",
            "list_scene_proposals",
            '{"category": "table"}',
        ],
    ]

    procs = [
        subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=PROJECT_ROOT,
        )
        for cmd in commands
    ]

    for proc in procs:
        stdout, stderr = proc.communicate(timeout=15)
        assert proc.returncode == 0, (stdout, stderr)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    names = [item["tool_name"] for item in trace["tool_trace"]]
    assert "inspect_proposal" in names
    assert "list_scene_proposals" in names
