from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle, Stage2TaskSpec
from agents.runtime.codex_sdk_agent import CodexSdkStage2Runtime

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _bundle(tmp_path) -> Stage2EvidenceBundle:
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    return Stage2EvidenceBundle(
        scene_id="scene0001_00",
        stage1_query="the square table by the door",
        extra_metadata={
            "gt_bbox_3d_9dof": [9999.0, 9998.0, 9997.0, 1, 1, 1, 0, 0, 0],
            "stage1_text_selector": {
                "scene_id": "scene0001_00",
                "phase8_data_root": str(tmp_path / "phase8"),
                "conceptgraph_root": str(
                    tmp_path / "phase8" / "scene0001_00" / "conceptgraph"
                ),
                "llm_model": "gemini-2.5-pro",
            },
            "scene_catalog": {
                "scene_id": "scene0001_00",
                "scene_category": "room",
                "total_frames": 12,
                "frame_id_range": [0, 11],
                "proposals": [
                    {
                        "proposal_id": 6,
                        "category": "table",
                        "position_3d": [1.0, 2.0, 0.4],
                        "bbox_3d_9dof": [1, 2, 0.4, 1.2, 1.1, 0.8, 0, 0, 0],
                        "enriched_category": "square table",
                        "compact_note": "Square table near the door.",
                    }
                ],
            },
            "vg_proposal_pool": {
                "source": "gt",
                "annotated_image_dir": str(annotated),
                "frame_index": {"10": [6]},
                "proposal_index": {"6": [10]},
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
                    }
                ],
            },
        },
    )


def test_codex_prompt_excludes_gt_fields(tmp_path) -> None:
    runtime = CodexSdkStage2Runtime(config=Stage2DeepAgentConfig())
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )

    prompt = runtime.build_decision_prompt(
        task,
        _bundle(tmp_path),
        tool_state_path=tmp_path / "sample.state.json",
        tool_trace_path=tmp_path / "sample.trace.json",
    )

    assert "gt_bbox_3d_9dof" not in prompt
    assert "9999" not in prompt
    assert "#6" in prompt
    assert "Square table near the door" in prompt
    assert "nr3d_tools_cli.py" in prompt
    assert "MCP tools are mounted" not in prompt
    assert "select_by_text" in prompt
    assert "lazily" in prompt
    assert "list_skills" not in prompt
    assert "load_skill" not in prompt


def test_codex_prompt_includes_cli_evidence_policy_and_state_paths(tmp_path) -> None:
    runtime = CodexSdkStage2Runtime(config=Stage2DeepAgentConfig())
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )
    state_path = tmp_path / "sample.state.json"
    trace_path = tmp_path / "sample.trace.json"

    prompt = runtime.build_decision_prompt(
        task,
        _bundle(tmp_path),
        tool_state_path=state_path,
        tool_trace_path=trace_path,
    )

    assert "nr3d_tools_cli.py" in prompt
    assert str(state_path) in prompt
    assert str(trace_path) in prompt
    assert "call inspect_proposal" in prompt
    assert "call select_by_proposal" in prompt
    assert "call mark_frame_with_bbox" in prompt
    assert "call compare_proposals_spatial" in prompt
    assert "call select_by_text" in prompt
    assert "Required CLI evidence policy" in prompt
    assert "CLI tools are the evidence interface" in prompt
    assert "run at least one CLI evidence command" not in prompt


def test_codex_cli_tool_note_has_valid_inspect_example(tmp_path) -> None:
    runtime = CodexSdkStage2Runtime(config=Stage2DeepAgentConfig())

    note = runtime.build_cli_tool_note(
        tmp_path / "sample.state.json",
        tmp_path / "sample.trace.json",
    )

    assert "call inspect_proposal --trace" not in note
    assert note.count("call inspect_proposal") == 1


def test_codex_runtime_defaults_to_cli_only_tools() -> None:
    runtime = CodexSdkStage2Runtime(config=Stage2DeepAgentConfig())

    assert runtime.enable_mcp_tools is False
    assert runtime.enable_cli_tools is True


def test_codex_runtime_wraps_json_decision(monkeypatch, tmp_path) -> None:
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(),
        mcp_state_dir=tmp_path / "mcp_state",
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )

    def fake_run_turn(*_args, **kwargs):
        assert kwargs["mcp_state_path"].exists()
        assert str(kwargs["mcp_state_path"]).startswith(str(tmp_path))
        assert kwargs["mcp_trace_path"] is None
        return (
            json.dumps(
                {
                    "proposal_id": 6,
                    "confidence": 0.88,
                    "summary": "Proposal 6 is the square table closest to the door.",
                    "uncertainties": [],
                    "cited_frame_indices": [10],
                }
            ),
            {"id": "turn_mock", "status": "completed"},
        )

    monkeypatch.setattr(runtime, "_run_codex_turn", fake_run_turn)

    result = runtime.run(task, _bundle(tmp_path))

    assert result.result.status.value == "completed"
    assert result.result.payload["proposal_id"] == 6
    assert result.result.payload["selected_object_id"] == 6
    assert result.result.confidence == 0.88
    assert result.tool_trace[0].tool_name == "codex_sdk_turn"
    assert result.tool_trace[0].tool_input["mcp_server_path"] is None
    assert result.tool_trace[0].tool_input["cli_trace_path"] is not None
    assert result.raw_state["mcp_tools_enabled"] is False
    assert result.raw_state["cli_tools_enabled"] is True


def test_codex_runtime_removes_temporary_tool_state_after_run(
    monkeypatch,
    tmp_path,
) -> None:
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(),
        mcp_state_dir=tmp_path / "mcp_state",
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )

    def fake_run_turn(*_args, **kwargs):
        state_path = kwargs["mcp_state_path"]
        assert state_path.exists()
        mcp_trace_path = state_path.with_name(
            state_path.name.replace(".state.json", ".trace.json")
        )
        cli_trace_path = runtime.cli_trace_path_for(mcp_trace_path)
        mcp_trace_path.write_text(
            json.dumps({"tool_trace": []}),
            encoding="utf-8",
        )
        cli_trace_path.write_text(
            json.dumps({"tool_trace": []}),
            encoding="utf-8",
        )
        cli_trace_path.with_name(f"{cli_trace_path.name}.lock").write_text(
            "",
            encoding="utf-8",
        )
        return (
            json.dumps(
                {
                    "proposal_id": 6,
                    "confidence": 0.88,
                    "summary": "Proposal 6 is the square table closest to the door.",
                    "uncertainties": [],
                    "cited_frame_indices": [10],
                }
            ),
            {"id": "turn_mock", "status": "completed"},
        )

    monkeypatch.setattr(runtime, "_run_codex_turn", fake_run_turn)

    result = runtime.run(task, _bundle(tmp_path))
    state_path = Path(result.tool_trace[-1].tool_input["mcp_state_path"])
    cli_trace_path = Path(result.tool_trace[-1].tool_input["cli_trace_path"])
    mcp_trace_path = state_path.with_name(
        state_path.name.replace(".state.json", ".trace.json")
    )

    assert result.raw_state["mcp_trace"]["tool_count"] == 0
    assert not state_path.exists()
    assert not mcp_trace_path.exists()
    assert not cli_trace_path.exists()
    assert not cli_trace_path.with_name(f"{cli_trace_path.name}.lock").exists()


def test_codex_runtime_retries_non_json_final_response_in_same_thread(
    monkeypatch, tmp_path
) -> None:
    calls: list[list[object]] = []
    configs: list[SimpleNamespace] = []

    class FakeTextInput:
        def __init__(self, text):
            self.text = text

    class FakeSkillInput:
        def __init__(self, name, path):
            self.name = name
            self.path = path

    class FakeLocalImageInput:
        def __init__(self, path):
            self.path = path

    class FakeSandboxValue:
        value = "workspace-write"

    class FakeSandbox:
        full_access = FakeSandboxValue()
        workspace_write = FakeSandboxValue()
        read_only = FakeSandboxValue()

    class FakeThread:
        def run(self, input_items, **_kwargs):
            calls.append(input_items)
            if len(calls) == 1:
                return SimpleNamespace(
                    final_response="I am still inspecting the candidates.",
                    id="turn_1",
                    status="completed",
                    duration_ms=10,
                    usage=None,
                )
            return SimpleNamespace(
                final_response=json.dumps(
                    {
                        "proposal_id": 6,
                        "confidence": 0.82,
                        "summary": "Proposal 6 best matches the query.",
                        "uncertainties": [],
                        "cited_frame_indices": [10],
                    }
                ),
                id="turn_2",
                status="completed",
                duration_ms=12,
                usage=None,
            )

    class FakeCodex:
        def __init__(self, config):
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def thread_start(self, **_kwargs):
            return FakeThread()

    def fake_codex_config(**kwargs):
        config = SimpleNamespace(**kwargs)
        configs.append(config)
        return config

    fake_module = types.SimpleNamespace(
        Codex=FakeCodex,
        CodexConfig=fake_codex_config,
        LocalImageInput=FakeLocalImageInput,
        Sandbox=FakeSandbox,
        SkillInput=FakeSkillInput,
        TextInput=FakeTextInput,
    )
    monkeypatch.setitem(sys.modules, "openai_codex", fake_module)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    codex_home = tmp_path / "codex_home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'model = "mock"\n',
        encoding="utf-8",
    )
    (codex_home / "installation_id").write_text(
        "mock-installation-id",
        encoding="utf-8",
    )
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(),
        codex_home=codex_home,
    )

    response_text, metadata = runtime._run_codex_turn("pick the proposal", [])

    assert json.loads(response_text)["proposal_id"] == 6
    assert metadata["id"] == "turn_2"
    assert len(calls) == 2
    assert isinstance(calls[1][0], FakeTextInput)
    assert "Return only one JSON object" in calls[1][0].text
    assert "CODEX_HOME" not in os.environ
    assert len(configs) == 1
    run_home = Path(configs[0].env["CODEX_HOME"])
    assert run_home.parent == codex_home / "runs"
    assert not run_home.exists()


def test_codex_runtime_uses_full_access_sandbox_when_cli_trace_is_enabled() -> None:
    runtime = CodexSdkStage2Runtime(config=Stage2DeepAgentConfig())

    assert runtime.codex_sandbox_value() == "full-access"


def test_codex_runtime_can_override_sandbox_with_env(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_AGENT_SANDBOX", "workspace_write")
    runtime = CodexSdkStage2Runtime(config=Stage2DeepAgentConfig())

    assert runtime.codex_sandbox_value() == "workspace-write"


def test_codex_runtime_writes_mcp_state_without_api_keys(tmp_path) -> None:
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(
            api_keys=["secret-key"],
        ),
        mcp_state_dir=tmp_path / "mcp_state",
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )

    state_path, trace_path = runtime.write_mcp_state(task, _bundle(tmp_path))
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    config_overrides = runtime.build_codex_app_server_config_overrides(
        mcp_state_path=state_path,
        mcp_trace_path=trace_path,
    )

    assert state_payload["config"]["enable_stage1_text_retrieval"] is True
    assert "api_keys" not in state_payload["config"]
    assert "secret-key" not in state_path.read_text(encoding="utf-8")
    joined = "\n".join(config_overrides)
    assert "mcp_servers.nr3d_tools.command=" in joined
    assert "mcp_servers.nr3d_tools.cwd=" in joined
    assert "mcp_servers.nr3d_tools.env.PYTHONPATH=" in joined
    assert str(PROJECT_ROOT / "src") in joined
    assert "mcp_servers.nr3d_tools.required=true" in joined
    assert "mcp_servers.nr3d_tools.startup_timeout_sec=30" in joined
    assert "mcp_servers.nr3d_tools.default_tools_approval_mode=" in joined
    assert "--state" in joined
    assert str(state_path) in joined
    assert "secret-key" not in joined


def test_codex_runtime_writes_allowed_tool_env_to_mcp_state(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("STAGE1_TEXT_RETRIEVAL_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("STAGE1_TEXT_RETRIEVAL_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(),
        mcp_state_dir=tmp_path / "mcp_state",
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )

    state_path, _ = runtime.write_mcp_state(task, _bundle(tmp_path))
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert state_payload["tool_env"] == {
        "STAGE1_TEXT_RETRIEVAL_MAX_CONCURRENCY": "2",
        "STAGE1_TEXT_RETRIEVAL_LOCK_DIR": str(tmp_path / "locks"),
    }
    assert "OPENAI_API_KEY" not in state_path.read_text(encoding="utf-8")
    assert "must-not-leak" not in state_path.read_text(encoding="utf-8")


def test_codex_runtime_configures_modelhub_prefix_cache_headers(tmp_path) -> None:
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(
            session_id="nr3d_codex_cache_session",
        ),
        mcp_state_dir=tmp_path / "mcp_state",
    )
    task = Stage2TaskSpec(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        user_query="the square table by the door",
    )
    state_path, trace_path = runtime.write_mcp_state(task, _bundle(tmp_path))

    overrides = runtime.build_codex_app_server_config_overrides(
        mcp_state_path=state_path,
        mcp_trace_path=trace_path,
    )
    env = runtime.build_codex_app_server_env()

    joined = "\n".join(overrides)
    assert (
        "model_providers.modelhub_adapter.env_http_headers.extra="
        '"CODEX_AGENT_MODELHUB_EXTRA_HEADER"'
    ) in joined
    assert (
        "model_providers.modelhub_adapter.env_http_headers.X-TT-LOGID="
        '"CODEX_AGENT_MODELHUB_LOGID"'
    ) in joined
    extra = json.loads(env["CODEX_AGENT_MODELHUB_EXTRA_HEADER"])
    assert extra["session_id"] == "nr3d_codex_cache_session"
    assert extra["source"] == "codex_agent_sdk"
    assert extra["chat_run_id"].startswith("nr3d_codex_cache_session_")
    assert env["CODEX_AGENT_MODELHUB_LOGID"].startswith(
        "codexsdk_nr3d_codex_cache_session_"
    )


def test_codex_runtime_sanitizes_prefix_cache_session_id() -> None:
    runtime = CodexSdkStage2Runtime(
        config=Stage2DeepAgentConfig(
            session_id=" /unsafe session/id? ",
        ),
        enable_mcp_tools=False,
    )

    env = runtime.build_codex_app_server_env()

    assert json.loads(env["CODEX_AGENT_MODELHUB_EXTRA_HEADER"])["session_id"] == (
        "unsafe_session_id"
    )


def test_codex_output_schema_is_strict_for_sdk() -> None:
    from agents.runtime.codex_sdk_agent import CodexVisualGroundingDecision

    schema = CodexVisualGroundingDecision.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_codex_runtime_mounts_deepagents_playbook_skills() -> None:
    runtime = CodexSdkStage2Runtime(config=Stage2DeepAgentConfig())

    names = [name for name, _ in runtime.codex_skill_specs()]

    assert names == [
        "nr3d-codex-sdk",
        "scene-exploration-playbook",
        "vg-grounding-playbook",
        "vg-spatial-disambiguation",
    ]


def test_codex_sdk_skill_teaches_cli_fallback() -> None:
    text = (PROJECT_ROOT / ".agents/skills/nr3d-codex-sdk/SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "nr3d_tools_cli.py" in text
    assert "call inspect_proposal" in text
    assert "call select_by_proposal" in text
    assert "call mark_frame_with_bbox" in text
    assert "call compare_proposals_spatial" in text
    assert "select_by_text" in text
    assert "Required CLI evidence policy" in text
    assert "run at least one CLI" not in text


def test_codex_playbook_skills_are_synced_from_deepagents_sources() -> None:
    specs = {
        "scene-exploration-playbook": (
            PROJECT_ROOT
            / "src/agents/skills/shared_skills/scene_exploration_playbook.md"
        ),
        "vg-grounding-playbook": (
            PROJECT_ROOT
            / "src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md"
        ),
        "vg-spatial-disambiguation": (
            PROJECT_ROOT
            / "src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md"
        ),
    }

    for skill_name, source_path in specs.items():
        skill_path = PROJECT_ROOT / ".agents" / "skills" / skill_name / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")
        source = source_path.read_text(encoding="utf-8").strip()
        begin = (
            f"<!-- BEGIN_SYNCED_PLAYBOOK: {source_path.relative_to(PROJECT_ROOT)} -->"
        )
        end = "<!-- END_SYNCED_PLAYBOOK -->"

        assert f"name: {skill_name}" in text
        assert begin in text
        synced = text.split(begin, 1)[1].split(end, 1)[0].strip()
        assert synced
        if skill_name == "vg-grounding-playbook":
            assert "scene-exploration-playbook" in synced
        assert "list_skills" not in text
        assert "load_skill" not in text
        assert "nr3d_tools_cli.py" in text
        assert "select_by_proposal" in synced or skill_name != "vg-grounding-playbook"
        assert len(synced) >= len(source) * 0.95
