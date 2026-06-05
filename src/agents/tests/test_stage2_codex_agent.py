from __future__ import annotations

import json
from pathlib import Path

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

    prompt = runtime.build_decision_prompt(task, _bundle(tmp_path))

    assert "gt_bbox_3d_9dof" not in prompt
    assert "9999" not in prompt
    assert "#6" in prompt
    assert "Square table near the door" in prompt
    assert "nr3d_tools" in prompt
    assert "select_by_text" in prompt
    assert "lazily" in prompt
    assert "list_skills" not in prompt
    assert "load_skill" not in prompt


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
        assert str(kwargs["mcp_trace_path"]).startswith(str(tmp_path))
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
    assert result.raw_state["mcp_tools_enabled"] is True


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
        'model_providers.modelhub_adapter.env_http_headers.extra='
        '"CODEX_AGENT_MODELHUB_EXTRA_HEADER"'
    ) in joined
    assert (
        'model_providers.modelhub_adapter.env_http_headers.X-TT-LOGID='
        '"CODEX_AGENT_MODELHUB_LOGID"'
    ) in joined
    assert json.loads(env["CODEX_AGENT_MODELHUB_EXTRA_HEADER"]) == {
        "session_id": "nr3d_codex_cache_session",
        "source": "codex_agent_sdk",
    }
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
        begin = f"<!-- BEGIN_SYNCED_PLAYBOOK: {source_path.relative_to(PROJECT_ROOT)} -->"
        end = "<!-- END_SYNCED_PLAYBOOK -->"

        assert f"name: {skill_name}" in text
        assert begin in text
        synced = text.split(begin, 1)[1].split(end, 1)[0].strip()
        assert synced
        if skill_name == "vg-grounding-playbook":
            assert "scene-exploration-playbook" in synced
        assert "list_skills" not in text
        assert "load_skill" not in text
        assert "select_by_proposal" in synced or skill_name != "vg-grounding-playbook"
        assert len(synced) >= len(source) * 0.95
