"""Smoke tests for v9 migration: callback-style tools are gone."""

import importlib
from types import SimpleNamespace

from agents.core.agent_config import Stage2DeepAgentConfig, Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.deepagents_agent import DeepAgentsStage2Runtime, _collect_v9_tools


def test_view_keyframe_not_loaded():
    rs = SimpleNamespace(
        bundle=SimpleNamespace(
            extra_metadata={
                "scene_catalog": {
                    "scene_id": "s",
                    "proposals": [],
                    "valid_frame_ids": [],
                }
            }
        ),
        task_type=Stage2TaskType.VISUAL_GROUNDING,
    )
    names = {getattr(t, "name", "") for t in _collect_v9_tools(runtime=rs, task_type=rs.task_type)}
    assert "view_keyframe" not in names


def test_select_by_hypothesis_not_loaded():
    rs = SimpleNamespace(
        bundle=SimpleNamespace(
            extra_metadata={
                "scene_catalog": {
                    "scene_id": "s",
                    "proposals": [],
                    "valid_frame_ids": [],
                }
            }
        ),
        task_type=Stage2TaskType.VISUAL_GROUNDING,
    )
    names = {getattr(t, "name", "") for t in _collect_v9_tools(runtime=rs, task_type=rs.task_type)}
    assert "select_by_hypothesis" not in names


def test_runtime_does_not_expose_request_more_views():
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    for attr in (
        "_request_more_views_impl",
        "_create_request_more_views_tool",
        "_create_switch_or_expand_hypothesis_tool",
        "_create_inspect_stage1_metadata_tool",
        "more_views_callback",
        "hypothesis_callback",
    ):
        assert not hasattr(rt, attr), f"{attr} should be removed"


def test_stage1_callbacks_drops_old_factories():
    mod = importlib.import_module("agents.stage1_callbacks")
    assert not hasattr(mod, "create_more_views_callback")
    assert not hasattr(mod, "create_hypothesis_callback")
    assert hasattr(mod, "create_crop_callback")


def test_runtime_tool_list_does_not_include_dead_names():
    rt = DeepAgentsStage2Runtime(config=Stage2DeepAgentConfig())
    bundle = Stage2EvidenceBundle(scene_id="dummy")
    runtime_state = Stage2RuntimeState(
        bundle=bundle, task_type=Stage2TaskType.QA
    )
    names = {t.name for t in rt.build_runtime_tools(runtime_state)}
    for dead in (
        "request_more_views",
        "switch_or_expand_hypothesis",
        "inspect_stage1_metadata",
        "list_keyframes_with_proposals",
        "find_proposals_by_category",
        "view_keyframe_marked",
    ):
        assert dead not in names, f"{dead} should not be wired"


def test_pack_tools_module_no_longer_defines_dead_helpers():
    import agents.packs.vg_embodiedscan.tools as mod
    for dead in (
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
    ):
        assert not hasattr(mod, dead), f"{dead} still exported"
