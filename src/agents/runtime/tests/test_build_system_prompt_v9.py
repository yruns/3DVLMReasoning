"""Tests for v9 catalog-first build_system_prompt.

Per spec docs/superpowers/specs/2026-05-14-v9-catalog-first-scene-exploration-design.md §F.1.
"""

from agents.core.agent_config import (
    Stage2DeepAgentConfig,
    Stage2PlanMode,
    Stage2TaskType,
)
from agents.core.task_types import Stage2TaskSpec
from agents.runtime.base import BaseStage2Runtime


class _MinimalRuntime(BaseStage2Runtime):
    def run(self, task, bundle):  # type: ignore[override]
        raise NotImplementedError


def _task(task_type: Stage2TaskType = Stage2TaskType.VISUAL_GROUNDING) -> Stage2TaskSpec:
    return Stage2TaskSpec(
        user_query="this is a brown chair",
        task_type=task_type,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=8,
    )


def test_system_prompt_has_v9_catalog_first_header():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "You are the Stage-2 scene reasoning agent" in prompt
    assert "Scene perception model" in prompt


def test_system_prompt_lists_six_selectors_cheapest_first():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "select_by_proposal" in prompt
    assert "select_by_frame_neighbor" in prompt
    assert "select_by_region" in prompt
    assert "select_by_coverage" in prompt
    assert "select_by_text" in prompt
    assert "select_by_hypothesis" in prompt
    assert "Cheapest-first" in prompt


def test_system_prompt_no_longer_mentions_callback_tools():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "request_more_views" not in prompt
    assert "switch_or_expand_hypothesis" not in prompt
    assert "inspect_stage1_metadata" not in prompt
    assert "view_keyframe_marked" not in prompt
    assert "find_proposals_by_category" not in prompt
    assert "list_keyframes_with_proposals" not in prompt


def test_system_prompt_for_qa_uses_view_keyframe_rgb_default():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task(Stage2TaskType.QA))
    assert "view_keyframe" in prompt
    assert "mode='auto'" in prompt or 'mode="auto"' in prompt
    assert "QA" in prompt or "qa" in prompt


def test_system_prompt_mentions_scene_exploration_playbook_first():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "scene-exploration-playbook" in prompt


def test_system_prompt_drops_enable_temporal_fan_branch():
    rt = _MinimalRuntime(config=Stage2DeepAgentConfig())
    prompt = rt.build_system_prompt(_task())
    assert "temporal_fan" not in prompt
