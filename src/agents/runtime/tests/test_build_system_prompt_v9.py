"""Tests for v9.1 catalog-first build_system_prompt.

Per spec docs/superpowers/specs/2026-05-15-v9-1-selectors-return-images-design.md §7.6.
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


class _FakeSelector:
    """Minimal stand-in for KeyframeSelector — satisfies the v9.3
    construction-time guard without bringing in the real Stage-1 stack.
    """

    def select_keyframes_v2(self, **_kwargs):  # pragma: no cover - sanity stub
        from types import SimpleNamespace

        return SimpleNamespace(keyframe_indices=[], metadata={})


def _runtime_default_cfg() -> _MinimalRuntime:
    """Build a runtime with the default (text-retrieval-on) config.

    Mirrors the v9.1+ production wiring where a selector must be provided
    when text retrieval is enabled. Tests that don't care about
    select_by_text can build a runtime with text retrieval disabled via
    ``_runtime_text_off()`` instead.
    """
    return _MinimalRuntime(
        config=Stage2DeepAgentConfig(),  # default enable_stage1_text_retrieval=True
        keyframe_selector=_FakeSelector(),
    )


def _runtime_text_off() -> _MinimalRuntime:
    return _MinimalRuntime(
        config=Stage2DeepAgentConfig(enable_stage1_text_retrieval=False)
    )


def _task(task_type: Stage2TaskType = Stage2TaskType.VISUAL_GROUNDING) -> Stage2TaskSpec:
    return Stage2TaskSpec(
        user_query="this is a brown chair",
        task_type=task_type,
        plan_mode=Stage2PlanMode.BRIEF,
        max_reasoning_turns=8,
    )


def test_system_prompt_has_v9_catalog_first_header():
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "You are the Stage-2 scene reasoning agent" in prompt
    assert "Scene perception model" in prompt


def test_system_prompt_lists_v9_1_selectors():
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "select_by_text" in prompt
    assert "select_by_proposal" in prompt
    assert "select_by_frame_neighbor" in prompt
    assert "select_by_region" in prompt
    assert "select_by_coverage" in prompt
    assert "mark_frame_with_bbox" in prompt


def test_prompt_describes_selector_first_move():
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "select_by_text" in prompt
    assert "primary entry" in prompt.lower() or "first move" in prompt.lower()


def test_prompt_does_not_mention_deleted_tools():
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "view_keyframe" not in prompt
    assert "select_by_hypothesis" not in prompt


def test_system_prompt_no_longer_mentions_callback_tools():
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "request_more_views" not in prompt
    assert "switch_or_expand_hypothesis" not in prompt
    assert "inspect_stage1_metadata" not in prompt
    assert "view_keyframe_marked" not in prompt
    assert "find_proposals_by_category" not in prompt
    assert "list_keyframes_with_proposals" not in prompt


def test_system_prompt_workflow_line_mentions_mark_for_verification():
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "Workflow" in prompt
    assert "mark_frame_with_bbox" in prompt
    assert "verifying" in prompt or "verification" in prompt


def test_system_prompt_mentions_scene_exploration_playbook_first():
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "scene-exploration-playbook" in prompt


def test_system_prompt_drops_enable_temporal_fan_branch():
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "temporal_fan" not in prompt


def test_system_prompt_qa_and_vg_share_same_workflow_line():
    rt = _runtime_default_cfg()
    qa_prompt = rt.build_system_prompt(_task(Stage2TaskType.QA))
    vg_prompt = rt.build_system_prompt(_task(Stage2TaskType.VISUAL_GROUNDING))
    # v9.1: no per-task "Default view mode" branches; both prompts share
    # the same Workflow sentence.
    assert "mark_frame_with_bbox" in qa_prompt
    assert "mark_frame_with_bbox" in vg_prompt
    assert "Default view mode" not in qa_prompt
    assert "Default view mode" not in vg_prompt


def test_system_prompt_drops_select_by_text_when_flag_disabled():
    """v9.2 toggle: when the config flag is False, the prompt must not
    advertise select_by_text and the catalog-first selectors take its
    spot as primary entry. Other selectors remain."""
    rt = _runtime_text_off()
    prompt = rt.build_system_prompt(_task())
    assert "select_by_text" not in prompt
    assert "select_by_proposal" in prompt
    assert "select_by_region" in prompt
    assert "select_by_coverage" in prompt
    # Primary-entry tag now attached to select_by_proposal.
    assert "select_by_proposal" in prompt
    assert "primary entry" in prompt.lower()


def test_system_prompt_keeps_select_by_text_when_flag_default_true():
    """When the flag is True (default) AND a selector is supplied
    (v9.3 contract), the prompt advertises select_by_text."""
    rt = _runtime_default_cfg()
    prompt = rt.build_system_prompt(_task())
    assert "select_by_text" in prompt
    assert "primary entry" in prompt.lower()
