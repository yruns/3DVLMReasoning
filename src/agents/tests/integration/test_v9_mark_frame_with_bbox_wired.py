from __future__ import annotations

from types import SimpleNamespace

from agents.core.agent_config import Stage2TaskType
from agents.runtime.deepagents_agent import _collect_v9_tools


def _stub_runtime() -> object:
    bundle = SimpleNamespace(
        extra_metadata={"scene_catalog": {"scene_id": "s", "proposals": [], "valid_frame_ids": []}}
    )
    rs = SimpleNamespace(bundle=bundle)
    return rs


def test_mark_frame_with_bbox_is_loaded_for_vg():
    tools = _collect_v9_tools(runtime=_stub_runtime(), task_type=Stage2TaskType.VISUAL_GROUNDING)
    names = {getattr(t, "name", "") for t in tools}
    assert "mark_frame_with_bbox" in names


def test_mark_frame_with_bbox_is_loaded_for_qa():
    tools = _collect_v9_tools(runtime=_stub_runtime(), task_type=Stage2TaskType.QA)
    names = {getattr(t, "name", "") for t in tools}
    assert "mark_frame_with_bbox" in names


def test_legacy_view_keyframe_still_loaded_in_transition():
    """During the v9.1 transition (before Task 15 deletion), view_keyframe should still load."""
    tools = _collect_v9_tools(runtime=_stub_runtime(), task_type=Stage2TaskType.VISUAL_GROUNDING)
    names = {getattr(t, "name", "") for t in tools}
    assert "view_keyframe" in names
