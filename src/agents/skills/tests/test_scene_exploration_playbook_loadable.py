"""Tests for v9.1 scene_exploration_playbook shared skill.

Per spec docs/superpowers/specs/2026-05-15-v9-1-selectors-return-images-design.md §7.1.
"""

import importlib
from pathlib import Path

import agents.packs.qa_default
import agents.packs.vg_embodiedscan
from agents.core.agent_config import Stage2TaskType
from agents.skills import PACKS
from agents.skills.registry import skills_for


def _ensure_packs_registered() -> None:
    """Reload pack packages so PACKS contains them (other tests can clear it)."""
    if Stage2TaskType.VISUAL_GROUNDING not in PACKS:
        importlib.reload(agents.packs.vg_embodiedscan)
    if Stage2TaskType.QA not in PACKS:
        importlib.reload(agents.packs.qa_default)


_PLAYBOOK_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "shared_skills"
    / "scene_exploration_playbook.md"
)


def test_scene_exploration_playbook_exists_and_lists_v9_1_selectors():
    text = _PLAYBOOK_PATH.read_text()
    for tool in (
        "select_by_proposal",
        "select_by_frame_neighbor",
        "select_by_region",
        "select_by_coverage",
        "select_by_text",
        "mark_frame_with_bbox",
        "view_bev",
        "list_scene_proposals",
        "list_frame_proposals",
        "inspect_proposal",
    ):
        assert tool in text, f"{tool} missing from scene_exploration_playbook"


def test_playbook_mentions_v9_1_tools():
    body = _PLAYBOOK_PATH.read_text()
    assert "select_by_text" in body
    assert "mark_frame_with_bbox" in body
    assert "select_by_hypothesis" not in body
    assert "view_keyframe" not in body


def test_playbook_says_first_move_is_select_by_text():
    body = _PLAYBOOK_PATH.read_text()
    assert "First move" in body
    assert body.lower().count("select_by_text") >= 2


def test_scene_exploration_playbook_registered_for_vg_and_qa():
    _ensure_packs_registered()
    for task_type in (Stage2TaskType.VISUAL_GROUNDING, Stage2TaskType.QA):
        names = {s.name for s in skills_for(task_type)}
        assert "scene-exploration-playbook" in names
    spec = next(
        s
        for s in skills_for(Stage2TaskType.VISUAL_GROUNDING)
        if s.name == "scene-exploration-playbook"
    )
    body = spec.body_path.read_text()
    assert "BEV" in body
    assert "select_by_text" in body


def test_scene_exploration_playbook_does_not_reference_deleted_tools():
    text = _PLAYBOOK_PATH.read_text()
    for dead in (
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
        "inspect_stage1_metadata",
        "view_keyframe_marked",
        "view_keyframe",
        "select_by_hypothesis",
    ):
        assert dead not in text, f"deleted tool {dead} still referenced"
