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
    assert ("view_" + "key" + "frame") not in body


def test_playbook_says_first_move_is_select_by_text():
    body = _PLAYBOOK_PATH.read_text()
    assert "First move" in body
    assert body.lower().count("select_by_text") >= 2


def test_playbook_warns_against_anchor_hidden_categories():
    body = _PLAYBOOK_PATH.read_text()
    assert "hidden_categories" in body
    assert "Do not hide support/anchor/context categories" in body
    assert "Masked category leak detected" in body
    assert "hidden_categories=[]" in body


def test_playbook_requires_same_category_candidate_coverage():
    body = _PLAYBOOK_PATH.read_text()
    assert "every plausible same-category candidate" in body
    assert "marked evidence or an explicit elimination reason" in body


def test_no_text_variant_exists_and_drops_select_by_text():
    """v9.2 catalog-first variant for the audit A/B test."""
    no_text_path = _PLAYBOOK_PATH.with_name("scene_exploration_playbook_no_text.md")
    assert no_text_path.exists(), no_text_path
    body = no_text_path.read_text()
    # Tool-invocation form `select_by_text(...)` must not appear; URL
    # references to the audit doc filename are OK.
    assert "select_by_text(" not in body, body
    assert "select_by_proposal" in body
    assert "catalog-first" in body.lower()


def test_no_text_variant_swapped_when_runtime_flag_false():
    """`load_skill` must prefer the `_no_text.md` sibling when the
    runtime's enable_stage1_text_retrieval flag is False."""
    from types import SimpleNamespace

    from agents.skills.chassis_tools import build_chassis_tools

    _ensure_packs_registered()

    bundle = SimpleNamespace(extra_metadata={})
    runtime = SimpleNamespace(
        task_type=Stage2TaskType.VISUAL_GROUNDING,
        bundle=bundle,
        skills_loaded=set(),
        tool_trace=[],
        enable_stage1_text_retrieval=False,
        # the rest of the chassis path inspects these but won't fire here
        final_submission=None,
        record=lambda *a, **kw: None,
    )
    _, load_skill, _ = build_chassis_tools(runtime)
    body = load_skill.invoke({"skill_name": "scene-exploration-playbook"})
    assert "select_by_text(" not in body
    assert "select_by_proposal" in body
    # And with the flag True, the canonical body still wins (the tool
    # invocation form must reappear).
    runtime.enable_stage1_text_retrieval = True
    runtime.skills_loaded = set()
    body_text_first = load_skill.invoke({"skill_name": "scene-exploration-playbook"})
    assert "select_by_text(" in body_text_first


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


def test_vg_playbooks_treat_generic_category_as_subtype_compatible():
    _ensure_packs_registered()
    spec = next(
        s
        for s in skills_for(Stage2TaskType.VISUAL_GROUNDING)
        if s.name == "vg-grounding-playbook"
    )
    for path in (spec.body_path, spec.body_path.with_name("vg_grounding_playbook_no_text.md")):
        body = path.read_text()
        assert "Generic category words include subtype labels" in body
        assert "office chair" in body


def test_scene_exploration_playbook_does_not_reference_deleted_tools():
    text = _PLAYBOOK_PATH.read_text()
    for dead in (
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_" + "key" + "frames_with_proposals",
        "inspect_stage1_metadata",
        "view_" + "key" + "frame_marked",
        "view_" + "key" + "frame",
        "select_by_hypothesis",
    ):
        assert dead not in text, f"deleted tool {dead} still referenced"
