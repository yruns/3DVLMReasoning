"""Consistency tests for v9 VG playbooks (catalog-first)."""

from pathlib import Path

import pytest


_PB = (
    Path(__file__).resolve().parents[1] / "vg_grounding_playbook.md"
)
_SD = (
    Path(__file__).resolve().parents[1] / "vg_spatial_disambiguation.md"
)


@pytest.mark.parametrize("path", [_PB, _SD])
@pytest.mark.parametrize(
    "dead",
    [
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
        "inspect_stage1_metadata",
        "view_keyframe_marked",
    ],
)
def test_no_dead_tool_names(path: Path, dead: str):
    assert dead not in path.read_text(), f"{dead} still mentioned in {path}"


def test_vg_playbook_lists_v9_tools():
    text = _PB.read_text()
    for tool in (
        "view_keyframe",
        "list_frame_proposals",
        "list_scene_proposals",
        "inspect_proposal",
        "select_by_proposal",
        "view_bev",
    ):
        assert tool in text


def test_vg_playbook_mentions_tadg_and_guards():
    text = _PB.read_text()
    assert "TADG" in text
    assert "no_match_guard" in text
    assert "evidence_frame_guard" in text


def test_vg_playbook_mentions_ood_proposal_minus_one():
    text = _PB.read_text()
    assert "proposal_id" in text
    assert "-1" in text


def test_vg_spatial_disambiguation_uses_view_keyframe_mode_marked():
    text = _SD.read_text()
    assert "view_keyframe(mode='marked')" in text or 'mode="marked"' in text
