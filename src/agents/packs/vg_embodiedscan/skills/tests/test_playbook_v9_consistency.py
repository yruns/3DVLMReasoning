"""Consistency tests for v9.1 VG playbooks (selectors-return-images)."""

from pathlib import Path

import pytest

_PB = Path(__file__).resolve().parents[1] / "vg_grounding_playbook.md"
_PB_NO_TEXT = Path(__file__).resolve().parents[1] / "vg_grounding_playbook_no_text.md"
_SD = Path(__file__).resolve().parents[1] / "vg_spatial_disambiguation.md"
_SD_NO_TEXT = (
    Path(__file__).resolve().parents[1] / "vg_spatial_disambiguation_no_text.md"
)


@pytest.mark.parametrize("path", [_PB, _SD])
@pytest.mark.parametrize(
    "dead",
    [
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_" + "key" + "frames_with_proposals",
        "inspect_stage1_metadata",
        "view_" + "key" + "frame_marked",
        "view_" + "key" + "frame",
        "select_by_hypothesis",
    ],
)
def test_no_dead_tool_names(path: Path, dead: str):
    assert dead not in path.read_text(), f"{dead} still mentioned in {path}"


def test_vg_playbook_lists_v9_1_tools():
    text = _PB.read_text()
    for tool in (
        "select_by_text",
        "select_by_proposal",
        "mark_frame_with_bbox",
        "list_frame_proposals",
        "list_scene_proposals",
        "inspect_proposal",
        "view_bev",
    ):
        assert tool in text, f"{tool} missing from vg_grounding_playbook"


def test_vg_playbook_first_move_is_select_by_text():
    text = _PB.read_text()
    assert text.count("select_by_text") >= 2
    assert "first move" in text.lower() or "First move" in text


def test_vg_playbook_mentions_tadg_and_guards():
    text = _PB.read_text()
    assert "TADG" in text
    assert "no_match_guard" in text
    assert "evidence_frame_guard" in text


@pytest.mark.parametrize("path", [_PB, _PB_NO_TEXT, _SD, _SD_NO_TEXT])
def test_vg_playbooks_document_relation_evidence_binding(path: Path):
    text = path.read_text()
    assert "evidence_id" in text
    assert "relation_evidence" in text
    assert "compare_proposals_spatial" in text
    assert "submit_final" in text


def test_vg_playbook_mentions_ood_proposal_minus_one():
    text = _PB.read_text()
    assert "proposal_id" in text
    assert "-1" in text


def test_vg_spatial_disambiguation_uses_mark_frame_with_bbox():
    text = _SD.read_text()
    assert "mark_frame_with_bbox" in text
    assert "frame_id" in text


@pytest.mark.parametrize("path", [_PB, _PB_NO_TEXT, _SD, _SD_NO_TEXT])
def test_vg_playbooks_document_canonical_spatial_relations(path: Path):
    text = path.read_text()
    for relation in (
        "closest_to",
        "near",
        "next_to",
        "farthest_from",
        "above",
        "below",
        "left_of",
        "right_of",
    ):
        assert relation in text, f"{relation} missing from {path}"
    assert "relation='left_of'|'right_of'|'closer_to'" not in text
