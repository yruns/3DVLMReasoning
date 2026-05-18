"""Consistency tests for v9.1 QA playbook (selectors-as-RGB-source)."""

from pathlib import Path

import pytest

_PB = Path(__file__).resolve().parents[1] / "qa_answering_playbook.md"


@pytest.mark.parametrize(
    "dead",
    [
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_" + "key" + "frames_with_proposals",
        "inspect_stage1_metadata",
        "view_" + "key" + "frame_marked",
        "retrieve_object_context",
        "view_" + "key" + "frame(mode='rgb')",
        "view_" + "key" + "frame",
        "select_by_hypothesis",
    ],
)
def test_no_dead_tool_names(dead: str):
    assert dead not in _PB.read_text(), f"{dead} still in qa_answering_playbook"


def test_qa_playbook_recommends_select_by_text_as_first_move():
    text = _PB.read_text()
    assert "select_by_text" in text
    assert "First move" in text or "first move" in text.lower()


def test_qa_playbook_lists_v9_1_tools():
    text = _PB.read_text()
    for tool in (
        "select_by_text",
        "mark_frame_with_bbox",
        "request_crops",
    ):
        assert tool in text, f"{tool} missing from qa_answering_playbook"


def test_qa_playbook_mentions_supporting_claims():
    text = _PB.read_text()
    assert "supporting_claims" in text


def test_qa_playbook_mentions_select_by_text_for_attributes():
    text = _PB.read_text()
    assert "select_by_text" in text
