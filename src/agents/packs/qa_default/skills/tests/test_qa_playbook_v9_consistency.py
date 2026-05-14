"""Consistency tests for v9 QA playbook (catalog-first)."""

from pathlib import Path

import pytest


_PB = (
    Path(__file__).resolve().parents[1] / "qa_answering_playbook.md"
)


@pytest.mark.parametrize(
    "dead",
    [
        "request_more_views",
        "switch_or_expand_hypothesis",
        "find_proposals_by_category",
        "list_keyframes_with_proposals",
        "inspect_stage1_metadata",
        "view_keyframe_marked",
        "retrieve_object_context",
    ],
)
def test_no_dead_tool_names(dead: str):
    assert dead not in _PB.read_text(), f"{dead} still in qa_answering_playbook"


def test_qa_playbook_recommends_rgb_mode():
    text = _PB.read_text()
    assert "view_keyframe(frame_id, mode='rgb')" in text or 'mode="rgb"' in text


def test_qa_playbook_mentions_supporting_claims():
    text = _PB.read_text()
    assert "supporting_claims" in text


def test_qa_playbook_mentions_select_by_text_for_attributes():
    text = _PB.read_text()
    assert "select_by_text" in text
