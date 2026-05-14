from __future__ import annotations

from query_scene.frame_nms import hard_nms_ordered_views


def test_hard_nms_suppresses_near_duplicate_and_preserves_order() -> None:
    overlaps = {
        (1, 0): 0.92,
        (2, 0): 0.20,
        (2, 1): 0.30,
    }

    def overlap_fn(view_a: int, view_b: int) -> float:
        return overlaps.get((view_a, view_b), overlaps.get((view_b, view_a), 0.0))

    result = hard_nms_ordered_views(
        [0, 1, 2],
        max_views=2,
        overlap_fn=overlap_fn,
        score_fn=lambda view_id: {0: 0.9, 1: 0.8, 2: 0.7}[view_id],
        overlap_threshold=0.75,
    )

    assert result.selected == [0, 2]
    assert result.strict_selected == [0, 2]
    assert [(item.view_id, item.suppressed_by) for item in result.suppressed] == [
        (1, 0)
    ]
    assert result.relaxed_backfill == []


def test_hard_nms_relaxed_backfill_keeps_evidence_budget() -> None:
    def overlap_fn(view_a: int, view_b: int) -> float:
        if view_a == view_b:
            return 1.0
        return 0.9 if {view_a, view_b} != {0, 3} else 0.4

    result = hard_nms_ordered_views(
        [0, 1, 2, 3],
        max_views=3,
        overlap_fn=overlap_fn,
        score_fn=lambda view_id: float(10 - view_id),
        overlap_threshold=0.5,
    )

    assert result.strict_selected == [0, 3]
    assert result.selected == [0, 3, 1]
    assert [item.view_id for item in result.relaxed_backfill] == [1]
    assert result.relaxed_backfill[0].max_overlap == 0.9
