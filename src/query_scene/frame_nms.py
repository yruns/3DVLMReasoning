from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrameNmsSuppression:
    view_id: int
    suppressed_by: int
    overlap: float
    score: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "view_id": self.view_id,
            "suppressed_by": self.suppressed_by,
            "overlap": self.overlap,
            "score": self.score,
        }


@dataclass(frozen=True)
class FrameNmsBackfill:
    view_id: int
    max_overlap: float
    score: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "view_id": self.view_id,
            "max_overlap": self.max_overlap,
            "score": self.score,
        }


@dataclass(frozen=True)
class FrameNmsResult:
    selected: list[int]
    strict_selected: list[int]
    suppressed: list[FrameNmsSuppression] = field(default_factory=list)
    relaxed_backfill: list[FrameNmsBackfill] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected,
            "strict_selected": self.strict_selected,
            "suppressed": [item.to_dict() for item in self.suppressed],
            "relaxed_backfill": [item.to_dict() for item in self.relaxed_backfill],
        }


def hard_nms_ordered_views(
    ordered_view_ids: Sequence[int],
    *,
    max_views: int,
    overlap_fn: Callable[[int, int], float],
    score_fn: Callable[[int], float] | None = None,
    overlap_threshold: float = 0.75,
    allow_relaxed_backfill: bool = True,
) -> FrameNmsResult:
    """Suppress near-duplicate camera views while preserving selector order.

    The strict pass keeps a view only when its overlap with every already-kept
    view is <= ``overlap_threshold``. If strict NMS leaves fewer than
    ``max_views`` frames, a second pass backfills the least-overlapping remaining
    candidates so downstream agents still receive the expected evidence budget.
    """
    if max_views <= 0:
        raise ValueError("max_views must be positive")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in [0, 1]")

    candidates = _unique_ints(ordered_view_ids)
    score_fn = score_fn or (lambda _view_id: 0.0)

    selected: list[int] = []
    suppressed: list[FrameNmsSuppression] = []
    rejected: list[int] = []

    for view_id in candidates:
        if len(selected) >= max_views:
            rejected.append(view_id)
            continue

        suppressor, overlap = _max_overlap(view_id, selected, overlap_fn)
        if suppressor is not None and overlap > overlap_threshold:
            suppressed.append(
                FrameNmsSuppression(
                    view_id=view_id,
                    suppressed_by=suppressor,
                    overlap=overlap,
                    score=float(score_fn(view_id)),
                )
            )
            rejected.append(view_id)
            continue

        selected.append(view_id)

    strict_selected = list(selected)
    relaxed_backfill: list[FrameNmsBackfill] = []
    if allow_relaxed_backfill and len(selected) < max_views:
        selected_set = set(selected)
        remaining = [view_id for view_id in candidates if view_id not in selected_set]
        ranked_remaining = []
        for view_id in remaining:
            _suppressor, max_overlap = _max_overlap(view_id, selected, overlap_fn)
            ranked_remaining.append(
                (
                    max_overlap,
                    -float(score_fn(view_id)),
                    candidates.index(view_id),
                    view_id,
                )
            )
        ranked_remaining.sort()
        for max_overlap, _neg_score, _index, view_id in ranked_remaining:
            if len(selected) >= max_views:
                break
            selected.append(view_id)
            relaxed_backfill.append(
                FrameNmsBackfill(
                    view_id=view_id,
                    max_overlap=float(max_overlap),
                    score=float(score_fn(view_id)),
                )
            )

    backfilled_view_ids = {item.view_id for item in relaxed_backfill}
    final_suppressed = [
        item for item in suppressed if item.view_id not in backfilled_view_ids
    ]

    return FrameNmsResult(
        selected=selected[:max_views],
        strict_selected=strict_selected,
        suppressed=final_suppressed,
        relaxed_backfill=relaxed_backfill,
    )


def _unique_ints(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _max_overlap(
    view_id: int,
    selected: Sequence[int],
    overlap_fn: Callable[[int, int], float],
) -> tuple[int | None, float]:
    best_view: int | None = None
    best_overlap = 0.0
    for selected_view in selected:
        overlap = float(overlap_fn(view_id, selected_view))
        if overlap > best_overlap:
            best_view = selected_view
            best_overlap = overlap
    return best_view, best_overlap
