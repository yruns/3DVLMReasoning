"""Tests for NR3D leaderboard-track metrics aggregator.

Covers the pure ``aggregate`` function plus the small predicate helpers
``is_easy`` and ``is_view_dep``. The IO wrapper ``compute_leaderboard_metrics``
is exercised via a smoke run in Task 2; this file deliberately keeps tests
loader-free so they run in milliseconds.
"""

from __future__ import annotations

import pytest

from evaluation.scripts.nr3d_leaderboard_metrics import (
    _VIEW_DEP_TOKENS,
    aggregate,
    is_easy,
    is_view_dep,
    load_requested_sample_ids,
    restrict_to_sample_ids,
)


def _meta(
    sid: str,
    target: int,
    n_obj: int,
    tokens: list[str],
    mtc: bool = True,
) -> dict:
    return {
        "sample_id": sid,
        "target_id": target,
        "n_objects": n_obj,
        "tokens": tokens,
        "mentions_target_class": mtc,
    }


def test_view_dep_token_set_matches_referit3d():
    """Source: referit3d/analysis/utterances.py:103-105."""
    assert _VIEW_DEP_TOKENS == frozenset({
        "front", "behind", "back", "right", "left",
        "facing", "leftmost", "rightmost", "looking", "across",
    })


def test_is_easy_boundary():
    assert is_easy(2) is True
    assert is_easy(3) is False
    assert is_easy(1) is True


def test_is_view_dep_per_token_not_substring():
    assert is_view_dep(["front", "of", "table"]) is True
    assert is_view_dep(["frontier", "of", "table"]) is False
    assert is_view_dep(["the", "table"]) is False
    assert is_view_dep([]) is False


def test_is_view_dep_uppercase_does_not_match():
    """Upstream NR3D CSV tokens are pre-lowercased; uppercase MUST NOT match."""
    assert is_view_dep(["FRONT"]) is False
    assert is_view_dep(["Front"]) is False


def test_aggregate_all_correct():
    predictions = {"a": 0, "b": 1, "c": 2, "d": 3}
    meta = [
        _meta("a", 0, 2, ["table"]),
        _meta("b", 1, 2, ["chair"]),
        _meta("c", 2, 3, ["the", "front", "lamp"]),
        _meta("d", 3, 3, ["the", "left", "couch"]),
    ]
    filt = {"a", "b", "c", "d"}
    m = aggregate(predictions, meta, filt)
    assert m["n_full"] == 4
    assert m["n_filtered"] == 4
    assert m["classification_acc_full"] == 1.0
    assert m["classification_acc_filtered"] == 1.0
    assert m["acc_easy"] == 1.0
    assert m["acc_hard"] == 1.0
    assert m["acc_view_dep"] == 1.0
    assert m["acc_view_indep"] == 1.0


def test_aggregate_all_wrong():
    predictions = {"a": 99, "b": 99}
    meta = [
        _meta("a", 0, 2, ["table"]),
        _meta("b", 1, 3, ["the", "left", "lamp"]),
    ]
    filt = {"a", "b"}
    m = aggregate(predictions, meta, filt)
    assert m["classification_acc_full"] == 0.0
    assert m["classification_acc_filtered"] == 0.0
    assert m["acc_easy"] == 0.0
    assert m["acc_hard"] == 0.0


def test_aggregate_easy_hard_partition():
    predictions = {"a": 0, "b": 0, "c": 0, "d": 0}
    meta = [
        _meta("a", 0, 2, ["x"]),    # easy, correct
        _meta("b", 0, 2, ["y"]),    # easy, correct
        _meta("c", 5, 3, ["z"]),    # hard, wrong (target=5, pred=0)
        _meta("d", 5, 5, ["w"]),    # hard, wrong
    ]
    filt = {"a", "b", "c", "d"}
    m = aggregate(predictions, meta, filt)
    assert m["n_easy"] == 2
    assert m["n_hard"] == 2
    assert m["acc_easy"] == 1.0
    assert m["acc_hard"] == 0.0


def test_aggregate_view_dep_partition():
    predictions = {"a": 0, "b": 0, "c": 0, "d": 0}
    meta = [
        _meta("a", 0, 2, ["the", "left", "chair"]),    # v-dep, correct
        _meta("b", 5, 2, ["the", "right", "table"]),   # v-dep, wrong
        _meta("c", 0, 2, ["the", "tall", "lamp"]),     # v-indep, correct
        _meta("d", 5, 2, ["a", "small", "vase"]),      # v-indep, wrong
    ]
    filt = {"a", "b", "c", "d"}
    m = aggregate(predictions, meta, filt)
    assert m["n_view_dep"] == 2
    assert m["n_view_indep"] == 2
    assert m["acc_view_dep"] == 0.5
    assert m["acc_view_indep"] == 0.5


def test_aggregate_filtered_subset_of_full():
    predictions = {"a": 0, "b": 99}
    meta = [
        _meta("a", 0, 2, ["table"], mtc=True),
        _meta("b", 0, 2, ["table"], mtc=False),
    ]
    filt = {"a"}
    m = aggregate(predictions, meta, filt)
    assert m["n_full"] == 2
    assert m["n_filtered"] == 1
    assert m["classification_acc_full"] == 0.5
    assert m["classification_acc_filtered"] == 1.0


def test_aggregate_raises_on_missing_filtered_sample():
    """Filtered fold sample MUST be in side_by_side; missing → raise."""
    predictions = {"a": 0}
    meta = [
        _meta("a", 0, 2, ["table"]),
        _meta("b", 1, 2, ["chair"]),
    ]
    filt = {"a", "b"}
    with pytest.raises(ValueError, match="missing"):
        aggregate(predictions, meta, filt)


def test_aggregate_includes_failed_predictions_as_wrong():
    """v2 sentinel failures (selected_object_id=None) count as is_correct=False."""
    predictions = {"a": None, "b": 1}
    meta = [
        _meta("a", 0, 2, ["table"]),
        _meta("b", 1, 2, ["chair"]),
    ]
    filt = {"a", "b"}
    m = aggregate(predictions, meta, filt)
    assert m["classification_acc_filtered"] == 0.5


def test_aggregate_per_sample_complete():
    predictions = {"a": 0}
    meta = [_meta("a", 0, 2, ["the", "left", "chair"])]
    filt = {"a"}
    m = aggregate(predictions, meta, filt)
    assert len(m["per_sample"]) == 1
    s = m["per_sample"][0]
    assert s["sample_id"] == "a"
    assert s["selected_object_id"] == 0
    assert s["target_id"] == 0
    assert s["is_correct"] is True
    assert s["is_easy"] is True
    assert s["is_view_dep"] is True
    assert s["is_filtered_out"] is False


def test_aggregate_invariants():
    """n_easy+n_hard==n_filtered; n_view_dep+n_view_indep==n_filtered."""
    predictions = {"a": 0, "b": 0, "c": 0}
    meta = [
        _meta("a", 0, 2, ["left"]),
        _meta("b", 0, 3, ["x"]),
        _meta("c", 0, 4, ["y"], mtc=False),
    ]
    filt = {"a", "b"}
    m = aggregate(predictions, meta, filt)
    assert m["n_easy"] + m["n_hard"] == m["n_filtered"] == 2
    assert m["n_view_dep"] + m["n_view_indep"] == m["n_filtered"] == 2


def test_aggregate_with_subset_scores_only_requested_sample_ids():
    predictions = {"a": 0, "b": 99, "c": 2}
    meta = [
        _meta("a", 0, 2, ["table"]),
        _meta("b", 1, 2, ["chair"]),
        _meta("c", 2, 2, ["lamp"]),
    ]
    subset_predictions, subset_meta, subset_filtered = restrict_to_sample_ids(
        predictions,
        meta,
        filtered_sample_ids={"a", "b", "c"},
        requested_sample_ids={"a", "c"},
    )
    m = aggregate(subset_predictions, subset_meta, subset_filtered)

    assert m["n_full"] == 2
    assert m["n_filtered"] == 2
    assert m["classification_acc_filtered"] == 1.0
    assert {row["sample_id"] for row in m["per_sample"]} == {"a", "c"}


def test_restrict_to_sample_ids_preserves_sample_meta_order():
    predictions = {"a": 0, "b": 1, "c": 2}
    meta = [
        _meta("c", 2, 2, ["lamp"]),
        _meta("a", 0, 2, ["table"]),
        _meta("b", 1, 2, ["chair"]),
    ]

    _, subset_meta, _ = restrict_to_sample_ids(
        predictions,
        meta,
        filtered_sample_ids={"a", "b", "c"},
        requested_sample_ids={"a", "c"},
    )

    assert [row["sample_id"] for row in subset_meta] == ["c", "a"]


def test_restrict_to_sample_ids_requires_prediction_for_requested_id():
    with pytest.raises(ValueError, match="missing requested sample_ids"):
        restrict_to_sample_ids(
            predictions={"a": 0},
            sample_meta=[_meta("a", 0, 2, ["x"])],
            filtered_sample_ids={"a"},
            requested_sample_ids={"a", "b"},
        )


def test_restrict_to_sample_ids_requires_metadata_for_requested_id():
    with pytest.raises(ValueError, match="missing requested sample_ids"):
        restrict_to_sample_ids(
            predictions={"a": 0, "b": 1},
            sample_meta=[_meta("a", 0, 2, ["x"])],
            filtered_sample_ids={"a"},
            requested_sample_ids={"a", "b"},
        )


def test_load_requested_sample_ids_accepts_string_list(tmp_path):
    path = tmp_path / "sample_ids.json"
    path.write_text('["a", "b"]', encoding="utf-8")

    assert load_requested_sample_ids(path) == {"a", "b"}


def test_load_requested_sample_ids_accepts_object_list(tmp_path):
    path = tmp_path / "sample_ids.json"
    path.write_text(
        '[{"sample_id": "a"}, {"sample_id": "b", "extra": true}]',
        encoding="utf-8",
    )

    assert load_requested_sample_ids(path) == {"a", "b"}


def test_load_requested_sample_ids_rejects_duplicate_string_entries(tmp_path):
    path = tmp_path / "sample_ids.json"
    path.write_text('["a", "b", "a"]', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id.*a.*0.*2"):
        load_requested_sample_ids(path)


def test_load_requested_sample_ids_rejects_duplicate_mixed_entries(tmp_path):
    path = tmp_path / "sample_ids.json"
    path.write_text('["a", {"sample_id": "b"}, {"sample_id": "a"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sample_id.*a.*0.*2"):
        load_requested_sample_ids(path)


@pytest.mark.parametrize("raw", ['{"sample_id": "a"}', "[]", '[""]', "[{}]", "[1]"])
def test_load_requested_sample_ids_rejects_invalid_json_shapes(tmp_path, raw):
    path = tmp_path / "sample_ids.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError):
        load_requested_sample_ids(path)
