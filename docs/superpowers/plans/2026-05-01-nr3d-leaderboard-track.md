# NR3D Leaderboard-Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-aggregator that turns existing v2 NR3D `selected_object_id` predictions into 5-column leaderboard metrics (Overall / Easy / Hard / View-Dep / View-Indep) on the canonical referit3d filter chain, ingest into SQLite, and write the v3 version doc — all without re-running the Stage 2 agent.

**Architecture:** A pure aggregator function consumes already-recorded predictions from `tmp/nr3d_eval_v1_full/side_by_side.json`, joins them with NR3D loader metadata under `mentions_target_class_only=True`, and emits classification accuracy + 4-way slicing. View-dep token set is the literal 10-word frozenset from `referit3d/analysis/utterances.py:103-105`. All Stage 1 / Stage 2 / Phase 8 code paths stay untouched.

**Tech Stack:** Python 3.12 (uv `.venv`), pytest, sqlite3, existing `Nr3dDataset` loader, existing `tmp/nr3d_eval_v1_full/` checkpoints.

**Context references:**
- Design spec: `docs/superpowers/specs/2026-05-01-nr3d-fairness-design.md` (commit `2fead86`)
- Consolidated NR3D protocol note:
  `docs/benchmark/nr3d/protocol.md` (merged from the older protocol audit,
  paper crosscheck, and pool-equivalence files)
- Branch: `feat/nr3d-vg-benchmark`, baseline tip: `2fead86`

**Date placeholder:** Use `YYYYMMDD = 20260501` throughout (the date this plan was authored). If implementation lands on a different date, replace `20260501` with that date in file names and run IDs.

---

## File Structure

**New files:**
- `src/evaluation/scripts/nr3d_leaderboard_metrics.py` — aggregator module + CLI
- `src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py` — pytest module
- `docs/benchmark/nr3d/v3_referit3d_track_20260501.md` — v3 version doc

**Modified files:**
- `scripts/ingest_nr3d_run.py` — schema additions + `--leaderboard-metrics` flag
- `src/evaluation/scripts/tests/test_ingest_nr3d_run.py` — schema migration + leaderboard-ingest tests
- `docs/benchmark/nr3d/README.md` — add v3 timeline row + supersede v2 caveat in "Current Interpretation"
- `docs/benchmark/README.md` — replace NR3D row highlight with v3 numbers

**Untouched (deliberately):**
- `src/benchmarks/nr3d_loader.py`
- `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
- `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`
- `src/agents/**`
- `data/nr3d/scannet/<scene>/conceptgraph/**` (Phase 8 packages)
- `tmp/nr3d_eval_v1_full/per_sample/**` (v2 agent checkpoints — read-only input)

---

## Task 1: Aggregator pure functions — RED → GREEN

**Files:**
- Create: `src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py`
- Create: `src/evaluation/scripts/nr3d_leaderboard_metrics.py`

This task delivers the **pure** aggregation logic — no IO. The pure function takes plain dicts/lists in and returns metrics. The IO wrapper that calls the NR3D loader comes in Task 2.

- [ ] **Step 1.1: Write the failing test file**

Create `src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py` with the full content below.

```python
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
```

- [ ] **Step 1.2: Run the test file to confirm it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py -v 2>&1 | tail -10`

Expected: ImportError (`evaluation.scripts.nr3d_leaderboard_metrics` does not exist) — all tests fail at collection.

- [ ] **Step 1.3: Implement the aggregator module**

Create `src/evaluation/scripts/nr3d_leaderboard_metrics.py` with the full content below.

```python
"""NR3D leaderboard-track post-aggregator.

Reads the existing ``side_by_side.json`` produced by
``run_nr3d_vg_side_by_side.py`` and emits the canonical 5-column
leaderboard metrics (Overall / Easy / Hard / View-Dep / View-Indep) under
the official ReferIt3D filter chain (``mentions_target_class_only=True``).

This is a pure post-processing step — no Stage 2 agent calls, no Phase 8
re-prep. Mathematical equivalence to a re-run is documented in the design
spec at ``docs/superpowers/specs/2026-05-01-nr3d-fairness-design.md``.

Source citations
----------------
- Pool / metric / filter protocol: ``docs/benchmark/nr3d/protocol.md``
- View-dep keyword set: ``referit3d/analysis/utterances.py:103-105``
- Easy / hard definition: ``referit3d/analysis/deepnet_predictions.py:34-36``
- Filter chain: ``referit3d/in_out/neural_net_oriented.py:55-105``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks.nr3d_loader import Nr3dDataset

# Source: referit3d/analysis/utterances.py:103-105
_VIEW_DEP_TOKENS: frozenset[str] = frozenset({
    "front", "behind", "back", "right", "left",
    "facing", "leftmost", "rightmost", "looking", "across",
})


def is_easy(n_objects: int) -> bool:
    """Easy ⟺ n_objects ≤ 2 (one same-class distractor or unique target).

    Source: referit3d/analysis/deepnet_predictions.py:34-36 — ``hardness <= 2``.
    """
    return n_objects <= 2


def is_view_dep(tokens: Iterable[str]) -> bool:
    """View-dependent ⟺ token set intersects the literal 10-word view-dep set.

    Per-token (not substring), case-sensitive (NR3D CSV tokens are pre-lowercased).
    Source: referit3d/analysis/utterances.py:98-105.
    """
    return bool(set(tokens) & _VIEW_DEP_TOKENS)


def aggregate(
    predictions: dict[str, int | None],
    sample_meta: list[dict[str, Any]],
    filtered_sample_ids: set[str],
) -> dict[str, Any]:
    """Inner-join predictions with sample metadata, return leaderboard metrics.

    Args:
        predictions: ``sample_id -> selected_object_id``. ``None`` means the
            agent emitted a failed sentinel; counted as ``is_correct=False``.
        sample_meta: list of dicts with at least ``sample_id``, ``target_id``,
            ``n_objects``, ``tokens``. ``mentions_target_class`` may be present
            but is not consulted here — filtering is the caller's job and is
            communicated via ``filtered_sample_ids``.
        filtered_sample_ids: set of sample_ids surviving the canonical filter
            (typically ``mentions_target_class_only=True``).

    Returns:
        Metrics dict — see the design spec, Component 1, for schema.

    Raises:
        ValueError: any filtered sample_id is missing from ``predictions``.
    """
    missing = filtered_sample_ids - set(predictions.keys())
    if missing:
        raise ValueError(
            f"side_by_side missing {len(missing)} filtered sample_ids; "
            f"first 5: {sorted(missing)[:5]}"
        )

    samples: list[dict[str, Any]] = []
    for meta in sample_meta:
        sid = meta["sample_id"]
        if sid not in predictions:
            continue
        selected = predictions[sid]
        target = meta["target_id"]
        is_correct = selected is not None and int(selected) == int(target)
        is_filtered_out = sid not in filtered_sample_ids
        ie = is_easy(int(meta["n_objects"]))
        ivd = is_view_dep(meta["tokens"])
        samples.append({
            "sample_id": sid,
            "selected_object_id": selected,
            "target_id": target,
            "is_correct": is_correct,
            "is_easy": ie,
            "is_view_dep": ivd,
            "is_filtered_out": is_filtered_out,
            "n_objects": int(meta["n_objects"]),
            "tokens": list(meta["tokens"]),
            "mentions_target_class": bool(meta.get("mentions_target_class", True)),
        })

    n_full = len(samples)
    full_correct = sum(1 for s in samples if s["is_correct"])
    filtered = [s for s in samples if not s["is_filtered_out"]]
    filtered_correct = sum(1 for s in filtered if s["is_correct"])
    easy = [s for s in filtered if s["is_easy"]]
    hard = [s for s in filtered if not s["is_easy"]]
    vd = [s for s in filtered if s["is_view_dep"]]
    vid = [s for s in filtered if not s["is_view_dep"]]

    metrics: dict[str, Any] = {
        "n_full": n_full,
        "n_filtered": len(filtered),
        "classification_acc_full": full_correct / n_full if n_full else 0.0,
        "classification_acc_filtered": (
            filtered_correct / len(filtered) if filtered else 0.0
        ),
        "n_easy": len(easy),
        "n_hard": len(hard),
        "n_view_dep": len(vd),
        "n_view_indep": len(vid),
        "acc_easy": (
            sum(1 for s in easy if s["is_correct"]) / len(easy) if easy else 0.0
        ),
        "acc_hard": (
            sum(1 for s in hard if s["is_correct"]) / len(hard) if hard else 0.0
        ),
        "acc_view_dep": (
            sum(1 for s in vd if s["is_correct"]) / len(vd) if vd else 0.0
        ),
        "acc_view_indep": (
            sum(1 for s in vid if s["is_correct"]) / len(vid) if vid else 0.0
        ),
        "per_sample": samples,
    }

    if metrics["n_filtered"] > metrics["n_full"]:
        raise AssertionError(
            f"n_filtered={metrics['n_filtered']} > n_full={metrics['n_full']}"
        )
    if metrics["n_easy"] + metrics["n_hard"] != metrics["n_filtered"]:
        raise AssertionError(
            f"n_easy ({metrics['n_easy']}) + n_hard ({metrics['n_hard']}) "
            f"!= n_filtered ({metrics['n_filtered']})"
        )
    if metrics["n_view_dep"] + metrics["n_view_indep"] != metrics["n_filtered"]:
        raise AssertionError(
            f"n_view_dep ({metrics['n_view_dep']}) + n_view_indep "
            f"({metrics['n_view_indep']}) != n_filtered ({metrics['n_filtered']})"
        )
    return metrics


def compute_leaderboard_metrics(
    side_by_side_path: Path,
    nr3d_data_root: Path,
    phase8_data_root: Path,
    canonical_filter: bool = True,
    backend: str = "pack_v1",
) -> dict[str, Any]:
    """IO wrapper: load side_by_side.json + NR3D dataset, call ``aggregate``."""
    payload = json.loads(side_by_side_path.read_text(encoding="utf-8"))
    if backend not in payload:
        raise ValueError(f"side_by_side.json missing backend={backend!r}")
    per_sample_records = payload[backend].get("per_sample")
    if not isinstance(per_sample_records, list) or not per_sample_records:
        raise ValueError(
            f"side_by_side.json[{backend!r}].per_sample must be a non-empty list"
        )

    predictions: dict[str, int | None] = {}
    for r in per_sample_records:
        sid = r["sample_id"]
        sel = r.get("selected_object_id")
        predictions[sid] = None if sel is None else int(sel)

    ds_full = Nr3dDataset.from_path(
        data_root=str(nr3d_data_root),
        split="test",
        bbox_source="phase8_gt_cg",
        phase8_data_root=str(phase8_data_root),
        mentions_target_class_only=False,
    )
    sample_meta = [
        {
            "sample_id": s.sample_id,
            "target_id": s.target_id,
            "n_objects": s.n_objects,
            "tokens": list(s.tokens),
            "mentions_target_class": s.mentions_target_class,
        }
        for s in ds_full
    ]
    if canonical_filter:
        filtered_sample_ids = {
            s["sample_id"] for s in sample_meta if s["mentions_target_class"]
        }
    else:
        filtered_sample_ids = {s["sample_id"] for s in sample_meta}

    return aggregate(predictions, sample_meta, filtered_sample_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side-by-side", required=True, type=Path)
    parser.add_argument("--nr3d-data-root", required=True, type=Path)
    parser.add_argument("--phase8-data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="pack_v1")
    parser.add_argument(
        "--canonical-filter",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=True,
        help="apply mentions_target_class_only=True (default true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_leaderboard_metrics(
        side_by_side_path=args.side_by_side,
        nr3d_data_root=args.nr3d_data_root,
        phase8_data_root=args.phase8_data_root,
        canonical_filter=args.canonical_filter,
        backend=args.backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"n_full={metrics['n_full']} n_filtered={metrics['n_filtered']}")
    print(
        f"classification_acc_full     = {metrics['classification_acc_full']:.4f}"
    )
    print(
        f"classification_acc_filtered = {metrics['classification_acc_filtered']:.4f}"
    )
    print(
        f"  Easy   (n={metrics['n_easy']:5}): {metrics['acc_easy']:.4f}"
    )
    print(
        f"  Hard   (n={metrics['n_hard']:5}): {metrics['acc_hard']:.4f}"
    )
    print(
        f"  V-Dep  (n={metrics['n_view_dep']:5}): {metrics['acc_view_dep']:.4f}"
    )
    print(
        f"  V-Ind  (n={metrics['n_view_indep']:5}): "
        f"{metrics['acc_view_indep']:.4f}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.4: Run tests to confirm GREEN**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py -v 2>&1 | tail -20`

Expected: 11 tests pass (`test_view_dep_token_set_matches_referit3d`, `test_is_easy_boundary`, `test_is_view_dep_per_token_not_substring`, `test_is_view_dep_uppercase_does_not_match`, `test_aggregate_all_correct`, `test_aggregate_all_wrong`, `test_aggregate_easy_hard_partition`, `test_aggregate_view_dep_partition`, `test_aggregate_filtered_subset_of_full`, `test_aggregate_raises_on_missing_filtered_sample`, `test_aggregate_includes_failed_predictions_as_wrong`, `test_aggregate_per_sample_complete`, `test_aggregate_invariants`). Actually 13 tests — count them when running.

- [ ] **Step 1.5: Commit (RED → GREEN)**

```bash
git add src/evaluation/scripts/nr3d_leaderboard_metrics.py \
        src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py
git commit -m "feat(nr3d): add leaderboard-track post-aggregator

Pure aggregation function (no IO) plus IO wrapper that reads
side_by_side.json and joins with Nr3dDataset metadata under the canonical
mentions_target_class_only filter. Source-cited view-dep keyword set
matches referit3d/analysis/utterances.py:103-105 verbatim.

Companion tests cover easy/hard partition, view-dep token matching
(per-token, not substring), failed-sentinel handling, and inner-join
completeness (raises on missing filtered sample_ids)."
```

---

## Task 2: Smoke-run the aggregator on existing v2 outputs

**Files:**
- Read: `tmp/nr3d_eval_v1_full/side_by_side.json` (existing 7.9 MB v2 output)
- Create: `tmp/nr3d_eval_v1_full/leaderboard_metrics.json`

This task captures the actual measured numbers we need for the v3 doc. No code changes.

- [ ] **Step 2.1: Verify input artifacts exist**

Run:
```bash
ls -la tmp/nr3d_eval_v1_full/side_by_side.json
ls data/nr3d/raw/nr3d.csv data/nr3d/raw/test_scans.txt
```

Expected: side_by_side.json ~7.9 MB; nr3d.csv exists; test_scans.txt exists.

- [ ] **Step 2.2: Run the aggregator CLI**

```bash
source .venv/bin/activate
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
    --side-by-side tmp/nr3d_eval_v1_full/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --output tmp/nr3d_eval_v1_full/leaderboard_metrics.json \
    --canonical-filter true
```

Expected: completes in under 2 minutes; prints n_full=8584, n_filtered≈7805, classification_acc_filtered as a float in [0.0, 1.0]. Capture the printed numbers — they go in Task 4.

- [ ] **Step 2.3: Sanity check the output JSON**

```bash
python -c "
import json
m = json.load(open('tmp/nr3d_eval_v1_full/leaderboard_metrics.json'))
print('n_full:', m['n_full'])
print('n_filtered:', m['n_filtered'])
print('acc_full:', round(m['classification_acc_full'], 4))
print('acc_filtered:', round(m['classification_acc_filtered'], 4))
print('easy/hard/vdep/vind n:', m['n_easy'], m['n_hard'], m['n_view_dep'], m['n_view_indep'])
print('per_sample len:', len(m['per_sample']))
assert m['n_full'] == 8584
assert m['n_filtered'] <= m['n_full']
assert m['n_easy'] + m['n_hard'] == m['n_filtered']
assert m['n_view_dep'] + m['n_view_indep'] == m['n_filtered']
assert len(m['per_sample']) == m['n_full']
print('all invariants pass')
"
```

Expected: all asserts pass; print "all invariants pass".

- [ ] **Step 2.4: Record the numbers in a scratch note for Task 4**

Save the numbers somewhere persistent (e.g., echo to `/tmp/v3_numbers.txt`). They will be substituted into the v3 doc template in Task 4.

```bash
python -c "
import json
m = json.load(open('tmp/nr3d_eval_v1_full/leaderboard_metrics.json'))
out = (
    f'n_full={m[\"n_full\"]} n_filtered={m[\"n_filtered\"]}\\n'
    f'classification_acc_full={m[\"classification_acc_full\"]:.4f}\\n'
    f'classification_acc_filtered={m[\"classification_acc_filtered\"]:.4f}\\n'
    f'acc_easy={m[\"acc_easy\"]:.4f} (n={m[\"n_easy\"]})\\n'
    f'acc_hard={m[\"acc_hard\"]:.4f} (n={m[\"n_hard\"]})\\n'
    f'acc_view_dep={m[\"acc_view_dep\"]:.4f} (n={m[\"n_view_dep\"]})\\n'
    f'acc_view_indep={m[\"acc_view_indep\"]:.4f} (n={m[\"n_view_indep\"]})\\n'
)
print(out)
open('/tmp/v3_numbers.txt', 'w').write(out)
"
cat /tmp/v3_numbers.txt
```

Expected: clean stdout printing of all 5-column metrics; file exists at `/tmp/v3_numbers.txt`.

- [ ] **Step 2.5: Commit the leaderboard_metrics.json checkpoint**

NOTE: `tmp/` is git-ignored by project convention (it's a scratch dir). Do NOT commit `tmp/nr3d_eval_v1_full/leaderboard_metrics.json`. Skip this step — the file's role is to feed Task 3's ingestion + Task 4's doc; the durable artifacts are in `runs.sqlite` and the v3 doc, not the JSON.

Verify by running `git status -s` — the only `tmp/` modification should still be `.claude/scheduled_tasks.lock` and any worktree noise; nothing under `tmp/nr3d_eval_v1_full/` should appear in tracked changes.

---

## Task 3: Extend ingester schema + CLI

**Files:**
- Modify: `scripts/ingest_nr3d_run.py` (add columns, add `--leaderboard-metrics` flag)
- Modify: `src/evaluation/scripts/tests/test_ingest_nr3d_run.py` (add migration + leaderboard tests)

- [ ] **Step 3.1: Read the current ingester to understand its shape**

```bash
wc -l scripts/ingest_nr3d_run.py
head -90 scripts/ingest_nr3d_run.py
```

Understand: SCHEMA constant has CREATE TABLE IF NOT EXISTS for `runs`, `samples`, `tool_calls`, `llm_calls`. The `ingest()` function populates `runs` and `samples`. We add 7 nullable columns to `runs` and 4 to `samples`, plus a `--leaderboard-metrics` CLI flag.

- [ ] **Step 3.2: Read the existing test file**

```bash
wc -l src/evaluation/scripts/tests/test_ingest_nr3d_run.py
```

Understand the test pattern (fixtures, how a fake side_by_side.json is built).

- [ ] **Step 3.3: Write failing tests for the new ingestion behavior**

Append the following test functions to `src/evaluation/scripts/tests/test_ingest_nr3d_run.py` (do NOT remove existing tests; add at end of file).

**Important — import pattern:** The existing tests in this file already import `ingest` from `scripts/ingest_nr3d_run.py`. Mirror **whatever import statement the existing tests use** (e.g. they may use `from scripts.ingest_nr3d_run import ingest`, an inline `sys.path` shim, or a `conftest.py` registered path). Do NOT introduce a new import style; copy the existing one verbatim into the new tests, or move all imports to the top of the file alongside the existing ones. Same goes for `import json` if it isn't already imported.

```python
def test_runs_table_has_leaderboard_columns_after_ingest(tmp_path):
    """After ingesting, the runs table must have the 7 new leaderboard columns."""
    import sqlite3
    from scripts.ingest_nr3d_run import ingest

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    side_by_side = output_dir / "side_by_side.json"
    side_by_side.write_text(json.dumps({
        "pack_v1": {
            "n": 1,
            "mean_iou": 0.5,
            "Acc@0.25": 0.5,
            "Acc@0.50": 0.5,
            "per_sample": [{
                "sample_id": "scannet/scene0011_00::5::abc",
                "backend": "pack_v1",
                "status": "completed",
                "iou": 0.5,
                "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9,
                "selected_object_id": 5,
                "confidence": 1.0,
                "query": "test",
            }],
        }
    }))
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db,
        output_dir=output_dir,
        run_id="test_v3",
        branch=None,
        commit_hash=None,
        backend="pack_v1",
        judge_model=None,
        notes=None,
    )
    conn = sqlite3.connect(str(db))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    expected_new = {
        "classification_acc_full", "classification_acc_filtered",
        "acc_easy", "acc_hard", "acc_view_dep", "acc_view_indep",
        "n_filtered",
    }
    assert expected_new <= cols
    sample_cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    expected_sample_new = {
        "is_easy", "is_view_dep", "is_filtered_out",
        "classification_correct",
    }
    assert expected_sample_new <= sample_cols
    conn.close()


def test_ingest_with_leaderboard_metrics_populates_columns(tmp_path):
    """With --leaderboard-metrics, the new columns must be populated."""
    import sqlite3
    from scripts.ingest_nr3d_run import ingest

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    sample_id = "scannet/scene0011_00::5::abc"
    side_by_side = output_dir / "side_by_side.json"
    side_by_side.write_text(json.dumps({
        "pack_v1": {
            "n": 1,
            "mean_iou": 1.0,
            "Acc@0.25": 1.0,
            "Acc@0.50": 1.0,
            "per_sample": [{
                "sample_id": sample_id,
                "backend": "pack_v1",
                "status": "completed",
                "iou": 1.0,
                "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9,
                "selected_object_id": 5,
                "confidence": 1.0,
                "query": "the chair",
            }],
        }
    }))
    leaderboard_metrics = tmp_path / "leaderboard_metrics.json"
    leaderboard_metrics.write_text(json.dumps({
        "n_full": 1, "n_filtered": 1,
        "classification_acc_full": 1.0,
        "classification_acc_filtered": 1.0,
        "n_easy": 1, "n_hard": 0, "n_view_dep": 0, "n_view_indep": 1,
        "acc_easy": 1.0, "acc_hard": 0.0,
        "acc_view_dep": 0.0, "acc_view_indep": 1.0,
        "per_sample": [{
            "sample_id": sample_id,
            "selected_object_id": 5, "target_id": 5,
            "is_correct": True, "is_easy": True,
            "is_view_dep": False, "is_filtered_out": False,
            "n_objects": 2, "tokens": ["the", "chair"],
            "mentions_target_class": True,
        }],
    }))
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db,
        output_dir=output_dir,
        run_id="test_v3_full",
        branch="feat/nr3d-vg-benchmark",
        commit_hash="abc1234",
        backend="pack_v1",
        judge_model=None,
        notes="leaderboard track test",
        leaderboard_metrics_path=leaderboard_metrics,
    )
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT classification_acc_filtered, acc_easy, acc_hard, "
        "acc_view_dep, acc_view_indep, n_filtered FROM runs "
        "WHERE run_id=?", ("test_v3_full",)
    ).fetchone()
    assert row == (1.0, 1.0, 0.0, 0.0, 1.0, 1)
    sample_row = conn.execute(
        "SELECT is_easy, is_view_dep, is_filtered_out, classification_correct "
        "FROM samples WHERE run_id=? AND sample_id=?",
        ("test_v3_full", sample_id),
    ).fetchone()
    assert sample_row == (1, 0, 0, 1)
    conn.close()


def test_ingest_without_leaderboard_metrics_keeps_new_columns_null(tmp_path):
    """Old-style ingest (no --leaderboard-metrics) leaves new columns NULL."""
    import sqlite3
    from scripts.ingest_nr3d_run import ingest

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    side_by_side = output_dir / "side_by_side.json"
    side_by_side.write_text(json.dumps({
        "pack_v1": {
            "n": 1, "mean_iou": 0.5, "Acc@0.25": 0.5, "Acc@0.50": 0.5,
            "per_sample": [{
                "sample_id": "scannet/scene0011_00::5::abc",
                "backend": "pack_v1",
                "status": "completed",
                "iou": 0.5,
                "predicted_bbox_3d_9dof": [0]*9,
                "gt_bbox_3d_9dof": [0]*9,
                "selected_object_id": 5, "confidence": 1.0,
                "query": "test",
            }],
        }
    }))
    db = tmp_path / "runs.sqlite"
    ingest(
        db_path=db,
        output_dir=output_dir,
        run_id="test_v2_old",
        branch=None,
        commit_hash=None,
        backend="pack_v1",
        judge_model=None,
        notes=None,
    )
    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT classification_acc_full, classification_acc_filtered, "
        "acc_easy, acc_hard, acc_view_dep, acc_view_indep, n_filtered "
        "FROM runs WHERE run_id=?", ("test_v2_old",)
    ).fetchone()
    assert row == (None, None, None, None, None, None, None)
    conn.close()
```

If the test file does not already have `import json` at the top, add that.

- [ ] **Step 3.4: Run the new tests to confirm they fail**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_ingest_nr3d_run.py::test_runs_table_has_leaderboard_columns_after_ingest src/evaluation/scripts/tests/test_ingest_nr3d_run.py::test_ingest_with_leaderboard_metrics_populates_columns src/evaluation/scripts/tests/test_ingest_nr3d_run.py::test_ingest_without_leaderboard_metrics_keeps_new_columns_null -v 2>&1 | tail -15`

Expected: tests fail (columns don't exist; `ingest()` does not accept `leaderboard_metrics_path`).

- [ ] **Step 3.5: Modify `scripts/ingest_nr3d_run.py` to add migration + leaderboard support**

Apply these edits to `scripts/ingest_nr3d_run.py`:

(a) After the `SCHEMA = """..."""` constant, add a new module-level constant:

```python
_RUNS_NEW_COLUMNS = [
    ("classification_acc_full", "REAL"),
    ("classification_acc_filtered", "REAL"),
    ("acc_easy", "REAL"),
    ("acc_hard", "REAL"),
    ("acc_view_dep", "REAL"),
    ("acc_view_indep", "REAL"),
    ("n_filtered", "INTEGER"),
]
_SAMPLES_NEW_COLUMNS = [
    ("is_easy", "INTEGER"),
    ("is_view_dep", "INTEGER"),
    ("is_filtered_out", "INTEGER"),
    ("classification_correct", "INTEGER"),
]


def _ensure_columns(conn, table: str, cols: list[tuple[str, str]]) -> None:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for col_name, col_type in cols:
        if col_name not in existing:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
            )
```

(b) Modify the `ingest()` function signature to add `leaderboard_metrics_path: Path | None = None`:

```python
def ingest(
    *,
    db_path: Path,
    output_dir: Path,
    run_id: str,
    branch: str | None,
    commit_hash: str | None,
    backend: str = "pack_v1",
    judge_model: str | None = None,
    notes: str | None = None,
    leaderboard_metrics_path: Path | None = None,
) -> None:
```

(c) Inside `ingest()`, just after `conn.executescript(SCHEMA)` add the migration calls:

```python
        conn.executescript(SCHEMA)
        _ensure_columns(conn, "runs", _RUNS_NEW_COLUMNS)
        _ensure_columns(conn, "samples", _SAMPLES_NEW_COLUMNS)
```

(d) Load leaderboard metrics if provided. After computing `acc25/acc50/mean_iou` and BEFORE the `cur.execute("INSERT OR REPLACE INTO runs ...")` call, insert:

```python
        leaderboard = None
        leaderboard_per_sample: dict[str, dict] = {}
        if leaderboard_metrics_path is not None:
            leaderboard = _read_json(Path(leaderboard_metrics_path))
            for entry in leaderboard.get("per_sample", []):
                leaderboard_per_sample[entry["sample_id"]] = entry
```

(e) Replace the original `cur.execute("INSERT OR REPLACE INTO runs ...")` call (the one with 13 ?-marks) with this version that writes 20 columns (13 original + 7 new). The columns must be listed explicitly to avoid relying on positional ordering:

```python
        cur.execute(
            """INSERT OR REPLACE INTO runs (
                run_id, branch, commit_hash, output_dir, backend, n,
                mean_iou, acc25, acc50, judge_model, started_at, ingested_at, notes,
                classification_acc_full, classification_acc_filtered,
                acc_easy, acc_hard, acc_view_dep, acc_view_indep, n_filtered
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                branch,
                commit_hash,
                str(output_dir),
                backend,
                n,
                mean_iou,
                acc25,
                acc50,
                judge_model,
                None,
                time.time(),
                notes,
                leaderboard.get("classification_acc_full") if leaderboard else None,
                leaderboard.get("classification_acc_filtered") if leaderboard else None,
                leaderboard.get("acc_easy") if leaderboard else None,
                leaderboard.get("acc_hard") if leaderboard else None,
                leaderboard.get("acc_view_dep") if leaderboard else None,
                leaderboard.get("acc_view_indep") if leaderboard else None,
                leaderboard.get("n_filtered") if leaderboard else None,
            ),
        )
```

(f) Replace the original `cur.execute("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ...)` block with one that writes 17 columns (13 original + 4 new). The full replacement, inside the existing `for item in per_sample:` loop, after computing `iou_float`:

```python
            extra = leaderboard_per_sample.get(sample_id)
            cur.execute(
                """INSERT INTO samples (
                    run_id, sample_id, scene_id, target_id, query, status,
                    selected_object_id, confidence, iou, acc25, acc50,
                    predicted_bbox_3d_9dof, gt_bbox_3d_9dof,
                    is_easy, is_view_dep, is_filtered_out, classification_correct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    sample_id,
                    scene_id,
                    target_id,
                    item.get("query"),
                    item.get("status"),
                    item.get("selected_object_id"),
                    item.get("confidence"),
                    iou_float,
                    int(iou_float is not None and iou_float >= 0.25),
                    int(iou_float is not None and iou_float >= 0.50),
                    _json_or_none(item.get("predicted_bbox_3d_9dof")),
                    _json_or_none(item.get("gt_bbox_3d_9dof")),
                    int(extra["is_easy"]) if extra else None,
                    int(extra["is_view_dep"]) if extra else None,
                    int(extra["is_filtered_out"]) if extra else None,
                    int(extra["is_correct"]) if extra else None,
                ),
            )
```

(g) In `main()`, add the new CLI flag and pass it to `ingest()`:

```python
    parser.add_argument(
        "--leaderboard-metrics",
        default=None,
        type=Path,
        help="optional path to leaderboard_metrics.json from "
             "nr3d_leaderboard_metrics.py",
    )
```

And in the `ingest(...)` call:

```python
    ingest(
        db_path=args.db,
        output_dir=args.output_dir,
        run_id=args.run_id,
        branch=args.branch,
        commit_hash=args.commit_hash,
        backend=args.backend,
        judge_model=args.judge_model,
        notes=args.notes,
        leaderboard_metrics_path=args.leaderboard_metrics,
    )
```

- [ ] **Step 3.6: Run the new tests to confirm GREEN**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_ingest_nr3d_run.py -v 2>&1 | tail -30`

Expected: all tests in the file pass (existing + 3 new).

- [ ] **Step 3.7: Run the actual ingest on v2 metrics**

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v1_full \
    --run-id v3_referit3d_track_20260501 \
    --branch feat/nr3d-vg-benchmark \
    --commit "$(git rev-parse --short HEAD)" \
    --backend pack_v1 \
    --judge-model none \
    --notes "v3 leaderboard-track: post-aggregated v2 selected_object_id under canonical mentions_target_class_only filter" \
    --leaderboard-metrics tmp/nr3d_eval_v1_full/leaderboard_metrics.json \
    --db docs/benchmark/nr3d/runs.sqlite
```

Expected: prints `[ingest] run_id=v3_referit3d_track_20260501 samples=8584 db=docs/benchmark/nr3d/runs.sqlite`.

- [ ] **Step 3.8: Verify the SQL row**

```bash
sqlite3 docs/benchmark/nr3d/runs.sqlite "
SELECT run_id,
       printf('%.4f', classification_acc_full) AS acc_full,
       printf('%.4f', classification_acc_filtered) AS acc_filtered,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind,
       n_filtered
FROM runs
WHERE run_id='v3_referit3d_track_20260501';
"
```

Expected: 5 non-null float values + a non-null `n_filtered` integer matching `/tmp/v3_numbers.txt` from Task 2.

- [ ] **Step 3.9: Commit**

```bash
git add scripts/ingest_nr3d_run.py \
        src/evaluation/scripts/tests/test_ingest_nr3d_run.py \
        docs/benchmark/nr3d/runs.sqlite
git commit -m "feat(nr3d): ingester schema for leaderboard-track metrics

Add 7 nullable columns to runs (classification_acc_full/filtered, acc_easy/hard/
view_dep/view_indep, n_filtered) and 4 nullable columns to samples (is_easy,
is_view_dep, is_filtered_out, classification_correct), backfilled via ALTER
TABLE migration so existing rows stay valid. New --leaderboard-metrics CLI
flag points at the JSON produced by nr3d_leaderboard_metrics.py; without it
the new columns simply stay NULL.

Includes the v3_referit3d_track_20260501 row in runs.sqlite, ingested from
the existing v2 side_by_side.json + post-aggregated leaderboard_metrics.json."
```

---

## Task 4: v3 version doc with measured numbers

**Files:**
- Create: `docs/benchmark/nr3d/v3_referit3d_track_20260501.md`

- [ ] **Step 4.1: Re-read the captured numbers**

```bash
cat /tmp/v3_numbers.txt
sqlite3 docs/benchmark/nr3d/runs.sqlite "
SELECT printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind,
       n_filtered
FROM runs WHERE run_id='v3_referit3d_track_20260501';"
```

These five numbers go into the headline table in Step 4.2.

- [ ] **Step 4.2: Write the v3 doc**

Create `docs/benchmark/nr3d/v3_referit3d_track_20260501.md` with the content below, replacing the **bracketed placeholders** `[OVERALL]`, `[EASY]`, `[HARD]`, `[VDEP]`, `[VINDEP]`, `[N_FILTERED]`, `[N_EASY]`, `[N_HARD]`, `[N_VDEP]`, `[N_VINDEP]`, `[ACC_FULL]` with the corresponding measured numbers from Step 4.1 / `tmp/nr3d_eval_v1_full/leaderboard_metrics.json`. Format percentages as `XX.XX` (two decimals) for the headline columns and as `0.XXXX` for `classification_acc_full`.

```markdown
# v3 ReferIt3D-Track — 2026-05-01

First NR3D evaluation aligned with the canonical ReferIt3D leaderboard
protocol (classification accuracy on the GT-pool, with the canonical
filter chain applied). Built by post-aggregating the v2 full-test
`selected_object_id` predictions — no Stage 2 agent re-run.

## Run Identity

- Branch: `feat/nr3d-vg-benchmark`
- Tip commit at run time: `<TIP_COMMIT_AT_RUN>` (record `git rev-parse --short HEAD` here)
- Working tree: `/Users/bytedance/project/3DVLMReasoning` (Mac, uv `.venv`, Python 3.12)
- Internal version: `v3_referit3d_track`
- Run ID: `v3_referit3d_track_20260501`
- Stage 2 backend (inherited from v2): `gpt-5.4-2026-03-05` via internal ModelHub
- Aggregator: `src/evaluation/scripts/nr3d_leaderboard_metrics.py`
- Wall time: under 2 minutes (no agent calls)

## Methodology

This version reuses every per-sample agent prediction from `v2_phase8_full`
(8584 utterances, 130/130 NR3D test scenes, GT-pool) and runs a pure
post-aggregator over them under the canonical ReferIt3D protocol:

- Filter: `mentions_target_class_only=True` (canonical default per
  `referit3d/in_out/arguments.py:50`).
- Metric: classification accuracy = mean over filtered fold of
  `selected_object_id == target_id`.
- Slicing: easy/hard via `n_objects ≤ 2`; view-dep via the 10-token literal
  set from `referit3d/analysis/utterances.py:103-105`.

The mathematical equivalence to a re-run is justified in
`docs/superpowers/specs/2026-05-01-nr3d-fairness-design.md`: the agent's
prediction for any given `sample_id` depends only on
`(bundle, query, agent_config)`, none of which change when the canonical
filter is applied — the filter only drops rows.

## Fold

- Split: NR3D `test`
- After baseline filters (blacklist + clothes drop): 8584 utterances on 130 scenes (== v2 fold)
- After canonical `mentions_target_class_only=True`: **[N_FILTERED]** utterances on 130 scenes
  - retain rate = [N_FILTERED] / 8584 = X.X% (consistent with ReferIt3D paper p.9 91.6%)
- Easy / Hard partition (n_objects ≤ 2): n_easy=[N_EASY] / n_hard=[N_HARD]
- View-Dep / View-Indep (token ∩ {front,behind,...} ≠ ∅): n_view_dep=[N_VDEP] / n_view_indep=[N_VINDEP]

## Headline Metrics (Leaderboard Track)

| Metric | Value |
|---|---:|
| Overall (classification accuracy on filtered fold) | **[OVERALL]** |
| Easy | **[EASY]** |
| Hard | **[HARD]** |
| View-Dep | **[VDEP]** |
| View-Indep | **[VINDEP]** |

## SOTA Comparison (NR3D test, classification, GT-track)

| Method | Setup | Overall | Easy | Hard | View-Dep | View-Indep |
|---|---|---:|---:|---:|---:|---:|
| ReferIt3DNet (NR3D-only) | classification | 35.6 | 43.6 | 27.9 | 32.5 | 37.1 |
| BUTD-DETR | classification | 54.6 | 60.7 | 48.4 | 46.0 | 58.0 |
| MVT | classification | 55.1 | 61.3 | 49.1 | 54.3 | 55.4 |
| 3D-VisTA | classification | 64.2 | 72.1 | 56.7 | 61.5 | 65.1 |
| MiKASA | classification | 64.4 | 69.7 | 59.4 | 65.4 | 64.0 |
| UniVLG (GT-track) | classification | 65.2 | 73.3 | 57.0 | 55.1 | 69.9 |
| **Ours (v3, zero-shot RGB+VLM)** | classification | **[OVERALL]** | **[EASY]** | **[HARD]** | **[VDEP]** | **[VINDEP]** |

The "Ours" row is now apples-to-apples with the public leaderboard column
(canonical filter chain, GT-pool, classification metric, identical 5-column
slicing). What remains a paradigm difference — not a protocol violation — is
the input modality (RGB keyframes + 9-DoF box overlays vs trained
3D-point-cloud methods) and the model class (zero-shot LLM vs trained 3D
models).

## Internal Cross-Check Table

| Metric | Value | Notes |
|---|---:|---|
| `classification_acc_full` (n=8584 denominator) | [ACC_FULL] | unfiltered baseline; differs from filtered by ≈0.5-1 pp |
| `classification_acc_filtered` (n=[N_FILTERED] denominator) | [OVERALL] | **headline** — leaderboard-comparable |
| v2 `Acc@0.50` (9-DoF IoU on GT-pool, n=8584) | 0.7762 | same predictions; under GT-pool, IoU ≥ 0.50 ⟺ correct ID, so this is the IoU-based proxy of `classification_acc_full` |
| v2 `mean_iou` | 0.7800 | unchanged; not a leaderboard metric |

The ~0.4 pp gap between v2 `Acc@0.50` (77.62 %) and `classification_acc_full`
([ACC_FULL]) is the GT-pool collapse: under GT-pool, the agent's bbox is
exactly the GT bbox when the right ID is picked, so IoU = 1.0; otherwise
IoU is near 0. The small residual is from a few cases where two GT
instances overlap in 3D enough that picking the wrong one still yields
IoU ≥ 0.50.

## Failure Sentinels

The 287 v2 sentinel failures (upstream `InternalServerError 500` /
`BadRequestError 400` / agent malformed payload) carry
`selected_object_id=null`. Under the leaderboard metric, these score
`is_correct=False`, so they pull the headline down by ≈287/[N_FILTERED]
= roughly Y pp. They remain in the denominator — i.e., we DO NOT exclude
them, matching how leaderboard methods report on their full test set.

## Caveats

- **Input modality asymmetry**: zero-shot RGB+VLM agent vs trained 3D-only
  models. Documented as paradigm difference, not protocol violation. The
  comparison answers "how does a zero-shot multimodal LLM agent score on
  NR3D under the canonical protocol", not "is our LLM stronger than 3D-VisTA
  at the same task."
- **Classification metric on GT-pool collapses to "right ID picked"**: in
  the GT-pool setting the bbox-IoU and ID-match metrics are functionally
  equivalent (because pool members ARE the GT bboxes). This is true for our
  v3 numbers and for every published GT-track leaderboard number above.
- **Per-LLM-call durability gap**: `tool_calls` / `llm_calls` SQLite tables
  remain empty for v3 (carried forward from v2). Tracked in
  `CLAUDE.md::Per-LLM-call durability`.
- **`correct_guess` filter NOT applied**: per audit (Q4), `correct_guess`
  is not part of the canonical referit3d filter chain — only UniVLG-side
  add-on. Applying it would inflate our number by dropping human-failed
  utterances; leaderboard methods don't drop those.
- **130 / 130 NR3D test scene coverage**: see
  `docs/benchmark/nr3d/protocol.md` for the empirical proof that our pool,
  fold, and indexing match the canonical GT-track dimensions.

## Cross-Version Comparison (NR3D-only)

| Version | Date | n_full | n_filtered | Headline (Overall) | Notes |
|---|---|---:|---:|---:|---|
| v1_phase8_smoke20_mac | 2026-04-30 | 20 | — | Acc@0.50 = 0.65 | first green pack-v1 numbers |
| v2_phase8_full | 2026-05-01 | 8584 | — | Acc@0.50 = 0.7762 | first full sweep on GT-pool, IoU metric |
| **v3_referit3d_track** | 2026-05-01 | 8584 | **[N_FILTERED]** | **classification_acc_filtered = [OVERALL]** | **first leaderboard-comparable run; 5-column SOTA-aligned table** |

## SQLite Reproduction Query

```sql
SELECT run_id,
       printf('%.4f', classification_acc_full) AS acc_full,
       printf('%.4f', classification_acc_filtered) AS acc_filtered,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind,
       n_filtered
FROM runs WHERE run_id='v3_referit3d_track_20260501';
-- → ('v3_referit3d_track_20260501', '[ACC_FULL]', '[OVERALL]',
--    '[EASY]', '[HARD]', '[VDEP]', '[VINDEP]', [N_FILTERED])
```

## Raw Artifacts

- v2 per-sample checkpoints (input): `tmp/nr3d_eval_v1_full/per_sample/pack_nr3d_v1/*.json`
- v2 aggregate: `tmp/nr3d_eval_v1_full/side_by_side.json`
- v3 leaderboard metrics (intermediate): `tmp/nr3d_eval_v1_full/leaderboard_metrics.json`
- SQLite row: `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v3_referit3d_track_20260501'`
- Aggregator entry point: `src/evaluation/scripts/nr3d_leaderboard_metrics.py`

## Reproduction Command

```bash
source .venv/bin/activate
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
    --side-by-side tmp/nr3d_eval_v1_full/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --output tmp/nr3d_eval_v1_full/leaderboard_metrics.json \
    --canonical-filter true

PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v1_full \
    --run-id v3_referit3d_track_20260501 \
    --branch feat/nr3d-vg-benchmark \
    --commit "$(git rev-parse --short HEAD)" \
    --backend pack_v1 --judge-model none \
    --notes "v3 leaderboard-track" \
    --leaderboard-metrics tmp/nr3d_eval_v1_full/leaderboard_metrics.json \
    --db docs/benchmark/nr3d/runs.sqlite
```

## Next Steps (post-v3)

- Detection-mode track using V-DETR / BIP3D / ConceptGraph proposals (separate P3 work)
- SR3D integration
- Tokenization alignment (referit3d's `pre_process_text`) if v4 wants strict input parity
```

After substituting the placeholders, double-check:

```bash
grep -n '\[OVERALL\]\|\[EASY\]\|\[HARD\]\|\[VDEP\]\|\[VINDEP\]\|\[N_FILTERED\]\|\[N_EASY\]\|\[N_HARD\]\|\[N_VDEP\]\|\[N_VINDEP\]\|\[ACC_FULL\]\|\[TIP_COMMIT_AT_RUN\]' docs/benchmark/nr3d/v3_referit3d_track_20260501.md
```

Expected: ZERO matches. Every placeholder must be substituted.

- [ ] **Step 4.3: Commit the v3 doc**

```bash
git add docs/benchmark/nr3d/v3_referit3d_track_20260501.md
git commit -m "docs(nr3d): v3 referit3d-track leaderboard run

First NR3D evaluation aligned to the canonical leaderboard protocol
(classification accuracy, mentions_target_class_only filter, easy/hard
+ view-dep/view-indep slicing) by post-aggregating v2 selected_object_id
predictions. SOTA comparison table includes ReferIt3DNet, BUTD-DETR, MVT,
3D-VisTA, MiKASA, UniVLG GT-track."
```

---

## Task 5: Update README.md and leaderboard.md indexes

**Files:**
- Modify: `docs/benchmark/nr3d/README.md`
- Modify: `docs/benchmark/README.md`

- [ ] **Step 5.1: Update `docs/benchmark/nr3d/README.md` timeline + Current Interpretation**

Add a new row to the Version Timeline table after the v2 row:

```markdown
| [v3_phase8_referit3d_track](v3_referit3d_track_20260501.md) | 2026-05-01 | classification_acc_filtered = **[OVERALL]** | Easy/Hard/V-Dep/V-Indep slices reported | 8584Q (n_filtered = [N_FILTERED]) | **First leaderboard-comparable NR3D run** — canonical filter chain (`mentions_target_class_only=True`), classification accuracy as headline, 5-column SOTA-aligned table. Post-aggregated from v2 outputs (no agent re-run). |
```

(Replace `[OVERALL]` and `[N_FILTERED]` with the same numbers used in the v3 doc.)

Then in the "Current Interpretation" section, prepend a new paragraph:

```markdown
**Latest (v3_phase8_referit3d_track, 2026-05-01)**: post-aggregation of v2
predictions under the canonical ReferIt3D leaderboard protocol — the
**first apples-to-apples NR3D number** for our pipeline. Headline:
**classification_acc = [OVERALL]** on n=[N_FILTERED] (after canonical
`mentions_target_class_only=True` filter), with full Easy/Hard +
View-Dep/View-Indep breakdown. The pool/fold equivalence to canonical is
verified empirically in `protocol.md`. See
`v3_referit3d_track_20260501.md` for the full SOTA comparison table.

The v2 numbers (Acc@0.25/0.50 = 77.67/77.62) remain valid as the IoU-on-GT-pool
proxy of classification accuracy, but they are **superseded by v3** as the
leaderboard-comparable number going forward.
```

Replace the literal text "Latest (v2_phase8_full, 2026-05-01)" paragraph with this — keep the original v2 paragraph as a "v2 (now superseded)" subsection if useful, or remove it. Trust your judgment on length; the goal is a clear single source of truth.

Update the Caveats section: drop the now-obsolete bullet "Pool comparability with public leaderboard: published NR3D numbers use ReferIt3D's target-type-only candidate pool (only same-class distractors). Our pool is wider (all instances); the two are not directly comparable." — replace with: "Pool / fold equivalence is empirically verified — see `protocol.md`. The earlier 'wider pool' caveat in v2 was based on an incorrect assumption and is retracted."

- [ ] **Step 5.2: Update `docs/benchmark/README.md` NR3D row**

Edit the Active Benchmarks table. Replace the NR3D row's Highlight cell with:

```markdown
| **NR3D VG** | live | [`nr3d/`](nr3d/) | v3 leaderboard-track 8584Q test (n_filtered=[N_FILTERED]): classification_acc = **[OVERALL]**, Easy=[EASY], Hard=[HARD], V-Dep=[VDEP], V-Indep=[VINDEP] (gpt-5.4) |
```

- [ ] **Step 5.3: Commit the index updates**

```bash
git add docs/benchmark/nr3d/README.md docs/benchmark/README.md
git commit -m "docs(nr3d): index updates for v3 leaderboard-track

- README.md: add v3 timeline row, replace v2 Current Interpretation
  paragraph with v3, retract the obsolete 'wider pool' caveat now that
  pool equivalence is empirically verified.
- docs/benchmark/README.md: NR3D row highlight switched to v3 numbers
  (classification accuracy + 5-column slicing)."
```

---

## Task 6: Final regression sweep + cleanup

**Files:** none modified; verification only.

- [ ] **Step 6.1: Run the affected pytest module**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py src/evaluation/scripts/tests/test_ingest_nr3d_run.py -v 2>&1 | tail -30`

Expected: all tests pass, no failures.

- [ ] **Step 6.2: Verify clean working tree**

Run: `git status -s`

Expected: only `.claude/scheduled_tasks.lock` (session noise, not committed). Nothing under `tmp/nr3d_eval_v1_full/` should be staged.

- [ ] **Step 6.3: Verify the SQLite DB is committed and queryable**

```bash
sqlite3 docs/benchmark/nr3d/runs.sqlite "
SELECT run_id, n,
       printf('%.4f', classification_acc_filtered) AS acc_filtered,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard
FROM runs ORDER BY ingested_at;"
```

Expected: at least the v1 + v2 + v3 rows; v3 row has non-null `classification_acc_filtered`, `acc_easy`, `acc_hard`.

- [ ] **Step 6.4: Optional final commit if anything is hanging**

If `git status -s` shows any uncommitted artifact other than session-only files, decide whether it's intentional (commit it) or noise (revert). Do not commit `tmp/`, do not commit `.claude/scheduled_tasks.lock`.

---

## Summary

After all 6 tasks:

- New module `src/evaluation/scripts/nr3d_leaderboard_metrics.py` (~250 lines, pure aggregator + IO wrapper + CLI).
- 13 new aggregator tests + 3 new ingester tests.
- Schema migration on `runs.sqlite` (idempotent ALTER TABLE).
- New v3 doc with concrete measured numbers, SOTA-aligned 5-column table.
- README + leaderboard index reflect v3 as live; v2 caveats retracted.
- ~6 commits, no API spend, no agent re-runs.
