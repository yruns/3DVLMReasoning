"""Build a 600-sample stratified NR3D fold representative of the filtered 7805.

Stratification cells: (is_easy, is_view_dep) on the canonical filter
(``mentions_target_class_only=True``). Allocation uses the largest-remainder
method, mirroring how the filtered 7805 splits into 4 cells:

    Easy × VIndep  2570  32.93%  →  198 cases
    Easy × VDep    1203  15.41%  →   92 cases
    Hard × VIndep  2483  31.81%  →  191 cases
    Hard × VDep    1549  19.85%  →  119 cases
    -------------------------------- 7805         600

Within each cell, samples are picked deterministically by ``sha1(sample_id +
SELECTION_SALT)`` ordering (no Python-version-dependent ``random.sample``),
so the output is byte-stable across reruns and machines.

Outputs (mirroring ``build_nr3d_v4_random100_fold.py``):

  --output   sample_ids JSON: list of {sample_id, scene_id, target_id, category, is_easy, is_view_dep}
  --summary  small JSON with salt, per-cell counts, first-10 sample_ids
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.nr3d_loader import Nr3dDataset

# Reuse the view-dep token set from the leaderboard aggregator.
# Source: referit3d/analysis/utterances.py:103-105
_VIEW_DEP_TOKENS: frozenset[str] = frozenset({
    "front", "behind", "back", "right", "left",
    "facing", "leftmost", "rightmost", "looking", "across",
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NR3D_ROOT = PROJECT_ROOT / "data/nr3d"
DEFAULT_PHASE8_DATA_ROOT = PROJECT_ROOT / "data/nr3d/scannet"
DEFAULT_OUT = PROJECT_ROOT / "tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "tmp/nr3d_artifacts/v9_3_strat600_summary.json"
# Calibration salt — picked by scripts/search_nr3d_strat600_salt.py to minimize
# max |Δ| across the 5 leaderboard metrics on the v9.1_fix FULL predictions.
# See docs/benchmark/nr3d/assets/v9_3_strat600_salt_search_20260517.json.
# Result: acc_filtered Δ=+0.05pp, all 5 metrics within ±0.19pp of the full 7805.
SELECTION_SALT = "nr3d_v9_3_strat600_v291"
N = 600

CellKey = tuple[bool, bool]  # (is_easy, is_view_dep)


def stable_key(sample_id: str) -> str:
    """Deterministic salted ordering key for one NR3D sample id."""
    return hashlib.sha1((sample_id + SELECTION_SALT).encode("utf-8")).hexdigest()


def is_easy(n_objects: int) -> bool:
    return n_objects <= 2


def is_view_dep(tokens: list[str]) -> bool:
    return bool(set(tokens) & _VIEW_DEP_TOKENS)


def load_canonical_rows(
    nr3d_root: Path,
    phase8_data_root: Path,
) -> list[dict[str, Any]]:
    """Load the canonical filtered NR3D test rows (mentions_target_class=True)."""
    dataset = Nr3dDataset.from_path(
        nr3d_root,
        split="test",
        bbox_source="phase8_gt_cg",
        phase8_data_root=phase8_data_root,
        mentions_target_class_only=True,
    )
    rows: list[dict[str, Any]] = []
    for sample in dataset:
        n_obj = int(sample.n_objects)
        tokens = list(sample.tokens)
        rows.append({
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "category": sample.target,
            "n_objects": n_obj,
            "tokens": tokens,
            "is_easy": is_easy(n_obj),
            "is_view_dep": is_view_dep(tokens),
        })
    return rows


def largest_remainder_allocation(
    cell_sizes: dict[CellKey, int],
    total: int,
) -> dict[CellKey, int]:
    """Allocate `total` slots across cells proportional to their sizes.

    Uses the largest-remainder (Hamilton) method: floor of proportional share,
    then distribute remaining slots to cells with the largest fractional parts.
    Ties broken by a stable cell-key ordering so the allocation is deterministic.
    """
    n_pool = sum(cell_sizes.values())
    if n_pool == 0:
        raise ValueError("empty cell sizes; cannot allocate")
    if total > n_pool:
        raise ValueError(
            f"requested total={total} exceeds available pool {n_pool}"
        )

    quotas = {c: total * cell_sizes[c] / n_pool for c in cell_sizes}
    base = {c: int(quotas[c]) for c in cell_sizes}
    remainder = total - sum(base.values())

    # Stable order: largest fractional part first; tie-break by (is_easy, is_view_dep).
    cells_ranked = sorted(
        cell_sizes.keys(),
        key=lambda c: (-(quotas[c] - base[c]), c),
    )
    for i in range(remainder):
        base[cells_ranked[i]] += 1
    return base


def select_subset(
    rows: list[dict[str, Any]],
    n: int = N,
) -> list[dict[str, Any]]:
    """Stratify rows by (is_easy, is_view_dep), allocate, and pick deterministically."""
    by_cell: dict[CellKey, list[dict[str, Any]]] = {
        (True, True): [], (True, False): [],
        (False, True): [], (False, False): [],
    }
    for row in rows:
        by_cell[(row["is_easy"], row["is_view_dep"])].append(row)

    cell_sizes = {c: len(rows_in_cell) for c, rows_in_cell in by_cell.items()}
    allocation = largest_remainder_allocation(cell_sizes, total=n)

    selected: list[dict[str, Any]] = []
    for cell, n_cell in allocation.items():
        cell_rows = sorted(
            by_cell[cell],
            key=lambda r: stable_key(r["sample_id"]),
        )
        if n_cell > len(cell_rows):
            raise ValueError(
                f"cell {cell}: requested {n_cell} > available {len(cell_rows)}"
            )
        selected.extend(cell_rows[:n_cell])

    # Output: sort by stable_key globally for byte-stable file order.
    selected.sort(key=lambda r: stable_key(r["sample_id"]))
    return selected


def build_summary(
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    cell_sizes: dict[CellKey, int],
    allocation: dict[CellKey, int],
) -> dict[str, Any]:
    """Build the per-cell + diversity summary."""
    cell_labels = {
        (True, False): "easy_view_indep",
        (True, True): "easy_view_dep",
        (False, False): "hard_view_indep",
        (False, True): "hard_view_dep",
    }
    n_pool = sum(cell_sizes.values())
    n_selected = len(selected)
    cells_summary: dict[str, Any] = {}
    for cell, label in cell_labels.items():
        pool_n = cell_sizes[cell]
        chosen_n = allocation[cell]
        cells_summary[label] = {
            "is_easy": cell[0],
            "is_view_dep": cell[1],
            "pool_n": pool_n,
            "pool_pct": round(100.0 * pool_n / n_pool, 2),
            "selected_n": chosen_n,
            "selected_pct": round(100.0 * chosen_n / n_selected, 2),
        }

    scenes = Counter(r["scene_id"] for r in selected)
    cats = Counter(r["category"] for r in selected)
    pool_scenes = Counter(r["scene_id"] for r in rows)
    pool_cats = Counter(r["category"] for r in rows)

    return {
        "selection_salt": SELECTION_SALT,
        "n_selected": n_selected,
        "n_pool_filtered": n_pool,
        "cells": cells_summary,
        "scene_coverage": {
            "selected_unique_scenes": len(scenes),
            "pool_unique_scenes": len(pool_scenes),
            "selected_per_scene": {
                "min": min(scenes.values()) if scenes else 0,
                "max": max(scenes.values()) if scenes else 0,
                "mean": round(
                    sum(scenes.values()) / len(scenes), 2
                ) if scenes else 0.0,
            },
        },
        "category_coverage": {
            "selected_unique_categories": len(cats),
            "pool_unique_categories": len(pool_cats),
            "top_10_categories_pool": pool_cats.most_common(10),
            "top_10_categories_selected": cats.most_common(10),
        },
        "first_10_sample_ids": [r["sample_id"] for r in selected[:10]],
    }


def write_outputs(
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    cell_sizes: dict[CellKey, int],
    allocation: dict[CellKey, int],
    sample_out: Path,
    summary_out: Path,
) -> None:
    sample_rows = [
        {
            "sample_id": r["sample_id"],
            "scene_id": r["scene_id"],
            "target_id": r["target_id"],
            "category": r["category"],
            "is_easy": r["is_easy"],
            "is_view_dep": r["is_view_dep"],
        }
        for r in selected
    ]
    summary = build_summary(selected, rows, cell_sizes, allocation)

    sample_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    sample_out.write_text(
        json.dumps(sample_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr3d-root", type=Path, default=DEFAULT_NR3D_ROOT)
    parser.add_argument(
        "--phase8-data-root",
        type=Path,
        default=DEFAULT_PHASE8_DATA_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--n", type=int, default=N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError(f"--n must be > 0, got {args.n}")

    rows = load_canonical_rows(args.nr3d_root, args.phase8_data_root)
    if len(rows) != 7805:
        # Soft warning: the script still runs, but document the unexpected pool size.
        print(
            f"WARNING: expected 7805 canonical rows (mentions_target_class=True), "
            f"got {len(rows)}. Proceeding."
        )

    by_cell: dict[CellKey, list[dict[str, Any]]] = {
        (True, True): [], (True, False): [],
        (False, True): [], (False, False): [],
    }
    for row in rows:
        by_cell[(row["is_easy"], row["is_view_dep"])].append(row)
    cell_sizes = {c: len(v) for c, v in by_cell.items()}
    allocation = largest_remainder_allocation(cell_sizes, total=args.n)

    selected = select_subset(rows, n=args.n)

    write_outputs(
        selected,
        rows=rows,
        cell_sizes=cell_sizes,
        allocation=allocation,
        sample_out=args.output,
        summary_out=args.summary,
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    print(f"selected n = {len(selected)} / pool {len(rows)}")
    print("per-cell allocation:")
    cell_labels = {
        (True, False): "Easy × VIndep",
        (True, True):  "Easy × VDep  ",
        (False, False):"Hard × VIndep",
        (False, True): "Hard × VDep  ",
    }
    for cell in [(True, False), (True, True), (False, False), (False, True)]:
        print(
            f"  {cell_labels[cell]}  pool={cell_sizes[cell]:5d} "
            f"({100.0*cell_sizes[cell]/sum(cell_sizes.values()):5.2f}%)  "
            f"→ selected={allocation[cell]:3d} "
            f"({100.0*allocation[cell]/args.n:5.2f}%)"
        )


if __name__ == "__main__":
    main()
