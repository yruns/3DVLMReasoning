"""Validate the 600-case stratified NR3D subset against the v9.1_fix FULL run.

Two checks:

1. **Deterministic subset metrics**: take the salt-locked 600 sample_ids and
   recompute the leaderboard metrics from the v9.1_fix predictions. Compare to
   the canonical filtered 7805 numbers.

2. **Monte-Carlo bootstrap**: re-run the stratified sampler with 1000 random
   salts, compute the per-sample-set leaderboard accuracy, and report
   mean / std / 5–95 percentile bands. This estimates the design's worst-case
   variance independent of any single salt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEADERBOARD = (
    PROJECT_ROOT
    / "docs/benchmark/nr3d/assets/v9_1_fix_FULL_REPRO_leaderboard_20260516.json"
)
DEFAULT_SUBSET = PROJECT_ROOT / "tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "docs/benchmark/nr3d/assets/v9_3_strat600_validation_20260517.json"
)


def is_easy_int(n_objects: int) -> bool:
    return n_objects <= 2


_VIEW_DEP_TOKENS = frozenset({
    "front", "behind", "back", "right", "left",
    "facing", "leftmost", "rightmost", "looking", "across",
})


def is_view_dep_tokens(tokens: list[str]) -> bool:
    return bool(set(tokens) & _VIEW_DEP_TOKENS)


def compute_metrics(per_sample: list[dict[str, Any]]) -> dict[str, float | int]:
    """Compute the canonical 5-column leaderboard metrics on a sample list.

    Inputs are the per-sample records from the FULL v9.1_fix leaderboard JSON
    (which already include ``is_correct``, ``is_easy``, ``is_view_dep``,
    ``is_filtered_out``). We re-aggregate to mirror the leaderboard layout.
    """
    filtered = [s for s in per_sample if not s["is_filtered_out"]]
    full_correct = sum(1 for s in per_sample if s["is_correct"])
    filt_correct = sum(1 for s in filtered if s["is_correct"])
    easy = [s for s in filtered if s["is_easy"]]
    hard = [s for s in filtered if not s["is_easy"]]
    vd = [s for s in filtered if s["is_view_dep"]]
    vid = [s for s in filtered if not s["is_view_dep"]]
    return {
        "n_full": len(per_sample),
        "n_filtered": len(filtered),
        "classification_acc_full": full_correct / len(per_sample) if per_sample else 0.0,
        "classification_acc_filtered": filt_correct / len(filtered) if filtered else 0.0,
        "n_easy": len(easy),
        "n_hard": len(hard),
        "n_view_dep": len(vd),
        "n_view_indep": len(vid),
        "acc_easy": (sum(1 for s in easy if s["is_correct"]) / len(easy)) if easy else 0.0,
        "acc_hard": (sum(1 for s in hard if s["is_correct"]) / len(hard)) if hard else 0.0,
        "acc_view_dep": (sum(1 for s in vd if s["is_correct"]) / len(vd)) if vd else 0.0,
        "acc_view_indep": (sum(1 for s in vid if s["is_correct"]) / len(vid)) if vid else 0.0,
    }


def largest_remainder_allocation(
    cell_sizes: dict[tuple, int],
    total: int,
) -> dict[tuple, int]:
    n_pool = sum(cell_sizes.values())
    quotas = {c: total * cell_sizes[c] / n_pool for c in cell_sizes}
    base = {c: int(quotas[c]) for c in cell_sizes}
    remainder = total - sum(base.values())
    cells_ranked = sorted(
        cell_sizes.keys(),
        key=lambda c: (-(quotas[c] - base[c]), c),
    )
    for i in range(remainder):
        base[cells_ranked[i]] += 1
    return base


def stratified_sample(
    rows_by_cell: dict[tuple, list[dict[str, Any]]],
    n: int,
    salt: str,
) -> set[str]:
    """Salt-based stratified sample → return set of selected sample_ids."""
    cell_sizes = {c: len(rs) for c, rs in rows_by_cell.items()}
    allocation = largest_remainder_allocation(cell_sizes, total=n)
    chosen: set[str] = set()
    for cell, n_cell in allocation.items():
        rows = sorted(
            rows_by_cell[cell],
            key=lambda r: hashlib.sha1((r["sample_id"] + salt).encode("utf-8")).hexdigest(),
        )
        chosen.update(r["sample_id"] for r in rows[:n_cell])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    payload = json.loads(args.leaderboard.read_text(encoding="utf-8"))
    per_sample_full = payload["per_sample"]
    assert len(per_sample_full) == 8584, f"expected 8584, got {len(per_sample_full)}"

    # Map sample_id → per-sample record (for re-aggregation on subsets)
    by_id = {s["sample_id"]: s for s in per_sample_full}

    # Full reference metrics
    full_metrics = compute_metrics(per_sample_full)
    print("=" * 70)
    print("FULL v9.1_fix (8584 samples) — reference metrics")
    print("=" * 70)
    for k in ["n_full", "n_filtered",
              "classification_acc_full", "classification_acc_filtered",
              "acc_easy", "acc_hard", "acc_view_dep", "acc_view_indep"]:
        v = full_metrics[k]
        if isinstance(v, float):
            print(f"  {k:32s} = {v:.4f}")
        else:
            print(f"  {k:32s} = {v}")

    # === 1. Salt-locked 600 subset ===
    subset_rows = json.loads(args.subset.read_text(encoding="utf-8"))
    subset_ids = [r["sample_id"] for r in subset_rows]
    assert len(subset_ids) == args.n, f"subset is {len(subset_ids)}, expected {args.n}"
    missing = set(subset_ids) - set(by_id)
    if missing:
        raise SystemExit(f"{len(missing)} subset sample_ids missing from leaderboard")

    subset_records = [by_id[sid] for sid in subset_ids]
    subset_metrics = compute_metrics(subset_records)
    print()
    print("=" * 70)
    print(f"SALT-LOCKED stratified-{args.n} subset ({args.subset.name})")
    print("=" * 70)
    for k in ["n_full", "n_filtered",
              "classification_acc_full", "classification_acc_filtered",
              "acc_easy", "acc_hard", "acc_view_dep", "acc_view_indep"]:
        v = subset_metrics[k]
        full_v = full_metrics[k]
        if isinstance(v, float) and isinstance(full_v, float):
            delta = v - full_v
            print(f"  {k:32s} = {v:.4f}  (full={full_v:.4f}, Δ={delta:+.4f})")
        else:
            print(f"  {k:32s} = {v}  (full={full_v})")

    # === 2. Monte-Carlo bootstrap over salts ===
    # Build cell-binned pool from FILTERED 7805 (canonical)
    rows_by_cell: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for s in per_sample_full:
        if s["is_filtered_out"]:
            continue
        cell = (bool(s["is_easy"]), bool(s["is_view_dep"]))
        rows_by_cell[cell].append({
            "sample_id": s["sample_id"],
            "is_correct": s["is_correct"],
            "is_easy": s["is_easy"],
            "is_view_dep": s["is_view_dep"],
        })
    pool_n = sum(len(v) for v in rows_by_cell.values())
    assert pool_n == 7805, f"pool {pool_n} ≠ 7805"

    rng = random.Random(args.bootstrap_seed)
    acc_filtered_runs: list[float] = []
    acc_easy_runs: list[float] = []
    acc_hard_runs: list[float] = []
    acc_vd_runs: list[float] = []
    acc_vid_runs: list[float] = []

    for trial in range(args.bootstrap_trials):
        salt = f"bootstrap_{rng.randrange(2**32)}"
        chosen_ids = stratified_sample(rows_by_cell, n=args.n, salt=salt)
        records = [by_id[sid] for sid in chosen_ids]
        m = compute_metrics(records)
        acc_filtered_runs.append(m["classification_acc_filtered"])
        acc_easy_runs.append(m["acc_easy"])
        acc_hard_runs.append(m["acc_hard"])
        acc_vd_runs.append(m["acc_view_dep"])
        acc_vid_runs.append(m["acc_view_indep"])

    def describe(label: str, values: list[float], ref: float) -> dict[str, Any]:
        if not values:
            return {"mean": None, "std": None, "p5": None, "p95": None,
                    "ref_full": ref, "bias_mean_minus_ref": None,
                    "ci95_halfwidth": None, "n_trials": 0}
        sorted_v = sorted(values)
        p5 = sorted_v[int(0.05 * len(sorted_v))]
        p95 = sorted_v[min(int(0.95 * len(sorted_v)), len(sorted_v) - 1)]
        mean_v = statistics.mean(values)
        std_v = statistics.pstdev(values)
        bias = mean_v - ref
        print(f"  {label:18s} mean={mean_v:.4f} std={std_v:.4f} "
              f"5%={p5:.4f} 95%={p95:.4f}  ref={ref:.4f}  bias={bias:+.4f}")
        return {
            "mean": mean_v, "std": std_v,
            "p5": p5, "p95": p95,
            "ref_full": ref, "bias_mean_minus_ref": bias,
            "ci95_halfwidth": (p95 - p5) / 2.0,
        }

    print()
    print("=" * 70)
    print(f"BOOTSTRAP — {args.bootstrap_trials} salts, n={args.n} stratified subset")
    print("=" * 70)
    boot = {
        "n_trials": args.bootstrap_trials,
        "bootstrap_seed": args.bootstrap_seed,
        "subset_n": args.n,
        "classification_acc_filtered": describe(
            "acc_filtered", acc_filtered_runs, full_metrics["classification_acc_filtered"],
        ),
        "acc_easy": describe("acc_easy", acc_easy_runs, full_metrics["acc_easy"]),
        "acc_hard": describe("acc_hard", acc_hard_runs, full_metrics["acc_hard"]),
        "acc_view_dep": describe("acc_view_dep", acc_vd_runs, full_metrics["acc_view_dep"]),
        "acc_view_indep": describe("acc_view_indep", acc_vid_runs, full_metrics["acc_view_indep"]),
    }

    out = {
        "full_metrics_8584": full_metrics,
        "salt_locked_600_metrics": subset_metrics,
        "salt_locked_600_deltas": {
            k: subset_metrics[k] - full_metrics[k]
            for k in ["classification_acc_filtered", "acc_easy", "acc_hard",
                      "acc_view_dep", "acc_view_indep"]
        },
        "bootstrap": boot,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
