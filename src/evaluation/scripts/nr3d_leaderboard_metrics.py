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
- Pool / metric / filter protocol: ``docs/benchmark/nr3d/protocol_audit_20260501.md``
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


def load_requested_sample_ids(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"sample ids JSON must be a list: {path}")

    out: set[str] = set()
    first_index_by_id: dict[str, int] = {}
    for index, item in enumerate(raw):
        if isinstance(item, str):
            sample_id = item
        elif isinstance(item, dict):
            sample_id = item.get("sample_id")
        else:
            raise ValueError(f"sample_ids[{index}] must be string or object")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"sample_ids[{index}] missing non-empty sample_id")
        if sample_id in first_index_by_id:
            raise ValueError(
                f"duplicate sample_id {sample_id!r} at "
                f"sample_ids[{first_index_by_id[sample_id]}] and "
                f"sample_ids[{index}]"
            )
        first_index_by_id[sample_id] = index
        out.add(sample_id)

    if not out:
        raise ValueError(f"sample ids JSON is empty: {path}")
    return out


def restrict_to_sample_ids(
    predictions: dict[str, int | None],
    sample_meta: list[dict[str, Any]],
    filtered_sample_ids: set[str],
    requested_sample_ids: set[str],
) -> tuple[dict[str, int | None], list[dict[str, Any]], set[str]]:
    missing_predictions = requested_sample_ids - set(predictions)
    if missing_predictions:
        raise ValueError(
            "missing requested sample_ids in predictions: "
            f"{sorted(missing_predictions)[:5]}"
        )

    meta_ids = {str(row["sample_id"]) for row in sample_meta}
    missing_meta = requested_sample_ids - meta_ids
    if missing_meta:
        raise ValueError(
            "missing requested sample_ids in NR3D metadata: "
            f"{sorted(missing_meta)[:5]}"
        )

    subset_predictions = {
        sample_id: predictions[sample_id]
        for sample_id in sorted(requested_sample_ids)
    }
    subset_meta = [
        row for row in sample_meta if str(row["sample_id"]) in requested_sample_ids
    ]
    subset_filtered = filtered_sample_ids & requested_sample_ids
    return subset_predictions, subset_meta, subset_filtered


def compute_leaderboard_metrics(
    side_by_side_path: Path,
    nr3d_data_root: Path,
    phase8_data_root: Path,
    canonical_filter: bool = True,
    backend: str = "pack_v1",
    sample_ids_path: Path | None = None,
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

    if sample_ids_path is not None:
        predictions, sample_meta, filtered_sample_ids = restrict_to_sample_ids(
            predictions,
            sample_meta,
            filtered_sample_ids,
            load_requested_sample_ids(sample_ids_path),
        )

    return aggregate(predictions, sample_meta, filtered_sample_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side-by-side", required=True, type=Path)
    parser.add_argument("--nr3d-data-root", required=True, type=Path)
    parser.add_argument("--phase8-data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="pack_v1")
    parser.add_argument("--sample-ids", type=Path, default=None)
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
        sample_ids_path=args.sample_ids,
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
