"""ScanRefer leaderboard-track post-aggregator.

Reads side_by_side.json from run_scanrefer_vg_side_by_side.py + sample
metadata from ScanRefVGDataset, emits canonical ScanRefer metrics:
Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }.

Source citations:
- ScanRefer paper Table 1 — canonical Unique/Multiple slicing
- ZSVG3D visprog_scanrefer.py:51 — `unique = (...class_ids == target_class_id).sum() == 1`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def aggregate(
    predictions: list[dict[str, Any]],
    sample_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Unique/Multiple/Overall × @0.25/0.50 from predictions.

    Args:
        predictions: per-sample records from side_by_side.json.
            Each must have ``sample_id``, ``iou``, ``status``.
            ``iou`` may be None for failed sentinels (counted as 0).
        sample_meta: per-sample metadata records from ScanRefVGDataset.
            Each must have ``sample_id``, ``is_unique``.

    Returns:
        Metrics dict with the leaderboard columns and per_sample list.

    Raises:
        ValueError: if any meta sample_id is missing from predictions.
    """
    pred_by_sid = {p["sample_id"]: p for p in predictions}
    meta_sids = {m["sample_id"] for m in sample_meta}
    missing = meta_sids - set(pred_by_sid.keys())
    if missing:
        raise ValueError(
            f"side_by_side missing {len(missing)} sample_ids; "
            f"first 5: {sorted(missing)[:5]}"
        )

    samples: list[dict[str, Any]] = []
    for meta in sample_meta:
        sid = meta["sample_id"]
        rec = pred_by_sid[sid]
        iou_raw = rec.get("iou")
        iou = float(iou_raw) if iou_raw is not None else 0.0
        if str(rec.get("status", "")).lower() == "failed":
            iou = 0.0
        is_unique = bool(meta.get("is_unique", False))
        samples.append({
            "sample_id": sid,
            "iou": iou,
            "is_unique": is_unique,
            "acc25": int(iou >= 0.25),
            "acc50": int(iou >= 0.50),
            "target_id": meta.get("target_id"),
            "target": meta.get("target"),
            "status": rec.get("status"),
        })

    n_total = len(samples)
    unique = [s for s in samples if s["is_unique"]]
    multi = [s for s in samples if not s["is_unique"]]

    def _mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "n_total": n_total,
        "n_unique": len(unique),
        "n_multiple": len(multi),
        "acc25_overall": _mean([s["acc25"] for s in samples]),
        "acc50_overall": _mean([s["acc50"] for s in samples]),
        "acc25_unique": _mean([s["acc25"] for s in unique]),
        "acc50_unique": _mean([s["acc50"] for s in unique]),
        "acc25_multiple": _mean([s["acc25"] for s in multi]),
        "acc50_multiple": _mean([s["acc50"] for s in multi]),
        "mean_iou_overall": _mean([s["iou"] for s in samples]) if samples else 0.0,
        "per_sample": samples,
    }
    if metrics["n_unique"] + metrics["n_multiple"] != metrics["n_total"]:
        raise AssertionError(
            f"n_unique + n_multiple != n_total: {metrics['n_unique']} + "
            f"{metrics['n_multiple']} != {metrics['n_total']}"
        )
    return metrics


def compute_leaderboard_metrics(
    side_by_side_path: Path,
    scanrefer_data_root: Path,
    phase8_data_root: Path,
    backend: str = "pack_v1",
) -> dict[str, Any]:
    """IO wrapper: load side_by_side + ScanRefer loader, call aggregate."""
    from benchmarks.scanrefer_loader import ScanRefVGDataset

    payload = json.loads(side_by_side_path.read_text(encoding="utf-8"))
    if backend not in payload:
        raise ValueError(f"side_by_side.json missing backend={backend!r}")
    per_sample = payload[backend].get("per_sample") or []

    ds = ScanRefVGDataset.from_path(
        data_root=str(scanrefer_data_root),
        phase8_data_root=str(phase8_data_root),
        split="val",
    )
    sample_meta = [
        {
            "sample_id": s.sample_id,
            "is_unique": s.is_unique,
            "target_id": s.target_id,
            "target": s.target,
        }
        for s in ds
    ]
    return aggregate(per_sample, sample_meta)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side-by-side", required=True, type=Path)
    p.add_argument("--scanrefer-data-root", default=Path("data/scanrefer"), type=Path)
    p.add_argument("--phase8-data-root", default=Path("data/nr3d/scannet"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backend", default="pack_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_leaderboard_metrics(
        side_by_side_path=args.side_by_side,
        scanrefer_data_root=args.scanrefer_data_root,
        phase8_data_root=args.phase8_data_root,
        backend=args.backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"n_total={metrics['n_total']} (Unique={metrics['n_unique']}, "
          f"Multiple={metrics['n_multiple']})")
    print(f"acc25  overall={metrics['acc25_overall']:.4f} "
          f"unique={metrics['acc25_unique']:.4f} "
          f"multiple={metrics['acc25_multiple']:.4f}")
    print(f"acc50  overall={metrics['acc50_overall']:.4f} "
          f"unique={metrics['acc50_unique']:.4f} "
          f"multiple={metrics['acc50_multiple']:.4f}")


if __name__ == "__main__":
    main()
