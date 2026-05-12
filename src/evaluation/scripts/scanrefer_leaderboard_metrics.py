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
import gzip
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


_PHASE8_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz")


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
        samples.append(
            {
                "sample_id": sid,
                "iou": iou,
                "is_unique": is_unique,
                "acc25": int(iou >= 0.25),
                "acc50": int(iou >= 0.50),
                "target_id": meta.get("target_id"),
                "target": meta.get("target"),
                "status": rec.get("status"),
            }
        )

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


def load_sample_id_filter(path: Path) -> set[str]:
    """Load a sample-id allow-list from the same JSON shapes used by runners."""
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"sample_ids JSON must be a list: {path}")
    out: set[str] = set()
    for index, item in enumerate(items):
        if isinstance(item, str):
            sample_id = item
        elif isinstance(item, dict):
            sample_id = item.get("sample_id")
        else:
            raise ValueError(
                f"sample_ids[{index}] must be a string or object with sample_id"
            )
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"sample_ids[{index}] missing non-empty sample_id")
        out.add(sample_id)
    return out


def _chars_after_backend_per_sample_array(
    side_by_side_path: Path,
    *,
    backend: str,
) -> Iterator[str]:
    """Yield chars inside side_by_side[backend].per_sample without full-file load."""
    backend_token = json.dumps(backend)
    seen_backend = False
    seen_per_sample = False

    with side_by_side_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not seen_backend:
                if backend_token not in line:
                    continue
                seen_backend = True

            if not seen_per_sample:
                idx = line.find('"per_sample"')
                if idx < 0:
                    continue
                bracket = line.find("[", idx)
                if bracket < 0:
                    for continuation in fh:
                        bracket = continuation.find("[")
                        if bracket >= 0:
                            line = continuation
                            break
                    else:
                        raise ValueError(
                            f"side_by_side.json[{backend}].per_sample missing '['"
                        )
                seen_per_sample = True
                yield from line[bracket + 1 :]
                continue

            yield from line

    if not seen_backend:
        raise ValueError(f"side_by_side.json missing backend={backend!r}")
    if not seen_per_sample:
        raise ValueError(f"side_by_side.json[{backend}].per_sample missing")


def _iter_json_array_objects(chars: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Incrementally decode objects from a JSON array body."""
    in_record = False
    in_string = False
    escaped = False
    depth = 0
    buf: list[str] = []

    for ch in chars:
        if not in_record:
            if ch.isspace() or ch == ",":
                continue
            if ch == "]":
                return
            if ch != "{":
                raise ValueError(f"unexpected character in per_sample array: {ch!r}")
            in_record = True
            depth = 1
            buf = [ch]
            in_string = False
            escaped = False
            continue

        buf.append(ch)
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield json.loads("".join(buf))
                in_record = False
                buf = []

    if in_record:
        raise ValueError("unterminated object in per_sample array")


def load_side_by_side_predictions(
    side_by_side_path: Path,
    *,
    backend: str = "pack_v1",
) -> Iterator[dict[str, Any]]:
    """Stream minimal prediction records from a large side_by_side.json.

    Full ScanRefer runs can produce multi-GB side-by-side files because each
    sample carries tool traces. Leaderboard metrics only need sample_id, iou,
    and status, so keep the resident set bounded by decoding one record at a
    time and discarding the trace-heavy payload immediately.
    """
    yield from load_side_by_side_compact_records(
        side_by_side_path,
        backend=backend,
        fields=("iou", "status"),
    )


def load_side_by_side_compact_records(
    side_by_side_path: Path,
    *,
    backend: str = "pack_v1",
    fields: tuple[str, ...] = (),
) -> Iterator[dict[str, Any]]:
    """Stream records from side_by_side.json and retain only selected fields."""
    requested = ("sample_id", *fields)
    chars = _chars_after_backend_per_sample_array(
        side_by_side_path,
        backend=backend,
    )
    for rec in _iter_json_array_objects(chars):
        if not isinstance(rec.get("sample_id"), str) or not rec["sample_id"]:
            raise ValueError("per_sample item missing sample_id")
        yield {field: rec.get(field) for field in requested}


def _scanrefer_json_path(data_root: Path, split: str) -> Path:
    json_path = data_root / "raw" / f"ScanRefer_filtered_{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"ScanRefer JSON missing: {json_path}")
    return json_path


def _sample_id_from_scanrefer_row(row: dict[str, Any]) -> str:
    scene = row["scene_id"]
    target_id = int(row["object_id"])
    ann_id = row["ann_id"]
    return f"scannet/{scene}::{target_id}::{ann_id}"


def _load_scene_class_counts_and_valid_targets(
    phase8_data_root: Path,
    scene: str,
) -> tuple[Counter[str], set[int]] | None:
    pkl_path = phase8_data_root / scene / _PHASE8_PCD_REL
    if not pkl_path.exists():
        return None

    with gzip.open(pkl_path, "rb") as f:
        objs = pickle.load(f).get("objects") or []

    counts: Counter[str] = Counter()
    valid_targets: set[int] = set()
    for index, obj in enumerate(objs):
        class_name = obj.get("class_name")
        if isinstance(class_name, list) and class_name:
            counts[str(class_name[0]).lower()] += 1
        elif isinstance(class_name, str) and class_name:
            counts[class_name.lower()] += 1
        if obj.get("bbox_np") is not None:
            valid_targets.add(index)
    return counts, valid_targets


def load_scanrefer_leaderboard_meta(
    *,
    scanrefer_data_root: Path,
    phase8_data_root: Path,
    split: str = "val",
    sample_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load only metadata needed for ScanRefer leaderboard columns.

    `ScanRefVGDataset.from_path()` also materializes Phase8 GT bboxes and keeps
    every scene's object list cached. That is appropriate for evaluation, but
    full leaderboard post-processing only needs Unique/Multiple class counts.
    This loader reads each Phase8 scene once, stores small counters/valid-id
    sets, and discards the trace-heavy object payload immediately.
    """
    json_path = _scanrefer_json_path(scanrefer_data_root, split)
    utterances = json.loads(json_path.read_text(encoding="utf-8"))
    requested = set(sample_filter) if sample_filter is not None else None

    rows: list[dict[str, Any]] = []
    scenes: set[str] = set()
    for row in utterances:
        sample_id = _sample_id_from_scanrefer_row(row)
        if requested is not None and sample_id not in requested:
            continue
        rows.append(row)
        scenes.add(row["scene_id"])

    scene_stats: dict[str, tuple[Counter[str], set[int]]] = {}
    for scene in sorted(scenes):
        stats = _load_scene_class_counts_and_valid_targets(phase8_data_root, scene)
        if stats is not None:
            scene_stats[scene] = stats

    sample_meta: list[dict[str, Any]] = []
    for row in rows:
        scene = row["scene_id"]
        target_id = int(row["object_id"])
        stats = scene_stats.get(scene)
        if stats is None:
            continue
        counts, valid_targets = stats
        if target_id not in valid_targets:
            continue
        target_name = row["object_name"]
        sample_meta.append(
            {
                "sample_id": _sample_id_from_scanrefer_row(row),
                "is_unique": counts.get(target_name.lower(), 0) == 1,
                "target_id": target_id,
                "target": target_name,
            }
        )

    if requested is not None:
        found = {m["sample_id"] for m in sample_meta}
        missing = requested - found
        if missing:
            raise ValueError(
                f"sample_ids not found in ScanRefer metadata: {sorted(missing)[:5]}"
            )
    return sample_meta


def compute_leaderboard_metrics(
    side_by_side_path: Path,
    scanrefer_data_root: Path,
    phase8_data_root: Path,
    backend: str = "pack_v1",
    sample_ids_path: Path | None = None,
) -> dict[str, Any]:
    """IO wrapper: load side_by_side + ScanRefer loader, call aggregate."""
    per_sample = list(
        load_side_by_side_predictions(side_by_side_path, backend=backend)
    )

    sample_filter = (
        load_sample_id_filter(sample_ids_path) if sample_ids_path is not None else None
    )
    sample_meta = load_scanrefer_leaderboard_meta(
        scanrefer_data_root=scanrefer_data_root,
        phase8_data_root=phase8_data_root,
        split="val",
        sample_filter=sample_filter,
    )
    return aggregate(per_sample, sample_meta)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side-by-side", required=True, type=Path)
    p.add_argument("--scanrefer-data-root", default=Path("data/scanrefer"), type=Path)
    p.add_argument("--phase8-data-root", default=Path("data/nr3d/scannet"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backend", default="pack_v1")
    p.add_argument(
        "--sample-ids",
        type=Path,
        default=None,
        help=(
            "Optional sample-id JSON allow-list. Use for random100/smoke folds; "
            "without it the canonical full val set is expected."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_leaderboard_metrics(
        side_by_side_path=args.side_by_side,
        scanrefer_data_root=args.scanrefer_data_root,
        phase8_data_root=args.phase8_data_root,
        backend=args.backend,
        sample_ids_path=args.sample_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"n_total={metrics['n_total']} (Unique={metrics['n_unique']}, "
        f"Multiple={metrics['n_multiple']})"
    )
    print(
        f"acc25  overall={metrics['acc25_overall']:.4f} "
        f"unique={metrics['acc25_unique']:.4f} "
        f"multiple={metrics['acc25_multiple']:.4f}"
    )
    print(
        f"acc50  overall={metrics['acc50_overall']:.4f} "
        f"unique={metrics['acc50_unique']:.4f} "
        f"multiple={metrics['acc50_multiple']:.4f}"
    )


if __name__ == "__main__":
    main()
