"""Run EmbodiedScan VG legacy + pack-v1 backends side-by-side and report."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Sequence

from loguru import logger


def run_one_sample(sample_id: str, backend: str) -> dict:
    """Adapter: run one sample through the chosen backend; return
    {sample_id, iou, status}. Wires into your existing runner; for the
    test, this is monkeypatched."""
    raise NotImplementedError("wire into your existing pilot runner")


def compare_backends(
    *,
    sample_ids: Sequence[str],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for backend in ("legacy", "pack_v1"):
        per_sample = [run_one_sample(s, backend) for s in sample_ids]
        ious = [r["iou"] for r in per_sample if r.get("iou") is not None]
        acc25 = sum(1 for v in ious if v >= 0.25) / max(len(ious), 1)
        acc50 = sum(1 for v in ious if v >= 0.50) / max(len(ious), 1)
        results[backend] = {
            "n": len(per_sample),
            "mean_iou": statistics.mean(ious) if ious else 0.0,
            "Acc@0.25": acc25,
            "Acc@0.50": acc50,
            "per_sample": per_sample,
        }
    (output_dir / "side_by_side.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "legacy: Acc@0.25={:.3f}  pack_v1: Acc@0.25={:.3f}",
        results["legacy"]["Acc@0.25"], results["pack_v1"]["Acc@0.25"],
    )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-ids", required=True, type=Path,
                   help="JSON file with [sample_id, ...]")
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = json.loads(args.sample_ids.read_text())
    compare_backends(sample_ids=sample_ids, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
