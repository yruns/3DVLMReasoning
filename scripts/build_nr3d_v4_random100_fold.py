"""Build the frozen NR3D v4 fair-view random100 pilot fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.nr3d_loader import Nr3dDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIDE_BY_SIDE = PROJECT_ROOT / "tmp/nr3d_eval_v1_full/side_by_side.json"
DEFAULT_OUT = (
    PROJECT_ROOT
    / "tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_summary.json"
)
DEFAULT_NR3D_ROOT = PROJECT_ROOT / "data/nr3d"
DEFAULT_PHASE8_DATA_ROOT = PROJECT_ROOT / "data/nr3d/scannet"
SELECTION_SALT = "nr3d_v4_agent_guards_fair_views_20260512"
N = 100


def stable_key(sample_id: str) -> str:
    """Return the deterministic salted ordering key for one NR3D sample id."""
    return hashlib.sha1((sample_id + SELECTION_SALT).encode("utf-8")).hexdigest()


def load_prediction_ids(side_by_side: Path, backend: str = "pack_v1") -> set[str]:
    """Load sample ids that have existing predictions for ``backend``."""
    payload = json.loads(side_by_side.read_text(encoding="utf-8"))
    if backend not in payload:
        raise ValueError(f"{side_by_side} missing backend={backend!r}")
    per_sample = payload[backend].get("per_sample")
    if not isinstance(per_sample, list):
        raise ValueError(
            f"{side_by_side}[{backend!r}].per_sample must be a list"
        )

    sample_ids: set[str] = set()
    for idx, row in enumerate(per_sample):
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            raise ValueError(
                f"{side_by_side}[{backend!r}].per_sample[{idx}] missing sample_id"
            )
        sample_ids.add(row["sample_id"])
    return sample_ids


def load_canonical_rows(
    nr3d_root: Path,
    phase8_data_root: Path,
) -> list[dict[str, Any]]:
    """Load canonical filtered NR3D test rows from the Phase 8 GT-CG source."""
    dataset = Nr3dDataset.from_path(
        nr3d_root,
        split="test",
        bbox_source="phase8_gt_cg",
        phase8_data_root=phase8_data_root,
        mentions_target_class_only=True,
    )
    return [
        {
            "sample_id": sample.sample_id,
            "scene_id": sample.scene_id,
            "target_id": sample.target_id,
            "category": sample.target,
        }
        for sample in dataset
    ]


def _candidate_rows(
    rows: list[dict[str, Any]],
    prediction_ids: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if row["sample_id"] in prediction_ids]


def select_subset(
    rows: list[dict[str, Any]],
    prediction_ids: set[str],
    n: int = N,
) -> list[dict[str, Any]]:
    """Inner-join canonical rows with predictions and select a frozen subset."""
    candidates = sorted(
        _candidate_rows(rows, prediction_ids),
        key=lambda row: stable_key(row["sample_id"]),
    )
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} candidates available, need {n}")
    return candidates[:n]


def write_outputs(
    selected: list[dict[str, Any]],
    sample_out: Path,
    summary_out: Path,
    n_candidates: int,
) -> None:
    """Write selected sample rows and a compact deterministic-selection summary."""
    sample_rows = [
        {
            "sample_id": row["sample_id"],
            "scene_id": row["scene_id"],
            "target_id": row["target_id"],
            "category": row["category"],
        }
        for row in selected
    ]
    summary = {
        "selection_salt": SELECTION_SALT,
        "n_selected": len(sample_rows),
        "n_candidates": n_candidates,
        "sample_ids_path": str(sample_out),
        "first_10_sample_ids": [row["sample_id"] for row in sample_rows[:10]],
    }

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
    parser.add_argument("--side-by-side", type=Path, default=DEFAULT_SIDE_BY_SIDE)
    parser.add_argument("--nr3d-root", type=Path, default=DEFAULT_NR3D_ROOT)
    parser.add_argument(
        "--phase8-data-root",
        type=Path,
        default=DEFAULT_PHASE8_DATA_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--backend", default="pack_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError(f"--n must be > 0, got {args.n}")

    prediction_ids = load_prediction_ids(args.side_by_side, backend=args.backend)
    rows = load_canonical_rows(args.nr3d_root, args.phase8_data_root)
    n_candidates = len(_candidate_rows(rows, prediction_ids))
    selected = select_subset(rows, prediction_ids, n=args.n)
    write_outputs(
        selected,
        sample_out=args.output,
        summary_out=args.summary,
        n_candidates=n_candidates,
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
