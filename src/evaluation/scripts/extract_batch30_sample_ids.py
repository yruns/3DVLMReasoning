"""Extract unique batch30 sample ids from feasibility CSV scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = ("scene_id", "target_id", "category")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def extract_sample_ids(csv_path: Path) -> list[dict[str, Any]]:
    """Read batch30_scores.csv and return unique targets in first-seen order."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for column in REQUIRED_COLUMNS:
            if column not in (reader.fieldnames or []):
                raise ValueError(f"Missing required CSV column: {column}")

        for row_number, row in enumerate(reader, start=2):
            scene_id = (row.get("scene_id") or "").strip()
            if not scene_id or scene_id.lower() == "nan":
                raise ValueError(f"Row {row_number} has invalid scene_id: {row!r}")

            raw_target_id = (row.get("target_id") or "").strip()
            try:
                target_id = int(raw_target_id)
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_number} has invalid target_id: {row!r}"
                ) from exc

            key = (scene_id, target_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "sample_id": f"{scene_id}::{target_id}",
                    "scene_id": scene_id,
                    "target_id": target_id,
                    "category": (row.get("category") or "").strip(),
                }
            )

    if not out:
        raise ValueError(f"CSV is empty: {csv_path}")
    return out


def main() -> None:
    args = parse_args()
    sample_ids = extract_sample_ids(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sample_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
