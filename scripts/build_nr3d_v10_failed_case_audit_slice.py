#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIT_SAMPLE_IDS = [
    "scannet/scene0696_00::21::29050",
    "scannet/scene0704_00::10::7262",
    "scannet/scene0699_00::26::40486",
    "scannet/scene0025_00::26::38922",
    "scannet/scene0095_00::23::37291",
    "scannet/scene0208_00::106::14558",
    "scannet/scene0222_00::20::39339",
    "scannet/scene0025_00::17::37165",
    "scannet/scene0030_00::18::30641",
    "scannet/scene0231_00::45::38236",
    "scannet/scene0565_00::4::9604",
    "scannet/scene0629_00::14::17248",
    "scannet/scene0011_00::23::15020",
    "scannet/scene0030_00::23::28411",
    "scannet/scene0081_00::5::26430",
    "scannet/scene0144_00::17::37726",
    "scannet/scene0221_00::13::4514",
    "scannet/scene0329_00::39::35152",
    "scannet/scene0378_00::41::26109",
    "scannet/scene0249_00::36::39983",
    "scannet/scene0249_00::23::30460",
    "scannet/scene0690_00::17::19402",
    "scannet/scene0221_00::46::36120",
    "scannet/scene0343_00::16::35014",
    "scannet/scene0644_00::35::19090",
    "scannet/scene0208_00::17::21496",
    "scannet/scene0222_00::7::10110",
    "scannet/scene0496_00::29::10819",
    "scannet/scene0565_00::23::30788",
    "scannet/scene0568_00::0::41395",
    "scannet/scene0647_00::17::34181",
    "scannet/scene0591_00::5::17135",
    "scannet/scene0655_00::19::31620",
    "scannet/scene0651_00::7::6342",
    "scannet/scene0222_00::13::738",
    "scannet/scene0351_00::22::23771",
    "scannet/scene0081_00::0::10619",
    "scannet/scene0246_00::39::37886",
    "scannet/scene0329_00::6::26630",
    "scannet/scene0095_00::29::20881",
    "scannet/scene0490_00::13::36559",
]


def _sample_ids_from_json(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise SystemExit(f"{path} must contain a JSON list")
    sample_ids: set[str] = set()
    for idx, item in enumerate(payload):
        if isinstance(item, str):
            sample_ids.add(item)
            continue
        if isinstance(item, dict) and isinstance(item.get("sample_id"), str):
            sample_ids.add(item["sample_id"])
            continue
        raise SystemExit(
            f"{path} item {idx} must be a sample-id string or dict with sample_id"
        )
    return sample_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strat600", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    strat600 = _sample_ids_from_json(args.strat600)
    missing = [sid for sid in AUDIT_SAMPLE_IDS if sid not in strat600]
    if missing:
        raise SystemExit(f"audit sample ids missing from strat600: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(AUDIT_SAMPLE_IDS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(AUDIT_SAMPLE_IDS)} sample ids to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
