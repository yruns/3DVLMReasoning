from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conceptgraph import generate_conceptgraph_proposals
from .evaluator import evaluate_records
from .targets import load_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EmbodiedScan 3D bbox feasibility harness"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    smoke = subcommands.add_parser("smoke", help="Run ConceptGraph smoke evaluation")
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument("--scene-data-root", type=Path, required=True)
    smoke.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embodiedscan_bbox_feasibility"),
    )
    smoke.add_argument("--max-targets", type=int, default=50)
    smoke.add_argument("--mini", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        run_smoke(args)


def run_smoke(args: argparse.Namespace) -> None:
    targets = load_targets(
        str(args.data_root),
        split="val",
        source_filter="scannet",
        max_samples=None,
        mini=args.mini,
    )[: args.max_targets]

    records = []
    seen_scan_ids: set[str] = set()
    for target in targets:
        if target.scan_id in seen_scan_ids:
            continue
        seen_scan_ids.add(target.scan_id)
        scene_name = target.scan_id.split("/")[-1]
        scene_path = args.scene_data_root / scene_name / "conceptgraph"
        records.append(
            generate_conceptgraph_proposals(
                scene_path=scene_path,
                scan_id=target.scan_id,
                scene_id=target.scene_id,
            )
        )

    result = evaluate_records(targets, records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "smoke_metrics.json"
    scores_path = args.output_dir / "smoke_scores.jsonl"

    metrics_path.write_text(result.metrics.model_dump_json(indent=2), encoding="utf-8")
    with scores_path.open("w", encoding="utf-8") as handle:
        for score in result.scores:
            handle.write(json.dumps(score.model_dump(), ensure_ascii=False) + "\n")

    print(f"Wrote {metrics_path}")
    print(f"Wrote {scores_path}")


if __name__ == "__main__":
    main()
