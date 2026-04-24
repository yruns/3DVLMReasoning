from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conceptgraph import generate_conceptgraph_proposals
from .detector_runner import run_detector_command
from .evaluator import evaluate_records
from .models import DetectorInputRecord
from .pointcloud_inputs import SUPPORTED_CONDITIONS, materialize_detector_input
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

    prepare = subcommands.add_parser(
        "prepare-inputs",
        help="Materialize detector point-cloud inputs or explicit blocked records",
    )
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--scene-data-root", type=Path, required=True)
    prepare.add_argument("--scannet-root", type=Path, default=None)
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embodiedscan_bbox_feasibility/detector_inputs"),
    )
    prepare.add_argument("--max-targets", type=int, default=50)
    prepare.add_argument("--mini", action="store_true")
    prepare.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted(SUPPORTED_CONDITIONS),
        default=[
            "single_frame_recon",
            "multi_frame_recon",
            "scannet_pose_crop",
            "scannet_full",
        ],
    )
    prepare.add_argument("--multi-frame-size", type=int, default=5)
    prepare.add_argument("--crop-padding", type=float, default=1.5)
    prepare.add_argument("--depth-scale", type=float, default=1000.0)
    prepare.add_argument("--max-points", type=int, default=40000)

    run = subcommands.add_parser(
        "run-detector",
        help="Run an external detector command template over prepared inputs",
    )
    run.add_argument("--inputs-jsonl", type=Path, required=True)
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embodiedscan_bbox_feasibility/detector_outputs"),
    )
    run.add_argument("--method", default="3d-vdetr")
    run.add_argument("--command-template", default=None)
    run.add_argument("--cwd", type=Path, default=None)
    run.add_argument("--cuda-device", default="0")
    run.add_argument("--timeout-seconds", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        run_smoke(args)
    elif args.command == "prepare-inputs":
        run_prepare_inputs(args)
    elif args.command == "run-detector":
        run_detector(args)


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


def run_prepare_inputs(args: argparse.Namespace) -> None:
    targets = load_targets(
        str(args.data_root),
        split="val",
        source_filter="scannet",
        max_samples=None,
        mini=args.mini,
    )[: args.max_targets]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "detector_inputs.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for target in targets:
            for condition in args.conditions:
                record = materialize_detector_input(
                    target=target,
                    condition=condition,
                    scene_data_root=args.scene_data_root,
                    output_dir=args.output_dir,
                    scannet_root=args.scannet_root,
                    multi_frame_size=args.multi_frame_size,
                    crop_padding=args.crop_padding,
                    depth_scale=args.depth_scale,
                    max_points=args.max_points,
                )
                handle.write(
                    json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
                )

    print(f"Wrote {records_path}")


def run_detector(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "detector_records.jsonl"
    with args.inputs_jsonl.open("r", encoding="utf-8") as in_handle:
        input_records = [
            DetectorInputRecord.model_validate(json.loads(line))
            for line in in_handle
            if line.strip()
        ]

    with records_path.open("w", encoding="utf-8") as out_handle:
        for input_record in input_records:
            output_path = _detector_prediction_path(args.output_dir, input_record)
            proposal_record = run_detector_command(
                input_record=input_record,
                command_template=args.command_template,
                output_path=output_path,
                method=args.method,
                cwd=args.cwd,
                cuda_device=args.cuda_device,
                timeout_seconds=args.timeout_seconds,
            )
            out_handle.write(
                json.dumps(proposal_record.model_dump(), ensure_ascii=False) + "\n"
            )

    print(f"Wrote {records_path}")


def _detector_prediction_path(
    output_dir: Path,
    input_record: DetectorInputRecord,
) -> Path:
    target = (
        "scene"
        if input_record.target_id is None
        else f"target{input_record.target_id}"
    )
    return (
        output_dir
        / "predictions"
        / f"{input_record.scene_id}_{target}_{input_record.input_condition}.json"
    )


if __name__ == "__main__":
    main()
