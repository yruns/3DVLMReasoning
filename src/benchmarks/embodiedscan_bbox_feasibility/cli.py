from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conceptgraph import generate_conceptgraph_proposals
from .detector_runner import run_detector_command
from .evaluator import evaluate_records
from .models import DetectorInputRecord, ProposalRecord
from .pointcloud_inputs import (
    SUPPORTED_CONDITIONS,
    materialize_detector_input,
    materialize_scene_full_detector_input,
)
from .targets import load_targets
from .vdetr import build_vdetr_command_template


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
    prepare.add_argument(
        "--target-categories",
        nargs="+",
        default=None,
        help="Keep only targets whose category matches these names.",
    )
    prepare.add_argument(
        "--require-visible-frames",
        action="store_true",
        help="Keep only targets with at least one annotated visible frame.",
    )
    prepare.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Keep targets from at most this many highest-yield scenes.",
    )
    prepare.add_argument(
        "--max-targets-per-scene",
        type=int,
        default=None,
        help="Keep at most this many selected targets per scene.",
    )
    prepare.add_argument(
        "--scene-level-full",
        action="store_true",
        help="For scannet_full, write one scene-level detector input per scene.",
    )

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
    run.add_argument(
        "--detector-profile",
        choices=["custom", "vdetr"],
        default="custom",
        help="Use a built-in detector command template when command-template is omitted.",
    )
    run.add_argument("--vdetr-repo-dir", type=Path, default=Path("external/V-DETR"))
    run.add_argument(
        "--vdetr-checkpoint",
        type=Path,
        default=Path("external/V-DETR/checkpoints/scannet_540ep.pth"),
    )
    run.add_argument("--vdetr-python", default="python")
    run.add_argument("--vdetr-num-points", type=int, default=40000)
    run.add_argument("--vdetr-conf-thresh", type=float, default=0.05)
    run.add_argument("--vdetr-top-k", type=int, default=256)
    run.add_argument("--cwd", type=Path, default=None)
    run.add_argument("--cuda-device", default="0")
    run.add_argument("--timeout-seconds", type=int, default=None)

    evaluate = subcommands.add_parser(
        "evaluate-records",
        help="Evaluate existing proposal records and write per-condition metrics",
    )
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--records-jsonl", type=Path, required=True)
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/embodiedscan_bbox_feasibility/eval"),
    )
    evaluate.add_argument("--max-targets", type=int, default=None)
    evaluate.add_argument("--mini", action="store_true")
    evaluate.add_argument("--split", default="val")
    evaluate.add_argument("--source-filter", default="scannet")
    evaluate.add_argument(
        "--records-targets-only",
        action="store_true",
        help="Evaluate only targets that appear in the proposal records.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        run_smoke(args)
    elif args.command == "prepare-inputs":
        run_prepare_inputs(args)
    elif args.command == "run-detector":
        run_detector(args)
    elif args.command == "evaluate-records":
        run_evaluate_records(args)


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
                axis_align_matrix=target.axis_align_matrix,
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
    )
    targets = select_targets_for_detector_batch(
        targets,
        target_categories=getattr(args, "target_categories", None),
        require_visible_frames=getattr(args, "require_visible_frames", False),
        max_scenes=getattr(args, "max_scenes", None),
        max_targets_per_scene=getattr(args, "max_targets_per_scene", None),
        max_targets=args.max_targets,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "detector_inputs.jsonl"
    conditions = list(args.conditions)
    write_scene_level_full = (
        getattr(args, "scene_level_full", False) and "scannet_full" in conditions
    )
    target_conditions = [
        condition
        for condition in conditions
        if not (write_scene_level_full and condition == "scannet_full")
    ]
    with records_path.open("w", encoding="utf-8") as handle:
        for target in targets:
            for condition in target_conditions:
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
        if write_scene_level_full:
            for target in _first_target_per_scene(targets):
                record = materialize_scene_full_detector_input(
                    target=target,
                    output_dir=args.output_dir,
                    scannet_root=args.scannet_root,
                    max_points=args.max_points,
                )
                handle.write(
                    json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
                )

    print(f"Wrote {records_path}")


def select_targets_for_detector_batch(
    targets: list,
    *,
    target_categories: list[str] | None = None,
    require_visible_frames: bool = False,
    max_scenes: int | None = None,
    max_targets_per_scene: int | None = None,
    max_targets: int | None = None,
) -> list:
    if max_scenes is not None and max_scenes <= 0:
        raise ValueError("max_scenes must be positive")
    if max_targets_per_scene is not None and max_targets_per_scene <= 0:
        raise ValueError("max_targets_per_scene must be positive")
    if max_targets is not None and max_targets < 0:
        raise ValueError("max_targets must be non-negative")

    allowed_categories = (
        {_normalize_category(category) for category in target_categories}
        if target_categories
        else None
    )

    eligible = []
    for target in targets:
        if allowed_categories is not None and (
            _normalize_category(target.target_category) not in allowed_categories
        ):
            continue
        if require_visible_frames and not target.visible_frame_ids:
            continue
        eligible.append(target)

    if max_scenes is not None:
        scene_counts: dict[str, int] = {}
        for target in eligible:
            scene_counts[target.scene_id] = scene_counts.get(target.scene_id, 0) + 1
        selected_scenes = {
            scene_id
            for scene_id, _count in sorted(
                scene_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:max_scenes]
        }
        eligible = [target for target in eligible if target.scene_id in selected_scenes]

    if max_targets_per_scene is not None:
        scene_seen: dict[str, int] = {}
        capped = []
        for target in eligible:
            seen = scene_seen.get(target.scene_id, 0)
            if seen >= max_targets_per_scene:
                continue
            scene_seen[target.scene_id] = seen + 1
            capped.append(target)
        eligible = capped

    if max_targets is not None:
        eligible = eligible[:max_targets]
    return eligible


def run_detector(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "detector_records.jsonl"
    command_template = _resolve_detector_command_template(args)
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
                command_template=command_template,
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


def run_evaluate_records(args: argparse.Namespace) -> None:
    targets = load_targets(
        str(args.data_root),
        split=args.split,
        source_filter=args.source_filter,
        max_samples=None,
        mini=args.mini,
    )
    with args.records_jsonl.open("r", encoding="utf-8") as handle:
        records = [
            ProposalRecord.model_validate(json.loads(line))
            for line in handle
            if line.strip()
        ]
    if not records:
        raise ValueError(f"records-jsonl contains no records: {args.records_jsonl}")
    if getattr(args, "records_targets_only", False):
        targets = _filter_targets_present_in_records(targets, records)
    if args.max_targets is not None:
        targets = targets[: args.max_targets]

    grouped_records: dict[tuple[str, str], list[ProposalRecord]] = {}
    for record in records:
        grouped_records.setdefault((record.method, record.input_condition), []).append(
            record
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    all_scores_path = args.output_dir / "scores.jsonl"
    metrics_rows = []
    with metrics_path.open("w", encoding="utf-8") as metrics_handle, all_scores_path.open(
        "w", encoding="utf-8"
    ) as all_scores_handle:
        for (method, condition), group in grouped_records.items():
            result = evaluate_records(targets, group)
            metrics_row = result.metrics.model_dump()
            metrics_rows.append(metrics_row)
            metrics_handle.write(json.dumps(metrics_row, ensure_ascii=False) + "\n")

            scores_path = args.output_dir / f"scores_{_safe_name(method)}_{_safe_name(condition)}.jsonl"
            with scores_path.open("w", encoding="utf-8") as scores_handle:
                for score in result.scores:
                    row = score.model_dump()
                    scores_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    all_scores_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "records_jsonl": str(args.records_jsonl),
                "num_targets": len(targets),
                "num_groups": len(metrics_rows),
                "metrics": metrics_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {all_scores_path}")


def _resolve_detector_command_template(args: argparse.Namespace) -> str | None:
    if args.command_template:
        return args.command_template
    if getattr(args, "detector_profile", "custom") == "vdetr":
        return build_vdetr_command_template(
            repo_dir=args.vdetr_repo_dir,
            checkpoint_path=args.vdetr_checkpoint,
            python_executable=args.vdetr_python,
            num_points=args.vdetr_num_points,
            conf_thresh=args.vdetr_conf_thresh,
            top_k=args.vdetr_top_k,
        )
    return None


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


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _filter_targets_present_in_records(
    targets: list,
    records: list[ProposalRecord],
) -> list:
    target_keys = {
        (record.scan_id, record.target_id)
        for record in records
        if record.target_id is not None
    }
    scene_level_scan_ids = {
        record.scan_id for record in records if record.target_id is None
    }
    return [
        target
        for target in targets
        if (target.scan_id, target.target_id) in target_keys
        or target.scan_id in scene_level_scan_ids
    ]


def _first_target_per_scene(targets: list) -> list:
    first_by_scene = {}
    for target in targets:
        first_by_scene.setdefault(target.scene_id, target)
    return list(first_by_scene.values())


def _normalize_category(value: str) -> str:
    return value.lower().replace(" ", "").replace("_", "")


if __name__ == "__main__":
    main()
