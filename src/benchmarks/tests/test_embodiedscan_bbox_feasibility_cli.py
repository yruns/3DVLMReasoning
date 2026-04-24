import json
from argparse import Namespace

from benchmarks.embodiedscan_bbox_feasibility.cli import build_parser, run_smoke
from benchmarks.embodiedscan_bbox_feasibility.models import (
    AggregateMetrics,
    EmbodiedScanTarget,
    ProposalRecord,
    TargetScore,
)


def test_parser_accepts_smoke_mode() -> None:
    args = build_parser().parse_args(
        [
            "smoke",
            "--data-root",
            "data/embodiedscan",
            "--scene-data-root",
            "data/embodiedscan/scannet",
            "--max-targets",
            "5",
        ]
    )
    assert args.command == "smoke"
    assert args.max_targets == 5


def test_run_smoke_writes_metrics_and_scores(tmp_path, monkeypatch) -> None:
    from benchmarks.embodiedscan_bbox_feasibility import cli

    target = EmbodiedScanTarget(
        sample_ids=["sample-a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=1,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1],
    )

    def fake_load_targets(*_args, **_kwargs):
        return [target]

    def fake_generate_conceptgraph_proposals(*, scene_path, scan_id, scene_id):
        assert scene_path.name == "conceptgraph"
        return ProposalRecord(
            scene_id=scene_id,
            scan_id=scan_id,
            method="2d-cg",
            input_condition="conceptgraph_scene",
        )

    def fake_evaluate_records(targets, records):
        assert targets == [target]
        assert records[0].scan_id == "scannet/scene0001_00"
        return Namespace(
            metrics=AggregateMetrics(
                method="2d-cg",
                input_condition="conceptgraph_scene",
                num_targets=1,
                mean_best_iou=0.5,
                median_best_iou=0.5,
                acc_025=1.0,
                acc_050=1.0,
                mean_proposals_per_record=0.0,
                non_degenerate_box_ratio=0.0,
            ),
            scores=[
                TargetScore(
                    scan_id=target.scan_id,
                    scene_id=target.scene_id,
                    target_id=target.target_id,
                    method="2d-cg",
                    input_condition="conceptgraph_scene",
                    best_iou=0.5,
                )
            ],
        )

    monkeypatch.setattr(cli, "load_targets", fake_load_targets)
    monkeypatch.setattr(
        cli,
        "generate_conceptgraph_proposals",
        fake_generate_conceptgraph_proposals,
    )
    monkeypatch.setattr(cli, "evaluate_records", fake_evaluate_records)

    output_dir = tmp_path / "smoke"
    run_smoke(
        Namespace(
            data_root=tmp_path / "data",
            scene_data_root=tmp_path / "scannet",
            output_dir=output_dir,
            max_targets=5,
            mini=False,
        )
    )

    metrics = json.loads((output_dir / "smoke_metrics.json").read_text())
    scores = (output_dir / "smoke_scores.jsonl").read_text().splitlines()
    assert metrics["mean_best_iou"] == 0.5
    assert len(scores) == 1
    assert json.loads(scores[0])["best_iou"] == 0.5
