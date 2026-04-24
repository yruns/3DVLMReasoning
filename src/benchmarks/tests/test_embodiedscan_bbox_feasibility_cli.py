import json
from argparse import Namespace

from benchmarks.embodiedscan_bbox_feasibility.cli import build_parser, run_smoke
from benchmarks.embodiedscan_bbox_feasibility.models import (
    AggregateMetrics,
    BBox3DProposal,
    DetectorInputRecord,
    EmbodiedScanTarget,
    FailureTag,
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


def test_parser_accepts_prepare_inputs_mode() -> None:
    args = build_parser().parse_args(
        [
            "prepare-inputs",
            "--data-root",
            "data/embodiedscan",
            "--scene-data-root",
            "data/embodiedscan/scannet",
            "--scannet-root",
            "/home/ysh/Datasets/ScanNet",
            "--conditions",
            "single_frame_recon",
            "scannet_pose_crop",
            "--max-targets",
            "3",
        ]
    )
    assert args.command == "prepare-inputs"
    assert args.conditions == ["single_frame_recon", "scannet_pose_crop"]
    assert args.max_targets == 3


def test_parser_accepts_run_detector_mode() -> None:
    args = build_parser().parse_args(
        [
            "run-detector",
            "--inputs-jsonl",
            "inputs.jsonl",
            "--output-dir",
            "outputs/detector",
            "--method",
            "3d-vdetr",
            "--cuda-device",
            "0",
        ]
    )
    assert args.command == "run-detector"
    assert args.method == "3d-vdetr"
    assert str(args.inputs_jsonl) == "inputs.jsonl"


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


def test_run_prepare_inputs_writes_jsonl_records(tmp_path, monkeypatch) -> None:
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

    def fake_materialize_detector_input(**kwargs):
        assert kwargs["target"] == target
        return DetectorInputRecord(
            scan_id=target.scan_id,
            scene_id=target.scene_id,
            target_id=target.target_id,
            input_condition=kwargs["condition"],
            failure_tag=FailureTag.INPUT_BLOCKED,
            metadata={"reason": "unit blocked"},
        )

    monkeypatch.setattr(cli, "load_targets", fake_load_targets)
    monkeypatch.setattr(cli, "materialize_detector_input", fake_materialize_detector_input)

    output_dir = tmp_path / "prepared"
    cli.run_prepare_inputs(
        Namespace(
            data_root=tmp_path / "data",
            scene_data_root=tmp_path / "scenes",
            scannet_root=tmp_path / "scannet",
            output_dir=output_dir,
            max_targets=5,
            mini=False,
            conditions=["single_frame_recon", "scannet_full"],
            multi_frame_size=5,
            crop_padding=1.5,
            depth_scale=1000.0,
            max_points=None,
        )
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "detector_inputs.jsonl").read_text().splitlines()
    ]
    assert [row["input_condition"] for row in rows] == [
        "single_frame_recon",
        "scannet_full",
    ]
    assert rows[0]["failure_tag"] == "input_blocked"


def test_run_detector_writes_proposal_records(tmp_path, monkeypatch) -> None:
    from benchmarks.embodiedscan_bbox_feasibility import cli

    inputs_path = tmp_path / "inputs.jsonl"
    inputs_path.write_text(
        DetectorInputRecord(
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
            target_id=1,
            input_condition="scannet_full",
            pointcloud_path=str(tmp_path / "cloud.ply"),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    def fake_run_detector_command(**kwargs):
        output_path = kwargs["output_path"]
        assert output_path.name == "scene0001_00_target1_scannet_full.json"
        return ProposalRecord(
            scene_id="scene0001_00",
            scan_id="scannet/scene0001_00",
            target_id=1,
            method=kwargs["method"],
            input_condition="scannet_full",
            proposals=[
                BBox3DProposal(
                    bbox_3d=[0, 0, 0, 1, 1, 1],
                    score=0.7,
                    source="unit",
                )
            ],
        )

    monkeypatch.setattr(cli, "run_detector_command", fake_run_detector_command)

    output_dir = tmp_path / "detector"
    cli.run_detector(
        Namespace(
            inputs_jsonl=inputs_path,
            output_dir=output_dir,
            method="3d-vdetr",
            command_template="unit",
            cwd=None,
            cuda_device="0",
            timeout_seconds=None,
        )
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "detector_records.jsonl").read_text().splitlines()
    ]
    assert rows[0]["method"] == "3d-vdetr"
    assert rows[0]["proposals"][0]["score"] == 0.7
