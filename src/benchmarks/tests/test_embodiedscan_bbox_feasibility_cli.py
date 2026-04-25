import json
from argparse import Namespace

from benchmarks.embodiedscan_bbox_feasibility.cli import (
    build_parser,
    run_smoke,
    select_targets_for_detector_batch,
)
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


def test_parser_accepts_prepare_inputs_batch_filters() -> None:
    args = build_parser().parse_args(
        [
            "prepare-inputs",
            "--data-root",
            "data/embodiedscan",
            "--scene-data-root",
            "data/embodiedscan/scannet",
            "--target-categories",
            "chair",
            "cabinet",
            "--require-visible-frames",
            "--max-scenes",
            "10",
            "--max-targets-per-scene",
            "3",
            "--scene-level-full",
        ]
    )

    assert args.target_categories == ["chair", "cabinet"]
    assert args.require_visible_frames is True
    assert args.max_scenes == 10
    assert args.max_targets_per_scene == 3
    assert args.scene_level_full is True


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


def test_parser_accepts_evaluate_records_mode() -> None:
    args = build_parser().parse_args(
        [
            "evaluate-records",
            "--data-root",
            "data/embodiedscan",
            "--records-jsonl",
            "detector_records.jsonl",
            "--output-dir",
            "outputs/eval",
            "--max-targets",
            "10",
            "--records-targets-only",
        ]
    )
    assert args.command == "evaluate-records"
    assert str(args.records_jsonl) == "detector_records.jsonl"
    assert args.max_targets == 10
    assert args.records_targets_only is True


def test_parser_accepts_vdetr_detector_profile() -> None:
    args = build_parser().parse_args(
        [
            "run-detector",
            "--inputs-jsonl",
            "inputs.jsonl",
            "--detector-profile",
            "vdetr",
            "--vdetr-repo-dir",
            "external/V-DETR",
            "--vdetr-checkpoint",
            "external/V-DETR/checkpoints/scannet_540ep.pth",
            "--vdetr-python",
            "/opt/vdetr/bin/python",
        ]
    )

    assert args.detector_profile == "vdetr"
    assert str(args.vdetr_repo_dir) == "external/V-DETR"
    assert str(args.vdetr_python) == "/opt/vdetr/bin/python"


def test_run_smoke_writes_metrics_and_scores(tmp_path, monkeypatch) -> None:
    from benchmarks.embodiedscan_bbox_feasibility import cli

    target = EmbodiedScanTarget(
        sample_ids=["sample-a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=1,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1],
        axis_align_matrix=[
            [1, 0, 0, 10],
            [0, 1, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1],
        ],
    )

    def fake_load_targets(*_args, **_kwargs):
        return [target]

    def fake_generate_conceptgraph_proposals(
        *,
        scene_path,
        scan_id,
        scene_id,
        axis_align_matrix,
    ):
        assert scene_path.name == "conceptgraph"
        assert axis_align_matrix == target.axis_align_matrix
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


def test_select_targets_for_detector_batch_filters_and_caps_by_scene() -> None:
    def target(scene_id: str, target_id: int, category: str, visible: list[int]):
        return EmbodiedScanTarget(
            sample_ids=[f"{scene_id}-{target_id}"],
            scan_id=f"scannet/{scene_id}",
            scene_id=scene_id,
            target_id=target_id,
            target_category=category,
            gt_bbox_3d=[0, 0, 0, 1, 1, 1],
            visible_frame_ids=visible,
        )

    selected = select_targets_for_detector_batch(
        [
            target("scene_a", 1, "chair", [0]),
            target("scene_a", 2, "chair", [1]),
            target("scene_a", 3, "chair", [2]),
            target("scene_b", 4, "cabinet", [0]),
            target("scene_b", 5, "door", [1]),
            target("scene_c", 6, "cabinet", []),
        ],
        target_categories=["chair", "cabinet"],
        require_visible_frames=True,
        max_scenes=2,
        max_targets_per_scene=2,
    )

    assert [(target.scene_id, target.target_id) for target in selected] == [
        ("scene_a", 1),
        ("scene_a", 2),
        ("scene_b", 4),
    ]


def test_run_prepare_inputs_can_write_scene_level_scannet_full_once_per_scene(
    tmp_path,
    monkeypatch,
) -> None:
    from benchmarks.embodiedscan_bbox_feasibility import cli

    targets = [
        EmbodiedScanTarget(
            sample_ids=["sample-a"],
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
            target_id=1,
            target_category="chair",
            gt_bbox_3d=[0, 0, 0, 1, 1, 1],
            visible_frame_ids=[0],
        ),
        EmbodiedScanTarget(
            sample_ids=["sample-b"],
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
            target_id=2,
            target_category="cabinet",
            gt_bbox_3d=[0, 0, 0, 1, 1, 1],
            visible_frame_ids=[1],
        ),
    ]

    def fake_materialize_detector_input(**kwargs):
        target = kwargs["target"]
        return DetectorInputRecord(
            scan_id=target.scan_id,
            scene_id=target.scene_id,
            target_id=target.target_id,
            input_condition=kwargs["condition"],
            pointcloud_path=str(tmp_path / f"{target.target_id}.ply"),
        )

    def fake_materialize_scene_full_detector_input(**kwargs):
        target = kwargs["target"]
        return DetectorInputRecord(
            scan_id=target.scan_id,
            scene_id=target.scene_id,
            target_id=None,
            input_condition="scannet_full",
            pointcloud_path=str(tmp_path / "scene.ply"),
        )

    monkeypatch.setattr(cli, "load_targets", lambda *_args, **_kwargs: targets)
    monkeypatch.setattr(cli, "materialize_detector_input", fake_materialize_detector_input)
    monkeypatch.setattr(
        cli,
        "materialize_scene_full_detector_input",
        fake_materialize_scene_full_detector_input,
    )

    cli.run_prepare_inputs(
        Namespace(
            data_root=tmp_path / "data",
            scene_data_root=tmp_path / "scenes",
            scannet_root=tmp_path / "scannet",
            output_dir=tmp_path / "prepared",
            max_targets=10,
            mini=False,
            conditions=["single_frame_recon", "scannet_full"],
            multi_frame_size=5,
            crop_padding=1.5,
            depth_scale=1000.0,
            max_points=None,
            target_categories=["chair", "cabinet"],
            require_visible_frames=True,
            max_scenes=None,
            max_targets_per_scene=None,
            scene_level_full=True,
        )
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "prepared" / "detector_inputs.jsonl")
        .read_text()
        .splitlines()
    ]

    assert [row["input_condition"] for row in rows] == [
        "single_frame_recon",
        "single_frame_recon",
        "scannet_full",
    ]
    assert [row["target_id"] for row in rows] == [1, 2, None]


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


def test_run_detector_uses_vdetr_profile_when_template_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    from benchmarks.embodiedscan_bbox_feasibility import cli

    inputs_path = tmp_path / "inputs.jsonl"
    inputs_path.write_text(
        DetectorInputRecord(
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
            target_id=1,
            input_condition="single_frame_recon",
            pointcloud_path=str(tmp_path / "cloud.ply"),
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    seen_templates = []

    def fake_run_detector_command(**kwargs):
        seen_templates.append(kwargs["command_template"])
        return ProposalRecord(
            scene_id="scene0001_00",
            scan_id="scannet/scene0001_00",
            target_id=1,
            method=kwargs["method"],
            input_condition="single_frame_recon",
        )

    monkeypatch.setattr(cli, "run_detector_command", fake_run_detector_command)

    cli.run_detector(
        Namespace(
            inputs_jsonl=inputs_path,
            output_dir=tmp_path / "detector",
            method="3d-vdetr",
            command_template=None,
            detector_profile="vdetr",
            vdetr_repo_dir=tmp_path / "V-DETR",
            vdetr_checkpoint=tmp_path / "scannet_540ep.pth",
            vdetr_python="/opt/vdetr/bin/python",
            vdetr_num_points=40000,
            vdetr_conf_thresh=0.05,
            vdetr_top_k=128,
            cwd=None,
            cuda_device="0",
            timeout_seconds=None,
        )
    )

    assert seen_templates
    assert seen_templates[0].startswith("/opt/vdetr/bin/python ")
    assert "vdetr_export_predictions.py" in seen_templates[0]
    assert "--top-k 128" in seen_templates[0]


def test_run_evaluate_records_groups_metrics_by_method_and_condition(
    tmp_path,
    monkeypatch,
) -> None:
    from benchmarks.embodiedscan_bbox_feasibility import cli

    target = EmbodiedScanTarget(
        sample_ids=["sample-a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=1,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1],
    )
    records_path = tmp_path / "detector_records.jsonl"
    records_path.write_text(
        "\n".join(
            [
                ProposalRecord(
                    scene_id=target.scene_id,
                    scan_id=target.scan_id,
                    target_id=target.target_id,
                    method="3d-vdetr",
                    input_condition="single_frame_recon",
                ).model_dump_json(),
                ProposalRecord(
                    scene_id=target.scene_id,
                    scan_id=target.scan_id,
                    target_id=target.target_id,
                    method="3d-vdetr",
                    input_condition="scannet_full",
                ).model_dump_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_load_targets(*_args, **_kwargs):
        return [target]

    def fake_evaluate_records(targets, records):
        return Namespace(
            metrics=AggregateMetrics(
                method=records[0].method,
                input_condition=records[0].input_condition,
                num_targets=len(targets),
                mean_best_iou=0.25,
                median_best_iou=0.25,
                acc_025=1.0,
                acc_050=0.0,
                mean_proposals_per_record=0.0,
                non_degenerate_box_ratio=0.0,
            ),
            scores=[
                TargetScore(
                    scan_id=target.scan_id,
                    scene_id=target.scene_id,
                    target_id=target.target_id,
                    method=records[0].method,
                    input_condition=records[0].input_condition,
                    best_iou=0.25,
                )
            ],
        )

    monkeypatch.setattr(cli, "load_targets", fake_load_targets)
    monkeypatch.setattr(cli, "evaluate_records", fake_evaluate_records)

    output_dir = tmp_path / "eval"
    cli.run_evaluate_records(
        Namespace(
            data_root=tmp_path / "data",
            records_jsonl=records_path,
            output_dir=output_dir,
            max_targets=None,
            mini=False,
            split="val",
            source_filter="scannet",
        )
    )

    metrics_rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["input_condition"] for row in metrics_rows] == [
        "single_frame_recon",
        "scannet_full",
    ]
    assert json.loads((output_dir / "metrics_summary.json").read_text())[
        "num_groups"
    ] == 2
    assert (output_dir / "scores_3d-vdetr_single_frame_recon.jsonl").exists()


def test_run_evaluate_records_can_filter_to_targets_present_in_records(
    tmp_path,
    monkeypatch,
) -> None:
    from benchmarks.embodiedscan_bbox_feasibility import cli

    targets = [
        EmbodiedScanTarget(
            sample_ids=["sample-a"],
            scan_id="scannet/scene0001_00",
            scene_id="scene0001_00",
            target_id=1,
            target_category="chair",
            gt_bbox_3d=[0, 0, 0, 1, 1, 1],
        ),
        EmbodiedScanTarget(
            sample_ids=["sample-b"],
            scan_id="scannet/scene0002_00",
            scene_id="scene0002_00",
            target_id=8,
            target_category="table",
            gt_bbox_3d=[0, 0, 0, 1, 1, 1],
        ),
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        ProposalRecord(
            scene_id="scene0002_00",
            scan_id="scannet/scene0002_00",
            target_id=8,
            method="3d-vdetr",
            input_condition="scannet_full",
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    seen_target_ids = []

    def fake_evaluate_records(filtered_targets, records):
        seen_target_ids.extend(target.target_id for target in filtered_targets)
        return Namespace(
            metrics=AggregateMetrics(
                method=records[0].method,
                input_condition=records[0].input_condition,
                num_targets=len(filtered_targets),
                mean_best_iou=0.0,
                median_best_iou=0.0,
                acc_025=0.0,
                acc_050=0.0,
                mean_proposals_per_record=0.0,
                non_degenerate_box_ratio=0.0,
            ),
            scores=[],
        )

    monkeypatch.setattr(cli, "load_targets", lambda *_args, **_kwargs: targets)
    monkeypatch.setattr(cli, "evaluate_records", fake_evaluate_records)

    cli.run_evaluate_records(
        Namespace(
            data_root=tmp_path / "data",
            records_jsonl=records_path,
            output_dir=tmp_path / "eval",
            max_targets=None,
            mini=False,
            split="val",
            source_filter="scannet",
            records_targets_only=True,
        )
    )

    assert seen_target_ids == [8]
