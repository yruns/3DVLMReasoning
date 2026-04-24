from __future__ import annotations

import sys

import pytest

from benchmarks.embodiedscan_bbox_feasibility.detector_runner import (
    render_detector_command,
    run_detector_command,
)
from benchmarks.embodiedscan_bbox_feasibility.models import (
    DetectorInputRecord,
    FailureTag,
)


def _input_record(pointcloud_path: str) -> DetectorInputRecord:
    return DetectorInputRecord(
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=7,
        input_condition="scannet_full",
        pointcloud_path=pointcloud_path,
    )


def test_render_detector_command_rejects_broken_gpu_1(tmp_path) -> None:
    record = _input_record(str(tmp_path / "cloud.ply"))

    with pytest.raises(ValueError, match="GPU 1"):
        render_detector_command(
            command_template="python infer.py --input {pointcloud_path}",
            input_record=record,
            output_path=tmp_path / "pred.json",
            cuda_device="1",
        )


def test_render_detector_command_substitutes_known_fields(tmp_path) -> None:
    cloud = tmp_path / "cloud.ply"
    record = _input_record(str(cloud))

    command = render_detector_command(
        command_template=(
            "python infer.py --scene {scene_id} "
            "--input {pointcloud_path} --output {output_path}"
        ),
        input_record=record,
        output_path=tmp_path / "pred.json",
        cuda_device="0",
    )

    assert command == [
        "python",
        "infer.py",
        "--scene",
        "scene0001_00",
        "--input",
        str(cloud),
        "--output",
        str(tmp_path / "pred.json"),
    ]


def test_run_detector_command_returns_model_blocked_without_template(tmp_path) -> None:
    record = run_detector_command(
        input_record=_input_record(str(tmp_path / "cloud.ply")),
        command_template=None,
        output_path=tmp_path / "pred.json",
        method="3d-vdetr",
    )

    assert record.failure_tag == FailureTag.MODEL_BLOCKED
    assert "command template" in record.metadata["reason"]


def test_run_detector_command_executes_template_and_loads_output(tmp_path) -> None:
    cloud = tmp_path / "cloud.ply"
    cloud.write_text("ply\n", encoding="utf-8")
    script = tmp_path / "write_pred.py"
    script.write_text(
        "import json, sys\n"
        "out = sys.argv[2]\n"
        "with open(out, 'w', encoding='utf-8') as f:\n"
        "    json.dump({'proposals': [{'bbox_3d': [0, 0, 0, 1, 1, 1], 'score': 0.8}]}, f)\n",
        encoding="utf-8",
    )

    record = run_detector_command(
        input_record=_input_record(str(cloud)),
        command_template=f"{sys.executable} {script} {{pointcloud_path}} {{output_path}}",
        output_path=tmp_path / "pred.json",
        method="3d-vdetr",
        cuda_device="0",
    )

    assert record.failure_tag is None
    assert len(record.proposals) == 1
    assert record.proposals[0].score == 0.8
    assert record.metadata["detector_input_path"] == str(cloud)
