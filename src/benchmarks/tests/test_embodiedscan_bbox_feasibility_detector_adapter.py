import json
from pathlib import Path

import pytest

from benchmarks.embodiedscan_bbox_feasibility.detector_adapter import (
    load_detector_proposals_json,
    model_blocked_record,
)
from benchmarks.embodiedscan_bbox_feasibility.models import FailureTag


def test_load_detector_proposals_json(tmp_path: Path) -> None:
    path = tmp_path / "pred.json"
    path.write_text(
        json.dumps(
            {
                "proposals": [
                    {
                        "bbox_3d": [0, 0, 0, 1, 1, 1],
                        "score": 0.9,
                        "label": "chair",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    record = load_detector_proposals_json(
        path=path,
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        method="3d-scannet-full-detector",
        input_condition="scannet_full",
    )

    assert len(record.proposals) == 1
    assert record.proposals[0].source == "detector"
    assert record.proposals[0].metadata["label"] == "chair"
    assert record.proposals[0].metadata["path"] == str(path)
    assert record.failure_tag is None


def test_load_detector_proposals_json_marks_empty_file_as_no_proposal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pred.json"
    path.write_text(json.dumps({"proposals": []}), encoding="utf-8")

    record = load_detector_proposals_json(
        path=path,
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        method="3d-scannet-full-detector",
        input_condition="scannet_full",
        target_id=3,
    )

    assert record.target_id == 3
    assert record.proposals == []
    assert record.failure_tag == FailureTag.NO_PROPOSAL


def test_load_detector_proposals_json_rejects_missing_proposals_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pred.json"
    path.write_text(json.dumps({"boxes": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="proposals"):
        load_detector_proposals_json(
            path=path,
            scene_id="scene0001_00",
            scan_id="scannet/scene0001_00",
            method="3d-scannet-full-detector",
            input_condition="scannet_full",
        )


def test_model_blocked_record_is_explicit() -> None:
    record = model_blocked_record(
        scene_id="scene0001_00",
        scan_id="scannet/scene0001_00",
        method="3d-sf-recon-dest-vdetr",
        input_condition="single_frame_recon",
        reason="DEST-VDETR checkpoint unavailable",
    )
    assert record.failure_tag == FailureTag.MODEL_BLOCKED
    assert record.metadata["reason"] == "DEST-VDETR checkpoint unavailable"
    assert record.proposals == []
