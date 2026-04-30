from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

EXTRACTOR_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "extract_nr3d_test_frames.py"
)
SPEC = importlib.util.spec_from_file_location("extract_nr3d_test_frames", EXTRACTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)

ExtractStats = extractor.ExtractStats
_normalize_raw_temp = extractor._normalize_raw_temp
_write_frame_extraction_log = extractor._write_frame_extraction_log


def _write_reader_frame(raw_temp: Path, frame_id: int, pose: np.ndarray) -> None:
    (raw_temp / "color").mkdir(parents=True, exist_ok=True)
    (raw_temp / "depth").mkdir(parents=True, exist_ok=True)
    (raw_temp / "pose").mkdir(parents=True, exist_ok=True)
    stem = str(frame_id)
    Image.new("RGB", (4, 3), color=(frame_id % 255, 0, 0)).save(
        raw_temp / "color" / f"{stem}.jpg"
    )
    Image.new("I;16", (4, 3), color=frame_id).save(
        raw_temp / "depth" / f"{stem}.png"
    )
    np.savetxt(raw_temp / "pose" / f"{stem}.txt", pose, fmt="%.8f")


def _write_intrinsics(raw_temp: Path) -> None:
    intrinsic_dir = raw_temp / "intrinsic"
    intrinsic_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "intrinsic_color.txt",
        "intrinsic_depth.txt",
        "extrinsic_color.txt",
        "extrinsic_depth.txt",
    ]:
        np.savetxt(intrinsic_dir / name, np.eye(4), fmt="%.8f")


def test_normalize_raw_temp_drops_invalid_pose_frames_and_keeps_original_ids(
    tmp_path: Path,
) -> None:
    raw_temp = tmp_path / "raw_temp"
    raw_dir = tmp_path / "raw"
    _write_intrinsics(raw_temp)
    valid_pose = np.eye(4)
    invalid_pose = np.eye(4)
    invalid_pose[0, 0] = np.nan
    _write_reader_frame(raw_temp, 0, valid_pose)
    _write_reader_frame(raw_temp, 10, invalid_pose)
    _write_reader_frame(raw_temp, 20, valid_pose)

    kept, dropped_bad_pose, raw_total = _normalize_raw_temp(
        "scene0001_00",
        raw_temp,
        raw_dir,
        stride=10,
        sens_path=tmp_path / "scene0001_00.sens",
        axis_alignment=np.eye(4),
        invalid_pose_policy="drop",
        max_bad_pose_ratio=0.5,
    )

    assert (kept, dropped_bad_pose, raw_total) == (2, 1, 3)
    assert (raw_dir / "000000-rgb.png").exists()
    assert not (raw_dir / "000010-rgb.png").exists()
    assert (raw_dir / "000020-rgb.png").exists()
    assert (raw_dir / "000020.txt").exists()
    assert "000020" not in (raw_dir / "traj.txt").read_text(encoding="utf-8")
    scene_info = (raw_dir / "scene_info.json").read_text(encoding="utf-8")
    assert '"kept_frame_ids": [\n    0,\n    20\n  ]' in scene_info
    assert '"dropped_bad_pose_frame_ids": [\n    10\n  ]' in scene_info


def test_normalize_raw_temp_can_fail_on_invalid_pose(tmp_path: Path) -> None:
    raw_temp = tmp_path / "raw_temp"
    _write_intrinsics(raw_temp)
    invalid_pose = np.eye(4)
    invalid_pose[0, 0] = np.inf
    _write_reader_frame(raw_temp, 0, invalid_pose)

    with pytest.raises(ValueError, match="contains NaN/Inf"):
        _normalize_raw_temp(
            "scene0001_00",
            raw_temp,
            tmp_path / "raw",
            stride=10,
            sens_path=tmp_path / "scene0001_00.sens",
            axis_alignment=np.eye(4),
            invalid_pose_policy="fail",
            max_bad_pose_ratio=0.5,
        )


def test_write_frame_extraction_log_records_scene_audit_rows(tmp_path: Path) -> None:
    _write_frame_extraction_log(
        tmp_path,
        [
            ExtractStats(
                scene_id="scene0002_00",
                frame_count=7,
                dropped_bad_pose=1,
                raw_total=8,
                elapsed_seconds=1.0,
                status="extracted",
            ),
            ExtractStats(
                scene_id="scene0001_00",
                frame_count=3,
                dropped_bad_pose=0,
                raw_total=3,
                elapsed_seconds=1.0,
                status="already_complete",
            ),
        ],
    )

    log_text = (tmp_path / "scannet" / ".frame_extraction_log.json").read_text(
        encoding="utf-8"
    )
    assert log_text.index("scene0001_00") < log_text.index("scene0002_00")
    assert '"dropped_bad_pose": 1' in log_text
