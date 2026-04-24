from pathlib import Path

import pytest

from benchmarks.embodiedscan_bbox_feasibility.models import EmbodiedScanTarget
from benchmarks.embodiedscan_bbox_feasibility.observations import (
    centered_frame_window,
    list_raw_frame_ids,
    make_observation,
)


def test_centered_frame_window_clamps_to_available_frames() -> None:
    assert centered_frame_window(center=2, available=[0, 2, 4, 6, 8], size=3) == [
        0,
        2,
        4,
    ]
    assert centered_frame_window(center=0, available=[0, 2, 4, 6, 8], size=3) == [
        0,
        2,
        4,
    ]
    assert centered_frame_window(center=8, available=[0, 2, 4, 6, 8], size=3) == [
        4,
        6,
        8,
    ]


def test_centered_frame_window_uses_nearest_available_center() -> None:
    assert centered_frame_window(center=5, available=[0, 2, 4, 6, 8], size=3) == [
        2,
        4,
        6,
    ]


def test_centered_frame_window_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        centered_frame_window(center=0, available=[0, 1], size=0)


def test_make_observation_records_target_policy() -> None:
    target = EmbodiedScanTarget(
        sample_ids=["a"],
        scan_id="scannet/scene0001_00",
        scene_id="scene0001_00",
        target_id=4,
        target_category="chair",
        gt_bbox_3d=[0, 0, 0, 1, 1, 1, 0, 0, 0],
    )
    obs = make_observation(
        target,
        best_frame_id=10,
        available_frame_ids=[6, 8, 10, 12, 14],
        window_size=3,
    )
    assert obs.policy == "target_best_visible_centered_window"
    assert obs.frame_ids == [8, 10, 12]
    assert obs.metadata["target_id"] == 4
    assert obs.metadata["scan_id"] == "scannet/scene0001_00"
    assert obs.metadata["best_frame_id"] == 10


def test_list_raw_frame_ids_finds_conceptgraph_rgb_frames(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "10-rgb.png").write_text("", encoding="utf-8")
    (raw / "2-rgb.jpg").write_text("", encoding="utf-8")
    (raw / "bad-rgb.png").write_text("", encoding="utf-8")
    (raw / "4-depth.png").write_text("", encoding="utf-8")

    assert list_raw_frame_ids(tmp_path) == [2, 10]
