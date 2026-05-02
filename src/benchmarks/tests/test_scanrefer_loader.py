"""Tests for ScanRefer VG loader."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from benchmarks.scanrefer_loader import (
    ScanRefVGDataset,
    ScanRefVGSample,
    parse_scanrefer_sample_id,
)


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Build a fake ScanRefer-shaped data tree under tmp_path.

    Returns the root that should be passed as `data_root`.
    """
    # ScanRefer JSON
    raw = tmp_path / "scanrefer/raw"
    raw.mkdir(parents=True)
    (raw / "ScanRefer_filtered_val.json").write_text(
        json.dumps([
            {"scene_id": "scene_a", "object_id": "0", "object_name": "chair",
             "ann_id": "0", "description": "the red chair", "token": ["the","red","chair"]},
            {"scene_id": "scene_a", "object_id": "1", "object_name": "chair",
             "ann_id": "1", "description": "the blue chair", "token": ["the","blue","chair"]},
            {"scene_id": "scene_a", "object_id": "2", "object_name": "table",
             "ann_id": "0", "description": "the wooden table", "token": ["the","wooden","table"]},
        ]),
        encoding="utf-8",
    )

    # Phase-8 GT-CG pkl mock for scene_a (3 GT instances: 2 chairs + 1 table)
    cg = tmp_path / "phase8/scene_a/conceptgraph/pcd_saves"
    cg.mkdir(parents=True)
    objs = []
    for i, label in enumerate(["chair", "chair", "table"]):
        corners = np.array([[i, 0, 0],[i+1, 0, 0],[i, 1, 0],[i+1, 1, 0],
                            [i, 0, 1],[i+1, 0, 1],[i, 1, 1],[i+1, 1, 1]], dtype=np.float64)
        objs.append({"bbox_np": corners, "class_name": [label], "class_id": [i]})
    with gzip.open(cg / "full_pcd_gt_axisaligned_post.pkl.gz", "wb") as f:
        pickle.dump({"objects": objs, "bg_objects": []}, f)

    return tmp_path


def test_parse_scanrefer_sample_id_canonical_form():
    parsed = parse_scanrefer_sample_id("scannet/scene0088_00::5::3")
    assert parsed.scan_id == "scannet/scene0088_00"
    assert parsed.scene_id == "scene0088_00"
    assert parsed.target_id == 5
    assert parsed.ann_id == "3"


def test_parse_scanrefer_sample_id_rejects_malformed():
    with pytest.raises(ValueError, match="format"):
        parse_scanrefer_sample_id("scannet/scene0088_00::5")


def test_load_returns_3_samples(fake_dataset: Path):
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    assert len(ds) == 3


def test_sample_id_format(fake_dataset: Path):
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    sids = [s.sample_id for s in ds]
    assert "scannet/scene_a::0::0" in sids
    assert "scannet/scene_a::1::1" in sids


def test_is_unique_field_chair_count_equals_two(fake_dataset: Path):
    """Two chairs in scene_a → both chair samples have is_unique=False; table is_unique=True."""
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    by_target = {s.target: [] for s in ds}
    for s in ds:
        by_target[s.target].append(s.is_unique)
    assert all(u is False for u in by_target["chair"])
    assert by_target["table"] == [True]


def test_gt_bbox_present_and_correct_shape(fake_dataset: Path):
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    s = ds[0]
    assert s.gt_bbox_3d is not None
    assert len(s.gt_bbox_3d) == 9   # 9-DoF (Euler=0 for axis-aligned)


def test_filter_by_sample_ids(fake_dataset: Path):
    """When sample_ids is provided, loader returns only matching utterances."""
    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
        sample_ids={"scannet/scene_a::2::0"},
    )
    assert len(ds) == 1
    assert ds[0].target == "table"


def test_skipped_when_phase8_missing(fake_dataset: Path):
    """If a sample's scene has no Phase 8 pkl, it's skipped (not crashed)."""
    # Add an utterance for a scene without a Phase 8 pkl
    raw = fake_dataset / "scanrefer/raw/ScanRefer_filtered_val.json"
    data = json.loads(raw.read_text())
    data.append({
        "scene_id": "scene_b", "object_id": "0", "object_name": "lamp",
        "ann_id": "0", "description": "the lamp", "token": ["the", "lamp"],
    })
    raw.write_text(json.dumps(data), encoding="utf-8")

    ds = ScanRefVGDataset.from_path(
        data_root=fake_dataset / "scanrefer",
        phase8_data_root=fake_dataset / "phase8",
        split="val",
    )
    # scene_b utterance is skipped
    assert len(ds) == 3
    assert ds.stats["skipped_missing_scene"] == 1
