"""Unit tests for EmbodiedScan benchmark loader and base classes."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.base import BenchmarkAdapter, BenchmarkSample
from benchmarks.embodiedscan_loader import (
    EmbodiedScanDataset,
    EmbodiedScanVGSample,
    _find_instance_bbox,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic annotation data
# ---------------------------------------------------------------------------

def _make_categories() -> dict[str, int]:
    return {
        "bag": 7,
        "chair": 42,
        "table": 100,
        "bathtub": 15,
        "lamp": 55,
    }


def _make_instance(
    bbox_id: int, label: int, center: tuple[float, ...] = (0.0, 0.0, 0.0)
) -> dict[str, Any]:
    cx, cy, cz = center
    return {
        "bbox_3d": [cx, cy, cz, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        "bbox_label_3d": label,
        "bbox_id": bbox_id,
    }


def _make_scene(
    scan_id: str, instances: list[dict], n_images: int = 3
) -> dict[str, Any]:
    images = [
        {
            "img_path": f"{scan_id}/posed_images/{i:05d}.jpg",
            "cam2global": np.eye(4).tolist(),
            "visible_instance_ids": [inst["bbox_id"] for inst in instances],
            "depth_path": f"{scan_id}/posed_images/{i:05d}.png",
        }
        for i in range(n_images)
    ]
    return {
        "sample_idx": scan_id,
        "instances": instances,
        "images": images,
        "cam2img": np.eye(4).tolist(),
        "axis_align_matrix": np.eye(4).tolist(),
        "depth_cam2img": np.eye(4).tolist(),
    }


def _make_vg_entry(
    scan_id: str,
    target_id: int,
    text: str,
    target: str,
    distractor_ids: list[int] | None = None,
    anchors: list[str] | None = None,
    anchor_ids: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "scan_id": scan_id,
        "target_id": target_id,
        "distractor_ids": distractor_ids or [],
        "text": text,
        "target": target,
        "anchors": anchors or [],
        "anchor_ids": anchor_ids or [],
        "tokens_positive": [[0, 3]],
    }


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with synthetic EmbodiedScan data."""
    categories = _make_categories()

    # Build scenes
    scenes = [
        _make_scene(
            "scannet/scene0001_00",
            [
                _make_instance(1, 7, (1.0, 0.0, 0.0)),   # bag
                _make_instance(2, 42, (0.0, 1.0, 0.0)),   # chair
                _make_instance(3, 15, (2.0, 0.0, 0.0)),   # bathtub
            ],
        ),
        _make_scene(
            "scannet/scene0002_00",
            [
                _make_instance(1, 100, (0.0, 0.0, 0.0)),  # table
                _make_instance(2, 55, (1.0, 1.0, 1.0)),   # lamp
            ],
        ),
        _make_scene(
            "3rscan/rscan0001",
            [
                _make_instance(1, 42, (0.0, 0.0, 0.0)),   # chair
            ],
        ),
    ]

    # Write PKL
    pkl_data = {
        "metainfo": {"categories": categories, "DATASET": "EmbodiedScan"},
        "data_list": scenes,
    }
    with open(tmp_path / "embodiedscan_infos_val.pkl", "wb") as f:
        pickle.dump(pkl_data, f)

    # Write VG JSON
    vg_entries = [
        _make_vg_entry(
            "scannet/scene0001_00", 1,
            "find the bag near the bathtub", "bag",
            distractor_ids=[], anchors=["bathtub"], anchor_ids=[3],
        ),
        _make_vg_entry(
            "scannet/scene0001_00", 2,
            "the chair in the room", "chair",
        ),
        _make_vg_entry(
            "scannet/scene0002_00", 2,
            "the lamp on the table", "lamp",
            anchors=["table"], anchor_ids=[1],
        ),
        _make_vg_entry(
            "3rscan/rscan0001", 1,
            "the only chair", "chair",
        ),
    ]
    with open(tmp_path / "embodiedscan_val_vg.json", "w") as f:
        json.dump(vg_entries, f)

    # Write mini VG JSON (subset)
    with open(tmp_path / "embodiedscan_val_mini_vg.json", "w") as f:
        json.dump(vg_entries[:2], f)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: BenchmarkSample base class
# ---------------------------------------------------------------------------


class TestBenchmarkSample:
    """Tests for the BenchmarkSample dataclass."""

    def test_basic_creation(self) -> None:
        sample = BenchmarkSample(
            sample_id="s1", scene_id="scene0001_00", query="where is the chair?"
        )
        assert sample.sample_id == "s1"
        assert sample.scene_id == "scene0001_00"
        assert sample.query == "where is the chair?"
        assert sample.metadata == {}

    def test_with_metadata(self) -> None:
        sample = BenchmarkSample(
            sample_id="s2",
            scene_id="scene0002_00",
            query="find the table",
            metadata={"source": "scannet"},
        )
        assert sample.metadata["source"] == "scannet"


class TestBenchmarkAdapter:
    """Tests for the BenchmarkAdapter ABC."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            BenchmarkAdapter()  # type: ignore[abstract]

    def test_subclass_must_implement(self) -> None:
        class Incomplete(BenchmarkAdapter):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_get_scene_path_raises(self) -> None:
        class Minimal(BenchmarkAdapter):
            def load_samples(self, split="val", **kw):
                return []

            def build_task_spec(self, sample):
                return None  # type: ignore[return-value]

            def extract_prediction(self, sample, result):
                return {}

            def evaluate(self, predictions, samples):
                return {}

        adapter = Minimal()
        sample = BenchmarkSample(sample_id="s1", scene_id="sc", query="q")
        with pytest.raises(NotImplementedError, match="Minimal"):
            adapter.get_scene_path(sample)


# ---------------------------------------------------------------------------
# Tests: EmbodiedScanVGSample
# ---------------------------------------------------------------------------


class TestEmbodiedScanVGSample:
    """Tests for the EmbodiedScanVGSample dataclass."""

    def test_creation_defaults(self) -> None:
        s = EmbodiedScanVGSample(
            sample_id="es_vg_val_0",
            scene_id="scene0001_00",
            query="find the bag",
        )
        assert s.scan_id == ""
        assert s.target_id == -1
        assert s.gt_bbox_3d is None
        assert s.distractor_ids == []
        assert s.anchors == []

    def test_full_creation(self) -> None:
        bbox = [1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        s = EmbodiedScanVGSample(
            sample_id="es_vg_val_0",
            scene_id="scene0001_00",
            query="the bag near the bathtub",
            scan_id="scannet/scene0001_00",
            text="the bag near the bathtub",
            target_id=1,
            target="bag",
            distractor_ids=[2],
            anchors=["bathtub"],
            anchor_ids=[3],
            tokens_positive=[[4, 7]],
            gt_bbox_3d=bbox,
        )
        assert s.target == "bag"
        assert len(s.gt_bbox_3d) == 9
        assert s.gt_bbox_3d[0] == 1.0

    def test_is_benchmark_sample_subclass(self) -> None:
        s = EmbodiedScanVGSample(
            sample_id="x", scene_id="y", query="z"
        )
        assert isinstance(s, BenchmarkSample)


# ---------------------------------------------------------------------------
# Tests: _find_instance_bbox helper
# ---------------------------------------------------------------------------


class TestFindInstanceBbox:
    """Tests for the _find_instance_bbox helper."""

    def test_found(self) -> None:
        instances = [
            _make_instance(1, 7, (1.0, 2.0, 3.0)),
            _make_instance(2, 42, (4.0, 5.0, 6.0)),
        ]
        bbox = _find_instance_bbox(instances, 1)
        assert bbox is not None
        assert bbox[0] == pytest.approx(1.0)
        assert len(bbox) == 9

    def test_not_found(self) -> None:
        instances = [_make_instance(1, 7)]
        assert _find_instance_bbox(instances, 99) is None

    def test_empty_instances(self) -> None:
        assert _find_instance_bbox([], 1) is None

    def test_numpy_array_conversion(self) -> None:
        instances = [{
            "bbox_3d": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float),
            "bbox_label_3d": 0,
            "bbox_id": 5,
        }]
        bbox = _find_instance_bbox(instances, 5)
        assert bbox is not None
        assert isinstance(bbox, list)
        assert len(bbox) == 9


# ---------------------------------------------------------------------------
# Tests: EmbodiedScanDataset loading
# ---------------------------------------------------------------------------


class TestEmbodiedScanDataset:
    """Tests for EmbodiedScanDataset.from_path and access methods."""

    def test_load_all_sources(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        assert len(ds) == 4

    def test_load_scannet_only(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter="scannet")
        assert len(ds) == 3
        for s in ds:
            assert s.scan_id.startswith("scannet/")

    def test_load_3rscan_only(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter="3rscan")
        assert len(ds) == 1
        assert ds[0].target == "chair"

    def test_load_mini(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(
            data_dir, source_filter=None, mini=True
        )
        assert len(ds) == 2

    def test_max_samples(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(
            data_dir, source_filter=None, max_samples=2
        )
        assert len(ds) == 2

    def test_sample_fields(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter="scannet")
        s = ds[0]
        assert s.sample_id == "es_vg_val_0"
        assert s.scene_id == "scene0001_00"
        assert s.scan_id == "scannet/scene0001_00"
        assert s.text == "find the bag near the bathtub"
        assert s.query == s.text
        assert s.target == "bag"
        assert s.target_id == 1
        assert s.anchors == ["bathtub"]
        assert s.anchor_ids == [3]

    def test_gt_bbox_populated(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter="scannet")
        s = ds[0]
        assert s.gt_bbox_3d is not None
        assert len(s.gt_bbox_3d) == 9
        # center = (1.0, 0.0, 0.0) from fixture
        assert s.gt_bbox_3d[0] == pytest.approx(1.0)
        assert s.gt_bbox_3d[1] == pytest.approx(0.0)

    def test_iteration(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        samples = list(ds)
        assert len(samples) == 4
        assert all(isinstance(s, EmbodiedScanVGSample) for s in samples)

    def test_missing_pkl_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="PKL not found"):
            EmbodiedScanDataset.from_path(tmp_path, split="val")

    def test_missing_vg_json_raises(self, tmp_path: Path) -> None:
        # Create PKL but no VG JSON
        pkl_data = {"metainfo": {"categories": {}}, "data_list": []}
        with open(tmp_path / "embodiedscan_infos_val.pkl", "wb") as f:
            pickle.dump(pkl_data, f)

        with pytest.raises(FileNotFoundError, match="VG annotations"):
            EmbodiedScanDataset.from_path(tmp_path, split="val")

    def test_invalid_source_filter(self, data_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown source_filter"):
            EmbodiedScanDataset.from_path(data_dir, source_filter="invalid")

    def test_skips_entries_without_matching_scene(self, data_dir: Path) -> None:
        """VG entries referencing non-existent scenes are skipped."""
        # Add an extra VG entry with a scan_id not in PKL
        vg_path = data_dir / "embodiedscan_val_vg.json"
        with open(vg_path) as f:
            entries = json.load(f)
        entries.append(
            _make_vg_entry("scannet/scene9999_00", 1, "ghost chair", "chair")
        )
        with open(vg_path, "w") as f:
            json.dump(entries, f)

        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        # 4 original entries loaded, 1 skipped
        assert len(ds) == 4


# ---------------------------------------------------------------------------
# Tests: Dataset access methods
# ---------------------------------------------------------------------------


class TestDatasetAccessMethods:
    """Tests for EmbodiedScanDataset query/filter methods."""

    def test_categories(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        cats = ds.categories
        assert "bag" in cats
        assert cats["bag"] == 7

    def test_label_to_name(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        assert ds.label_to_name[7] == "bag"
        assert ds.label_to_name[42] == "chair"

    def test_get_scene_info(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        info = ds.get_scene_info("scannet/scene0001_00")
        assert "instances" in info
        assert len(info["instances"]) == 3

    def test_get_scene_info_not_found(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        with pytest.raises(KeyError):
            ds.get_scene_info("scannet/nonexistent")

    def test_get_gt_bbox(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        bbox = ds.get_gt_bbox("scannet/scene0001_00", 1)
        assert len(bbox) == 9
        assert bbox[0] == pytest.approx(1.0)

    def test_get_gt_bbox_not_found(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        with pytest.raises(ValueError, match="target_id=99"):
            ds.get_gt_bbox("scannet/scene0001_00", 99)

    def test_get_instances_for_scene(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        insts = ds.get_instances_for_scene("scannet/scene0001_00")
        assert len(insts) == 3
        assert insts[0]["category_name"] == "bag"
        assert "bbox_3d" in insts[0]
        assert "bbox_id" in insts[0]

    def test_filter_by_scene(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        filtered = ds.filter_by_scene("scene0001_00")
        assert len(filtered) == 2

    def test_filter_by_target(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        chairs = ds.filter_by_target("chair")
        assert len(chairs) == 2

    def test_filter_by_target_case_insensitive(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        chairs = ds.filter_by_target("CHAIR")
        assert len(chairs) == 2

    def test_get_target_categories(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        cats = ds.get_target_categories()
        assert cats["chair"] == 2
        assert cats["bag"] == 1
        assert cats["lamp"] == 1

    def test_get_scenes(self, data_dir: Path) -> None:
        ds = EmbodiedScanDataset.from_path(data_dir, source_filter=None)
        scenes = ds.get_scenes()
        assert "scene0001_00" in scenes
        assert "scene0002_00" in scenes
        assert "rscan0001" in scenes
        assert scenes == sorted(scenes)
