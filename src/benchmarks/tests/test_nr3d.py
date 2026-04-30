"""Unit tests for the NR3D benchmark loader."""

from __future__ import annotations

import csv
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.base import BenchmarkSample
from benchmarks.nr3d_loader import (
    Nr3dDataset,
    Nr3dVGSample,
    _phase8_corners_to_9dof,
    decode_stimulus_string,
)

CSV_COLUMNS = [
    "assignmentid",
    "stimulus_id",
    "utterance",
    "correct_guess",
    "speaker_id",
    "listener_id",
    "scan_id",
    "instance_type",
    "target_id",
    "tokens",
    "dataset",
    "mentions_target_class",
    "uses_object_lang",
    "uses_spatial_lang",
    "uses_color_lang",
    "uses_shape_lang",
]


def _make_csv_row(
    scan_id: str,
    target_id: int,
    instance_type: str,
    utterance: str = "the chair",
    n_objects: int = 2,
    distractor_ids: list[int] | None = None,
    correct_guess: bool = True,
    mentions_target_class: bool = True,
    assignment_id: str = "A0",
) -> dict[str, str]:
    distractors = distractor_ids if distractor_ids is not None else [target_id + 1]
    if len(distractors) != n_objects - 1:
        raise ValueError("distractor_ids length must equal n_objects - 1")
    label = instance_type.replace(" ", "_")
    tail = "".join(f"-{item}" for item in distractors)
    return {
        "assignmentid": assignment_id,
        "stimulus_id": f"{scan_id}-{label}-{n_objects}-{target_id}{tail}",
        "utterance": utterance,
        "correct_guess": str(correct_guess),
        "speaker_id": "10",
        "listener_id": "20",
        "scan_id": scan_id,
        "instance_type": instance_type,
        "target_id": str(target_id),
        "tokens": repr(utterance.lower().split()),
        "dataset": "nr3d",
        "mentions_target_class": str(mentions_target_class),
        "uses_object_lang": "True",
        "uses_spatial_lang": "False",
        "uses_color_lang": "False",
        "uses_shape_lang": "False",
    }


def _write_nr3d_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    raw = tmp_path / "nr3d" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "nr3d.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_scan_lists(tmp_path: Path, train: list[str], test: list[str]) -> None:
    raw = tmp_path / "nr3d" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "train_scans.txt").write_text(json.dumps(train))
    (raw / "test_scans.txt").write_text(json.dumps(test))


def _write_blacklist(tmp_path: Path, contexts: list[tuple[str, str, int]]) -> None:
    raw = tmp_path / "nr3d" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with open(raw / "manually_inspected_bad_contexts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scan_id", "instance_type", "target_id"])
        writer.writeheader()
        for scan_id, instance_type, target_id in contexts:
            writer.writerow(
                {
                    "scan_id": scan_id,
                    "instance_type": instance_type,
                    "target_id": str(target_id),
                }
            )


def _make_instance(
    bbox_id: int,
    label: int = 42,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    return {
        "bbox_id": bbox_id,
        "bbox_label_3d": label,
        "bbox_3d": [center[0], center[1], center[2], 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    }


def _make_scene(scan_id: str, instances: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_idx": scan_id,
        "instances": instances,
        "images": [],
        "cam2img": np.eye(4).tolist(),
        "axis_align_matrix": np.eye(4).tolist(),
        "depth_cam2img": np.eye(4).tolist(),
    }


def _write_es_split(es_root: Path, split: str, scenes: list[dict[str, Any]]) -> None:
    categories = {"chair": 42, "table": 100, "clothes": 111, "lamp": 55}
    pkl_data = {
        "metainfo": {"categories": categories, "DATASET": "EmbodiedScan"},
        "data_list": scenes,
    }
    with open(es_root / f"embodiedscan_infos_{split}.pkl", "wb") as f:
        pickle.dump(pkl_data, f)
    entries = []
    for scene in scenes:
        if scene["instances"]:
            inst = scene["instances"][0]
            entries.append(
                {
                    "scan_id": scene["sample_idx"],
                    "target_id": int(inst["bbox_id"]),
                    "distractor_ids": [],
                    "text": "fixture",
                    "target": "chair",
                    "anchors": [],
                    "anchor_ids": [],
                    "tokens_positive": [[0, 1]],
                }
            )
    with open(es_root / f"embodiedscan_{split}_vg.json", "w") as f:
        json.dump(entries, f)


def _write_es_test_split_without_instances(es_root: Path, scan_ids: list[str]) -> None:
    categories = {"chair": 42, "table": 100, "clothes": 111, "lamp": 55}
    scenes = []
    for scan_id in scan_ids:
        scene = _make_scene(scan_id, [])
        scene.pop("instances")
        scenes.append(scene)
    pkl_data = {
        "metainfo": {"categories": categories, "DATASET": "EmbodiedScan"},
        "data_list": scenes,
    }
    with open(es_root / "embodiedscan_infos_test.pkl", "wb") as f:
        pickle.dump(pkl_data, f)
    with open(es_root / "embodiedscan_test_vg.json", "w") as f:
        json.dump([], f)


def _make_es_pkl(tmp_path: Path, scenes: dict[str, list[dict[str, Any]]]) -> Path:
    es_root = tmp_path / "embodiedscan"
    es_root.mkdir(parents=True, exist_ok=True)
    val_scenes = []
    train_scenes = []
    for scan_id, instances in scenes.items():
        scene = _make_scene(scan_id, instances)
        if scan_id.endswith("scene0002_00"):
            train_scenes.append(scene)
        else:
            val_scenes.append(scene)
    _write_es_split(es_root, "val", val_scenes)
    _write_es_split(es_root, "train", train_scenes)
    _write_es_split(es_root, "test", [])
    return es_root


def _box_corners(
    center: tuple[float, float, float] = (1.0, 2.0, 4.0),
    size: tuple[float, float, float] = (2.0, 4.0, 8.0),
) -> np.ndarray:
    cx, cy, cz = center
    dx, dy, dz = size
    mins = np.array([cx - dx / 2, cy - dy / 2, cz - dz / 2], dtype=np.float64)
    maxs = np.array([cx + dx / 2, cy + dy / 2, cz + dz / 2], dtype=np.float64)
    return np.array(
        [
            [x, y, z]
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        ],
        dtype=np.float64,
    )


def _write_phase8_scene(
    phase8_root: Path,
    scene_id: str,
    objects: list[dict[str, Any]],
) -> Path:
    pkl_path = (
        phase8_root
        / scene_id
        / "conceptgraph"
        / "pcd_saves"
        / "full_pcd_gt_axisaligned_post.pkl.gz"
    )
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "objects": objects,
                "bg_objects": None,
                "cfg": {},
                "class_names": [],
                "class_colors": {},
            },
            f,
        )
    return pkl_path


@pytest.fixture()
def nr3d_dirs(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        _make_csv_row(
            "scene0001_00", 1, "chair", "the chair near the door", assignment_id="A1"
        ),
        _make_csv_row(
            "scene0001_00", 2, "table", "the table in front", assignment_id="A2"
        ),
        _make_csv_row(
            "scene0002_00", 3, "lamp", "the lamp on the desk", assignment_id="A3"
        ),
    ]
    _write_nr3d_csv(tmp_path, rows)
    _write_scan_lists(tmp_path, train=["scene0002_00"], test=["scene0001_00"])
    _write_blacklist(tmp_path, [])
    es_root = _make_es_pkl(
        tmp_path,
        {
            "scannet/scene0001_00": [
                _make_instance(1, 42, (1.0, 0.0, 0.0)),
                _make_instance(2, 100, (2.0, 0.0, 0.0)),
            ],
            "scannet/scene0002_00": [
                _make_instance(3, 55, (0.0, 3.0, 0.0)),
            ],
        },
    )
    return tmp_path / "nr3d", es_root


class TestNr3dVGSample:
    def test_sample_dataclass_defaults(self) -> None:
        sample = Nr3dVGSample(sample_id="s", scene_id="scene0001_00", query="q")
        assert sample.scan_id == ""
        assert sample.target_id == -1
        assert sample.distractor_ids == []
        assert sample.tokens == []
        assert sample.gt_bbox_3d is None
        assert sample.text == "q"

    def test_sample_full_construction(self) -> None:
        bbox = [1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        sample = Nr3dVGSample(
            sample_id="scannet/scene0001_00::1::A1",
            scene_id="scene0001_00",
            query="the chair",
            scan_id="scannet/scene0001_00",
            target_id=1,
            target="chair",
            distractor_ids=[2],
            n_objects=2,
            stimulus_id="scene0001_00-chair-2-1-2",
            assignment_id="A1",
            tokens=["the", "chair"],
            correct_guess=True,
            mentions_target_class=True,
            gt_bbox_3d=bbox,
        )
        assert sample.target == "chair"
        assert sample.gt_bbox_3d == bbox
        assert sample.text == "the chair"

    def test_inherits_benchmark_sample(self) -> None:
        sample = Nr3dVGSample(sample_id="s", scene_id="scene0001_00", query="q")
        assert isinstance(sample, BenchmarkSample)


class TestNr3dDataset:
    def test_from_path_happy_test_split(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        dataset = Nr3dDataset.from_path(nr3d_root, es_root, split="test")
        assert len(dataset) == 2
        assert dataset.split == "test"
        sample = dataset[0]
        assert sample.scene_id == "scene0001_00"
        assert sample.scan_id == "scannet/scene0001_00"
        assert sample.sample_id == "scannet/scene0001_00::1::A1"
        assert sample.tokens == ["the", "chair", "near", "the", "door"]
        assert sample.gt_bbox_3d == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    def test_from_path_happy_train_split(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        dataset = Nr3dDataset.from_path(nr3d_root, es_root, split="train")
        assert len(dataset) == 1
        assert dataset[0].scene_id == "scene0002_00"
        assert dataset[0].target == "lamp"

    def test_from_path_max_samples(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        dataset = Nr3dDataset.from_path(nr3d_root, es_root, split="test", max_samples=1)
        assert len(dataset) == 1

    def test_from_path_sample_ids_filter(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        dataset = Nr3dDataset.from_path(
            nr3d_root,
            es_root,
            split="test",
            sample_ids={"scannet/scene0001_00::2::A2"},
        )
        assert len(dataset) == 1
        assert dataset[0].sample_id == "scannet/scene0001_00::2::A2"

    def test_from_path_invalid_split(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        with pytest.raises(ValueError, match="Unknown NR3D split"):
            Nr3dDataset.from_path(nr3d_root, es_root, split="val")

    def test_from_path_missing_csv(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        (nr3d_root / "raw" / "nr3d.csv").unlink()
        with pytest.raises(FileNotFoundError, match="NR3D CSV"):
            Nr3dDataset.from_path(nr3d_root, es_root, split="test")

    def test_from_path_missing_train_scans(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        (nr3d_root / "raw" / "train_scans.txt").unlink()
        with pytest.raises(FileNotFoundError, match="train scene list"):
            Nr3dDataset.from_path(nr3d_root, es_root, split="test")

    def test_from_path_missing_test_scans(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        (nr3d_root / "raw" / "test_scans.txt").unlink()
        with pytest.raises(FileNotFoundError, match="test scene list"):
            Nr3dDataset.from_path(nr3d_root, es_root, split="test")

    def test_from_path_missing_es_pkl(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        (es_root / "embodiedscan_infos_val.pkl").unlink()
        with pytest.raises(FileNotFoundError, match="PKL not found"):
            Nr3dDataset.from_path(nr3d_root, es_root, split="test")

    def test_skips_missing_scene(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene9999_00", 1, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene9999_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(tmp_path, {"scannet/scene0001_00": [_make_instance(1)]})
        dataset = Nr3dDataset.from_path(tmp_path / "nr3d", es_root, split="test")
        assert len(dataset) == 0
        assert dataset.stats["skipped_missing_scene"] == 1

    def test_skips_scene_without_instances(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0007_00", 7, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene0007_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(tmp_path, {})
        _write_es_test_split_without_instances(es_root, ["scannet/scene0007_00"])
        dataset = Nr3dDataset.from_path(tmp_path / "nr3d", es_root, split="test")
        assert len(dataset) == 0
        assert dataset.stats["skipped_missing_scene"] == 0
        assert dataset.stats["skipped_missing_or_ambiguous_bbox"] == 1

    def test_skips_ambiguous_bbox(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 1, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(
            tmp_path,
            {"scannet/scene0001_00": [_make_instance(1), _make_instance(1)]},
        )
        dataset = Nr3dDataset.from_path(tmp_path / "nr3d", es_root, split="test")
        assert len(dataset) == 0
        assert dataset.stats["skipped_missing_or_ambiguous_bbox"] == 1

    def test_drops_clothes(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 1, "clothes")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(tmp_path, {"scannet/scene0001_00": [_make_instance(1)]})
        dataset = Nr3dDataset.from_path(tmp_path / "nr3d", es_root, split="test")
        assert len(dataset) == 0
        assert dataset.stats["skipped_clothes"] == 1

    def test_drops_clothing(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 1, "clothing")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(tmp_path, {"scannet/scene0001_00": [_make_instance(1)]})
        dataset = Nr3dDataset.from_path(tmp_path / "nr3d", es_root, split="test")
        assert len(dataset) == 0
        assert dataset.stats["skipped_clothes"] == 1

    def test_drop_clothes_disabled(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 1, "clothes")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(tmp_path, {"scannet/scene0001_00": [_make_instance(1)]})
        dataset = Nr3dDataset.from_path(
            tmp_path / "nr3d", es_root, split="test", drop_clothes=False
        )
        assert len(dataset) == 1

    def test_blacklist_drop(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 1, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [("scene0001_00", "chair", 1)])
        es_root = _make_es_pkl(tmp_path, {"scannet/scene0001_00": [_make_instance(1)]})
        dataset = Nr3dDataset.from_path(tmp_path / "nr3d", es_root, split="test")
        assert len(dataset) == 0
        assert dataset.stats["skipped_blacklist"] == 1

    def test_blacklist_disabled(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 1, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [("scene0001_00", "chair", 1)])
        es_root = _make_es_pkl(tmp_path, {"scannet/scene0001_00": [_make_instance(1)]})
        dataset = Nr3dDataset.from_path(
            tmp_path / "nr3d", es_root, split="test", apply_blacklist=False
        )
        assert len(dataset) == 1

    def test_correct_guess_only(self, tmp_path: Path) -> None:
        rows = [
            _make_csv_row(
                "scene0001_00", 1, "chair", correct_guess=True, assignment_id="A1"
            ),
            _make_csv_row(
                "scene0001_00", 2, "chair", correct_guess=False, assignment_id="A2"
            ),
        ]
        _write_nr3d_csv(tmp_path, rows)
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(
            tmp_path,
            {"scannet/scene0001_00": [_make_instance(1), _make_instance(2)]},
        )
        dataset = Nr3dDataset.from_path(
            tmp_path / "nr3d", es_root, split="test", correct_guess_only=True
        )
        assert len(dataset) == 1
        assert dataset[0].target_id == 1
        assert dataset.stats["skipped_correct_guess_filter"] == 1

    def test_mentions_target_class_only(self, tmp_path: Path) -> None:
        rows = [
            _make_csv_row(
                "scene0001_00",
                1,
                "chair",
                mentions_target_class=True,
                assignment_id="A1",
            ),
            _make_csv_row(
                "scene0001_00",
                2,
                "chair",
                mentions_target_class=False,
                assignment_id="A2",
            ),
        ]
        _write_nr3d_csv(tmp_path, rows)
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(
            tmp_path,
            {"scannet/scene0001_00": [_make_instance(1), _make_instance(2)]},
        )
        dataset = Nr3dDataset.from_path(
            tmp_path / "nr3d",
            es_root,
            split="test",
            mentions_target_class_only=True,
        )
        assert len(dataset) == 1
        assert dataset[0].target_id == 1
        assert dataset.stats["skipped_mentions_target_class_filter"] == 1

    def test_stimulus_decode(self, nr3d_dirs: tuple[Path, Path]) -> None:
        assert decode_stimulus_string("scene0001_00-office_chair-3-4-5-6") == (
            "scene0001_00",
            "office chair",
            3,
            4,
            [5, 6],
        )
        nr3d_root, es_root = nr3d_dirs
        dataset = Nr3dDataset.from_path(nr3d_root, es_root, split="test")
        assert dataset[0].n_objects == 2
        assert dataset[0].distractor_ids == [2]
        assert dataset[0].target_id == 1

    def test_stimulus_decode_invalid_distractor_count(self) -> None:
        with pytest.raises(ValueError, match="distractor count 1 does not match"):
            decode_stimulus_string("scene0001_00-chair-3-1-2")

    def test_duplicate_embodiedscan_scene_across_pkls_raises(
        self, tmp_path: Path
    ) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 1, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        es_root = tmp_path / "embodiedscan"
        es_root.mkdir(parents=True)
        duplicate_scene = _make_scene("scannet/scene0001_00", [_make_instance(1)])
        _write_es_split(es_root, "train", [duplicate_scene])
        _write_es_split(es_root, "val", [duplicate_scene])
        _write_es_split(es_root, "test", [])

        with pytest.raises(ValueError, match="Duplicate EmbodiedScan scene"):
            Nr3dDataset.from_path(tmp_path / "nr3d", es_root, split="test")

    def test_assignment_id_uniquifier(self, tmp_path: Path) -> None:
        rows = [
            _make_csv_row("scene0001_00", 1, "chair", assignment_id="A1"),
            _make_csv_row("scene0001_00", 1, "chair", assignment_id="A2"),
        ]
        _write_nr3d_csv(tmp_path, rows)
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        es_root = _make_es_pkl(tmp_path, {"scannet/scene0001_00": [_make_instance(1)]})
        dataset = Nr3dDataset.from_path(tmp_path / "nr3d", es_root, split="test")
        assert [sample.sample_id for sample in dataset] == [
            "scannet/scene0001_00::1::A1",
            "scannet/scene0001_00::1::A2",
        ]

    def test_get_gt_bbox(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        dataset = Nr3dDataset.from_path(nr3d_root, es_root, split="test")
        assert dataset.get_gt_bbox("scannet/scene0001_00", 2) == [
            2.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ]

    def test_filter_by_scene_and_target(self, nr3d_dirs: tuple[Path, Path]) -> None:
        nr3d_root, es_root = nr3d_dirs
        dataset = Nr3dDataset.from_path(nr3d_root, es_root, split="test")
        assert len(dataset.filter_by_scene("scene0001_00")) == 2
        assert [sample.target_id for sample in dataset.filter_by_target("TABLE")] == [2]
        assert dataset.get_scenes() == ["scene0001_00"]
        assert dataset.get_target_categories() == {"chair": 1, "table": 1}

    def test_stats_counts(self, tmp_path: Path) -> None:
        rows = [
            _make_csv_row("scene0001_00", 1, "chair", assignment_id="valid"),
            _make_csv_row("scene9999_00", 1, "chair", assignment_id="missing_scene"),
            _make_csv_row("scene0002_00", 2, "chair", assignment_id="ambiguous"),
            _make_csv_row("scene0003_00", 3, "chair", assignment_id="blacklist"),
            _make_csv_row("scene0004_00", 4, "clothes", assignment_id="clothes"),
            _make_csv_row(
                "scene0005_00", 5, "chair", correct_guess=False, assignment_id="guess"
            ),
            _make_csv_row(
                "scene0006_00",
                6,
                "chair",
                mentions_target_class=False,
                assignment_id="mention",
            ),
        ]
        _write_nr3d_csv(tmp_path, rows)
        _write_scan_lists(
            tmp_path,
            train=[],
            test=[
                "scene0001_00",
                "scene9999_00",
                "scene0002_00",
                "scene0003_00",
                "scene0004_00",
                "scene0005_00",
                "scene0006_00",
            ],
        )
        _write_blacklist(tmp_path, [("scene0003_00", "chair", 3)])
        es_root = _make_es_pkl(
            tmp_path,
            {
                "scannet/scene0001_00": [_make_instance(1)],
                "scannet/scene0002_00": [_make_instance(2), _make_instance(2)],
                "scannet/scene0003_00": [_make_instance(3)],
                "scannet/scene0004_00": [_make_instance(4)],
                "scannet/scene0005_00": [_make_instance(5)],
                "scannet/scene0006_00": [_make_instance(6)],
            },
        )
        dataset = Nr3dDataset.from_path(
            tmp_path / "nr3d",
            es_root,
            split="test",
            correct_guess_only=True,
            mentions_target_class_only=True,
        )
        assert dataset.stats == {
            "total_loaded": 1,
            "skipped_missing_scene": 1,
            "skipped_missing_or_ambiguous_bbox": 1,
            "skipped_blacklist": 1,
            "skipped_clothes": 1,
            "skipped_correct_guess_filter": 1,
            "skipped_mentions_target_class_filter": 1,
        }

    def test_from_path_bbox_source_phase8_gt_cg(self, tmp_path: Path) -> None:
        _write_nr3d_csv(
            tmp_path,
            [_make_csv_row("scene0001_00", 1, "chair", assignment_id="A1")],
        )
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        phase8_root = tmp_path / "phase8"
        _write_phase8_scene(
            phase8_root,
            "scene0001_00",
            [
                {"bbox_np": _box_corners(center=(0.0, 0.0, 5.0), size=(1.0, 1.0, 1.0))},
                {"bbox_np": _box_corners(center=(1.0, 2.0, 4.0), size=(2.0, 4.0, 8.0))},
            ],
        )

        dataset = Nr3dDataset.from_path(
            tmp_path / "nr3d",
            split="test",
            bbox_source="phase8_gt_cg",
            phase8_data_root=phase8_root,
        )

        assert len(dataset) == 1
        assert dataset[0].sample_id == "scannet/scene0001_00::1::A1"
        assert dataset[0].gt_bbox_3d == pytest.approx(
            [1.0, 2.0, 4.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0]
        )
        assert dataset.stats["total_loaded"] == 1

    def test_phase8_corners_to_9dof_conversion(self) -> None:
        bbox = _phase8_corners_to_9dof(
            _box_corners(center=(1.0, 2.0, 4.0), size=(2.0, 4.0, 8.0)),
            field_name="fixture.bbox_np",
        )

        assert bbox == pytest.approx([1.0, 2.0, 4.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0])

    def test_phase8_corners_to_9dof_recovers_yaw_obb(self) -> None:
        """Rotated 8-corner input must round-trip via oriented_bbox_to_corners.

        Locks down the OBB-recovery behavior introduced after the AABB
        shortcut was found to inflate volume by ~3x mean (median 2.6x) on
        real Phase 8 outputs.
        """
        import numpy as np

        from benchmarks.embodiedscan_eval import oriented_bbox_to_corners

        # 30° yaw on a 2x4x1 box centered at (5, -2, 3)
        cx, cy, cz = 5.0, -2.0, 3.0
        dx, dy, dz = 2.0, 4.0, 1.0
        yaw = np.deg2rad(30.0)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        local = np.array(
            [
                [+dx / 2, +dy / 2, +dz / 2],
                [+dx / 2, +dy / 2, -dz / 2],
                [+dx / 2, -dy / 2, +dz / 2],
                [+dx / 2, -dy / 2, -dz / 2],
                [-dx / 2, +dy / 2, +dz / 2],
                [-dx / 2, +dy / 2, -dz / 2],
                [-dx / 2, -dy / 2, +dz / 2],
                [-dx / 2, -dy / 2, -dz / 2],
            ]
        )
        corners = (R @ local.T).T + np.array([cx, cy, cz])

        bbox = _phase8_corners_to_9dof(corners, field_name="fixture.bbox_np")

        # Center + extent must be exact.
        assert bbox[:3] == pytest.approx([cx, cy, cz], abs=1e-9)
        assert sorted(bbox[3:6]) == pytest.approx(sorted([dx, dy, dz]), abs=1e-9)

        # Round-trip: 9-DOF → 8 corners covers the same point set as input.
        reconstructed = oriented_bbox_to_corners(bbox)
        sorted_in = np.sort(corners, axis=0)
        sorted_out = np.sort(reconstructed, axis=0)
        np.testing.assert_array_almost_equal(sorted_in, sorted_out, decimal=6)

    def test_phase8_missing_pkl_skips_with_counter(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 1, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])

        dataset = Nr3dDataset.from_path(
            tmp_path / "nr3d",
            split="test",
            bbox_source="phase8_gt_cg",
            phase8_data_root=tmp_path / "phase8",
        )

        assert len(dataset) == 0
        assert dataset.stats["skipped_missing_scene"] == 1

    def test_phase8_missing_target_id_skips_with_counter(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 7, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        phase8_root = tmp_path / "phase8"
        _write_phase8_scene(
            phase8_root,
            "scene0001_00",
            [{"bbox_np": _box_corners(center=(0.0, 0.0, 5.0), size=(1.0, 1.0, 1.0))}],
        )

        dataset = Nr3dDataset.from_path(
            tmp_path / "nr3d",
            split="test",
            bbox_source="phase8_gt_cg",
            phase8_data_root=phase8_root,
        )

        assert len(dataset) == 0
        assert dataset.stats["skipped_missing_or_ambiguous_bbox"] == 1

    def test_phase8_bbox_shape_mismatch_raises(self, tmp_path: Path) -> None:
        _write_nr3d_csv(tmp_path, [_make_csv_row("scene0001_00", 0, "chair")])
        _write_scan_lists(tmp_path, train=[], test=["scene0001_00"])
        _write_blacklist(tmp_path, [])
        phase8_root = tmp_path / "phase8"
        _write_phase8_scene(
            phase8_root,
            "scene0001_00",
            [{"bbox_np": np.zeros((7, 3), dtype=np.float64)}],
        )

        with pytest.raises(
            ValueError,
            match=r"scene0001_00::0 bbox_np shape is \(7, 3\), expected \(8,3\)",
        ):
            Nr3dDataset.from_path(
                tmp_path / "nr3d",
                split="test",
                bbox_source="phase8_gt_cg",
                phase8_data_root=phase8_root,
            )
