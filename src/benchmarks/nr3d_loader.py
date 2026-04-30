"""NR3D visual grounding benchmark loader.

NR3D: natural-language ScanNet references from ReferIt3D.
https://referit3d.github.io/

This loader parses the canonical NR3D CSV and joins each utterance to a
9-DOF ground-truth box from the local EmbodiedScan train+val+test PKLs.
"""

from __future__ import annotations

import ast
import csv
import json
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from .base import BenchmarkSample
from .embodiedscan_loader import (
    EmbodiedScanDataset,
    _build_bbox_dict,
)

_CSV_COLUMNS = [
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

_STATS_KEYS = [
    "total_loaded",
    "skipped_missing_scene",
    "skipped_missing_or_ambiguous_bbox",
    "skipped_blacklist",
    "skipped_clothes",
    "skipped_correct_guess_filter",
    "skipped_mentions_target_class_filter",
]

_CLOTHES_LABELS = {"clothes", "clothing"}


@dataclass
class Nr3dVGSample(BenchmarkSample):
    """NR3D visual-grounding sample."""

    scan_id: str = ""
    target_id: int = -1
    target: str = ""
    distractor_ids: list[int] = field(default_factory=list)
    n_objects: int = 0
    stimulus_id: str = ""
    assignment_id: str = ""
    tokens: list[str] = field(default_factory=list)
    correct_guess: bool = False
    mentions_target_class: bool = False
    gt_bbox_3d: list[float] | None = None

    @property
    def text(self) -> str:
        """VG referring expression (alias for query)."""
        return self.query


class Nr3dDataset:
    """NR3D VG dataset backed by EmbodiedScan 9-DOF bbox metadata."""

    def __init__(
        self,
        samples: list[Nr3dVGSample],
        bbox_oracle: EmbodiedScanDataset,
        scenes_train: set[str],
        scenes_test: set[str],
        split: str,
        stats: dict[str, int],
    ) -> None:
        self._samples = samples
        self._bbox_oracle = bbox_oracle
        self._scenes_train = scenes_train
        self._scenes_test = scenes_test
        self._split = split
        self._stats = stats

    @classmethod
    def from_path(
        cls,
        data_root: str | Path,
        embodiedscan_data_root: str | Path,
        split: str = "test",
        max_samples: int | None = None,
        correct_guess_only: bool = False,
        mentions_target_class_only: bool = False,
        apply_blacklist: bool = True,
        drop_clothes: bool = True,
    ) -> Nr3dDataset:
        """Load NR3D VG samples from disk.

        Args:
            data_root: Directory containing ``raw/nr3d.csv`` and split files.
            embodiedscan_data_root: Directory containing EmbodiedScan PKLs.
            split: NR3D split, either ``"train"`` or ``"test"``.
            max_samples: Optional cap on loaded, valid samples.
            correct_guess_only: Keep only utterances solved by the human listener.
            mentions_target_class_only: Keep only utterances mentioning target class.
            apply_blacklist: Drop manually inspected bad contexts.
            drop_clothes: Drop ``clothes`` and ``clothing`` rows.

        Raises:
            FileNotFoundError: If any required file is missing.
            ValueError: If inputs are malformed or split is unknown.
        """
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown NR3D split {split!r}; expected 'train' or 'test'")
        if max_samples is not None and max_samples < 0:
            raise ValueError(f"max_samples must be non-negative, got {max_samples}")

        data_root = Path(data_root)
        raw_dir = data_root / "raw"
        csv_path = raw_dir / "nr3d.csv"
        train_path = raw_dir / "train_scans.txt"
        test_path = raw_dir / "test_scans.txt"
        blacklist_path = raw_dir / "manually_inspected_bad_contexts.csv"

        _require_file(csv_path, "NR3D CSV")
        _require_file(train_path, "NR3D train scene list")
        _require_file(test_path, "NR3D test scene list")
        if apply_blacklist:
            _require_file(blacklist_path, "NR3D bad-context blacklist")

        scenes_train = _load_scene_list(train_path)
        scenes_test = _load_scene_list(test_path)
        split_scenes = scenes_train if split == "train" else scenes_test
        blacklist = _load_blacklist(blacklist_path) if apply_blacklist else set()
        bbox_oracle = _load_bbox_oracle(Path(embodiedscan_data_root))

        bbox_lookup: dict[str, dict[int, list[float]]] = {}
        for scan_id, scene_data in bbox_oracle._scene_index.items():
            bbox_lookup[scan_id] = _build_bbox_dict(scene_data.get("instances", []))

        stats = {key: 0 for key in _STATS_KEYS}
        samples: list[Nr3dVGSample] = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != _CSV_COLUMNS:
                raise ValueError(
                    f"Unexpected NR3D CSV columns in {csv_path}: {reader.fieldnames!r}"
                )
            for row_idx, row in enumerate(reader, start=2):
                if max_samples is not None and len(samples) >= max_samples:
                    break

                bare_scan_id = row["scan_id"]
                if bare_scan_id not in split_scenes:
                    continue

                target = row["instance_type"]
                target_id = _parse_int(row["target_id"], "target_id", row_idx)
                correct_guess = _parse_bool(row["correct_guess"], "correct_guess", row_idx)
                mentions_target_class = _parse_bool(
                    row["mentions_target_class"],
                    "mentions_target_class",
                    row_idx,
                )
                (
                    stimulus_scene_id,
                    stimulus_target,
                    n_objects,
                    stimulus_target_id,
                    distractor_ids,
                ) = decode_stimulus_string(row["stimulus_id"])
                if stimulus_scene_id != bare_scan_id:
                    raise ValueError(
                        f"Row {row_idx}: stimulus scene {stimulus_scene_id!r} "
                        f"does not match scan_id {bare_scan_id!r}"
                    )
                if stimulus_target != target:
                    raise ValueError(
                        f"Row {row_idx}: stimulus target {stimulus_target!r} "
                        f"does not match instance_type {target!r}"
                    )
                if stimulus_target_id != target_id:
                    raise ValueError(
                        f"Row {row_idx}: stimulus target_id {stimulus_target_id} "
                        f"does not match target_id {target_id}"
                    )

                if drop_clothes and target in _CLOTHES_LABELS:
                    stats["skipped_clothes"] += 1
                    continue

                blacklist_key = (bare_scan_id, target, target_id)
                if apply_blacklist and blacklist_key in blacklist:
                    stats["skipped_blacklist"] += 1
                    continue

                if correct_guess_only and not correct_guess:
                    stats["skipped_correct_guess_filter"] += 1
                    continue

                if mentions_target_class_only and not mentions_target_class:
                    stats["skipped_mentions_target_class_filter"] += 1
                    continue

                scan_id = f"scannet/{bare_scan_id}"
                if scan_id not in bbox_lookup:
                    stats["skipped_missing_scene"] += 1
                    continue

                gt_bbox = bbox_lookup[scan_id].get(target_id)
                if gt_bbox is None:
                    stats["skipped_missing_or_ambiguous_bbox"] += 1
                    continue

                sample = Nr3dVGSample(
                    sample_id=f"{scan_id}::{target_id}::{row['assignmentid']}",
                    scene_id=bare_scan_id,
                    query=row["utterance"],
                    scan_id=scan_id,
                    target_id=target_id,
                    target=target,
                    distractor_ids=distractor_ids,
                    n_objects=n_objects,
                    stimulus_id=row["stimulus_id"],
                    assignment_id=row["assignmentid"],
                    tokens=_parse_tokens(row["tokens"], row_idx),
                    correct_guess=correct_guess,
                    mentions_target_class=mentions_target_class,
                    gt_bbox_3d=gt_bbox,
                    metadata={
                        "dataset": row["dataset"],
                        "speaker_id": row["speaker_id"],
                        "listener_id": row["listener_id"],
                        "uses_object_lang": _parse_bool(
                            row["uses_object_lang"], "uses_object_lang", row_idx
                        ),
                        "uses_spatial_lang": _parse_bool(
                            row["uses_spatial_lang"], "uses_spatial_lang", row_idx
                        ),
                        "uses_color_lang": _parse_bool(
                            row["uses_color_lang"], "uses_color_lang", row_idx
                        ),
                        "uses_shape_lang": _parse_bool(
                            row["uses_shape_lang"], "uses_shape_lang", row_idx
                        ),
                    },
                )
                samples.append(sample)

        stats["total_loaded"] = len(samples)
        logger.info(
            "Built {} NR3D samples (split={}, stats={})",
            len(samples),
            split,
            stats,
        )
        return cls(
            samples=samples,
            bbox_oracle=bbox_oracle,
            scenes_train=scenes_train,
            scenes_test=scenes_test,
            split=split,
            stats=stats,
        )

    def __iter__(self) -> Iterator[Nr3dVGSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Nr3dVGSample:
        return self._samples[idx]

    @property
    def split(self) -> str:
        """Dataset split used to build this instance."""
        return self._split

    @property
    def stats(self) -> dict[str, int]:
        """Loader skip counters and total loaded count."""
        return dict(self._stats)

    @property
    def bbox_oracle(self) -> EmbodiedScanDataset:
        """Underlying EmbodiedScan dataset used for scene metadata and boxes."""
        return self._bbox_oracle

    def filter_by_scene(self, scene_id: str) -> list[Nr3dVGSample]:
        """Get all VG samples for a bare scene ID."""
        return [s for s in self._samples if s.scene_id == scene_id]

    def filter_by_target(self, target: str) -> list[Nr3dVGSample]:
        """Get all VG samples for a specific target category."""
        target_lower = target.lower()
        return [s for s in self._samples if s.target.lower() == target_lower]

    def get_scenes(self) -> list[str]:
        """Get sorted unique bare scene IDs."""
        return sorted({s.scene_id for s in self._samples})

    def get_target_categories(self) -> dict[str, int]:
        """Get frequency count of each target category."""
        counts: dict[str, int] = {}
        for sample in self._samples:
            counts[sample.target] = counts.get(sample.target, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_gt_bbox(self, scan_id: str, target_id: int) -> list[float]:
        """Get GT 9-DOF bbox from the EmbodiedScan oracle.

        Args:
            scan_id: Full scan ID, e.g. ``"scannet/scene0525_00"``.
            target_id: ScanNet objectId / EmbodiedScan bbox_id.
        """
        scene_info = self._bbox_oracle.get_scene_info(scan_id)
        if "instances" not in scene_info:
            raise ValueError(f"scan_id {scan_id!r} has no GT instances in bbox oracle")
        return self._bbox_oracle.get_gt_bbox(scan_id, target_id)

    def get_scene_info(self, scan_id: str) -> dict[str, Any]:
        """Get full scene metadata from the EmbodiedScan oracle."""
        return self._bbox_oracle.get_scene_info(scan_id)

    def get_instances_for_scene(self, scan_id: str) -> list[dict[str, Any]]:
        """Get all scene instances from the EmbodiedScan oracle."""
        scene_info = self._bbox_oracle.get_scene_info(scan_id)
        if "instances" not in scene_info:
            raise ValueError(f"scan_id {scan_id!r} has no GT instances in bbox oracle")
        return self._bbox_oracle.get_instances_for_scene(scan_id)


def decode_stimulus_string(s: str) -> tuple[str, str, int, int, list[int]]:
    """Decode ReferIt3D's packed stimulus id."""
    if len(s.split("-", maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = s.split("-", maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = s.split(
            "-", maxsplit=4
        )
    instance_label = instance_label.replace("_", " ")
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split("-") if i != ""]
    if len(distractors_ids) != n_objects - 1:
        raise ValueError(
            f"stimulus_id {s!r}: distractor count {len(distractors_ids)} "
            f"does not match n_objects - 1 = {n_objects - 1}"
        )
    return scene_id, instance_label, n_objects, target_id, distractors_ids


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not a file: {path}")


def _load_scene_list(path: Path) -> set[str]:
    try:
        scenes = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON scene list: {path}") from exc
    if not isinstance(scenes, list) or not all(isinstance(s, str) for s in scenes):
        raise ValueError(f"Scene list must be a JSON array of strings: {path}")
    return set(scenes)


def _load_blacklist(path: Path) -> set[tuple[str, str, int]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == ["stimulus_id"]:
            result = set()
            for row_idx, row in enumerate(reader, start=2):
                scene_id, instance_label, _n_objects, target_id, _distractors = (
                    decode_stimulus_string(row["stimulus_id"])
                )
                result.add((scene_id, instance_label, target_id))
            return result
        required = {"scan_id", "instance_type", "target_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Blacklist {path} must contain stimulus_id or {sorted(required)} columns"
            )
        return {
            (
                row["scan_id"],
                row["instance_type"],
                _parse_int(row["target_id"], "target_id", row_idx),
            )
            for row_idx, row in enumerate(reader, start=2)
        }


def _load_bbox_oracle(data_root: Path) -> EmbodiedScanDataset:
    scene_index: dict[str, dict[str, Any]] = {}
    categories: dict[str, int] | None = None
    for split in ("train", "val", "test"):
        pkl_path = data_root / f"embodiedscan_infos_{split}.pkl"
        _require_file(pkl_path, f"EmbodiedScan {split} PKL")
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        split_categories: dict[str, int] = pkl_data["metainfo"]["categories"]
        if categories is None:
            categories = split_categories
        elif split_categories != categories:
            raise ValueError("EmbodiedScan train/val/test category maps differ")
        for scene_data in pkl_data["data_list"]:
            scan_id = scene_data["sample_idx"]
            if not scan_id.startswith("scannet/"):
                continue
            if scan_id in scene_index:
                raise ValueError(f"Duplicate EmbodiedScan scene in bbox oracle: {scan_id}")
            scene_index[scan_id] = scene_data
    if categories is None:
        raise RuntimeError("No EmbodiedScan datasets were loaded")
    return EmbodiedScanDataset(
        samples=[],
        scene_index=scene_index,
        categories=categories,
    )


def _parse_int(value: str, field_name: str, row_idx: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Row {row_idx}: invalid integer {field_name}={value!r}") from exc


def _parse_bool(value: str, field_name: str, row_idx: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Row {row_idx}: invalid boolean {field_name}={value!r}")


def _parse_tokens(value: str, row_idx: int) -> list[str]:
    try:
        tokens = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Row {row_idx}: invalid tokens literal") from exc
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise ValueError(f"Row {row_idx}: tokens must be a list of strings")
    return tokens
