"""NR3D visual grounding benchmark loader.

NR3D: natural-language ScanNet references from ReferIt3D.
https://referit3d.github.io/

This loader parses the canonical NR3D CSV and joins each utterance to a
9-DOF ground-truth box from the local EmbodiedScan train+val+test PKLs.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
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
_PHASE8_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz")
BBoxSource = Literal["embodiedscan_pkl", "phase8_gt_cg"]


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
        bbox_oracle: EmbodiedScanDataset | None,
        scenes_train: set[str],
        scenes_test: set[str],
        split: str,
        stats: dict[str, int],
        bbox_source: BBoxSource = "embodiedscan_pkl",
        bbox_lookup: dict[str, dict[int, list[float]]] | None = None,
        phase8_data_root: Path | None = None,
    ) -> None:
        self._samples = samples
        self._bbox_oracle = bbox_oracle
        self._scenes_train = scenes_train
        self._scenes_test = scenes_test
        self._split = split
        self._stats = stats
        self._bbox_source = bbox_source
        self._bbox_lookup = bbox_lookup or {}
        self._phase8_data_root = phase8_data_root

    @classmethod
    def from_path(
        cls,
        data_root: str | Path,
        embodiedscan_data_root: str | Path | None = None,
        split: str = "test",
        max_samples: int | None = None,
        correct_guess_only: bool = False,
        mentions_target_class_only: bool = False,
        apply_blacklist: bool = True,
        drop_clothes: bool = True,
        bbox_source: BBoxSource = "embodiedscan_pkl",
        phase8_data_root: str | Path = "data/nr3d/scannet",
        sample_ids: set[str] | None = None,
    ) -> Nr3dDataset:
        """Load NR3D VG samples from disk.

        Args:
            data_root: Directory containing ``raw/nr3d.csv`` and split files.
            embodiedscan_data_root: Directory containing EmbodiedScan PKLs.
                Required when ``bbox_source="embodiedscan_pkl"``.
            split: NR3D split, either ``"train"`` or ``"test"``.
            max_samples: Optional cap on loaded, valid samples.
            correct_guess_only: Keep only utterances solved by the human listener.
            mentions_target_class_only: Keep only utterances mentioning target class.
            apply_blacklist: Drop manually inspected bad contexts.
            drop_clothes: Drop ``clothes`` and ``clothing`` rows.
            bbox_source: Ground-truth bbox source. The default preserves the
                Phase 6 EmbodiedScan-PKL behavior. ``"phase8_gt_cg"`` reads
                Phase 8 ConceptGraph GT objects.
            phase8_data_root: Root containing
                ``<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz``.
            sample_ids: Optional exact NR3D sample IDs to load. This keeps
                smoke prep from loading every Phase 8 scene.

        Raises:
            FileNotFoundError: If any required file is missing.
            ValueError: If inputs are malformed or split is unknown.
        """
        if split not in {"train", "test"}:
            raise ValueError(
                f"Unknown NR3D split {split!r}; expected 'train' or 'test'"
            )
        if max_samples is not None and max_samples < 0:
            raise ValueError(f"max_samples must be non-negative, got {max_samples}")
        if bbox_source not in ("embodiedscan_pkl", "phase8_gt_cg"):
            raise ValueError(
                f"Unknown bbox_source {bbox_source!r}; expected 'embodiedscan_pkl' "
                "or 'phase8_gt_cg'"
            )

        data_root = Path(data_root)
        phase8_root = Path(phase8_data_root)
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
        bbox_lookup: dict[str, dict[int, list[float]]] = {}
        bbox_oracle: EmbodiedScanDataset | None = None
        missing_phase8_scenes: set[str] = set()
        phase8_bbox_cache: dict[str, list[Any]] = {}
        if bbox_source == "embodiedscan_pkl":
            if embodiedscan_data_root is None:
                raise ValueError(
                    "embodiedscan_data_root is required when "
                    "bbox_source='embodiedscan_pkl'"
                )
            bbox_oracle = _load_bbox_oracle(Path(embodiedscan_data_root))
            for scan_id, scene_data in bbox_oracle._scene_index.items():
                bbox_lookup[scan_id] = _build_bbox_dict(scene_data.get("instances", []))

        stats = dict.fromkeys(_STATS_KEYS, 0)
        samples: list[Nr3dVGSample] = []
        requested_sample_ids = set(sample_ids) if sample_ids is not None else None

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
                scan_id = f"scannet/{bare_scan_id}"
                candidate_sample_id = f"{scan_id}::{target_id}::{row['assignmentid']}"
                if (
                    requested_sample_ids is not None
                    and candidate_sample_id not in requested_sample_ids
                ):
                    continue
                correct_guess = _parse_bool(
                    row["correct_guess"], "correct_guess", row_idx
                )
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

                if bbox_source == "phase8_gt_cg":
                    scene_lookup = bbox_lookup.setdefault(scan_id, {})
                    if bare_scan_id not in phase8_bbox_cache:
                        bbox_entries = _load_phase8_bbox_entries(
                            phase8_root,
                            bare_scan_id,
                        )
                        if bbox_entries is None:
                            missing_phase8_scenes.add(bare_scan_id)
                        else:
                            phase8_bbox_cache[bare_scan_id] = bbox_entries
                    if bare_scan_id not in missing_phase8_scenes and (
                        target_id not in scene_lookup
                    ):
                        bbox_entries = phase8_bbox_cache[bare_scan_id]
                        if target_id < 0 or target_id >= len(bbox_entries):
                            stats["skipped_missing_or_ambiguous_bbox"] += 1
                            continue
                        scene_lookup[target_id] = _phase8_object_bbox_9dof(
                            bbox_entries[target_id],
                            scene_id=bare_scan_id,
                            target_id=target_id,
                        )

                if scan_id not in bbox_lookup or bare_scan_id in missing_phase8_scenes:
                    stats["skipped_missing_scene"] += 1
                    continue

                gt_bbox = bbox_lookup[scan_id].get(target_id)
                if gt_bbox is None:
                    stats["skipped_missing_or_ambiguous_bbox"] += 1
                    continue

                sample = Nr3dVGSample(
                    sample_id=candidate_sample_id,
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
            bbox_source=bbox_source,
            bbox_lookup=bbox_lookup,
            phase8_data_root=phase8_root if bbox_source == "phase8_gt_cg" else None,
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
        if self._bbox_oracle is None:
            raise RuntimeError(
                "No EmbodiedScan bbox oracle is available when "
                "bbox_source='phase8_gt_cg'"
            )
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
        """Get GT 9-DOF bbox from the configured bbox source.

        Args:
            scan_id: Full scan ID, e.g. ``"scannet/scene0525_00"``.
            target_id: ScanNet objectId / EmbodiedScan bbox_id.
        """
        if self._bbox_source == "phase8_gt_cg":
            scene_lookup = self._bbox_lookup.get(scan_id)
            if scene_lookup is None:
                raise KeyError(f"scan_id {scan_id!r} not in Phase 8 bbox lookup")
            bbox = scene_lookup.get(int(target_id))
            if bbox is None:
                raise ValueError(f"target_id={target_id} not found in {scan_id}")
            return bbox
        if self._bbox_oracle is None:
            raise RuntimeError("EmbodiedScan bbox oracle is not loaded")
        scene_info = self._bbox_oracle.get_scene_info(scan_id)
        if "instances" not in scene_info:
            raise ValueError(f"scan_id {scan_id!r} has no GT instances in bbox oracle")
        return self._bbox_oracle.get_gt_bbox(scan_id, target_id)

    def get_scene_info(self, scan_id: str) -> dict[str, Any]:
        """Get full scene metadata from the EmbodiedScan oracle."""
        if self._bbox_oracle is None:
            raise RuntimeError(
                "Scene metadata is not available from Phase 8 bbox lookup; "
                "use the Phase 8 scene root directly."
            )
        return self._bbox_oracle.get_scene_info(scan_id)

    def get_instances_for_scene(self, scan_id: str) -> list[dict[str, Any]]:
        """Get all scene instances from the EmbodiedScan oracle."""
        if self._bbox_oracle is None:
            raise RuntimeError(
                "EmbodiedScan instances are not available when "
                "bbox_source='phase8_gt_cg'"
            )
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
            for row in reader:
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
                raise ValueError(
                    f"Duplicate EmbodiedScan scene in bbox oracle: {scan_id}"
                )
            scene_index[scan_id] = scene_data
    if categories is None:
        raise RuntimeError("No EmbodiedScan datasets were loaded")
    return EmbodiedScanDataset(
        samples=[],
        scene_index=scene_index,
        categories=categories,
    )


def _load_phase8_bbox_entries(
    phase8_data_root: Path,
    scene_id: str,
) -> list[Any] | None:
    pkl_path = phase8_data_root / scene_id / _PHASE8_PCD_REL
    if not pkl_path.exists():
        return None
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Phase 8 GT-CG pkl is not a file: {pkl_path}")
    with gzip.open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{pkl_path} must contain an objects list")
    bbox_entries: list[Any] = []
    for obj in objects:
        if not isinstance(obj, dict):
            bbox_entries.append(obj)
            continue
        bbox_entries.append(obj.get("bbox_np"))
    return bbox_entries


def _phase8_object_bbox_9dof(
    bbox_np: Any,
    *,
    scene_id: str,
    target_id: int,
) -> list[float]:
    if bbox_np is None:
        raise ValueError(f"{scene_id}::{target_id} missing bbox_np")
    corners = np.asarray(bbox_np, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError(
            f"{scene_id}::{target_id} bbox_np shape is {corners.shape}, "
            "expected (8,3)"
        )
    return _phase8_corners_to_9dof(
        corners,
        field_name=f"{scene_id}::{target_id}.bbox_np",
    )


def _phase8_corners_to_9dof(
    corners: Any,
    *,
    field_name: str = "bbox_np",
) -> list[float]:
    """Convert Phase 8 oriented 8-corner boxes to 9-DOF boxes.

    Phase 8's producer derived per-instance OBBs via Open3D's
    ``OrientedBoundingBox.create_from_points`` and stored the 8 box corners
    in ``bbox_np``. Despite the producer filename's ``axisaligned`` suffix
    (which refers to the post-axisAlignment frame), individual instance OBBs
    are typically yaw-rotated (~17% of ScanNet objects exceed 15° yaw).

    This function recovers exact 9-DOF parameters
    ``[cx, cy, cz, dx, dy, dz, alpha, beta, gamma]`` (ZXY Euler) by:

    1. Computing the centroid as the mean of the 8 corners.
    2. Running PCA on the centered corners to recover the box's orthonormal
       principal axes (exact for a degenerate 8-corner cloud).
    3. Aligning each PCA axis to its closest world axis so the resulting
       Euler is "minimal" — this preserves a zero rotation for AABB inputs.
    4. Projecting corners onto the aligned axes to derive extents.

    The recovery is exact for OBB inputs (round-trip:
    ``oriented_bbox_to_corners(out) == sorted_corners(input)`` within float
    tolerance).
    """
    from scipy.spatial.transform import Rotation as _SciR  # local import

    arr = np.asarray(corners, dtype=np.float64)
    if arr.shape != (8, 3):
        raise ValueError(f"{field_name} shape is {arr.shape}, expected (8,3)")

    center = arr.mean(axis=0)
    centered = arr - center
    cov = centered.T @ centered / 8.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.isfinite(eigvecs).all() or not np.isfinite(eigvals).all():
        raise ValueError(f"{field_name} eigendecomposition produced non-finite values")

    # Align each PCA axis (column of eigvecs) to its closest world axis so
    # that an AABB input yields R=I (zero Euler), and rotated inputs return
    # a small Euler. Greedy assignment by max |dot product|.
    axis_to_world = [-1, -1, -1]
    used_pca: set[int] = set()
    for world_idx in range(3):
        similarities = np.abs(eigvecs[world_idx])
        ranked = np.argsort(-similarities)
        for pca_idx in ranked:
            if int(pca_idx) not in used_pca:
                axis_to_world[world_idx] = int(pca_idx)
                used_pca.add(int(pca_idx))
                break

    aligned = eigvecs[:, axis_to_world]
    for i in range(3):
        if aligned[i, i] < 0:
            aligned[:, i] *= -1
    if np.linalg.det(aligned) < 0:
        aligned[:, 2] *= -1

    proj = centered @ aligned
    extent = proj.max(axis=0) - proj.min(axis=0)
    euler = _SciR.from_matrix(aligned).as_euler("ZXY")

    bbox = [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(extent[0]),
        float(extent[1]),
        float(extent[2]),
        float(euler[0]),
        float(euler[1]),
        float(euler[2]),
    ]
    if not np.isfinite(bbox).all():
        raise ValueError(f"{field_name} conversion produced non-finite bbox: {bbox}")
    return bbox


def _parse_int(value: str, field_name: str, row_idx: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_idx}: invalid integer {field_name}={value!r}"
        ) from exc


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
