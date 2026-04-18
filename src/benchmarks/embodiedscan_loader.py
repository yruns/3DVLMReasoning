"""EmbodiedScan visual grounding benchmark loader.

EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite
https://github.com/OpenRobotLab/EmbodiedScan

This loader provides:
- VG sample loading from official JSON + PKL annotation files
- 9-DOF oriented bounding box parsing
- Source filtering (scannet / 3rscan / matterport3d)
- GT bbox lookup via PKL instance join
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .base import BenchmarkSample

# Scan ID prefixes for each dataset source.
_SOURCE_PREFIXES: dict[str, str] = {
    "scannet": "scannet/",
    "3rscan": "3rscan/",
    "matterport3d": "matterport3d/",
}


@dataclass
class EmbodiedScanVGSample(BenchmarkSample):
    """EmbodiedScan visual grounding sample.

    Extends BenchmarkSample with VG-specific fields from the annotation
    JSON and PKL instance metadata.

    Attributes:
        scan_id: Full scan identifier (e.g., "scannet/scene0415_00").
        target_id: GT instance ID (bbox_id in PKL).
        target: GT object category name.
        distractor_ids: Instance IDs of distractor objects.
        anchors: Anchor object category names.
        anchor_ids: Instance IDs of anchor objects.
        tokens_positive: Token span indices for target in text.
        gt_bbox_3d: 9-DOF bbox [cx,cy,cz,dx,dy,dz,alpha,beta,gamma].
        text: Property alias for query (the VG referring expression).
    """

    scan_id: str = ""
    target_id: int = -1
    target: str = ""
    distractor_ids: list[int] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    anchor_ids: list[int] = field(default_factory=list)
    tokens_positive: list[list[int]] = field(default_factory=list)
    gt_bbox_3d: list[float] | None = None

    @property
    def text(self) -> str:
        """VG referring expression (alias for query)."""
        return self.query


class EmbodiedScanDataset:
    """EmbodiedScan VG dataset loader.

    Joins VG annotation JSON with scene-level PKL metadata to provide
    samples with full bounding box ground truth.

    Usage::

        ds = EmbodiedScanDataset.from_path("data/embodiedscan/", split="val")
        for sample in ds:
            print(sample.text, sample.gt_bbox_3d)
    """

    def __init__(
        self,
        samples: list[EmbodiedScanVGSample],
        scene_index: dict[str, dict[str, Any]],
        categories: dict[str, int],
    ) -> None:
        self._samples = samples
        self._scene_index = scene_index
        self._categories = categories
        self._label_to_name = {v: k for k, v in categories.items()}

    @classmethod
    def from_path(
        cls,
        data_root: str | Path,
        split: str = "val",
        source_filter: str | None = "scannet",
        max_samples: int | None = None,
        mini: bool = False,
    ) -> EmbodiedScanDataset:
        """Load EmbodiedScan VG dataset from disk.

        Args:
            data_root: Path to data/embodiedscan/ directory.
            split: Dataset split ("train", "val", "test").
            source_filter: Keep only scenes from this source.
                One of "scannet", "3rscan", "matterport3d", or None (all).
            max_samples: Cap on number of samples to load.
            mini: If True, load the mini VG annotation set.

        Returns:
            Loaded dataset ready for iteration.

        Raises:
            FileNotFoundError: If required annotation files are missing.
        """
        data_root = Path(data_root)

        # --- Load PKL scene metadata ---
        pkl_path = data_root / f"embodiedscan_infos_{split}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"PKL not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        categories: dict[str, int] = pkl_data["metainfo"]["categories"]
        data_list: list[dict[str, Any]] = pkl_data["data_list"]

        # Build scene index: scan_id → scene dict
        scene_index: dict[str, dict[str, Any]] = {}
        for scene in data_list:
            scan_id = scene["sample_idx"]
            scene_index[scan_id] = scene

        logger.info(
            "Loaded {} scenes from {}",
            len(scene_index),
            pkl_path.name,
        )

        # --- Load VG annotations ---
        suffix = "_mini_vg.json" if mini else "_vg.json"
        vg_path = data_root / f"embodiedscan_{split}{suffix}"
        if not vg_path.exists():
            raise FileNotFoundError(f"VG annotations not found: {vg_path}")

        with open(vg_path) as f:
            vg_entries: list[dict[str, Any]] = json.load(f)

        logger.info(
            "Loaded {} VG entries from {}",
            len(vg_entries),
            vg_path.name,
        )

        # --- Apply source filter ---
        if source_filter is not None:
            prefix = _SOURCE_PREFIXES.get(source_filter)
            if prefix is None:
                raise ValueError(
                    f"Unknown source_filter={source_filter!r}. "
                    f"Choose from: {list(_SOURCE_PREFIXES.keys())}"
                )
            vg_entries = [e for e in vg_entries if e["scan_id"].startswith(prefix)]
            logger.info(
                "After source_filter={!r}: {} entries",
                source_filter,
                len(vg_entries),
            )

        # --- Pre-build bbox lookup dicts per scene ---
        bbox_lookup: dict[str, dict[int, list[float]]] = {}
        for scan_id_key, scene_data in scene_index.items():
            bbox_lookup[scan_id_key] = _build_bbox_dict(
                scene_data["instances"]
            )

        # --- Build samples ---
        samples: list[EmbodiedScanVGSample] = []
        skipped = 0

        for idx, entry in enumerate(vg_entries):
            if max_samples is not None and len(samples) >= max_samples:
                break

            scan_id = entry["scan_id"]
            if scan_id not in scene_index:
                skipped += 1
                continue

            # Extract scene_id (last component, e.g. "scene0415_00")
            scene_id = scan_id.split("/")[-1]

            # O(1) GT bbox lookup
            gt_bbox = bbox_lookup[scan_id].get(entry["target_id"])

            sample = EmbodiedScanVGSample(
                sample_id=f"es_vg_{split}_{idx}",
                scene_id=scene_id,
                query=entry["text"],
                scan_id=scan_id,
                target_id=entry["target_id"],
                target=entry["target"],
                distractor_ids=entry.get("distractor_ids", []),
                anchors=entry.get("anchors", []),
                anchor_ids=entry.get("anchor_ids", []),
                tokens_positive=entry.get("tokens_positive", []),
                gt_bbox_3d=gt_bbox,
            )
            samples.append(sample)

        if skipped > 0:
            logger.warning(
                "Skipped {} entries with no matching scene in PKL", skipped
            )

        logger.info(
            "Built {} samples (split={}, source={}, mini={})",
            len(samples),
            split,
            source_filter,
            mini,
        )

        return cls(
            samples=samples,
            scene_index=scene_index,
            categories=categories,
        )

    def __iter__(self) -> Iterator[EmbodiedScanVGSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EmbodiedScanVGSample:
        return self._samples[idx]

    @property
    def categories(self) -> dict[str, int]:
        """Category name → label ID mapping."""
        return self._categories

    @property
    def label_to_name(self) -> dict[int, str]:
        """Label ID → category name mapping."""
        return self._label_to_name

    def get_scene_info(self, scan_id: str) -> dict[str, Any]:
        """Get full PKL scene info for a scan_id.

        Args:
            scan_id: Full scan ID (e.g. "scannet/scene0415_00").

        Returns:
            Scene dict with instances, images, cam2img, etc.

        Raises:
            KeyError: If scan_id not found in index.
        """
        if scan_id not in self._scene_index:
            raise KeyError(f"scan_id {scan_id!r} not in scene index")
        return self._scene_index[scan_id]

    def get_gt_bbox(self, scan_id: str, target_id: int) -> list[float]:
        """Get GT 9-DOF bounding box for a target instance.

        Args:
            scan_id: Full scan ID.
            target_id: Instance bbox_id.

        Returns:
            9-DOF bbox [cx,cy,cz,dx,dy,dz,alpha,beta,gamma].

        Raises:
            KeyError: If scan_id not found.
            ValueError: If target_id not found in scene instances.
        """
        scene_data = self.get_scene_info(scan_id)
        bbox = _find_instance_bbox(scene_data["instances"], target_id)
        if bbox is None:
            raise ValueError(
                f"target_id={target_id} not found in {scan_id}"
            )
        return bbox

    def get_instances_for_scene(
        self, scan_id: str
    ) -> list[dict[str, Any]]:
        """Get all instances for a scene with resolved category names.

        Returns list of dicts with keys: bbox_id, bbox_3d,
        bbox_label_3d, category_name.
        """
        scene_data = self.get_scene_info(scan_id)
        result = []
        for inst in scene_data["instances"]:
            result.append({
                "bbox_id": inst["bbox_id"],
                "bbox_3d": list(inst["bbox_3d"]),
                "bbox_label_3d": inst["bbox_label_3d"],
                "category_name": self._label_to_name.get(
                    inst["bbox_label_3d"], "unknown"
                ),
            })
        return result

    def filter_by_scene(self, scene_id: str) -> list[EmbodiedScanVGSample]:
        """Get all VG samples for a specific scene.

        Args:
            scene_id: Scene ID (e.g. "scene0415_00").
        """
        return [s for s in self._samples if s.scene_id == scene_id]

    def filter_by_target(self, target: str) -> list[EmbodiedScanVGSample]:
        """Get all VG samples for a specific target category."""
        target_lower = target.lower()
        return [s for s in self._samples if s.target.lower() == target_lower]

    def get_target_categories(self) -> dict[str, int]:
        """Get frequency count of each target category."""
        counts: dict[str, int] = {}
        for s in self._samples:
            counts[s.target] = counts.get(s.target, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_scenes(self) -> list[str]:
        """Get sorted list of unique scene IDs."""
        return sorted({s.scene_id for s in self._samples})


def _normalize_bbox(bbox: Any) -> list[float]:
    """Convert bbox_3d to a plain list of floats."""
    if isinstance(bbox, np.ndarray):
        return bbox.tolist()
    return list(bbox)


def _build_bbox_dict(
    instances: list[dict[str, Any]],
) -> dict[int, list[float]]:
    """Build {bbox_id: bbox_3d} lookup dict for O(1) access."""
    return {
        inst["bbox_id"]: _normalize_bbox(inst["bbox_3d"])
        for inst in instances
    }


def _find_instance_bbox(
    instances: list[dict[str, Any]], target_id: int
) -> list[float] | None:
    """Find bbox_3d for a target instance by bbox_id.

    Returns None if not found. Used by get_gt_bbox() for ad-hoc lookups.
    """
    for inst in instances:
        if inst["bbox_id"] == target_id:
            return _normalize_bbox(inst["bbox_3d"])
    return None
