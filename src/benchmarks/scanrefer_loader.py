"""ScanRefer benchmark loader.

ScanRefer: 3D Object Localization in RGB-D Scans Using Natural Language
ECCV 2020, https://daveredrum.github.io/ScanRefer/

This loader provides:
- Dataset loading from the official ScanRefer JSON format
- 3D bounding box parsing and conversion
- IoU-based evaluation (Acc@0.25, Acc@0.5)

ScanRefer is a visual grounding benchmark that requires localizing
objects in 3D scenes based on natural language descriptions.
"""

from __future__ import annotations

import gzip
import json
import pickle
import subprocess
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .base import BenchmarkSample


@dataclass
class BoundingBox3D:
    """A 3D axis-aligned bounding box.

    Attributes:
        center: Center point [x, y, z].
        size: Box dimensions [width, height, depth].
        orientation: Optional rotation (Euler angles in radians).
    """

    center: list[float]  # [x, y, z]
    size: list[float]  # [width, height, depth]
    orientation: list[float] | None = None  # [roll, pitch, yaw]

    @property
    def min_corner(self) -> np.ndarray:
        """Get minimum corner of bounding box."""
        center = np.array(self.center)
        half_size = np.array(self.size) / 2
        return center - half_size

    @property
    def max_corner(self) -> np.ndarray:
        """Get maximum corner of bounding box."""
        center = np.array(self.center)
        half_size = np.array(self.size) / 2
        return center + half_size

    def volume(self) -> float:
        """Compute box volume."""
        return float(np.prod(self.size))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "center": self.center,
            "size": self.size,
        }
        if self.orientation:
            result["orientation"] = self.orientation
        return result


def compute_iou_3d(box1: BoundingBox3D, box2: BoundingBox3D) -> float:
    """Compute 3D IoU (Intersection over Union) between two bounding boxes.

    Assumes axis-aligned bounding boxes (no rotation).

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        IoU score between 0 and 1.
    """
    # Get min/max corners
    min1, max1 = box1.min_corner, box1.max_corner
    min2, max2 = box2.min_corner, box2.max_corner

    # Compute intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(0, inter_max - inter_min)
    intersection = float(np.prod(inter_size))

    # Compute union
    vol1 = box1.volume()
    vol2 = box2.volume()
    union = vol1 + vol2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


@dataclass
class ScanReferSample:
    """A single ScanRefer referring expression sample.

    Attributes:
        sample_id: Unique identifier for this annotation.
        scene_id: ScanNet scene identifier (e.g., "scene0000_00").
        object_id: Target object ID within the scene.
        object_name: Category name of the target object.
        description: Natural language referring expression.
        target_bbox: Ground truth 3D bounding box.
        ann_id: Original annotation ID.
        token: Tokenized description (optional).
    """

    sample_id: str
    scene_id: str
    object_id: str
    object_name: str
    description: str
    target_bbox: BoundingBox3D
    ann_id: str = ""
    token: list[str] = field(default_factory=list)

    @property
    def query(self) -> str:
        """Get the referring expression as query."""
        return self.description


class ScanReferDataset:
    """ScanRefer dataset loader.

    Usage:
        dataset = ScanReferDataset.from_path("/path/to/ScanRefer")
        for sample in dataset:
            print(f"Description: {sample.description}")
            print(f"Target: {sample.object_name} at {sample.target_bbox.center}")
    """

    def __init__(
        self,
        samples: list[ScanReferSample],
        data_root: Path,
        split: str = "val",
    ) -> None:
        """Initialize dataset.

        Args:
            samples: List of ScanRefer samples.
            data_root: Root directory containing the dataset.
            split: Dataset split (train/val).
        """
        self.samples = samples
        self.data_root = data_root
        self.split = split

    @classmethod
    def from_path(
        cls,
        data_root: str | Path,
        split: str = "val",
        scene_id: str | None = None,
        object_name: str | None = None,
        max_samples: int | None = None,
    ) -> ScanReferDataset:
        """Load ScanRefer dataset from directory.

        Args:
            data_root: Path to the ScanRefer data directory.
            split: Dataset split to load (train/val).
            scene_id: Filter by specific ScanNet scene.
            object_name: Filter by object category.
            max_samples: Maximum number of samples to load.

        Returns:
            ScanReferDataset instance.
        """
        data_root = Path(data_root)

        # ScanRefer stores annotations in different possible locations
        possible_paths = [
            data_root / f"ScanRefer_filtered_{split}.json",
            data_root / "data" / f"ScanRefer_filtered_{split}.json",
            data_root / "ScanRefer" / f"ScanRefer_filtered_{split}.json",
        ]

        json_path = None
        for path in possible_paths:
            if path.exists():
                json_path = path
                break

        if json_path is None:
            raise FileNotFoundError(
                f"ScanRefer data file not found. Tried: {possible_paths}. "
                "Please download the dataset first using download_scanrefer()."
            )

        logger.info(f"Loading ScanRefer ({split}) from {json_path}")
        with open(json_path) as f:
            data = json.load(f)

        samples: list[ScanReferSample] = []

        for item in data:
            # Apply filters
            item_scene_id = item.get("scene_id", "")
            if scene_id and item_scene_id != scene_id:
                continue

            item_object_name = item.get("object_name", "")
            if object_name and item_object_name.lower() != object_name.lower():
                continue

            # Parse bounding box
            # ScanRefer uses axis-aligned bounding boxes
            # Format can be [center_x, center_y, center_z, size_x, size_y, size_z]
            # or separate center/size fields
            bbox_data = item.get("object_bbox", item.get("bbox", []))
            if len(bbox_data) == 6:
                center = bbox_data[:3]
                size = bbox_data[3:]
            else:
                # Try alternative format
                center = item.get("center", [0.0, 0.0, 0.0])
                size = item.get("size", [1.0, 1.0, 1.0])

            target_bbox = BoundingBox3D(
                center=[float(x) for x in center],
                size=[float(x) for x in size],
            )

            sample = ScanReferSample(
                sample_id=f"{item_scene_id}_{item.get('object_id', '')}_{item.get('ann_id', len(samples))}",
                scene_id=item_scene_id,
                object_id=str(item.get("object_id", "")),
                object_name=item_object_name,
                description=item.get("description", ""),
                target_bbox=target_bbox,
                ann_id=str(item.get("ann_id", "")),
                token=item.get("token", []),
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

        logger.info(f"Loaded {len(samples)} ScanRefer samples")
        return cls(samples, data_root, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ScanReferSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[ScanReferSample]:
        return iter(self.samples)

    def get_scenes(self) -> list[str]:
        """Get list of unique scene IDs."""
        return sorted({s.scene_id for s in self.samples})

    def get_object_names(self) -> list[str]:
        """Get list of unique object categories."""
        return sorted({s.object_name for s in self.samples})

    def filter_by_scene(self, scene_id: str) -> ScanReferDataset:
        """Create a new dataset filtered by scene."""
        filtered = [s for s in self.samples if s.scene_id == scene_id]
        return ScanReferDataset(filtered, self.data_root, self.split)

    def filter_by_object(self, object_name: str) -> ScanReferDataset:
        """Create a new dataset filtered by object category."""
        filtered = [
            s for s in self.samples if s.object_name.lower() == object_name.lower()
        ]
        return ScanReferDataset(filtered, self.data_root, self.split)


def download_scanrefer(
    output_dir: str | Path,
    include_scannet: bool = False,
) -> Path:
    """Download ScanRefer dataset.

    Args:
        output_dir: Directory to save the dataset.
        include_scannet: Whether to download ScanNet scenes (requires agreement).

    Returns:
        Path to the downloaded dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = output_dir / "ScanRefer"

    if repo_dir.exists():
        logger.info(f"ScanRefer already exists at {repo_dir}")
        return repo_dir

    logger.info("Cloning ScanRefer repository...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/daveredrum/ScanRefer.git",
            str(repo_dir),
        ],
        check=True,
    )

    if include_scannet:
        logger.warning(
            "ScanNet download requires manual agreement at "
            "http://www.scan-net.org/. Please download manually."
        )

    logger.success(f"ScanRefer downloaded to {repo_dir}")
    return repo_dir


# Evaluation utilities


@dataclass
class ScanReferEvaluationResult:
    """Result of evaluating a single ScanRefer prediction."""

    sample_id: str
    scene_id: str
    object_id: str
    description: str
    ground_truth_bbox: BoundingBox3D
    predicted_bbox: BoundingBox3D | None
    iou: float
    acc_at_025: bool  # IoU >= 0.25
    acc_at_050: bool  # IoU >= 0.5


def evaluate_scanrefer(
    predictions: list[tuple[ScanReferSample, BoundingBox3D | None]],
) -> list[ScanReferEvaluationResult]:
    """Evaluate ScanRefer predictions.

    Following the official ScanRefer evaluation protocol:
    - Acc@0.25: Fraction of predictions with IoU >= 0.25
    - Acc@0.5: Fraction of predictions with IoU >= 0.5

    Args:
        predictions: List of (sample, predicted_bbox) tuples.
            predicted_bbox can be None if no prediction was made.

    Returns:
        List of evaluation results.
    """
    results = []

    for sample, predicted_bbox in predictions:
        if predicted_bbox is None:
            iou = 0.0
        else:
            iou = compute_iou_3d(sample.target_bbox, predicted_bbox)

        results.append(
            ScanReferEvaluationResult(
                sample_id=sample.sample_id,
                scene_id=sample.scene_id,
                object_id=sample.object_id,
                description=sample.description,
                ground_truth_bbox=sample.target_bbox,
                predicted_bbox=predicted_bbox,
                iou=iou,
                acc_at_025=iou >= 0.25,
                acc_at_050=iou >= 0.5,
            )
        )

    return results


def compute_scanrefer_metrics(
    results: list[ScanReferEvaluationResult],
) -> dict[str, Any]:
    """Compute aggregate metrics from ScanRefer evaluation results.

    Args:
        results: List of evaluation results.

    Returns:
        Dictionary containing accuracy metrics.
    """
    if not results:
        return {
            "accuracy_at_025": 0.0,
            "accuracy_at_050": 0.0,
            "mean_iou": 0.0,
            "total": 0,
        }

    acc_025 = sum(1 for r in results if r.acc_at_025) / len(results)
    acc_050 = sum(1 for r in results if r.acc_at_050) / len(results)
    mean_iou = sum(r.iou for r in results) / len(results)

    return {
        "accuracy_at_025": acc_025,
        "accuracy_at_050": acc_050,
        "mean_iou": mean_iou,
        "total": len(results),
    }


def compute_scanrefer_metrics_by_category(
    results: list[ScanReferEvaluationResult],
    samples: list[ScanReferSample],
) -> dict[str, dict[str, Any]]:
    """Compute metrics per object category.

    Args:
        results: List of evaluation results.
        samples: List of corresponding samples.

    Returns:
        Dictionary mapping category name to metrics.
    """
    # Group by category
    sample_map = {s.sample_id: s for s in samples}
    category_results: dict[str, list[ScanReferEvaluationResult]] = {}

    for r in results:
        sample = sample_map.get(r.sample_id)
        if sample:
            category = sample.object_name
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(r)

    return {
        category: compute_scanrefer_metrics(cat_results)
        for category, cat_results in category_results.items()
    }


# ---------------------------------------------------------------------------
# ScanRefer VG loader (Phase 8 GT-CG backed) — Task 4 additions
# ---------------------------------------------------------------------------

_PHASE8_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz")

_STATS_KEYS = [
    "total_loaded",
    "skipped_missing_scene",
    "skipped_missing_or_ambiguous_bbox",
]


@dataclass(frozen=True)
class ParsedScanRefSampleId:
    scan_id: str
    scene_id: str
    target_id: int
    ann_id: str


def parse_scanrefer_sample_id(sample_id: str) -> ParsedScanRefSampleId:
    """Parse ``scannet/<scene>::<target_id>::<ann_id>`` form."""
    parts = sample_id.split("::")
    if len(parts) != 3:
        raise ValueError(
            "Expected sample_id in '<scan_id>::<target_id>::<ann_id>' "
            f"format, got {sample_id!r}"
        )
    scan_id, target_text, ann_id = parts
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"Invalid target_id in sample_id={sample_id!r}") from exc
    return ParsedScanRefSampleId(
        scan_id=scan_id,
        scene_id=scan_id.split("/")[-1],
        target_id=target_id,
        ann_id=ann_id,
    )


@dataclass
class ScanRefVGSample(BenchmarkSample):
    """ScanRefer visual-grounding sample."""

    scan_id: str = ""
    target_id: int = -1
    target: str = ""             # ScanRefer JSON's object_name
    ann_id: str = ""
    description: str = ""        # alias of query, kept for clarity
    is_unique: bool = False
    gt_bbox_3d: list[float] | None = None

    @property
    def text(self) -> str:
        """VG referring expression (alias for query)."""
        return self.query


def _phase8_corners_to_9dof(corners) -> list[float]:
    """Reuse the same conversion NR3D uses (axis-aligned, Euler=0)."""
    import numpy as np
    arr = np.asarray(corners, dtype=np.float64)
    if arr.shape != (8, 3):
        raise ValueError(f"corners shape {arr.shape}, expected (8,3)")
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    cx, cy, cz = ((mn + mx) / 2.0).tolist()
    dx, dy, dz = (mx - mn).tolist()
    return [cx, cy, cz, dx, dy, dz, 0.0, 0.0, 0.0]


class ScanRefVGDataset:
    """ScanRefer VG dataset backed by Phase 8 GT-CG pkl for GT bbox lookup."""

    def __init__(
        self,
        samples: list[ScanRefVGSample],
        stats: dict[str, int],
        split: str,
    ) -> None:
        self._samples = samples
        self._stats = stats
        self._split = split

    @classmethod
    def from_path(
        cls,
        data_root: str | Path,
        phase8_data_root: str | Path,
        split: str = "val",
        sample_ids: set[str] | None = None,
    ) -> ScanRefVGDataset:
        """Load ScanRefer VG samples joined to Phase 8 GT-CG GT bboxes.

        Args:
            data_root: Directory containing ``raw/ScanRefer_filtered_<split>.json``.
            phase8_data_root: Directory containing ``<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz``.
            split: ``"val"`` or ``"train"``.
            sample_ids: Optional restriction to specific sample_ids.

        Raises:
            FileNotFoundError: If ScanRefer JSON is missing.
        """
        if split not in {"val", "train"}:
            raise ValueError(f"Unknown ScanRefer split {split!r}")

        data_root = Path(data_root)
        phase8_data_root = Path(phase8_data_root)
        json_path = data_root / "raw" / f"ScanRefer_filtered_{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"ScanRefer JSON missing: {json_path}")

        utterances = json.loads(json_path.read_text(encoding="utf-8"))

        # Cache per-scene Phase 8 objects + class counts (for is_unique).
        scene_cache: dict[str, list[dict[str, Any]] | None] = {}
        scene_class_counts: dict[str, Counter] = {}

        def _load_scene(scene: str) -> list[dict[str, Any]] | None:
            if scene in scene_cache:
                return scene_cache[scene]
            pkl_path = phase8_data_root / scene / _PHASE8_PCD_REL
            if not pkl_path.exists():
                scene_cache[scene] = None
                return None
            with gzip.open(pkl_path, "rb") as f:
                objs = pickle.load(f).get("objects") or []
            scene_cache[scene] = objs
            counts: Counter = Counter()
            for o in objs:
                cn = o.get("class_name")
                if isinstance(cn, list) and cn:
                    counts[cn[0].lower()] += 1
            scene_class_counts[scene] = counts
            return objs

        stats = dict.fromkeys(_STATS_KEYS, 0)
        samples: list[ScanRefVGSample] = []
        requested = set(sample_ids) if sample_ids is not None else None

        for row in utterances:
            scene = row["scene_id"]
            target_id = int(row["object_id"])
            ann_id = row["ann_id"]
            target_name = row["object_name"]
            sample_id = f"scannet/{scene}::{target_id}::{ann_id}"
            if requested is not None and sample_id not in requested:
                continue

            objs = _load_scene(scene)
            if objs is None:
                stats["skipped_missing_scene"] += 1
                continue
            if target_id < 0 or target_id >= len(objs):
                stats["skipped_missing_or_ambiguous_bbox"] += 1
                continue

            corners = objs[target_id].get("bbox_np")
            if corners is None:
                stats["skipped_missing_or_ambiguous_bbox"] += 1
                continue
            try:
                gt_bbox = _phase8_corners_to_9dof(corners)
            except ValueError:
                stats["skipped_missing_or_ambiguous_bbox"] += 1
                continue

            counts = scene_class_counts.get(scene, Counter())
            is_unique = counts.get(target_name.lower(), 0) == 1

            description = row["description"]
            samples.append(
                ScanRefVGSample(
                    sample_id=sample_id,
                    scene_id=scene,
                    query=description,
                    scan_id=f"scannet/{scene}",
                    target_id=target_id,
                    target=target_name,
                    ann_id=ann_id,
                    description=description,
                    is_unique=is_unique,
                    gt_bbox_3d=gt_bbox,
                    metadata={"token": list(row.get("token", []))},
                )
            )

        stats["total_loaded"] = len(samples)
        logger.info(
            "Built {} ScanRefer samples (split={}, stats={})",
            len(samples), split, stats,
        )
        return cls(samples=samples, stats=stats, split=split)

    def __iter__(self) -> Iterator[ScanRefVGSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ScanRefVGSample:
        return self._samples[idx]

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def split(self) -> str:
        return self._split
