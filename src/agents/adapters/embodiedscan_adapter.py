"""EmbodiedScan 3D visual grounding adapter.

Bridges EmbodiedScan VG samples to the Stage 2 agent via the pluggable
BenchmarkAdapter interface. Handles:
- Sample loading from EmbodiedScanDataset
- Task spec construction with VG-specific settings
- Scene path mapping to ConceptGraph data
- VG candidate list building from scene graph objects
- Prediction extraction and evaluation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2AgentResult, Stage2TaskSpec
from benchmarks.base import BenchmarkAdapter, BenchmarkSample
from benchmarks.embodiedscan_eval import evaluate_vg_predictions
from benchmarks.embodiedscan_loader import (
    EmbodiedScanDataset,
    EmbodiedScanVGSample,
)


def _parse_bbox_3d(raw: Any) -> list[float] | None:
    """Parse bbox_3d from VLM output into a validated 9-float list.

    Handles: None, list[float], JSON string, incomplete lists,
    nested lists, and non-numeric values.
    """
    if raw is None:
        return None

    # If string, try JSON parse
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("bbox_3d is unparseable string: {}", raw[:100])
            return None

    # Flatten single-nested list (e.g. [[cx,cy,...]])
    if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
        raw = raw[0]

    if not isinstance(raw, (list, tuple)):
        logger.warning("bbox_3d is not a list: {}", type(raw))
        return None

    # Need at least 6 values (cx,cy,cz,dx,dy,dz); pad missing angles with 0
    if len(raw) < 6:
        logger.warning("bbox_3d too short ({} values): {}", len(raw), raw)
        return None

    try:
        values = [float(v) for v in raw[:9]]
    except (TypeError, ValueError) as exc:
        logger.warning("bbox_3d contains non-numeric values: {} — {}", raw, exc)
        return None

    # Pad to 9 elements (missing euler angles default to 0)
    while len(values) < 9:
        values.append(0.0)

    return values


class EmbodiedScanVGAdapter(BenchmarkAdapter):
    """Adapter for EmbodiedScan 3D Visual Grounding.

    Maps EmbodiedScan VG samples through the unified adapter pipeline:
    load_samples → build_task_spec → extract_prediction → evaluate.

    Args:
        data_root: Path to data/embodiedscan/ (annotations).
        scene_data_root: Path to scene ConceptGraph data (may differ
            from data_root on multi-machine setups).
    """

    def __init__(
        self,
        data_root: str | Path,
        scene_data_root: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.scene_data_root = Path(scene_data_root or data_root)
        self._dataset: EmbodiedScanDataset | None = None

    @property
    def dataset(self) -> EmbodiedScanDataset:
        """Access the loaded dataset. Raises if load_samples() not called."""
        if self._dataset is None:
            raise RuntimeError(
                "Dataset not loaded. Call load_samples() first."
            )
        return self._dataset

    def load_samples(
        self,
        split: str = "val",
        source_filter: str | None = "scannet",
        max_samples: int | None = None,
        mini: bool = False,
        **kwargs: Any,
    ) -> list[BenchmarkSample]:
        """Load EmbodiedScan VG samples.

        Args:
            split: Dataset split ("train", "val", "test").
            source_filter: Keep only scenes from this source.
            max_samples: Cap on number of samples.
            mini: Use mini annotation set.

        Returns:
            List of EmbodiedScanVGSample instances.
        """
        self._dataset = EmbodiedScanDataset.from_path(
            self.data_root,
            split=split,
            source_filter=source_filter,
            max_samples=max_samples,
            mini=mini,
        )
        return list(self._dataset)

    def build_task_spec(
        self, sample: BenchmarkSample
    ) -> Stage2TaskSpec:
        """Create VG task specification from a sample.

        Sets task_type=VISUAL_GROUNDING and uses the VG description
        as the user query.
        """
        return Stage2TaskSpec(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            user_query=sample.query,
            max_reasoning_turns=6,
        )

    def get_scene_path(self, sample: BenchmarkSample) -> Path:
        """Map sample to local ConceptGraph scene directory.

        Extracts scene name from scan_id (e.g., "scannet/scene0415_00"
        → "scene0415_00") and returns the scene root containing
        raw/ and conceptgraph/ subdirectories.
        """
        if isinstance(sample, EmbodiedScanVGSample) and sample.scan_id:
            scene_name = sample.scan_id.split("/")[-1]
        else:
            scene_name = sample.scene_id
        return self.scene_data_root / scene_name

    def get_axis_align_matrix(self, scan_id: str) -> np.ndarray | None:
        """Get the axis alignment matrix for a scene from EmbodiedScan metadata.

        This transforms ConceptGraph coordinates (ScanNet raw frame)
        to EmbodiedScan aligned coordinates.
        """
        if self._dataset is None:
            return None
        scene_info = self._dataset.get_scene_info(scan_id)
        if scene_info is None:
            return None
        mat = scene_info.get("axis_align_matrix")
        if mat is not None:
            return np.array(mat, dtype=np.float64)
        return None

    def build_vg_candidates(
        self,
        scene_objects: Any,
        axis_align_matrix: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Build VG candidate list from ConceptGraph scene objects.

        Converts scene graph objects into the format expected by the
        VLM's object candidate prompt section. Applies axis_align_matrix
        to transform from ConceptGraph to EmbodiedScan coordinates.

        Args:
            scene_objects: Iterable of objects with obj_id, category,
                and centroid attributes.
            axis_align_matrix: 4x4 transformation matrix (optional).

        Returns:
            List of candidate dicts for injection into extra_metadata.
        """
        candidates = []
        for obj in scene_objects:
            centroid = getattr(obj, "centroid", None)
            if centroid is None:
                continue

            # Transform centroid to EmbodiedScan coordinates
            ctr = np.array(centroid, dtype=np.float64)
            if axis_align_matrix is not None:
                ctr_h = np.append(ctr, 1.0)
                ctr = (axis_align_matrix @ ctr_h)[:3]

            # Try to get extent from pcd, bbox, or default
            extent = getattr(obj, "bbox_extent", None)
            if extent is None:
                pcd = getattr(obj, "pcd_np", None)
                if pcd is not None and len(pcd) > 0:
                    pts = np.array(pcd)
                    extent = pts.max(axis=0) - pts.min(axis=0)
                else:
                    extent = [0.3, 0.3, 0.3]

            category = getattr(obj, "category", "unknown")
            if category in ("wall", "floor", "ceiling"):
                continue

            desc = getattr(obj, "summary", "") or getattr(obj, "description", "")
            candidates.append({
                "obj_id": getattr(obj, "obj_id", id(obj)),
                "category": category,
                "cx": float(ctr[0]),
                "cy": float(ctr[1]),
                "cz": float(ctr[2]),
                "dx": float(extent[0]),
                "dy": float(extent[1]),
                "dz": float(extent[2]),
                "description": str(desc)[:200],
            })
        return candidates

    def extract_prediction(
        self, sample: BenchmarkSample, result: Stage2AgentResult
    ) -> dict[str, Any]:
        """Extract 3D bbox prediction from agent output.

        Prefers tool-filled data (from select_object tool stored in
        raw_state) over VLM text output. Falls back to payload parsing.
        """
        raw = result.raw_state or {}
        payload = result.result.payload

        # Prefer tool-filled bbox from select_object
        bbox_3d = raw.get("vg_selected_bbox_3d")
        if bbox_3d is None:
            bbox_3d = _parse_bbox_3d(payload.get("bbox_3d"))

        selected_id = (
            raw.get("vg_selected_object_id")
            or payload.get("selected_object_id")
        )

        return {
            "sample_id": sample.sample_id,
            "bbox_3d": bbox_3d,
            "selected_object_id": selected_id,
            "confidence": result.result.confidence,
        }

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        samples: list[BenchmarkSample],
    ) -> dict[str, Any]:
        """Compute VG evaluation metrics.

        Delegates to evaluate_vg_predictions for Acc@0.25, Acc@0.50,
        mean_iou, and per-category breakdown.
        """
        vg_samples = [
            s for s in samples if isinstance(s, EmbodiedScanVGSample)
        ]
        return evaluate_vg_predictions(predictions, vg_samples)
