"""NR3D 3D visual grounding adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from agents.adapters.embodiedscan_adapter import _parse_bbox_3d
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2AgentResult, Stage2TaskSpec
from benchmarks.base import BenchmarkAdapter, BenchmarkSample
from benchmarks.nr3d_eval import evaluate_vg_predictions
from benchmarks.nr3d_loader import Nr3dDataset, Nr3dVGSample


class Nr3dVGAdapter(BenchmarkAdapter):
    """Adapter for NR3D detection-style 3D visual grounding."""

    def __init__(
        self,
        data_root: str | Path,
        embodiedscan_data_root: str | Path | None = None,
        scene_data_root: str | Path | None = None,
        bbox_source: Literal["embodiedscan_pkl", "phase8_gt_cg"] = "embodiedscan_pkl",
        phase8_data_root: str | Path = "data/nr3d/scannet",
        default_split: str = "train",
    ) -> None:
        self.data_root = Path(data_root)
        self.embodiedscan_data_root = (
            Path(embodiedscan_data_root) if embodiedscan_data_root is not None else None
        )
        self.scene_data_root = Path(scene_data_root or phase8_data_root)
        self.bbox_source = bbox_source
        self.phase8_data_root = Path(phase8_data_root)
        self.default_split = default_split
        self._dataset: Nr3dDataset | None = None

    @classmethod
    def from_phase8_test(
        cls,
        data_root: str | Path = "data/nr3d",
        phase8_data_root: str | Path = "data/nr3d/scannet",
        scene_data_root: str | Path | None = None,
    ) -> Nr3dVGAdapter:
        """Build the common NR3D test-split adapter backed by Phase 8 GT-CG."""
        phase8_root = Path(phase8_data_root)
        return cls(
            data_root=data_root,
            embodiedscan_data_root=None,
            scene_data_root=scene_data_root or phase8_root,
            bbox_source="phase8_gt_cg",
            phase8_data_root=phase8_root,
            default_split="test",
        )

    @property
    def dataset(self) -> Nr3dDataset:
        """Access the loaded dataset. Raises if load_samples() was not called."""
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded. Call load_samples() first.")
        return self._dataset

    def load_samples(
        self,
        split: str | None = None,
        max_samples: int | None = None,
        correct_guess_only: bool = False,
        mentions_target_class_only: bool = False,
        apply_blacklist: bool = True,
        drop_clothes: bool = True,
        sample_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> list[BenchmarkSample]:
        """Load NR3D VG samples.

        Defaults to ``"train"`` since the EmbodiedScan test PKL withholds
        ``instances``; see ``docs/benchmark/nr3d/README.md`` Caveats.
        """
        split = split or self.default_split
        bbox_source = kwargs.pop("bbox_source", self.bbox_source)
        phase8_data_root = kwargs.pop("phase8_data_root", self.phase8_data_root)
        self._dataset = Nr3dDataset.from_path(
            data_root=self.data_root,
            embodiedscan_data_root=self.embodiedscan_data_root,
            split=split,
            max_samples=max_samples,
            correct_guess_only=correct_guess_only,
            mentions_target_class_only=mentions_target_class_only,
            apply_blacklist=apply_blacklist,
            drop_clothes=drop_clothes,
            bbox_source=bbox_source,
            phase8_data_root=phase8_data_root,
            sample_ids=sample_ids,
        )
        return list(self._dataset)

    def build_task_spec(self, sample: BenchmarkSample) -> Stage2TaskSpec:
        """Create a visual-grounding task specification from an NR3D sample."""
        return Stage2TaskSpec(
            task_type=Stage2TaskType.VISUAL_GROUNDING,
            user_query=sample.query,
            max_reasoning_turns=6,
        )

    def get_scene_path(self, sample: BenchmarkSample) -> Path:
        """Map an NR3D sample to its scene data directory."""
        if isinstance(sample, Nr3dVGSample) and sample.scan_id:
            return self.scene_data_root / sample.scan_id
        return self.scene_data_root / sample.scene_id

    def get_axis_align_matrix(self, scan_id: str) -> np.ndarray | None:
        """Get axis-alignment metadata from the loaded bbox oracle."""
        if self.bbox_source == "phase8_gt_cg":
            return None
        scene_info = self.dataset.get_scene_info(scan_id)
        mat = scene_info.get("axis_align_matrix")
        if mat is not None:
            return np.array(mat, dtype=np.float64)
        return None

    def build_vg_candidates(
        self,
        scene_objects: Any,
        axis_align_matrix: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Build VG candidate list from ConceptGraph scene objects."""
        candidates = []
        for obj in scene_objects:
            centroid = getattr(obj, "centroid", None)
            if centroid is None:
                continue

            ctr = np.array(centroid, dtype=np.float64)
            if axis_align_matrix is not None:
                ctr_h = np.append(ctr, 1.0)
                ctr = (axis_align_matrix @ ctr_h)[:3]

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
            candidates.append(
                {
                    "obj_id": getattr(obj, "obj_id", id(obj)),
                    "category": category,
                    "cx": float(ctr[0]),
                    "cy": float(ctr[1]),
                    "cz": float(ctr[2]),
                    "dx": float(extent[0]),
                    "dy": float(extent[1]),
                    "dz": float(extent[2]),
                    "description": str(desc)[:200],
                }
            )
        return candidates

    def extract_prediction(
        self,
        sample: BenchmarkSample,
        result: Stage2AgentResult,
    ) -> dict[str, Any]:
        """Extract 3D bbox prediction from agent output."""
        payload = result.result.payload
        return {
            "sample_id": sample.sample_id,
            "bbox_3d": _parse_bbox_3d(payload.get("bbox_3d")),
            "selected_object_id": payload.get("selected_object_id"),
            "confidence": result.result.confidence,
        }

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        samples: list[BenchmarkSample],
    ) -> dict[str, Any]:
        """Compute NR3D VG evaluation metrics."""
        vg_samples = [sample for sample in samples if isinstance(sample, Nr3dVGSample)]
        return evaluate_vg_predictions(predictions, vg_samples)
