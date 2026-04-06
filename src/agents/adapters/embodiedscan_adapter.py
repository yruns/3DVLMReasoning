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

from pathlib import Path
from typing import Any

from loguru import logger

from src.agents.core.agent_config import Stage2TaskType
from src.agents.core.task_types import Stage2AgentResult, Stage2TaskSpec
from src.benchmarks.base import BenchmarkAdapter, BenchmarkSample
from src.benchmarks.embodiedscan_eval import evaluate_vg_predictions
from src.benchmarks.embodiedscan_loader import (
    EmbodiedScanDataset,
    EmbodiedScanVGSample,
)


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

    def build_vg_candidates(
        self, scene_objects: Any
    ) -> list[dict[str, Any]]:
        """Build VG candidate list from ConceptGraph scene objects.

        Converts scene graph objects into the format expected by the
        VLM's object candidate prompt section.

        Args:
            scene_objects: Iterable of objects with obj_id, category,
                centroid, and bbox_extent attributes.

        Returns:
            List of candidate dicts for injection into extra_metadata.
        """
        candidates = []
        for obj in scene_objects:
            centroid = getattr(obj, "centroid", None)
            extent = getattr(obj, "bbox_extent", None)
            if centroid is None or extent is None:
                continue
            candidates.append({
                "obj_id": getattr(obj, "obj_id", id(obj)),
                "category": getattr(obj, "category", "unknown"),
                "cx": float(centroid[0]),
                "cy": float(centroid[1]),
                "cz": float(centroid[2]),
                "dx": float(extent[0]),
                "dy": float(extent[1]),
                "dz": float(extent[2]),
                "description": str(
                    getattr(obj, "description", "")
                )[:200],
            })
        return candidates

    def extract_prediction(
        self, sample: BenchmarkSample, result: Stage2AgentResult
    ) -> dict[str, Any]:
        """Extract 3D bbox prediction from agent output.

        Reads selected_object_id and bbox_3d from the agent's
        structured payload.
        """
        payload = result.result.payload
        return {
            "sample_id": sample.sample_id,
            "bbox_3d": payload.get("bbox_3d"),
            "selected_object_id": payload.get("selected_object_id"),
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
