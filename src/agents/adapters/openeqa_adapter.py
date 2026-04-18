"""OpenEQA QA benchmark adapter.

Thin wrapper around existing OpenEQA logic using the pluggable
BenchmarkAdapter interface. The existing OpenEQA entry points
continue to work unchanged — this adapter provides consistency
with the new unified adapter pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2AgentResult, Stage2TaskSpec
from benchmarks.base import BenchmarkAdapter, BenchmarkSample
from benchmarks.openeqa_loader import OpenEQADataset, OpenEQASample


class OpenEQAAdapter(BenchmarkAdapter):
    """Adapter for OpenEQA QA benchmark.

    Wraps existing OpenEQA dataset loading and evaluation in the
    pluggable BenchmarkAdapter interface. This is a lightweight
    wrapper, not a rewrite — existing OpenEQA entry points remain
    the primary interface for production evaluation.

    Args:
        data_root: Path to OpenEQA data directory.
        scene_data_root: Path to prepared scene data (ConceptGraph).
    """

    def __init__(
        self,
        data_root: str | Path,
        scene_data_root: str | Path | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.scene_data_root = Path(scene_data_root or data_root)
        self._dataset: OpenEQADataset | None = None

    def load_samples(
        self,
        split: str = "val",
        question_type: str | None = None,
        category: str | None = None,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> list[BenchmarkSample]:
        """Load OpenEQA samples.

        Args:
            split: Dataset split.
            question_type: Filter by question type.
            category: Filter by question category.
            max_samples: Cap on number of samples.

        Returns:
            List of OpenEQASample instances (subclass of BenchmarkSample
            via duck typing — OpenEQASample has sample_id-compatible
            question_id, scene_id, and query-compatible question fields).
        """
        self._dataset = OpenEQADataset.from_path(
            self.data_root,
            split=split,
            question_type=question_type,
            category=category,
            max_samples=max_samples,
        )
        # Wrap OpenEQASample into BenchmarkSample-compatible objects
        return [
            BenchmarkSample(
                sample_id=s.question_id,
                scene_id=s.scene_id,
                query=s.question,
                metadata={
                    "answer": s.answer,
                    "category": s.category,
                    "question_type": s.question_type,
                },
            )
            for s in self._dataset
        ]

    def build_task_spec(
        self, sample: BenchmarkSample
    ) -> Stage2TaskSpec:
        """Create QA task specification from a sample."""
        return Stage2TaskSpec(
            task_type=Stage2TaskType.QA,
            user_query=sample.query,
        )

    def get_scene_path(self, sample: BenchmarkSample) -> Path:
        """Map sample to local prepared scene directory."""
        return self.scene_data_root / sample.scene_id / "conceptgraph"

    def extract_prediction(
        self, sample: BenchmarkSample, result: Stage2AgentResult
    ) -> dict[str, Any]:
        """Extract QA answer from agent output."""
        payload = result.result.payload
        return {
            "question_id": sample.sample_id,
            "answer": payload.get("answer", result.result.summary),
        }

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        samples: list[BenchmarkSample],
    ) -> dict[str, Any]:
        """Compute QA evaluation metrics.

        Returns a placeholder dict. For production evaluation, use
        the existing openeqa_official_eval module which provides
        LLM-match scoring against the official benchmark.
        """
        if not predictions:
            return {"num_samples": 0}

        # Build ground-truth lookup from sample metadata
        gt_map = {s.sample_id: s.metadata.get("answer", "") for s in samples}

        correct = 0
        for pred in predictions:
            qid = pred.get("question_id", "")
            pred_answer = pred.get("answer", "").strip().lower()
            gt_answer = gt_map.get(qid, "").strip().lower()
            if pred_answer and gt_answer and pred_answer in gt_answer:
                correct += 1

        total = len(predictions)
        return {
            "contains_accuracy": correct / max(total, 1),
            "num_samples": total,
            "note": "Use openeqa_official_eval for LLM-match scoring",
        }
