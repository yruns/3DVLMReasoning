"""Base adapter interface for benchmark integration.

This module defines the abstract base class for adapters that bridge
benchmark datasets to the Stage-2 agent interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.task_types import (
    KeyframeEvidence,
    Stage2EvidenceBundle,
    Stage2TaskSpec,
)


class BenchmarkType(str, Enum):
    """Supported benchmark types."""

    OPENEQA = "openeqa"
    SCANREFER = "scanrefer"
    SQA3D = "sqa3d"
    REPLICA = "replica"
    SCANNET = "scannet"
    SPACE_3D = "space_3d"
    CUSTOM = "custom"


@dataclass
class BenchmarkSampleInfo:
    """Metadata for a single benchmark sample.

    Attributes:
        sample_id: Unique sample identifier
        benchmark_type: Type of benchmark
        scene_id: Scene identifier
        question: Question or task instruction
        answer: Ground truth answer (if available)
        choices: Answer choices for MC tasks
        metadata: Additional benchmark-specific metadata
    """

    sample_id: str
    benchmark_type: BenchmarkType
    scene_id: str
    question: str
    answer: str | None = None
    choices: list[str] | None = None
    metadata: dict[str, Any] | None = None


class FrameProvider(ABC):
    """Abstract base class for providing frames from a scene.

    Frame providers abstract away the details of loading frames
    from different dataset formats.
    """

    @abstractmethod
    def get_frame_count(self, scene_id: str) -> int:
        """Return total number of frames in a scene."""
        pass

    @abstractmethod
    def get_frame_path(self, scene_id: str, frame_id: int) -> Path:
        """Return path to a frame image."""
        pass

    @abstractmethod
    def get_frame_paths(self, scene_id: str, frame_ids: list[int]) -> list[Path]:
        """Return paths to multiple frames."""
        pass

    def get_bev_path(self, scene_id: str) -> Path | None:
        """Return path to bird's eye view image if available."""
        return None


class BenchmarkAdapter(ABC):
    """Abstract base class for benchmark adapters.

    Benchmark adapters convert benchmark samples into the Stage-2
    task specification and evidence bundle format.
    """

    def __init__(self, data_root: str | Path, **kwargs):
        """Initialize the adapter.

        Args:
            data_root: Root directory containing the benchmark data
            **kwargs: Additional adapter-specific configuration
        """
        self.data_root = Path(data_root)

    @property
    @abstractmethod
    def benchmark_type(self) -> BenchmarkType:
        """Return the benchmark type."""
        pass

    @abstractmethod
    def get_sample_ids(self) -> list[str]:
        """Return all available sample IDs."""
        pass

    @abstractmethod
    def load_sample(self, sample_id: str) -> BenchmarkSampleInfo:
        """Load a benchmark sample by ID.

        Args:
            sample_id: Sample identifier

        Returns:
            BenchmarkSampleInfo with sample data
        """
        pass

    @abstractmethod
    def build_task_spec(self, sample: BenchmarkSampleInfo) -> Stage2TaskSpec:
        """Convert a sample to a task specification.

        Args:
            sample: The benchmark sample

        Returns:
            Stage2TaskSpec for the agent
        """
        pass

    @abstractmethod
    def build_evidence_bundle(
        self,
        sample: BenchmarkSampleInfo,
        frame_ids: list[int],
        frame_provider: FrameProvider,
    ) -> Stage2EvidenceBundle:
        """Build an evidence bundle for a sample.

        Args:
            sample: The benchmark sample
            frame_ids: Selected keyframe indices
            frame_provider: Provider for loading frames

        Returns:
            Stage2EvidenceBundle with visual evidence
        """
        pass

    def evaluate_response(
        self,
        sample: BenchmarkSampleInfo,
        prediction: str,
    ) -> dict[str, Any]:
        """Evaluate a predicted response against ground truth.

        Args:
            sample: The benchmark sample with ground truth
            prediction: The predicted answer

        Returns:
            Dict with evaluation metrics
        """
        if sample.answer is None:
            return {"evaluated": False, "reason": "No ground truth"}

        # Default: exact match
        is_correct = prediction.strip().lower() == sample.answer.strip().lower()
        return {
            "evaluated": True,
            "correct": is_correct,
            "prediction": prediction,
            "ground_truth": sample.answer,
        }


def build_evidence_from_frames(
    frame_paths: list[Path],
    frame_ids: list[int] | None = None,
    scores: list[float] | None = None,
    bev_path: Path | None = None,
) -> Stage2EvidenceBundle:
    """Build an evidence bundle from frame paths.

    Args:
        frame_paths: Paths to keyframe images
        frame_ids: Frame indices (defaults to 0..N-1)
        scores: Relevance scores (defaults to 1.0)
        bev_path: Optional path to BEV image

    Returns:
        Stage2EvidenceBundle
    """
    if frame_ids is None:
        frame_ids = list(range(len(frame_paths)))
    if scores is None:
        scores = [1.0] * len(frame_paths)

    keyframes = [
        KeyframeEvidence(
            frame_id=fid,
            image_path=str(fpath),
            score=score,
        )
        for fid, fpath, score in zip(frame_ids, frame_paths, scores, strict=False)
    ]

    return Stage2EvidenceBundle(
        keyframes=keyframes,
        bev_path=str(bev_path) if bev_path else None,
    )


__all__ = [
    "BenchmarkType",
    "BenchmarkSampleInfo",
    "FrameProvider",
    "BenchmarkAdapter",
    "build_evidence_from_frames",
]
