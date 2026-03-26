"""Base adapter interface for benchmark integration.

This module defines the abstract base class for adapters that bridge
benchmark datasets to the Stage-2 agent interface.

The adapter pattern enables:
1. Unified interface for multiple benchmarks (OpenEQA, SQA3D, ScanRefer)
2. Flexible frame provision strategies (real data, mock data, episode history)
3. Benchmark-specific evaluation logic
4. Separation of concerns between Stage-1 retrieval and Stage-2 reasoning
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
    """Supported benchmark types.

    This enum provides type safety and autocomplete for benchmark names
    throughout the codebase.
    """

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

    This unified structure allows different benchmarks to be processed
    through a common interface while preserving benchmark-specific
    metadata in the extra field.

    Attributes:
        sample_id: Unique sample identifier within the benchmark
        benchmark_type: Type of benchmark (OpenEQA, SQA3D, etc.)
        scene_id: Scene identifier (e.g., "room0", "scene0000")
        question: Question text or task instruction
        answer: Ground truth answer (if available)
        choices: Answer choices for multiple-choice tasks
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
    from different dataset formats (ConceptGraph scenes, episode
    history, static frame directories, etc.).

    This allows the same Stage-2 agent to work with:
    - Real preprocessed 3D scenes (Replica, ScanNet)
    - Episode trajectories (OpenEQA)
    - Mock data for testing
    """

    @abstractmethod
    def get_frame_count(self, scene_id: str) -> int:
        """Return total number of frames available in a scene.

        Args:
            scene_id: Scene identifier

        Returns:
            Total frame count
        """
        pass

    @abstractmethod
    def get_frame_path(self, scene_id: str, frame_id: int) -> Path:
        """Return path to a specific frame image.

        Args:
            scene_id: Scene identifier
            frame_id: Frame index

        Returns:
            Path to the frame image file
        """
        pass

    @abstractmethod
    def get_frame_paths(self, scene_id: str, frame_ids: list[int]) -> list[Path]:
        """Return paths to multiple frames efficiently.

        This method allows for batch operations and caching optimizations.

        Args:
            scene_id: Scene identifier
            frame_ids: List of frame indices

        Returns:
            List of paths to frame images
        """
        pass

    def get_bev_path(self, scene_id: str) -> Path | None:
        """Return path to bird's eye view image if available.

        BEV images provide spatial context for navigation and
        spatial reasoning tasks.

        Args:
            scene_id: Scene identifier

        Returns:
            Path to BEV image, or None if not available
        """
        return None


class BenchmarkAdapter(ABC):
    """Abstract base class for benchmark adapters.

    Benchmark adapters convert benchmark samples into the Stage-2
    task specification and evidence bundle format. This allows the
    Stage-2 agent to work with different benchmarks through a
    unified interface.

    The adapter is responsible for:
    1. Loading benchmark samples
    2. Converting samples to Stage2TaskSpec
    3. Building evidence bundles (with or without Stage-1)
    4. Evaluating agent responses against ground truth

    Typical usage:
        adapter = OpenEQAAdapter(data_root="/path/to/openeqa")
        sample_ids = adapter.get_sample_ids()

        for sample_id in sample_ids:
            sample = adapter.load_sample(sample_id)
            task_spec = adapter.build_task_spec(sample)
            evidence = adapter.build_evidence_bundle(
                sample, frame_ids, frame_provider
            )

            # Run Stage-2 agent
            result = agent.run(task_spec, evidence)

            # Evaluate
            metrics = adapter.evaluate_response(sample, result.answer)
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
        """Return the benchmark type.

        Returns:
            BenchmarkType enum value
        """
        pass

    @abstractmethod
    def get_sample_ids(self) -> list[str]:
        """Return all available sample IDs in the benchmark.

        This enables iteration over the entire benchmark for
        batch evaluation.

        Returns:
            List of sample identifiers
        """
        pass

    @abstractmethod
    def load_sample(self, sample_id: str) -> BenchmarkSampleInfo:
        """Load a benchmark sample by ID.

        This method loads the full sample metadata including
        question, ground truth, and benchmark-specific info.

        Args:
            sample_id: Sample identifier

        Returns:
            BenchmarkSampleInfo with sample data

        Raises:
            ValueError: If sample_id is not found
        """
        pass

    @abstractmethod
    def build_task_spec(self, sample: BenchmarkSampleInfo) -> Stage2TaskSpec:
        """Convert a benchmark sample to a task specification.

        This method translates the benchmark's native format into
        the Stage-2 agent's expected format.

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

        This method constructs the visual evidence that the Stage-2
        agent will reason over. In full pipeline mode, frame_ids come
        from Stage-1 retrieval. In frame-based mode, they may be
        sampled from the episode history.

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

        This method implements benchmark-specific evaluation logic.
        The base implementation does exact string matching, but
        subclasses can override for more sophisticated metrics
        (IoU, GloVe similarity, etc.).

        Args:
            sample: The benchmark sample with ground truth
            prediction: The predicted answer from the agent

        Returns:
            Dict with evaluation metrics:
                - evaluated: bool, whether evaluation was performed
                - correct: bool, whether prediction matches ground truth
                - prediction: str, the prediction
                - ground_truth: str, the expected answer
                - Additional benchmark-specific metrics
        """
        if sample.answer is None:
            return {"evaluated": False, "reason": "No ground truth"}

        # Default: exact match (case-insensitive)
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

    Utility function to construct a Stage2EvidenceBundle from
    a list of frame paths. This is commonly used by adapters
    in their build_evidence_bundle implementations.

    Args:
        frame_paths: Paths to keyframe images
        frame_ids: Frame indices (defaults to 0..N-1 if not provided)
        scores: Relevance scores for each frame (defaults to 1.0)
        bev_path: Optional path to bird's eye view image

    Returns:
        Stage2EvidenceBundle ready for agent consumption

    Example:
        paths = [Path("frame0.jpg"), Path("frame1.jpg")]
        evidence = build_evidence_from_frames(
            paths,
            frame_ids=[0, 10],
            scores=[0.95, 0.87]
        )
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
