"""Benchmark adapter base classes.

Defines the pluggable adapter interface that all benchmarks implement.
Adding a new benchmark = implementing one BenchmarkAdapter subclass.

The pipeline calls adapters in this order:
1. load_samples() — get evaluation samples
2. build_task_spec(sample) — create Stage2TaskSpec
3. extract_prediction(result) — extract prediction from agent output
4. evaluate(predictions, samples) — compute metrics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.core.response_schema import Stage2AgentResult
    from src.agents.core.task_types import Stage2TaskSpec


@dataclass
class BenchmarkSample:
    """Minimal unified sample interface.

    All benchmark-specific sample types should extend this.

    Attributes:
        sample_id: Unique identifier for this sample.
        scene_id: Scene identifier (e.g., "scene0415_00").
        query: Natural language input (question or description).
        metadata: Arbitrary extra metadata.
    """

    sample_id: str
    scene_id: str
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    """Pluggable benchmark adapter interface.

    Implement this to add a new benchmark. The pipeline orchestrator
    calls methods in the documented order above. Each adapter owns:
    - Data loading and sample iteration
    - Task specification construction
    - Prediction extraction from agent output
    - Evaluation metric computation
    """

    @abstractmethod
    def load_samples(
        self, split: str = "val", **kwargs: Any
    ) -> list[BenchmarkSample]:
        """Load benchmark samples for a given split."""

    @abstractmethod
    def build_task_spec(self, sample: BenchmarkSample) -> Stage2TaskSpec:
        """Create task specification from a sample."""

    @abstractmethod
    def extract_prediction(
        self, sample: BenchmarkSample, result: Stage2AgentResult
    ) -> dict[str, Any]:
        """Extract benchmark-specific prediction from agent result."""

    @abstractmethod
    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        samples: list[BenchmarkSample],
    ) -> dict[str, Any]:
        """Compute evaluation metrics over all predictions."""

    def get_scene_path(self, sample: BenchmarkSample) -> Path:
        """Get path to scene data for Stage 1. Override per benchmark."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_scene_path()"
        )
