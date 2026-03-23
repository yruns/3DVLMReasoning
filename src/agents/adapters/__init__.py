"""Adapters package for benchmark integration.

This module provides adapters for connecting various 3D scene understanding
benchmarks to the Stage-2 agent interface.
"""

from .base import (
    BenchmarkAdapter,
    BenchmarkSampleInfo,
    BenchmarkType,
    FrameProvider,
    build_evidence_from_frames,
)

__all__ = [
    "BenchmarkType",
    "BenchmarkSampleInfo",
    "FrameProvider",
    "BenchmarkAdapter",
    "build_evidence_from_frames",
]
