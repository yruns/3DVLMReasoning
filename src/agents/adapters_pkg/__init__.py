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

# Note: Specific benchmark adapters remain in the parent directory
# for backward compatibility:
# - benchmark_adapters.py -> MultiBenchmarkAdapter, OpenEQAFrameProvider, etc.
# - space3d_bench_adapter.py -> Space3DBenchAdapter
#
# Import them from the parent module:
# from agents import MultiBenchmarkAdapter, OpenEQAFrameProvider

__all__ = [
    "BenchmarkType",
    "BenchmarkSampleInfo",
    "FrameProvider",
    "BenchmarkAdapter",
    "build_evidence_from_frames",
]
