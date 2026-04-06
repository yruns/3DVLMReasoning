"""Benchmark loaders for 3D scene understanding evaluation.

Supported benchmarks:
- OpenEQA: Embodied Question Answering (CVPR 2024)
- SQA3D: Situated Question Answering in 3D Scenes (CVPR 2023)
- ScanRefer: 3D Visual Grounding (ECCV 2020)
- EmbodiedScan: 3D Visual Grounding with Oriented BBoxes
- EAI: Embodied Agent Interface (NeurIPS 2024)
"""

from .base import BenchmarkAdapter, BenchmarkSample
from .embodiedscan_loader import EmbodiedScanDataset, EmbodiedScanVGSample
from .openeqa_loader import OpenEQADataset, OpenEQASample, download_openeqa
from .scanrefer_loader import (
    BoundingBox3D,
    ScanReferDataset,
    ScanReferEvaluationResult,
    ScanReferSample,
    compute_iou_3d,
    compute_scanrefer_metrics,
    compute_scanrefer_metrics_by_category,
    download_scanrefer,
    evaluate_scanrefer,
)
from .sqa3d_loader import (
    SQA3DDataset,
    SQA3DEvaluationResult,
    SQA3DSample,
    SQA3DSituation,
    compute_sqa3d_metrics,
    download_sqa3d,
    evaluate_sqa3d,
)

__all__ = [
    # Base
    "BenchmarkAdapter",
    "BenchmarkSample",
    # EmbodiedScan
    "EmbodiedScanDataset",
    "EmbodiedScanVGSample",
    # OpenEQA
    "OpenEQADataset",
    "OpenEQASample",
    "download_openeqa",
    # SQA3D
    "SQA3DDataset",
    "SQA3DEvaluationResult",
    "SQA3DSample",
    "SQA3DSituation",
    "compute_sqa3d_metrics",
    "download_sqa3d",
    "evaluate_sqa3d",
    # ScanRefer
    "BoundingBox3D",
    "ScanReferDataset",
    "ScanReferEvaluationResult",
    "ScanReferSample",
    "compute_iou_3d",
    "compute_scanrefer_metrics",
    "compute_scanrefer_metrics_by_category",
    "download_scanrefer",
    "evaluate_scanrefer",
]
