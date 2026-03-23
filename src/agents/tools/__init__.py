"""Stage 2 agent tool backends.

This package provides backend implementations for the Stage-2 evidence tools.
Each tool module implements callbacks that can be injected into the agent
to provide real evidence acquisition from the scene.
"""

from .hypothesis_repair import (
    HypothesisAction,
    HypothesisHistoryEntry,
    HypothesisRepairBackend,
    HypothesisRepairConfig,
    create_hypothesis_repair_callback,
)
from .request_crops import (
    CropBackend,
    CropRequest,
    CropResult,
    create_crop_callback,
)

__all__ = [
    # request_crops
    "CropRequest",
    "CropResult",
    "CropBackend",
    "create_crop_callback",
    # hypothesis_repair
    "HypothesisAction",
    "HypothesisHistoryEntry",
    "HypothesisRepairConfig",
    "HypothesisRepairBackend",
    "create_hypothesis_repair_callback",
]
