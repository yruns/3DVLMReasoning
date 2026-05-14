"""Stage 2 agent tool backends.

This package provides backend implementations for the Stage-2 evidence tools.
Each tool module implements callbacks that can be injected into the agent
to provide real evidence acquisition from the scene.

v9: the hypothesis-repair module was removed alongside the pre-v9 hypothesis
tool wrapper. The catalog-first selectors / view tools in this package
replace it.
"""

from .request_crops import (
    CropBackend,
    CropRequest,
    CropResult,
    create_crop_callback,
)

__all__ = [
    "CropRequest",
    "CropResult",
    "CropBackend",
    "create_crop_callback",
]
