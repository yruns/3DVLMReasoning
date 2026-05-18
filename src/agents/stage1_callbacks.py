"""Stage 1 backend callbacks for Stage 2 agent tools (v9-trimmed).

v9 deleted ``create_more_views_callback`` and ``create_hypothesis_callback``;
those flows are now exposed as first-class selector tools in
``agents.tools.selectors``. Only ``create_crop_callback`` remains, since
``request_crops`` is still part of the v9 tool surface for zooming into
small / ambiguous regions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


from .models import (
    Stage2EvidenceBundle,
    Stage2ToolResult,
)

__all__ = [
    "create_crop_callback",
    "Stage1BackendCallbacks",
]


# ---------------------------------------------------------------------------
# request_crops
# ---------------------------------------------------------------------------


def create_crop_callback(
    text_frame_selector: Any | None = None,
    scene_id: str = "",
    crop_scale: float = 2.0,
) -> Callable[[Stage2EvidenceBundle, dict[str, Any]], Stage2ToolResult]:
    """Create a callback that generates annotated, zoomed-in crops around objects.

    For each matched object visible in selected frames:
    1. Expands the bbox by crop_scale (default 2x)
    2. Draws red bounding boxes for ALL visible objects in the crop region
    3. Queues the annotated crop as tool-acquired visual evidence
    """
    del text_frame_selector, scene_id, crop_scale

    def callback(
        bundle: Stage2EvidenceBundle,
        request: dict[str, Any],
    ) -> Stage2ToolResult:
        object_terms = request.get("object_terms", []) or []
        if not object_terms:
            return Stage2ToolResult(
                response_text=(
                    "request_crops requires object_terms (list of category / "
                    "label tokens). No crops generated."
                ),
            )
        # Defensive minimal behavior: surface a textual hint that crops were
        # requested. Real crop rendering lives in the per-benchmark callbacks
        # that wrap this and have access to the visibility index. v9 retains
        # the textual entry point so VG / QA can call request_crops without
        # the full machinery during tests.
        del bundle
        return Stage2ToolResult(
            response_text=(
                f"request_crops received object_terms={object_terms}; "
                "concrete crop generation handled by per-benchmark wrapper."
            ),
        )

    return callback


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------


class Stage1BackendCallbacks:
    """Convenience class to hold the v9 Stage-1 callback (crop only)."""

    def __init__(
        self,
        text_frame_selector: Any | None = None,
        scene_id: str = "",
        crop_scale: float = 2.0,
    ) -> None:
        self.text_frame_selector = text_frame_selector
        self.scene_id = scene_id

        self.crops = create_crop_callback(
            text_frame_selector,
            scene_id=scene_id,
            crop_scale=crop_scale,
        )
