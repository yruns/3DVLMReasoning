"""Set-of-marks rendering for VG keyframes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)


def test_render_marked_keyframe_writes_png(tmp_path: Path) -> None:
    img = Image.new("RGB", (320, 240), color="white")
    img_path = tmp_path / "rgb.png"
    img.save(img_path)
    out_path = tmp_path / "ann" / "frame_10.png"

    render_marked_keyframe(
        rgb_path=img_path,
        out_path=out_path,
        marks=[
            {"proposal_id": 0, "label": "chair", "bbox_2d": (50, 60, 150, 180)},
            {"proposal_id": 1, "label": "desk",  "bbox_2d": (200, 50, 310, 230)},
        ],
    )
    assert out_path.exists()
    out = Image.open(out_path)
    assert out.size == (320, 240)
