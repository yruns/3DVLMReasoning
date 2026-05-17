from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.mark_frame_with_bbox import (
    _decide_label_anchor,
    build_mark_frame_with_bbox_tool,
)


def _runtime_with_one_box(tmp_path: Path, bbox: tuple[int, int, int, int]) -> Stage2RuntimeState:
    rgb_path = tmp_path / "frame_42.png"
    Image.new("RGB", (400, 300), (200, 200, 200)).save(rgb_path)
    bev_path = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev_path)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    42: FrameView(
                        frame_id=42, raw_rgb_path=str(rgb_path), bbox_2d=bbox
                    )
                },
            ),
        ],
        total_frames=100,
        frame_id_range=(0, 99),
        valid_frame_ids=[42],
        bev_image_path=str(bev_path),
    )
    bundle = SimpleNamespace(
        extra_metadata={"scene_catalog": catalog.model_dump(), "vg_pending_images": []}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_decide_label_anchor_topleft_inside_for_large_bbox():
    """v9.3: even on a large bbox, the label is anchored at the top-left
    INSIDE the bbox (was: centred for large bboxes)."""
    from agents.tools.mark_frame_with_bbox import LABEL_PADDING_PX

    anchor = _decide_label_anchor(
        bbox=(50, 50, 250, 200),
        text_w=60,
        text_h=18,
    )
    assert anchor.mode == "topleft_inside"
    pad = LABEL_PADDING_PX
    # Background top-left = bbox top-left; size = (text_w + 2*pad, text_h + 2*pad).
    assert anchor.bg_box == (50, 50, 50 + 60 + 2 * pad, 50 + 18 + 2 * pad)
    # Text origin = (bbox_x1 + pad, bbox_y1 + text_h + pad).
    assert anchor.origin == (50 + pad, 50 + 18 + pad)


def test_decide_label_anchor_above_when_bbox_too_small():
    """When the label doesn't fit inside the bbox (either too narrow or
    too short), v9.3 places it just ABOVE the bbox top, left-edge-aligned."""
    from agents.tools.mark_frame_with_bbox import LABEL_PADDING_PX

    anchor = _decide_label_anchor(
        bbox=(50, 50, 110, 80),  # 60 × 30 — too small for label_w = 60 + 2*pad
        text_w=60,
        text_h=18,
    )
    assert anchor.mode == "topleft_above"
    pad = LABEL_PADDING_PX
    label_h = 18 + 2 * pad
    # Label is placed in [y1 - label_h, y1].
    assert anchor.bg_box == (50, 50 - label_h, 50 + 60 + 2 * pad, 50)


def test_decide_label_anchor_above_clamps_to_image_top():
    """When the bbox top is too close to the image top edge to fit the
    label above, the label is clamped to start at row 0 (overlapping the
    bbox slightly is preferred over going negative)."""
    from agents.tools.mark_frame_with_bbox import LABEL_PADDING_PX

    anchor = _decide_label_anchor(
        bbox=(50, 4, 110, 30),
        text_w=60,
        text_h=18,
        img_shape=(300, 400, 3),
    )
    assert anchor.mode == "topleft_above"
    assert anchor.bg_box[1] == 0  # clamped to image top
    pad = LABEL_PADDING_PX
    assert anchor.bg_box == (50, 0, 50 + 60 + 2 * pad, 18 + 2 * pad)


def test_render_label_uses_bbox_colour_at_topleft(tmp_path: Path):
    """v9.3 contract: the label background colour matches the bbox stroke
    palette colour (was: black background + white text). Verifies that
    the very top-left corner of the bbox is the palette green of the first
    proposal, not black or white.
    """
    from agents.tools.mark_frame_with_bbox import BBOX_PALETTE

    rs = _runtime_with_one_box(tmp_path, bbox=(50, 50, 350, 250))
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [4]})
    rendered_path = Path(out.split("mark image at")[1].split(";", 1)[0].strip())
    img = np.asarray(Image.open(rendered_path).convert("RGB"))
    green = np.array(BBOX_PALETTE[0])
    # Look just inside the bbox top-left (a few px past the stroke + outline).
    # The label background should colour a contiguous region of pixels.
    label_patch = img[55:75, 55:120]
    green_count = int(np.all(label_patch == green, axis=-1).sum())
    assert green_count > 50, (
        f"expected the bbox palette green to fill a label background patch "
        f"in the bbox top-left; only {green_count} green pixels found"
    )


def test_render_label_centre_is_no_longer_painted_on_large_bbox(tmp_path: Path):
    """v9.3: a large bbox's CENTRE should now be empty (label moved to
    top-left). Regression for the user-requested unified positioning."""
    rs = _runtime_with_one_box(tmp_path, bbox=(50, 50, 350, 250))
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [4]})
    rendered_path = Path(out.split("mark image at")[1].split(";", 1)[0].strip())
    img = np.asarray(Image.open(rendered_path).convert("RGB"))
    # Centre of bbox ~ (200, 150). Background of the synthetic frame is grey
    # (200, 200, 200). The centre patch should still be that grey (no text).
    centre_patch = img[140:160, 180:220]
    grey = np.array([200, 200, 200])
    grey_share = float(np.all(centre_patch == grey, axis=-1).mean())
    assert grey_share > 0.95, (
        f"v9.3 label moved to top-left; bbox centre should remain unpainted "
        f"(>95 % grey); got {grey_share:.2%} grey share"
    )


def test_label_font_scale_is_roughly_20_percent_larger_than_v91():
    """The user-tuned font scale ratio: at 1500 px width, v9.3 hits ~1.0
    where v9.1 was 0.83. That is +20 % over the legacy formula."""
    from agents.tools.mark_frame_with_bbox import _label_font_scale

    v93_at_1500 = _label_font_scale(1500)
    v91_at_1500 = max(0.7, 1500 / 1800.0)
    assert v93_at_1500 == 1.0
    assert v93_at_1500 / v91_at_1500 == pytest.approx(1.2, abs=0.01)


def test_text_color_picked_by_luminance_for_each_palette_color():
    """White text must be picked against the dark palette colours (green,
    red, blue, purple); black text against yellow."""
    from agents.tools.mark_frame_with_bbox import (
        BBOX_PALETTE,
        _label_text_color_for_bg,
    )

    expected = {
        (34, 197, 94): (255, 255, 255),  # green → white
        (239, 68, 68): (255, 255, 255),  # red   → white
        (59, 130, 246): (255, 255, 255), # blue  → white
        (234, 179, 8): (0, 0, 0),        # yellow→ black
        (168, 85, 247): (255, 255, 255), # purple→ white
    }
    for bg in BBOX_PALETTE:
        assert _label_text_color_for_bg(bg) == expected[bg]
