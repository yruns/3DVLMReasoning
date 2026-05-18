from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.mark_frame_with_bbox import (
    BBOX_PALETTE,
    BLACK_OUTLINE_PAD,
    build_mark_frame_with_bbox_tool,
)


@pytest.fixture
def tiny_runtime(tmp_path: Path) -> tuple[Stage2RuntimeState, Path]:
    rgb_path = tmp_path / "frame_42.png"
    Image.new("RGB", (400, 300), (200, 200, 200)).save(rgb_path)
    bev_path = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev_path)
    catalog = SceneCatalog(
        scene_id="s_render",
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    42: FrameView(
                        frame_id=42,
                        raw_rgb_path=str(rgb_path),
                        bbox_2d=(50, 50, 250, 200),
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
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs, rgb_path


def test_render_writes_png_with_bbox_pixels(tiny_runtime):
    rs, _ = tiny_runtime
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [4]})
    assert "mark image at" in out
    rendered_path = Path(out.split("mark image at")[1].split(";", 1)[0].strip())
    assert rendered_path.exists()
    img = np.asarray(Image.open(rendered_path).convert("RGB"))
    # The first palette colour is green (34, 197, 94). It should appear somewhere on
    # the bbox border (around y=50 / y=200, x=50 / x=250).
    green = np.array(BBOX_PALETTE[0])
    matches = np.all(img == green, axis=2)
    assert matches.any(), "expected at least one pixel of the first palette colour"


def test_render_has_black_outline_outside_colour_stroke(tiny_runtime):
    """The colour stroke must be wrapped by a black silhouette for any-background contrast."""
    rs, _ = tiny_runtime
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [4]})
    rendered_path = Path(out.split("mark image at")[1].split(";", 1)[0].strip())
    img = np.asarray(Image.open(rendered_path).convert("RGB"))
    # Walk outward from the colour stroke at the top edge (y just above bbox y1=50).
    green = np.array(BBOX_PALETTE[0])
    black = np.array((0, 0, 0))
    found_green_then_black = False
    for x in (60, 120, 180, 240):
        for y in range(45, 56):
            if np.array_equal(img[y, x], green):
                # Look BLACK_OUTLINE_PAD pixels outside the green band.
                if np.array_equal(img[max(0, y - BLACK_OUTLINE_PAD - 1), x], black):
                    found_green_then_black = True
                    break
        if found_green_then_black:
            break
    assert found_green_then_black, "expected a black outline outside the colour stroke"


def test_image_metadata_is_recorded_for_trace_injection(tiny_runtime):
    rs, _ = tiny_runtime
    tool = build_mark_frame_with_bbox_tool(rs)
    tool.invoke({"frame_id": 42, "ids": [4]})
    assert rs.tool_trace[-1].image_metadata
    assert rs.tool_trace[-1].image_metadata[-1]["image_path"].endswith(".png")
