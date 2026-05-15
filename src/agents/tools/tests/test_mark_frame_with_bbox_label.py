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


def test_decide_label_anchor_centre_for_large_bbox():
    anchor = _decide_label_anchor(
        bbox=(50, 50, 250, 200),
        text_w=60,
        text_h=18,
    )
    cx, cy = (50 + 250) / 2, (50 + 200) / 2
    assert anchor.mode == "centre"
    # area_ratio = 60*18 / (200 * 150) = 1080 / 30000 = 0.036 < 0.15
    assert anchor.origin == (int(cx - 60 / 2), int(cy + 18 / 2))


def test_decide_label_anchor_topleft_for_small_bbox():
    anchor = _decide_label_anchor(
        bbox=(50, 50, 110, 80),
        text_w=60,
        text_h=18,
    )
    # area_ratio = 60*18 / (60*30) = 1080 / 1800 = 0.6 > 0.15 -> top-left inside
    assert anchor.mode == "topleft"
    assert anchor.origin == (50 + 4, 50 + 18 + 4)


def test_render_includes_white_label_pixels_in_centre_for_large_bbox(tmp_path: Path):
    rs = _runtime_with_one_box(tmp_path, bbox=(50, 50, 350, 250))
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [4]})
    rendered_path = Path(out.split("mark image at")[1].split(";", 1)[0].strip())
    img = np.asarray(Image.open(rendered_path).convert("RGB"))
    # Centre of bbox ~ (200, 150); expect a white-pixel cluster nearby (label text).
    centre_patch = img[140:160, 180:220]
    has_white = (centre_patch == np.array([255, 255, 255])).all(axis=-1).any()
    assert has_white, "expected centred label text in centre of large bbox"
