from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.mark_frame_with_bbox import build_mark_frame_with_bbox_tool


@pytest.fixture
def rs_with_two(tmp_path: Path) -> Stage2RuntimeState:
    rgb = tmp_path / "frame_42.png"
    Image.new("RGB", (400, 300), (200, 200, 200)).save(rgb)
    bev = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={42: FrameView(frame_id=42, raw_rgb_path=str(rgb), bbox_2d=(20, 50, 120, 150))},
            ),
            SceneProposal(
                proposal_id=7,
                category="table",
                position_3d=(1.0, 0.0, 0.0),
                source="mask3d",
                frame_views={42: FrameView(frame_id=42, raw_rgb_path=str(rgb), bbox_2d=(220, 50, 320, 150))},
            ),
        ],
        total_frames=100,
        frame_id_range=(0, 99),
        valid_frame_ids=[42],
        bev_image_path=str(bev),
    )
    bundle = SimpleNamespace(
        extra_metadata={"scene_catalog": catalog.model_dump(), "vg_pending_images": []}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_response_includes_guard_parseable_fields(rs_with_two):
    tool = build_mark_frame_with_bbox_tool(rs_with_two)
    out = tool.invoke({"frame_id": 42, "ids": [4, 7]})
    assert re.search(r"visible_proposals=\[\s*4\s*,\s*7\s*\]", out)
    assert re.search(r"categories=\['chair'\s*,\s*'table'\]", out)
    assert re.search(r"left_to_right=\['4:chair'\s*,\s*'7:table'\]", out)
    assert re.search(
        r"boxes_2d=\{4:\s*\[20,\s*50,\s*120,\s*150\]\s*,\s*7:\s*\[220,\s*50,\s*320,\s*150\]\}",
        out,
    )


def test_response_includes_filter_summary(rs_with_two):
    tool = build_mark_frame_with_bbox_tool(rs_with_two)
    out = tool.invoke({"frame_id": 42, "labels": ["table"]})
    assert "filtered_by={'labels': ['table'], 'ids': []}" in out
    assert "visible_proposals=[7]" in out
