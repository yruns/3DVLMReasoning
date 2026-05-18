from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.selectors import build_selector_tools


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev)
    rgbs = {f: tmp_path / f"frame_{f}.png" for f in [10, 20, 30, 40]}
    for r in rgbs.values():
        Image.new("RGB", (320, 240), (200, 200, 200)).save(r)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    10: FrameView(
                        frame_id=10, raw_rgb_path=str(rgbs[10]), bbox_2d=(0, 0, 20, 20)
                    ),
                    20: FrameView(
                        frame_id=20, raw_rgb_path=str(rgbs[20]), bbox_2d=(0, 0, 20, 20)
                    ),
                    30: FrameView(
                        frame_id=30, raw_rgb_path=str(rgbs[30]), bbox_2d=(0, 0, 20, 20)
                    ),
                    40: FrameView(
                        frame_id=40, raw_rgb_path=str(rgbs[40]), bbox_2d=(0, 0, 20, 20)
                    ),
                },
            ),
        ],
        total_frames=100,
        frame_id_range=(0, 99),
        valid_frame_ids=[10, 20, 30, 40],
        bev_image_path=str(bev),
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "camera_trajectory_xy_yaw": {
                f: [float(f), 0.0, 0.0] for f in [10, 20, 30, 40]
            },
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_select_by_proposal_returns_3_frames_with_images(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    raw = tool.invoke({"proposal_ids": [4]})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    for frame in payload["frames"]:
        assert "image_path" in frame
    assert len(rs.tool_trace[-1].image_metadata) == 3
    assert not any(
        key.startswith("vg_" + "pending") for key in rs.bundle.extra_metadata
    )


def test_select_by_proposal_respects_k_cap(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    raw = tool.invoke({"proposal_ids": [4], "k": 5})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
