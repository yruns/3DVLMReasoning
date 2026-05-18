from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.selectors import build_selector_tools


def _runtime_bev(tmp_path: Path) -> Stage2RuntimeState:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev)
    fids = [10, 20, 30, 40]
    rgbs = {f: tmp_path / f"frame_{f}.png" for f in fids}
    for r in rgbs.values():
        Image.new("RGB", (320, 240), (200, 200, 200)).save(r)
    traj = {
        10: [1.0, 1.0, 0.0],
        20: [2.0, 2.0, 0.0],
        30: [3.0, 3.0, 0.0],
        40: [4.0, 4.0, 0.0],
    }
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    f: FrameView(
                        frame_id=f, raw_rgb_path=str(rgbs[f]), bbox_2d=(0, 0, 20, 20)
                    )
                    for f in fids
                },
            ),
        ],
        total_frames=100,
        frame_id_range=(0, 99),
        valid_frame_ids=fids,
        bev_image_path=str(bev),
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "camera_trajectory_xy_yaw": traj,
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def _runtime_bbox(tmp_path: Path) -> Stage2RuntimeState:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev)
    fids = [10, 20, 30, 40]
    rgbs = {f: tmp_path / f"frame_{f}.png" for f in fids}
    for r in rgbs.values():
        Image.new("RGB", (320, 240), (200, 200, 200)).save(r)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=1,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    f: FrameView(
                        frame_id=f, raw_rgb_path=str(rgbs[f]), bbox_2d=(0, 0, 20, 20)
                    )
                    for f in fids
                },
            ),
        ],
        total_frames=100,
        frame_id_range=(0, 99),
        valid_frame_ids=fids,
        bev_image_path=str(bev),
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "camera_trajectory_xy_yaw": {f: [float(f), 0.0, 0.0] for f in fids},
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_select_by_region_bev_2d_returns_three_frames_with_images(tmp_path: Path):
    rs = _runtime_bev(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    raw = tool.invoke({"region_type": "bev_2d", "region": [0.0, 0.0, 100.0, 100.0]})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    for frame in payload["frames"]:
        assert frame.get("image_path")
    assert len(rs.tool_trace[-1].image_metadata) == 3


def test_select_by_region_bbox_3d_returns_three_frames_with_images(tmp_path: Path):
    rs = _runtime_bbox(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    raw = tool.invoke(
        {"region_type": "bbox_3d", "region": [-1.0, -1.0, -1.0, 2.0, 2.0, 2.0]}
    )
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    for frame in payload["frames"]:
        assert frame.get("image_path")
    assert len(rs.tool_trace[-1].image_metadata) == 3
