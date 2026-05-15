from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.selectors import build_selector_tools


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    bev = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev)
    fids = [10, 20, 30, 40, 50]
    rgbs = {f: tmp_path / f"frame_{f}.png" for f in fids}
    for r in rgbs.values():
        Image.new("RGB", (320, 240), (200, 200, 200)).save(r)
    fw = {
        f: FrameView(frame_id=f, raw_rgb_path=str(rgbs[f]), bbox_2d=(0, 0, 20, 20)) for f in fids
    }
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views=dict(fw),
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
            "vg_pending_images": [],
            "camera_trajectory_xy_yaw": {f: [float(f), 0.0, 0.0] for f in fids},
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_select_by_frame_neighbor_temporal_returns_three_frames_with_images(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    raw = tool.invoke({"anchor_frame_id": 20, "mode": "temporal"})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    for frame in payload["frames"]:
        assert "image_path" in frame
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert len(pending) == 3


def test_select_by_frame_neighbor_respects_k_cap_viewpoint_diverse(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    raw = tool.invoke({"anchor_frame_id": 20, "mode": "viewpoint_diverse", "k": 5})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
