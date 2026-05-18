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
    fids = [10, 20, 30, 40, 50]
    rgbs = {f: tmp_path / f"frame_{f}.png" for f in fids}
    for r in rgbs.values():
        Image.new("RGB", (320, 240), (200, 200, 200)).save(r)
    proposals = [
        SceneProposal(
            proposal_id=i,
            category="obj",
            position_3d=(float(fid), 0.0, 0.0),
            source="mask3d",
            frame_views={
                fid: FrameView(
                    frame_id=fid, raw_rgb_path=str(rgbs[fid]), bbox_2d=(0, 0, 20, 20)
                )
            },
        )
        for i, fid in enumerate(fids)
    ]
    catalog = SceneCatalog(
        scene_id="s",
        proposals=proposals,
        total_frames=100,
        frame_id_range=(0, 99),
        valid_frame_ids=fids,
        bev_image_path=str(bev),
    )
    traj = {
        10: [0.0, 0.0, 0.0],
        20: [1.0, 0.0, 0.0],
        30: [2.0, 0.0, 0.0],
        40: [3.0, 0.0, 0.0],
        50: [10.0, 0.0, 0.0],
    }
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


def test_select_by_coverage_obj_iou_without_seen_returns_three_with_images(
    tmp_path: Path,
):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    raw = tool.invoke({"method": "obj_iou"})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    assert {f["frame_id"] for f in payload["frames"]} == {10, 20, 30}
    for frame in payload["frames"]:
        assert frame.get("image_path")
    assert len(rs.tool_trace[-1].image_metadata) == 3


def test_select_by_coverage_pose_depth_excludes_seen_and_records_images(
    tmp_path: Path,
):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    raw = tool.invoke({"method": "pose_depth", "seen_frame_ids": [10]})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    assert 10 not in {f["frame_id"] for f in payload["frames"]}
    for frame in payload["frames"]:
        assert frame.get("image_path")
    assert len(rs.tool_trace[-1].image_metadata) == 3
