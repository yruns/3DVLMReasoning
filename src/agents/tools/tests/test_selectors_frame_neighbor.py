import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import make_runtime


def test_select_by_frame_neighbor_temporal_returns_adjacent_frames():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    payload = json.loads(tool.invoke({"anchor_frame_id": 30, "mode": "temporal", "k": 2}))
    fids = [f["frame_id"] for f in payload["frames"]]
    # valid_frame_ids = [10, 20, 30, 40, 50]; nearest to 30 by id are [20, 40]
    assert sorted(fids) == [20, 40]


def test_select_by_frame_neighbor_temporal_skips_anchor_itself():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    payload = json.loads(tool.invoke({"anchor_frame_id": 10, "mode": "temporal", "k": 3}))
    fids = [f["frame_id"] for f in payload["frames"]]
    assert 10 not in fids
    assert len(fids) == 3
    assert fids[0] == 20


def test_select_by_frame_neighbor_viewpoint_diverse_prefers_yaw_spread():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    payload = json.loads(tool.invoke({"anchor_frame_id": 10, "mode": "viewpoint_diverse", "k": 2}))
    fids = [f["frame_id"] for f in payload["frames"]]
    assert len(fids) == 2
    # anchor 10 has yaw=0.0; the two largest yaw deltas among others are 30 (yaw=1.0) and 50 (yaw=3.0)
    assert sorted(fids) == [30, 50]


def test_select_by_frame_neighbor_unknown_anchor_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    resp = tool.invoke({"anchor_frame_id": 999, "mode": "temporal"})
    assert resp.startswith("ERROR")


def test_select_by_frame_neighbor_invalid_mode_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_frame_neighbor")
    resp = tool.invoke({"anchor_frame_id": 10, "mode": "bogus"})
    assert resp.startswith("ERROR")
