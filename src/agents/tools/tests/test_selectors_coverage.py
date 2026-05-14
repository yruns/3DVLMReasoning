import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import make_runtime


def test_select_by_coverage_obj_iou_picks_most_jaccard_distinct():
    rs = make_runtime()
    rs.seen_image_paths.update(["r.png"])
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    payload = json.loads(
        tool.invoke({"method": "obj_iou", "k": 2, "seen_frame_ids": [10]})
    )
    fids = [f["frame_id"] for f in payload["frames"]]
    # Frame 10 contains proposals {0, 2}; the most different by Jaccard is frame 50 ({3})
    assert 50 in fids
    assert 10 not in fids


def test_select_by_coverage_pose_depth_picks_most_distant_camera():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    payload = json.loads(
        tool.invoke({"method": "pose_depth", "k": 1, "seen_frame_ids": [10]})
    )
    fids = [f["frame_id"] for f in payload["frames"]]
    # camera xy: 10->(0,0); farthest from (0,0) is 50->(-2,-2)
    assert fids == [50]


def test_select_by_coverage_invalid_method_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    resp = tool.invoke({"method": "bogus"})
    assert resp.startswith("ERROR")


def test_select_by_coverage_no_seen_frames_uses_runtime_seen_image_paths(monkeypatch):
    rs = make_runtime()
    rs.seen_image_paths = set()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_coverage")
    payload = json.loads(tool.invoke({"method": "obj_iou", "k": 2}))
    # When seen set is empty, every frame is "novel"; returns first k by valid_frame_ids order.
    fids = [f["frame_id"] for f in payload["frames"]]
    assert fids[:2] == [10, 20]
