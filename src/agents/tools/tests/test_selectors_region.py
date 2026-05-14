import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import make_runtime


def test_select_by_region_bev_2d_picks_frames_whose_camera_in_box():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    payload = json.loads(
        tool.invoke({"region": [-0.5, -0.5, 1.6, 1.0], "region_type": "bev_2d", "k": 5})
    )
    fids = sorted(f["frame_id"] for f in payload["frames"])
    # Camera xy: 10->(0,0), 20->(0.5,0), 30->(1.0,0), 40->(1.5,0.5), 50->(-2,-2)
    # In box [-0.5,-0.5, 1.6, 1.0]: frames 10, 20, 30, 40
    assert fids == [10, 20, 30, 40]


def test_select_by_region_bbox_3d_picks_frames_whose_proposals_inside():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    payload = json.loads(
        tool.invoke(
            {
                "region": [-0.5, -0.5, -0.5, 2.5, 0.5, 0.5],
                "region_type": "bbox_3d",
                "k": 10,
            }
        )
    )
    fids = sorted(f["frame_id"] for f in payload["frames"])
    assert fids == [10, 20, 30]


def test_select_by_region_invalid_region_length_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    resp = tool.invoke({"region": [0, 0], "region_type": "bev_2d"})
    assert resp.startswith("ERROR")
    assert "region" in resp


def test_select_by_region_unknown_region_type_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_region")
    resp = tool.invoke({"region": [0, 0, 1, 1], "region_type": "foo"})
    assert resp.startswith("ERROR")
