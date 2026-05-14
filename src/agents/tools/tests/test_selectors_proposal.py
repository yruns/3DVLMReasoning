import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import make_runtime


def test_select_by_proposal_union_default():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    payload = json.loads(tool.invoke({"proposal_ids": [0, 2]}))
    fids = sorted(f["frame_id"] for f in payload["frames"])
    assert fids == [10, 20, 40]


def test_select_by_proposal_intersection_when_require_all_true():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    payload = json.loads(tool.invoke({"proposal_ids": [0, 2], "require_all": True}))
    fids = [f["frame_id"] for f in payload["frames"]]
    assert fids == [10]


def test_select_by_proposal_k_caps_output():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    payload = json.loads(tool.invoke({"proposal_ids": [0, 2], "k": 2}))
    assert len(payload["frames"]) == 2


def test_select_by_proposal_unknown_id_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    resp = tool.invoke({"proposal_ids": [9999]})
    assert resp.startswith("ERROR")
    assert "9999" in resp


def test_select_by_proposal_empty_list_errors():
    rs = make_runtime()
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_proposal")
    resp = tool.invoke({"proposal_ids": []})
    assert resp.startswith("ERROR")
