import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import PRIMARY_SKILL, make_runtime


class _FakeKeyframeSelector:
    def __init__(self):
        self.calls: list[dict] = []

    def execute_hypothesis_dict(self, hypothesis_dict, k=3, hidden_categories=None):
        self.calls.append({"hypothesis": hypothesis_dict, "k": k, "hidden": hidden_categories})
        return {"keyframe_indices": [40, 50, 10], "summary": "executed table hypothesis"}


def _attach(rs):
    rs.keyframe_selector = _FakeKeyframeSelector()
    return rs.keyframe_selector


def test_select_by_hypothesis_dispatches_to_executor():
    rs = make_runtime()
    fake = _attach(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_hypothesis")
    hyp = {"hypotheses": [{"rank": 1, "kind": "direct", "grounding_query": {"root": {"category": "table"}}}]}
    payload = json.loads(tool.invoke({"hypothesis": hyp, "k": 2}))
    assert fake.calls and fake.calls[0]["k"] == 2
    fids = [f["frame_id"] for f in payload["frames"]]
    assert sorted(fids) == [10, 40, 50]


def test_select_by_hypothesis_invalid_payload_errors():
    rs = make_runtime()
    _attach(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_hypothesis")
    resp = tool.invoke({"hypothesis": "not a dict"})
    assert resp.startswith("ERROR")
    assert "hypothesis" in resp


def test_select_by_hypothesis_gates_on_skill():
    rs = make_runtime()
    rs.skills_loaded.discard(PRIMARY_SKILL)
    _attach(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_hypothesis")
    resp = tool.invoke({"hypothesis": {"hypotheses": []}})
    assert resp.startswith("ERROR")
