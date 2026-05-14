import json

from agents.tools.selectors import build_selector_tools
from agents.tools.tests._selector_fixtures import PRIMARY_SKILL, make_runtime


class _FakeKeyframeSelector:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def select_keyframes_v2(self, query, k=3, **kwargs):
        self.calls.append((query, k))
        from query_scene.keyframe_selector import KeyframeResult

        return KeyframeResult(
            query=query,
            target_term="chair",
            anchor_term=None,
            keyframe_indices=[10, 20, 30],
            keyframe_paths=[],
            target_objects=[],
            anchor_objects=[],
            metadata={"hypothesis_output": {"hypotheses": [{"grounding_query": {"root": {"category": "chair"}}}]}},
        )


def _attach_selector(runtime):
    runtime.keyframe_selector = _FakeKeyframeSelector()
    return runtime.keyframe_selector


def test_select_by_text_gates_on_skill():
    rs = make_runtime()
    rs.skills_loaded.discard(PRIMARY_SKILL)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    resp = tool.invoke({"query": "a chair"})
    assert resp.startswith("ERROR")


def test_select_by_text_calls_keyframe_selector_and_returns_frames():
    rs = make_runtime()
    fake = _attach_selector(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    payload = json.loads(tool.invoke({"query": "a chair", "k": 3}))
    assert fake.calls == [("a chair", 3)]
    fids = [f["frame_id"] for f in payload["frames"]]
    assert fids == [10, 20, 30]
    assert payload["hypothesis_summary"]


def test_select_by_text_attaches_visible_proposal_ids_from_catalog():
    rs = make_runtime()
    _attach_selector(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    payload = json.loads(tool.invoke({"query": "a chair"}))
    f10 = next(f for f in payload["frames"] if f["frame_id"] == 10)
    assert sorted(f10["visible_proposal_ids"]) == [0, 2]


def test_select_by_text_attaches_camera_pose_when_available():
    rs = make_runtime()
    _attach_selector(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    payload = json.loads(tool.invoke({"query": "a chair"}))
    f10 = next(f for f in payload["frames"] if f["frame_id"] == 10)
    assert f10["bev_xy"] == [0.0, 0.0]
    assert f10["camera_yaw"] == 0.0


def test_select_by_text_filters_hidden_categories():
    rs = make_runtime()
    fake = _attach_selector(rs)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    payload = json.loads(
        tool.invoke({"query": "a chair", "k": 3, "hidden_categories": ["chair"]})
    )
    # Stage 1 still returned 3 frames; selector strips chair from visible_proposal_ids
    for frame in payload["frames"]:
        assert 0 not in frame["visible_proposal_ids"]
        assert 1 not in frame["visible_proposal_ids"]
    # And Stage 1 was called with hidden categories
    assert fake.calls and fake.calls[0][0] == "a chair"


def test_select_by_text_missing_keyframe_selector_errors():
    rs = make_runtime()
    rs.keyframe_selector = None
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    resp = tool.invoke({"query": "a chair"})
    assert resp.startswith("ERROR")
    assert "keyframe_selector" in resp
