from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.selectors import build_selector_tools


class _FakeTextFrameSelector:
    def __init__(self, fids: list[int]) -> None:
        self._fids = fids

    def select_keyframes_v2(self, **_kwargs):
        return SimpleNamespace(
            keyframe_indices=list(self._fids),
            metadata={
                "hypothesis_output": {
                    "hypotheses": [
                        {
                            "grounding_query": {"root": {"category": "chair"}},
                            "kind": "direct",
                        }
                    ]
                }
            },
        )


class _LeakyThenOkTextFrameSelector:
    def __init__(self, fids: list[int]) -> None:
        self.calls: list[dict] = []
        self._fids = list(fids)

    def select_keyframes_v2(self, **kwargs):
        self.calls.append(dict(kwargs))
        hidden = list(kwargs.get("hidden_categories") or [])
        if hidden:
            raise ValueError("Masked category leak detected: 'wall'")
        return SimpleNamespace(
            keyframe_indices=list(self._fids),
            metadata={
                "hypothesis_output": {
                    "hypotheses": [
                        {
                            "grounding_query": {"root": {"category": "picture"}},
                            "kind": "direct",
                        }
                    ]
                }
            },
        )


def _runtime(tmp_path: Path, fids: list[int]) -> Stage2RuntimeState:
    bev_path = tmp_path / "bev.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(bev_path)
    proposals: list[SceneProposal] = []
    for idx, fid in enumerate(fids):
        rgb = tmp_path / f"frame_{fid}.png"
        Image.new("RGB", (320, 240), (200, 200, 200)).save(rgb)
        proposals.append(
            SceneProposal(
                proposal_id=10 + idx,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    fid: FrameView(
                        frame_id=fid, raw_rgb_path=str(rgb), bbox_2d=(0, 0, 20, 20)
                    )
                },
            )
        )
    catalog = SceneCatalog(
        scene_id="s",
        proposals=proposals,
        total_frames=max(fids) + 1,
        frame_id_range=(0, max(fids)),
        valid_frame_ids=list(fids),
        bev_image_path=str(bev_path),
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
            "vg_pending_images": [],
            "camera_trajectory_xy_yaw": {f: [float(f), float(f), 0.0] for f in fids},
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    rs.text_frame_selector = _FakeTextFrameSelector(fids)
    return rs


def test_select_by_text_returns_image_paths_and_queues_them(tmp_path: Path):
    rs = _runtime(tmp_path, fids=[1, 2, 3])
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    raw = tool.invoke({"query": "wooden chair"})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    for frame, fid in zip(payload["frames"], [1, 2, 3]):
        assert frame["frame_id"] == fid
        assert "image_path" in frame
        assert frame["already_seen"] is False
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert len(pending) == 3


def test_select_by_text_caps_k_at_3(tmp_path: Path):
    rs = _runtime(tmp_path, fids=[1, 2, 3, 4, 5])
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    raw = tool.invoke({"query": "chair", "k": 5})
    payload = json.loads(raw)
    assert len(payload["frames"]) == 3
    assert "k capped at 3" in raw.lower()


def test_select_by_text_marks_already_seen(tmp_path: Path):
    rs = _runtime(tmp_path, fids=[1, 2])
    seen_path = str(tmp_path / "frame_1.png")
    rs.seen_image_paths.add(seen_path)
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    raw = tool.invoke({"query": "chair"})
    payload = json.loads(raw)
    by_fid = {f["frame_id"]: f for f in payload["frames"]}
    assert by_fid[1]["already_seen"] is True
    assert by_fid[2]["already_seen"] is False
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending == [str(tmp_path / "frame_2.png")]


def test_select_by_text_omitted_when_flag_disabled(tmp_path: Path):
    """v9.2 toggle: when runtime.enable_stage1_text_retrieval is False,
    `select_by_text` must NOT appear in the registered tool set; the
    catalog-driven selectors remain so the agent can still ground."""
    rs = _runtime(tmp_path, fids=[1, 2, 3])
    rs.enable_stage1_text_retrieval = False
    tools = build_selector_tools(rs)
    names = {t.name for t in tools}
    assert "select_by_text" not in names
    assert {
        "select_by_proposal",
        "select_by_frame_neighbor",
        "select_by_region",
        "select_by_coverage",
    }.issubset(names)


def test_select_by_text_included_when_flag_enabled_default(tmp_path: Path):
    """The flag defaults to True; the tool must remain in the set unless
    explicitly disabled. This pins the default in case the field default
    is ever flipped."""
    rs = _runtime(tmp_path, fids=[1, 2, 3])
    assert rs.enable_stage1_text_retrieval is True  # default
    tools = build_selector_tools(rs)
    assert any(t.name == "select_by_text" for t in tools)


def test_select_by_text_force_to_error_short_circuits_before_selector(
    tmp_path: Path,
):
    """v9.4 cadence experiment: when
    `force_stage1_text_retrieval_to_error=True`, the tool must
    short-circuit to an explicit ERROR string BEFORE invoking the
    text-frame selector. This cleanly reproduces the v9.1_fix bug-state
    behaviour while keeping the text-first playbook and system prompt
    consistent. See
    docs/benchmark/nr3d/v9_1_fix_vs_v9_3_audit30_20260517.md §Experiment A.

    Contract:
      - Tool is still REGISTERED (so the text-first playbook prose
        about `select_by_text` matches the tool surface).
      - Tool body returns ERROR for every call.
      - `select_keyframes_v2` is NOT invoked (would be
        wasted compute and would muddy the deliberation cadence we
        want to force).
      - `runtime.record('select_by_text', ...)` is called exactly once
        per invocation with the ERROR string (so per-sample
        `tool_trace` accurately reflects the call).
    """
    rs = _runtime(tmp_path, fids=[1, 2, 3])
    rs.force_stage1_text_retrieval_to_error = True

    # Replace the fake selector with one that explodes if invoked. This
    # makes "selector was not called" verifiable.
    class _ExplodingSelector:
        def select_keyframes_v2(self, **_kwargs):
            raise AssertionError(
                "select_by_text must short-circuit to ERROR before "
                "calling the text-frame selector when "
                "force_stage1_text_retrieval_to_error=True"
            )

    rs.text_frame_selector = _ExplodingSelector()

    tools = build_selector_tools(rs)
    # Still registered (text-first contract preserved).
    assert any(t.name == "select_by_text" for t in tools)
    tool = next(t for t in tools if t.name == "select_by_text")

    raw = tool.invoke({"query": "wooden chair"})
    assert raw.startswith("ERROR:")
    assert "force-disabled" in raw or "force_stage1_text_retrieval_to_error" in raw
    # Fallback chain prose is included so the agent has a recovery hint.
    assert "scene-exploration-playbook" in raw

    # No frames were queued (pending images list is unchanged).
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending == []


def test_select_by_text_force_to_error_records_into_tool_trace(tmp_path: Path):
    """The forced-ERROR call must still be recorded into the per-sample
    tool_trace so audits can verify the cadence anchor. This is what
    distinguishes "select_by_text never called" from "select_by_text
    called once and got ERROR" in per-sample analyses.
    """
    rs = _runtime(tmp_path, fids=[1, 2, 3])
    rs.force_stage1_text_retrieval_to_error = True

    recorded: list[tuple[str, dict, str]] = []

    def _record(name: str, request: dict, response: str) -> None:
        recorded.append((name, request, response))

    rs.record = _record  # type: ignore[method-assign]

    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    tool.invoke({"query": "anywhere", "k": 2})

    assert len(recorded) == 1
    name, request, response = recorded[0]
    assert name == "select_by_text"
    assert request["query"] == "anywhere"
    assert request["k"] == 2
    assert response.startswith("ERROR:")


def test_select_by_text_force_to_error_default_is_off(tmp_path: Path):
    """The force flag defaults to False. Pin this so a future field-default
    flip doesn't silently turn every NR3D run into the v9.4-A experiment.
    """
    rs = _runtime(tmp_path, fids=[1])
    assert rs.force_stage1_text_retrieval_to_error is False


def test_select_by_text_auto_retries_masked_category_leak_with_empty_mask(
    tmp_path: Path,
) -> None:
    rs = _runtime(tmp_path, fids=[43, 49, 50])
    selector = _LeakyThenOkTextFrameSelector([43, 49, 50])
    rs.text_frame_selector = selector
    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")

    raw = tool.invoke(
        {
            "query": "The largest picture in the room.",
            "hidden_categories": ["wall", "floor"],
        }
    )

    payload = json.loads(raw)
    assert payload["masked_category_retry"]["retried_with_hidden_categories"] == []
    assert "Masked category leak detected" in payload["masked_category_retry"]["error"]
    assert [call["hidden_categories"] for call in selector.calls] == [
        ["wall", "floor"],
        [],
    ]
    assert [call["query"] for call in selector.calls] == [
        "The largest picture in the room.",
        "The largest picture in the room.",
    ]
    assert [frame["frame_id"] for frame in payload["frames"]] == [43, 49, 50]


def test_select_by_text_masked_retry_records_original_request(tmp_path: Path) -> None:
    rs = _runtime(tmp_path, fids=[1])
    selector = _LeakyThenOkTextFrameSelector([1])
    rs.text_frame_selector = selector
    recorded: list[tuple[str, dict, str]] = []
    rs.record = lambda name, request, response: recorded.append(  # type: ignore[method-assign]
        (name, request, response)
    )

    tool = next(t for t in build_selector_tools(rs) if t.name == "select_by_text")
    raw = tool.invoke({"query": "picture on wall", "hidden_categories": ["wall"]})

    assert len(recorded) == 1
    assert recorded[0][0] == "select_by_text"
    assert recorded[0][1]["hidden_categories"] == ["wall"]
    payload = json.loads(raw)
    assert payload["masked_category_retry"]["retried_with_hidden_categories"] == []
