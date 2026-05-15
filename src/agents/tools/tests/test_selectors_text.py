from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.selectors import build_selector_tools


class _FakeKeyframeSelector:
    def __init__(self, fids: list[int]) -> None:
        self._fids = fids

    def select_keyframes_v2(self, **_kwargs):
        return SimpleNamespace(
            keyframe_indices=list(self._fids),
            metadata={"hypothesis_output": {"hypotheses": [
                {"grounding_query": {"root": {"category": "chair"}}, "kind": "direct"}
            ]}},
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
                frame_views={fid: FrameView(frame_id=fid, raw_rgb_path=str(rgb), bbox_2d=(0, 0, 20, 20))},
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
    rs.keyframe_selector = _FakeKeyframeSelector(fids)
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
