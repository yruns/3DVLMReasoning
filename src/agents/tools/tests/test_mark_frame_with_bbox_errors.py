from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.runtime.base import Stage2RuntimeState
from agents.tools.mark_frame_with_bbox import build_mark_frame_with_bbox_tool


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    bev_path = tmp_path / "bev.png"
    bev_path.write_bytes(b"")
    catalog = SceneCatalog(
        scene_id="s_test",
        proposals=[
            SceneProposal(
                proposal_id=4,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={
                    42: FrameView(
                        frame_id=42,
                        raw_rgb_path=str(tmp_path / "frame_42.png"),
                        bbox_2d=(10, 10, 100, 100),
                    )
                },
            ),
        ],
        total_frames=100,
        frame_id_range=(0, 99),
        valid_frame_ids=[42],
        bev_image_path=str(bev_path),
    )
    bundle = SimpleNamespace(
        extra_metadata={
            "scene_catalog": catalog.model_dump(),
        }
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.seen_image_paths = set()
    rs.skills_loaded = {"scene-exploration-playbook"}
    return rs


def test_mark_frame_with_bbox_errors_when_no_filter(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42})
    assert out.startswith(
        "ERROR: mark_frame_with_bbox requires at least one of {labels, ids}"
    )


def test_mark_frame_with_bbox_errors_on_empty_lists(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "labels": [], "ids": []})
    assert out.startswith(
        "ERROR: mark_frame_with_bbox requires at least one of {labels, ids}"
    )


def test_mark_frame_with_bbox_errors_on_invalid_frame(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 999, "ids": [4]})
    assert out.startswith("ERROR: frame_id=999 not in valid_frame_ids")


def test_mark_frame_with_bbox_errors_when_filter_matches_nothing(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = build_mark_frame_with_bbox_tool(rs)
    out = tool.invoke({"frame_id": 42, "ids": [999]})
    assert "no visible proposals matched filters" in out
