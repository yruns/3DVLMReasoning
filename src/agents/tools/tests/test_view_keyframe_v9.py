from pathlib import Path

from PIL import Image

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.tools.view_keyframe import build_view_keyframe_tool

PRIMARY_SKILL = "scene-exploration-playbook"


def _runtime(tmp_path: Path, task_type: Stage2TaskType) -> Stage2RuntimeState:
    raw = tmp_path / "raw10.png"
    Image.new("RGB", (200, 200), (220, 220, 220)).save(raw)
    bev = tmp_path / "bev.png"
    Image.new("RGB", (100, 100), (10, 10, 10)).save(bev)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="chair",
                position_3d=(0, 0, 0),
                source="mask3d",
                frame_views={10: FrameView(frame_id=10, bbox_2d=(10, 20, 60, 80), raw_rgb_path=str(raw))},
            ),
            SceneProposal(
                proposal_id=1,
                category="table",
                position_3d=(1, 1, 0),
                source="mask3d",
                frame_views={10: FrameView(frame_id=10, bbox_2d=(100, 30, 180, 90), raw_rgb_path=str(raw))},
            ),
        ],
        total_frames=1,
        frame_id_range=(10, 10),
        valid_frame_ids=[10],
        bev_image_path=str(bev),
    )
    bundle = Stage2EvidenceBundle(extra_metadata={"scene_catalog": catalog.model_dump()})
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = task_type
    rs.skills_loaded.add(PRIMARY_SKILL)
    # Provide annotated dir path (used for unfiltered marked mode)
    annotated = tmp_path / "annotated"
    annotated.mkdir()
    Image.new("RGB", (200, 200), (200, 200, 200)).save(annotated / "frame_10.png")
    rs.bundle.extra_metadata = dict(rs.bundle.extra_metadata)
    rs.bundle.extra_metadata["annotated_image_dir"] = str(annotated)
    return rs


def test_view_keyframe_auto_vg_defaults_to_marked(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10})
    assert "marked image" in resp
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending[-1].endswith("frame_10.png")


def test_view_keyframe_auto_qa_defaults_to_rgb(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.QA)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10})
    assert "rgb image" in resp.lower()
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending[-1] == str(tmp_path / "raw10.png")


def test_view_keyframe_explicit_mode_rgb(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "rgb"})
    assert "rgb image" in resp.lower()
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending[-1] == str(tmp_path / "raw10.png")


def test_view_keyframe_explicit_mode_marked_categories_filter(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.QA)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "marked", "categories": ["chair"]})
    assert "filtered marked image" in resp
    assert "visible_proposals=[0]" in resp


def test_view_keyframe_explicit_mode_marked_proposal_ids_filter(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "marked", "proposal_ids": [1]})
    assert "filtered marked image" in resp
    assert "visible_proposals=[1]" in resp


def test_view_keyframe_marked_filter_no_match_errors(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "marked", "categories": ["lamp"]})
    assert resp.startswith("ERROR")
    assert "no visible proposals matched filters" in resp


def test_view_keyframe_unknown_frame_errors(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.QA)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 999})
    assert resp.startswith("ERROR")
    assert "frame_id=999" in resp


def test_view_keyframe_mode_rgb_ignores_filters(tmp_path: Path):
    """mode='rgb' is raw rgb; categories/proposal_ids are ignored (silent)."""
    rs = _runtime(tmp_path, Stage2TaskType.QA)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10, "mode": "rgb", "categories": ["chair"]})
    assert "rgb image" in resp.lower()
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending[-1] == str(tmp_path / "raw10.png")


def test_view_keyframe_gates_on_skill(tmp_path: Path):
    rs = _runtime(tmp_path, Stage2TaskType.VISUAL_GROUNDING)
    rs.skills_loaded.discard(PRIMARY_SKILL)
    tool = build_view_keyframe_tool(rs)
    resp = tool.invoke({"frame_id": 10})
    assert resp.startswith("ERROR")
    assert PRIMARY_SKILL in resp
