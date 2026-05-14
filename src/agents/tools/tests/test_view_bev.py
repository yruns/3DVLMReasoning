from pathlib import Path

import pytest

from agents.catalog import SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.tools.scene_perception import build_scene_perception_tools

PRIMARY_SKILL = "scene-exploration-playbook"


def _runtime(tmp_path: Path) -> Stage2RuntimeState:
    bev = tmp_path / "bev.png"
    from PIL import Image

    Image.new("RGB", (100, 80), (220, 220, 220)).save(bev)
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(proposal_id=0, category="chair", position_3d=(0, 0, 0), source="mask3d"),
            SceneProposal(proposal_id=1, category="table", position_3d=(1, 1, 0), source="mask3d"),
        ],
        total_frames=1,
        frame_id_range=(0, 0),
        valid_frame_ids=[0],
        bev_image_path=str(bev),
    )
    bundle = Stage2EvidenceBundle(extra_metadata={"scene_catalog": catalog.model_dump()})
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    rs.skills_loaded.add(PRIMARY_SKILL)
    return rs


def test_view_bev_default_returns_catalog_path(tmp_path: Path):
    rs = _runtime(tmp_path)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    response = tool.invoke({})
    assert "bev image" in response.lower()
    assert (tmp_path / "bev.png").as_posix() in response.replace("\\", "/")
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending == [str(tmp_path / "bev.png")]
    assert rs.evidence_updated is True


def test_view_bev_highlight_renders_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rs = _runtime(tmp_path)
    captured: dict = {}

    def _fake_render(catalog, highlight_ids, output_path):
        captured["highlight_ids"] = list(highlight_ids or [])
        from PIL import Image

        Image.new("RGB", (40, 40), (255, 0, 0)).save(output_path)
        return output_path

    monkeypatch.setattr(
        "agents.tools.scene_perception._render_highlighted_bev",
        _fake_render,
    )
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    resp = tool.invoke({"highlight": [1]})
    assert captured["highlight_ids"] == [1]
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    assert pending and pending[-1].endswith(".png")
    assert "highlight=[1]" in resp


def test_view_bev_gates_on_skill(tmp_path: Path):
    rs = _runtime(tmp_path)
    rs.skills_loaded.discard(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    resp = tool.invoke({})
    assert resp.startswith("ERROR")
