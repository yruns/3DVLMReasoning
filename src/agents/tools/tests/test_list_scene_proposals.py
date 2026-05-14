import json

from agents.catalog import SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.tools.scene_perception import build_scene_perception_tools

PRIMARY_SKILL = "scene-exploration-playbook"


def _runtime() -> Stage2RuntimeState:
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(proposal_id=0, category="chair", position_3d=(0.1, 0.2, 0), source="mask3d"),
            SceneProposal(proposal_id=1, category="chair", position_3d=(1.5, 0.8, 0), source="mask3d"),
            SceneProposal(proposal_id=2, category="table", position_3d=(3.0, 4.0, 0), source="mask3d"),
            SceneProposal(proposal_id=3, category="lamp", position_3d=(-2.0, -1.0, 0), source="mask3d"),
        ],
        total_frames=10,
        frame_id_range=(0, 90),
        valid_frame_ids=[0, 10, 20, 30],
        bev_image_path="bev.png",
    )
    bundle = Stage2EvidenceBundle(extra_metadata={"scene_catalog": catalog.model_dump()})
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    return rs


def test_list_scene_proposals_gates_on_skill():
    rs = _runtime()
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    resp = tool.invoke({})
    assert resp.startswith("ERROR")
    assert PRIMARY_SKILL in resp


def test_list_scene_proposals_no_filter_returns_all():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    payload = json.loads(tool.invoke({}))
    assert payload["count"] == 4
    assert {p["proposal_id"] for p in payload["proposals"]} == {0, 1, 2, 3}


def test_list_scene_proposals_filters_by_category():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    payload = json.loads(tool.invoke({"category": "chair"}))
    assert payload["count"] == 2
    assert {p["proposal_id"] for p in payload["proposals"]} == {0, 1}


def test_list_scene_proposals_filters_by_region_bev():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    payload = json.loads(tool.invoke({"region_bev": [0, 0, 2, 2]}))
    assert {p["proposal_id"] for p in payload["proposals"]} == {0, 1}


def test_list_scene_proposals_respects_limit():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    payload = json.loads(tool.invoke({"limit": 2}))
    assert payload["count"] == 2


def test_list_scene_proposals_records_trace():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "list_scene_proposals")
    tool.invoke({"category": "chair"})
    names = [obs.tool_name for obs in rs.tool_trace]
    assert "list_scene_proposals" in names
