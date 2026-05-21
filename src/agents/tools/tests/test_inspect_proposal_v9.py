import json

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.tools.scene_perception import build_scene_perception_tools

PRIMARY_SKILL = "scene-exploration-playbook"


def _runtime() -> Stage2RuntimeState:
    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=7,
                category="sofa",
                enriched_category="sectional sofa",
                compact_note="gray fabric sectional used for seating",
                enrichment={
                    "category": "sectional sofa",
                    "description": "A gray fabric sectional sofa with cushions.",
                    "location": "Against the living-room wall.",
                    "nearby_objects": ["pillow", "coffee table"],
                    "color": "gray",
                    "usability": "Provides seating for several people.",
                },
                position_3d=(1.0, 1.0, 0.3),
                bbox_3d_9dof=(1.0, 1.0, 0.3, 1.5, 0.8, 0.5, 0, 0, 0),
                frame_views={
                    20: FrameView(
                        frame_id=20, bbox_2d=(0, 0, 10, 10), raw_rgb_path="a.png"
                    ),
                    25: FrameView(
                        frame_id=25, bbox_2d=(5, 5, 20, 20), raw_rgb_path="b.png"
                    ),
                },
                source="vdetr",
            )
        ],
        total_frames=10,
        frame_id_range=(0, 90),
        valid_frame_ids=[0, 20, 25],
        bev_image_path="bev.png",
    )
    bundle = Stage2EvidenceBundle(
        extra_metadata={"scene_catalog": catalog.model_dump()}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    rs.task_type = Stage2TaskType.VISUAL_GROUNDING
    return rs


def test_inspect_proposal_returns_expected_fields():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(
        t for t in build_scene_perception_tools(rs) if t.name == "inspect_proposal"
    )
    payload = json.loads(tool.invoke({"proposal_id": 7}))
    assert payload["proposal_id"] == 7
    assert payload["category"] == "sofa"
    assert payload["position_3d"] == [1.0, 1.0, 0.3]
    assert payload["bbox_3d_9dof"][:3] == [1.0, 1.0, 0.3]
    assert sorted(payload["frames_appeared"]) == [20, 25]
    assert payload["source"] == "vdetr"
    assert payload["enriched_category"] == "sectional sofa"
    assert payload["compact_note"] == "gray fabric sectional used for seating"
    assert (
        payload["enrichment"]["description"]
        == "A gray fabric sectional sofa with cushions."
    )
    assert payload["enrichment"]["nearby_objects"] == ["pillow", "coffee table"]


def test_inspect_proposal_unknown_id_errors():
    rs = _runtime()
    rs.skills_loaded.add(PRIMARY_SKILL)
    tool = next(
        t for t in build_scene_perception_tools(rs) if t.name == "inspect_proposal"
    )
    resp = tool.invoke({"proposal_id": 999})
    assert resp.startswith("ERROR")
    assert "999" in resp
