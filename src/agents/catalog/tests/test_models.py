from agents.catalog.models import FrameView, SceneCatalog, SceneProposal


def test_frame_view_minimal_required_fields():
    fv = FrameView(frame_id=10, bbox_2d=(1, 2, 3, 4), raw_rgb_path="a.png")
    assert fv.frame_id == 10
    assert fv.bbox_2d == (1, 2, 3, 4)
    assert fv.raw_rgb_path == "a.png"
    assert fv.visibility_weight is None


def test_frame_view_with_visibility_weight():
    fv = FrameView(
        frame_id=11,
        bbox_2d=(0, 0, 100, 200),
        raw_rgb_path="b.png",
        visibility_weight=0.42,
    )
    assert fv.visibility_weight == 0.42


def test_scene_proposal_default_frame_views_empty():
    p = SceneProposal(
        proposal_id=7,
        category="chair",
        position_3d=(1.0, 2.0, 0.5),
        source="mask3d",
    )
    assert p.frame_views == {}
    assert p.bbox_3d_9dof is None
    assert p.source == "mask3d"


def test_scene_proposal_with_views_and_9dof():
    fv = FrameView(frame_id=20, bbox_2d=(0, 0, 50, 50), raw_rgb_path="c.png")
    p = SceneProposal(
        proposal_id=3,
        category="table",
        position_3d=(0.0, 0.0, 0.0),
        bbox_3d_9dof=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        frame_views={20: fv},
        source="gt",
    )
    assert p.frame_views[20].frame_id == 20
    assert len(p.bbox_3d_9dof) == 9


def test_scene_proposal_keeps_enrichment_fields():
    p = SceneProposal(
        proposal_id=25,
        category="cabinet",
        enriched_category="mini-fridge/cabinet",
        compact_note=(
            "small white cube covered by yellow-pattern cloth, near door/radiator; "
            "top used as storage surface"
        ),
        enrichment={
            "category": "mini-fridge",
            "description": "A small white cube-shaped mini-fridge.",
            "location": "Near the door and radiator.",
            "nearby_objects": ["door", "radiator"],
            "color": "white",
            "usability": "Used for refrigerating items and as a storage surface.",
        },
        position_3d=(1.0, 2.0, 0.5),
        source="gt",
    )

    reloaded = SceneProposal(**p.model_dump())
    assert reloaded.category == "cabinet"
    assert reloaded.enriched_category == "mini-fridge/cabinet"
    assert reloaded.compact_note.startswith("small white cube")
    assert reloaded.enrichment["nearby_objects"] == ["door", "radiator"]


def test_scene_catalog_roundtrip():
    proposals = [
        SceneProposal(
            proposal_id=0,
            category="chair",
            position_3d=(0.0, 0.0, 0.0),
            source="mask3d",
        ),
        SceneProposal(
            proposal_id=1,
            category="chair",
            position_3d=(1.0, 0.0, 0.0),
            source="mask3d",
        ),
        SceneProposal(
            proposal_id=2,
            category="table",
            position_3d=(2.0, 0.0, 0.0),
            source="mask3d",
        ),
    ]
    catalog = SceneCatalog(
        scene_id="scannet/scene0000_00",
        scene_category="kitchen",
        proposals=proposals,
        total_frames=100,
        frame_id_range=(0, 990),
        valid_frame_ids=[0, 10, 20, 30],
        bev_image_path="bev/scene_bev_v9.png",
    )
    dumped = catalog.model_dump()
    reloaded = SceneCatalog(**dumped)
    assert reloaded.scene_id == "scannet/scene0000_00"
    assert reloaded.scene_category == "kitchen"
    assert reloaded.total_frames == 100
    assert reloaded.frame_id_range == (0, 990)
    assert reloaded.valid_frame_ids == [0, 10, 20, 30]
    assert reloaded.bev_image_path == "bev/scene_bev_v9.png"
    assert len(reloaded.proposals) == 3


def test_scene_catalog_proposals_by_category_groups_ids():
    proposals = [
        SceneProposal(
            proposal_id=0, category="chair", position_3d=(0, 0, 0), source="mask3d"
        ),
        SceneProposal(
            proposal_id=1, category="chair", position_3d=(1, 0, 0), source="mask3d"
        ),
        SceneProposal(
            proposal_id=2, category="table", position_3d=(2, 0, 0), source="mask3d"
        ),
    ]
    catalog = SceneCatalog(
        scene_id="s",
        proposals=proposals,
        total_frames=10,
        frame_id_range=(0, 90),
        valid_frame_ids=[0],
        bev_image_path="x.png",
    )
    grouped = catalog.proposals_by_category()
    assert grouped == {"chair": [0, 1], "table": [2]}


def test_scene_proposal_rejects_unknown_source():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SceneProposal(
            proposal_id=0,
            category="x",
            position_3d=(0, 0, 0),
            source="foo",  # not in Literal
        )
