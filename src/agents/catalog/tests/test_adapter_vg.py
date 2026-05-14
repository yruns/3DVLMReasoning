from agents.catalog import SceneCatalog
from agents.catalog.adapters import from_vg_proposal_pool


def _fixture_pool() -> dict:
    return {
        "source": "vdetr",
        "proposals": [
            {
                "id": 0,
                "bbox_3d_9dof": [0.0, 1.0, 0.5, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0],
                "category": "chair",
                "score": 0.92,
                "frame_views": [
                    {
                        "frame_id": 10,
                        "bbox_2d": [12, 20, 30, 60],
                        "raw_rgb_path": "raw/000010-rgb.png",
                        "visibility_weight": 0.83,
                    },
                ],
            },
            {
                "id": 1,
                "bbox_3d_9dof": [1.2, 1.4, 0.5, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0],
                "category": "table",
                "score": 0.88,
                "frame_views": [
                    {
                        "frame_id": 10,
                        "bbox_2d": [80, 30, 140, 80],
                        "raw_rgb_path": "raw/000010-rgb.png",
                    },
                    {
                        "frame_id": 20,
                        "bbox_2d": [30, 30, 70, 70],
                        "raw_rgb_path": "raw/000020-rgb.png",
                    },
                ],
            },
        ],
        "frame_index": {10: [0, 1], 20: [1]},
        "proposal_index": {0: [10], 1: [10, 20]},
        "annotated_image_dir": "annotated",
    }


def test_from_vg_proposal_pool_returns_catalog_with_required_fields():
    catalog = from_vg_proposal_pool(
        pool=_fixture_pool(),
        scene_id="scannet/scene0000_00",
        bev_image_path="bev/scene_bev_v9.png",
        scene_category="kitchen",
        axis_align_matrix=None,
        valid_frame_ids=[10, 20],
    )
    assert isinstance(catalog, SceneCatalog)
    assert catalog.scene_id == "scannet/scene0000_00"
    assert catalog.scene_category == "kitchen"
    assert catalog.total_frames == 2
    assert catalog.frame_id_range == (10, 20)
    assert catalog.valid_frame_ids == [10, 20]
    assert catalog.bev_image_path == "bev/scene_bev_v9.png"
    assert catalog.axis_align_matrix is None


def test_from_vg_proposal_pool_drops_score_and_keeps_position_from_9dof():
    catalog = from_vg_proposal_pool(
        pool=_fixture_pool(),
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        axis_align_matrix=None,
        valid_frame_ids=[10, 20],
    )
    chair = next(p for p in catalog.proposals if p.proposal_id == 0)
    assert chair.category == "chair"
    assert chair.position_3d == (0.0, 1.0, 0.5)
    assert chair.bbox_3d_9dof == (0.0, 1.0, 0.5, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0)
    assert chair.source == "vdetr"
    assert not hasattr(chair, "score")


def test_from_vg_proposal_pool_normalizes_frame_views():
    catalog = from_vg_proposal_pool(
        pool=_fixture_pool(),
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        axis_align_matrix=None,
        valid_frame_ids=[10, 20],
    )
    table = next(p for p in catalog.proposals if p.proposal_id == 1)
    assert set(table.frame_views.keys()) == {10, 20}
    v10 = table.frame_views[10]
    assert v10.bbox_2d == (80, 30, 140, 80)
    assert v10.raw_rgb_path == "raw/000010-rgb.png"
    assert v10.visibility_weight is None


def test_from_vg_proposal_pool_passes_through_axis_align_matrix():
    pool = _fixture_pool()
    axis = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    catalog = from_vg_proposal_pool(
        pool=pool,
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        axis_align_matrix=axis,
        valid_frame_ids=[10, 20],
    )
    assert catalog.axis_align_matrix == axis


def test_from_vg_proposal_pool_source_maps_through():
    pool = _fixture_pool()
    pool["source"] = "gt"
    catalog = from_vg_proposal_pool(
        pool=pool,
        scene_id="s",
        bev_image_path="b.png",
        scene_category=None,
        axis_align_matrix=None,
        valid_frame_ids=[10, 20],
    )
    assert all(p.source == "gt" for p in catalog.proposals)


def test_from_vg_proposal_pool_empty_valid_frames_errors():
    import pytest

    with pytest.raises(ValueError, match="valid_frame_ids"):
        from_vg_proposal_pool(
            pool=_fixture_pool(),
            scene_id="s",
            bev_image_path="b.png",
            scene_category=None,
            axis_align_matrix=None,
            valid_frame_ids=[],
        )
