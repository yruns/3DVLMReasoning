import pytest

from agents.catalog import FrameView, SceneCatalog, SceneProposal
from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState
from agents.runtime.scene_runtime import (
    get_scene_catalog,
    make_tool_image_ref,
)


def _catalog() -> SceneCatalog:
    return SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=0,
                category="chair",
                position_3d=(0, 0, 0),
                source="mask3d",
                frame_views={
                    10: FrameView(
                        frame_id=10, bbox_2d=(0, 0, 10, 10), raw_rgb_path="r.png"
                    )
                },
            )
        ],
        total_frames=1,
        frame_id_range=(10, 10),
        valid_frame_ids=[10],
        bev_image_path="bev.png",
    )


def test_get_scene_catalog_hydrates_from_extra_metadata():
    bundle = Stage2EvidenceBundle(
        extra_metadata={"scene_catalog": _catalog().model_dump()}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    catalog = get_scene_catalog(rs)
    assert isinstance(catalog, SceneCatalog)
    assert catalog.scene_id == "s"
    assert catalog.proposals[0].proposal_id == 0


def test_get_scene_catalog_caches_on_runtime():
    bundle = Stage2EvidenceBundle(
        extra_metadata={"scene_catalog": _catalog().model_dump()}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    a = get_scene_catalog(rs)
    b = get_scene_catalog(rs)
    assert a is b


def test_get_scene_catalog_missing_errors():
    bundle = Stage2EvidenceBundle(extra_metadata={})
    rs = Stage2RuntimeState(bundle=bundle)
    with pytest.raises(ValueError, match="scene_catalog"):
        get_scene_catalog(rs)


def test_make_tool_image_ref_returns_metadata_without_runtime_queue(tmp_path):
    bundle = Stage2EvidenceBundle(
        extra_metadata={"scene_catalog": _catalog().model_dump()}
    )
    rs = Stage2RuntimeState(bundle=bundle)
    first = make_tool_image_ref(rs, str(tmp_path / "x.png"))
    second = make_tool_image_ref(rs, str(tmp_path / "y.png"))
    assert first == {"image_path": str(tmp_path / "x.png")}
    assert second == {"image_path": str(tmp_path / "y.png")}
    assert not any("pending" in name and "image" in name for name in dir(rs))
    assert rs.evidence_updated is False
