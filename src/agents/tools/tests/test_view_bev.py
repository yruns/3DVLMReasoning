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
    """v9.3: default ``view_bev()`` returns the catalog's pre-rendered base BEV
    (clean overview, no labels)."""
    rs = _runtime(tmp_path)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    response = tool.invoke({})
    assert "bev image" in response.lower()
    assert (tmp_path / "bev.png").as_posix() in response.replace("\\", "/")
    # v9.3: response advertises the no-text default; previous versions used
    # "highlight=ALL" wording that conflated default with "label everything".
    assert "no text labels" in response or "default view" in response
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


def test_view_bev_categories_resolves_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """v9.3: passing ``categories=["chair"]`` resolves to the matching
    proposal_ids in the catalog and renders the same focused-label BEV.
    Mirrors how `mark_frame_with_bbox` adds focused annotations to a frame.
    """
    rs = _runtime(tmp_path)
    captured: dict = {}

    def _fake_render(catalog, highlight_ids, output_path):
        captured["highlight_ids"] = list(highlight_ids or [])
        from PIL import Image

        Image.new("RGB", (40, 40), (0, 255, 0)).save(output_path)
        return output_path

    monkeypatch.setattr(
        "agents.tools.scene_perception._render_highlighted_bev",
        _fake_render,
    )
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    # Two proposals exist in the catalog: chair (#0), table (#1).
    resp = tool.invoke({"categories": ["chair"]})
    assert captured["highlight_ids"] == [0], (
        "categories=['chair'] should resolve to proposal_id=0 only"
    )
    assert "resolved_from_categories=['chair']" in resp


def test_view_bev_categories_union_with_highlight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both ``highlight`` and ``categories`` can be supplied; the result is
    the union of the explicit IDs and the category-resolved IDs.
    """
    rs = _runtime(tmp_path)
    captured: dict = {}

    def _fake_render(catalog, highlight_ids, output_path):
        captured["highlight_ids"] = list(highlight_ids or [])
        from PIL import Image

        Image.new("RGB", (40, 40), (0, 0, 255)).save(output_path)
        return output_path

    monkeypatch.setattr(
        "agents.tools.scene_perception._render_highlighted_bev",
        _fake_render,
    )
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    resp = tool.invoke({"highlight": [1], "categories": ["chair"]})
    # chair=#0, plus explicit #1: union sorted = [0, 1]
    assert captured["highlight_ids"] == [0, 1]
    assert "highlight=[0, 1]" in resp


def test_view_bev_categories_with_unknown_category_warns(tmp_path: Path):
    """Unknown category names are surfaced to the agent as
    ``categories_with_no_matches`` instead of silently producing an empty
    highlight set. With NO matches and no explicit highlight, the tool
    returns the default BEV plus an inline note.
    """
    rs = _runtime(tmp_path)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    resp = tool.invoke({"categories": ["unicorn"]})
    assert "no matches" in resp.lower()
    assert "unicorn" in resp
    pending = rs.bundle.extra_metadata["vg_pending_images"]
    # Returns the base BEV path because nothing was actually highlighted.
    assert pending == [str(tmp_path / "bev.png")]


def test_view_bev_categories_case_insensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Category match is case-insensitive on stripped strings."""
    rs = _runtime(tmp_path)
    captured: dict = {}

    def _fake_render(catalog, highlight_ids, output_path):
        captured["highlight_ids"] = list(highlight_ids or [])
        from PIL import Image

        Image.new("RGB", (40, 40), (255, 255, 0)).save(output_path)
        return output_path

    monkeypatch.setattr(
        "agents.tools.scene_perception._render_highlighted_bev",
        _fake_render,
    )
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    tool.invoke({"categories": ["  CHAIR  "]})
    assert captured["highlight_ids"] == [0]


def test_view_bev_gates_on_skill(tmp_path: Path):
    rs = _runtime(tmp_path)
    rs.skills_loaded.discard(PRIMARY_SKILL)
    tool = next(t for t in build_scene_perception_tools(rs) if t.name == "view_bev")
    resp = tool.invoke({})
    assert resp.startswith("ERROR")


def test_highlight_label_aligns_with_base_bev_via_view_params(tmp_path: Path):
    """Both the base BEV and the highlight BEV must project labels through the
    same perspective camera. Pre-Task-18 the highlight path used
    ``_project_centroid``'s linear-bounds fallback and landed labels tens of
    pixels off the actual proposal. Task 18 routes highlight through the
    persisted ``view_params`` sidecar so the projections agree pixel-for-pixel.
    """
    import json

    import numpy as np
    from PIL import Image

    from agents.catalog import SceneCatalog, SceneProposal
    from agents.tools.scene_perception import _render_highlighted_bev

    bev_path = tmp_path / "scene_bev.png"
    Image.new("RGB", (800, 800), (200, 200, 200)).save(bev_path)
    view = {
        "R": np.eye(3).tolist(),
        "t": [0.0, 0.0, 10.0],
        "f": 800.0,
        "c": 400.0,
        "image_size": 800,
        "crop_offset": [0, 0],
    }
    bev_path.with_suffix(".view.json").write_text(json.dumps(view))

    f, c = 800.0, 400.0
    cam = np.array([1.0, 1.0, 10.0])
    u_expected = int(f * cam[0] / cam[2] + c)
    v_expected = int(f * cam[1] / cam[2] + c)

    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=1,
                category="chair",
                position_3d=(1.0, 1.0, 0.0),
                source="mask3d",
                frame_views={},
            ),
        ],
        total_frames=100,
        frame_id_range=(0, 99),
        valid_frame_ids=[],
        bev_image_path=str(bev_path),
    )

    highlight_path = tmp_path / "highlight.png"
    _render_highlighted_bev(catalog, [1], highlight_path)

    img = np.asarray(Image.open(highlight_path).convert("RGB"))
    patch = img[
        max(0, v_expected - 10) : v_expected + 10,
        max(0, u_expected - 10) : u_expected + 10,
    ]
    diff_from_grey = np.abs(patch.astype(int) - 200).sum(axis=-1)
    assert (diff_from_grey > 30).any(), (
        f"expected a marker near projected (u, v) = "
        f"({u_expected}, {v_expected})"
    )


def test_highlight_render_fails_when_view_params_sidecar_missing(tmp_path: Path):
    """Missing sidecar -> hard error (no silent fallback to linear bounds)."""
    from PIL import Image

    from agents.catalog import SceneCatalog, SceneProposal
    from agents.tools.scene_perception import _render_highlighted_bev

    bev_path = tmp_path / "scene_bev.png"
    Image.new("RGB", (400, 400), (200, 200, 200)).save(bev_path)

    catalog = SceneCatalog(
        scene_id="s",
        proposals=[
            SceneProposal(
                proposal_id=1,
                category="chair",
                position_3d=(0.0, 0.0, 0.0),
                source="mask3d",
                frame_views={},
            ),
        ],
        total_frames=1,
        frame_id_range=(0, 0),
        valid_frame_ids=[0],
        bev_image_path=str(bev_path),
    )

    with pytest.raises(FileNotFoundError, match="view_params sidecar"):
        _render_highlighted_bev(catalog, [1], tmp_path / "highlight.png")
