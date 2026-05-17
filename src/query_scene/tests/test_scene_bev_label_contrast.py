import numpy as np

from agents.catalog import SceneProposal
from query_scene.scene_bev_builder import ScanNetSceneBEVBuilderBase, SceneBEVConfig


class _Stub(ScanNetSceneBEVBuilderBase):
    def resolve_paths(self, scene_id, data_root):
        raise NotImplementedError


def _proposals() -> list[SceneProposal]:
    return [
        SceneProposal(
            proposal_id=1,
            category="chair",
            position_3d=(0.0, 0.0, 0.0),
            source="mask3d",
            frame_views={},
        ),
        SceneProposal(
            proposal_id=2,
            category="table",
            position_3d=(3.0, 3.0, 0.0),
            source="mask3d",
            frame_views={},
        ),
    ]


def test_label_has_black_outline_around_white_text():
    """When labels are drawn, they must use a sharp black-outline-on-white-fill
    rendering (`putText` outline-then-core technique). v9.3 contract: text is
    drawn only when proposal_id is in `highlight_ids`.
    """
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    builder = _Stub(config=SceneBEVConfig(image_size=400))
    bounds = (0.0, 0.0, 10.0, 10.0)
    out = builder._overlay_proposal_labels(
        img, _proposals(), bounds, highlight_ids=[1]
    )
    is_black = out.max(axis=-1) <= 32
    is_white = out.min(axis=-1) >= 248
    adj_right = is_black[:, :-1] & is_white[:, 1:]
    adj_left = is_white[:, :-1] & is_black[:, 1:]
    assert adj_right.any() or adj_left.any(), (
        "expected black/white adjacency from text outline on a highlighted "
        "proposal"
    )


def test_default_highlight_none_draws_no_text_labels():
    """v9.3 contract: ``highlight_ids=None`` → mesh + dots only, no labels.

    Regression test for the cluttered-BEV bug where the default view labelled
    every proposal in the catalog, making 40+ proposal scenes unreadable. The
    fix: text labels are opt-in via ``highlight_ids=[...]``; the default view
    keeps dots so the agent can still see WHERE things are.
    """
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    builder = _Stub(config=SceneBEVConfig(image_size=400))
    bounds = (0.0, 0.0, 10.0, 10.0)
    out = builder._overlay_proposal_labels(
        img, _proposals(), bounds, highlight_ids=None
    )
    # No black text means the global "max black, min white" adjacency from
    # the outline-on-fill renderer should NOT appear: pixel patches above
    # 248 are the white label backgrounds, and they shouldn't exist either.
    is_white_bg = out.min(axis=-1) >= 248
    assert is_white_bg.sum() <= 2, (
        "default view (highlight_ids=None) must not draw white label "
        f"backgrounds; got {int(is_white_bg.sum())} white pixels in a "
        "starting grey-200 canvas"
    )


def test_legacy_label_all_when_highlight_is_none_restores_old_behavior():
    """Backward-compat escape hatch for historical traces. Setting
    ``label_all_when_highlight_is_none=True`` makes ``highlight_ids=None``
    label every proposal again (the pre-v9.3 default).
    """
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    builder = _Stub(
        config=SceneBEVConfig(
            image_size=400,
            label_all_when_highlight_is_none=True,
        )
    )
    bounds = (0.0, 0.0, 10.0, 10.0)
    out = builder._overlay_proposal_labels(
        img, _proposals(), bounds, highlight_ids=None
    )
    is_white_bg = out.min(axis=-1) >= 248
    assert is_white_bg.sum() > 100, (
        "legacy flag must restore label-every-proposal behaviour"
    )


def test_highlighted_proposal_has_enlarged_red_marker():
    """Highlighted proposals get a bigger red dot than non-highlighted ones,
    so the agent's eye is drawn to the requested subset even before reading
    the text label.
    """
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    builder = _Stub(config=SceneBEVConfig(image_size=400))
    bounds = (0.0, 0.0, 10.0, 10.0)
    out = builder._overlay_proposal_labels(
        img, _proposals(), bounds, highlight_ids=[1]
    )
    # Red-dominant pixels (label_color_highlight = 255,64,64) come only from
    # the highlighted marker. We don't compare exact counts because OpenCV
    # rendering produces blended anti-aliased edges; the presence of *any*
    # such pixel is sufficient.
    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    is_red = (r > 200) & (g < 100) & (b < 100)
    assert is_red.any(), "expected a red highlight marker for proposal_id=1"
