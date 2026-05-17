from pathlib import Path

import numpy as np
import pytest

from agents.catalog import SceneProposal
from query_scene.scene_bev_builder import (
    ScanNetSceneBEVBuilderBase,
    SceneBEVConfig,
)


class _DummyBuilder(ScanNetSceneBEVBuilderBase):
    benchmark = "dummy"

    def resolve_paths(self, scene_id, data_root):
        raise NotImplementedError


def test_overlay_proposal_labels_draws_id_and_category(tmp_path: Path):
    """When labels are requested via ``highlight_ids``, the overlay must
    draw the "#id category" text at each highlighted proposal centroid.

    v9.3: text labels are opt-in via ``highlight_ids``; the default
    ``highlight_ids=None`` path is exercised by
    ``test_default_no_text_labels_only_dots`` below.
    """
    builder = _DummyBuilder(config=SceneBEVConfig(image_size=400))
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    proposals = [
        SceneProposal(
            proposal_id=12,
            category="chair",
            position_3d=(0.5, 0.5, 0.0),
            source="mask3d",
        ),
        SceneProposal(
            proposal_id=31,
            category="desk",
            position_3d=(-0.5, -0.5, 0.0),
            source="mask3d",
        ),
    ]
    out_img = builder._overlay_proposal_labels(
        img, proposals, scene_bounds=(-1.0, -1.0, 1.0, 1.0),
        highlight_ids=[12, 31],
    )
    assert out_img.shape == (400, 400, 3)
    diff_count = int(np.any(out_img != img, axis=-1).sum())
    assert diff_count > 0


def test_default_no_text_labels_only_dots():
    """v9.3 contract: ``highlight_ids=None`` → small dot at each proposal,
    no text labels. The number of changed pixels stays small (a few dot
    discs) compared to the labelled path."""
    builder = _DummyBuilder(config=SceneBEVConfig(image_size=400))
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    proposals = [
        SceneProposal(proposal_id=1, category="chair", position_3d=(0.5, 0.5, 0.0), source="mask3d"),
        SceneProposal(proposal_id=2, category="chair", position_3d=(0.0, 0.0, 0.0), source="mask3d"),
        SceneProposal(proposal_id=3, category="chair", position_3d=(-0.5, -0.5, 0.0), source="mask3d"),
    ]
    out = builder._overlay_proposal_labels(img.copy(), proposals, (-1, -1, 1, 1), None)
    diff = int(np.any(out != img, axis=-1).sum())
    # 3 dots of radius 4 each ≈ 3 × π × 4² ≈ 150 pixels (anti-aliased ~180).
    # Labels would add several thousand additional pixels (text + bg rects).
    # We bound generously to allow rendering noise.
    assert 0 < diff < 600, (
        f"default view should draw only small dots; got {diff} changed pixels"
    )


def test_overlay_proposal_labels_highlight_subset_draws_more_than_default(tmp_path: Path):
    """v9.3 contract: with ``highlight_ids=[id]`` the overlay draws text +
    background panel for that proposal PLUS dots for every proposal.
    Therefore the highlighted-subset render writes more pixels than the
    default no-labels view.
    """
    builder = _DummyBuilder(config=SceneBEVConfig(image_size=400))
    # Off-white canvas so default white label panels register; pure white hides them.
    img = np.full((400, 400, 3), 240, dtype=np.uint8)
    proposals = [
        SceneProposal(proposal_id=1, category="chair", position_3d=(0.5, 0.5, 0.0), source="mask3d"),
        SceneProposal(proposal_id=2, category="chair", position_3d=(0.0, 0.0, 0.0), source="mask3d"),
        SceneProposal(proposal_id=3, category="chair", position_3d=(-0.5, -0.5, 0.0), source="mask3d"),
    ]
    default_view = builder._overlay_proposal_labels(
        img.copy(), proposals, (-1, -1, 1, 1), None
    )
    highlighted = builder._overlay_proposal_labels(
        img.copy(), proposals, (-1, -1, 1, 1), [2]
    )
    default_diff = int(np.any(default_view != img, axis=-1).sum())
    highlighted_diff = int(np.any(highlighted != img, axis=-1).sum())
    # Both have 3 dots; highlighted additionally has 1 label panel + text +
    # an enlarged red marker on proposal 2. So highlighted > default.
    assert 0 < default_diff < highlighted_diff, (
        f"highlighted view should add a label panel beyond the default dots; "
        f"default={default_diff}, highlighted={highlighted_diff}"
    )


def test_config_hash_is_deterministic():
    a = SceneBEVConfig(image_size=1200, perspective=True)
    b = SceneBEVConfig(image_size=1200, perspective=True)
    builder_a = _DummyBuilder(config=a)
    builder_b = _DummyBuilder(config=b)
    assert builder_a.config_hash() == builder_b.config_hash()


def test_build_with_labels_calls_resolve_paths(tmp_path: Path):
    class FakeBuilder(_DummyBuilder):
        def __init__(self):
            super().__init__(config=SceneBEVConfig(image_size=200))
            self.called = False

        def resolve_paths(self, scene_id, data_root):
            self.called = True
            return Path("missing.ply"), Path("missing_traj.txt"), Path("missing_intr.txt")

    builder = FakeBuilder()
    out = tmp_path / "bev.png"
    with pytest.raises(FileNotFoundError):
        builder.build_with_labels(
            scene_id="s",
            data_root=tmp_path,
            proposals=[],
            output_path=out,
        )
    assert builder.called is True


def test_render_hash_changes_with_proposals_and_highlights():
    builder = _DummyBuilder(config=SceneBEVConfig(image_size=400))
    p1 = [SceneProposal(proposal_id=1, category="chair", position_3d=(0, 0, 0), source="mask3d")]
    p2 = [SceneProposal(proposal_id=2, category="chair", position_3d=(0, 0, 0), source="mask3d")]
    h1 = builder.render_hash(p1, None)
    h2 = builder.render_hash(p2, None)
    h3 = builder.render_hash(p1, [1])
    assert h1 != h2  # different proposal id
    assert h1 != h3  # same proposals, different highlights
    assert h1 == builder.render_hash(p1, None)  # deterministic


def test_build_with_labels_cache_hit_skips_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """If a BEV with matching render_hash exists in <scene>/bev_cache/, the
    builder must NOT call _render_mesh_with_traj again. The cache slot is
    only considered complete when both the PNG and its ``.view.json``
    sidecar (perspective camera) exist; see Task 18."""
    from query_scene.scene_bev_builder import _view_params_path

    class CachingBuilder(_DummyBuilder):
        def __init__(self):
            super().__init__(config=SceneBEVConfig(image_size=200))
            self.render_calls = 0

        def resolve_paths(self, scene_id, data_root):
            # Won't be reached on cache hit.
            return Path(""), Path(""), Path("")

        def _render_mesh_with_traj(self, *a, **kw):  # pragma: no cover
            self.render_calls += 1
            raise AssertionError("should not be called on cache hit")

    builder = CachingBuilder()
    proposals = [
        SceneProposal(proposal_id=7, category="chair", position_3d=(0, 0, 0), source="mask3d"),
    ]
    cache_target = builder.cache_path(
        scene_id="scene_x",
        data_root=tmp_path,
        proposals=proposals,
        highlight_ids=None,
    )
    cache_target.parent.mkdir(parents=True, exist_ok=True)
    cache_target.write_bytes(b"\x89PNG\r\n\x1a\nCACHED")
    _view_params_path(cache_target).write_text(
        '{"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,1], "f": 1.0, '
        '"c": 100.0, "image_size": 200, "crop_offset": [0, 0]}'
    )

    out = tmp_path / "scene_x" / "bev_out.png"
    result = builder.build_with_labels(
        scene_id="scene_x",
        data_root=tmp_path,
        proposals=proposals,
        output_path=out,
    )
    assert result == out
    assert out.read_bytes() == b"\x89PNG\r\n\x1a\nCACHED"
    assert builder.render_calls == 0
    assert _view_params_path(out).exists(), "sidecar must also be copied to output_path"
    assert cache_target.name.startswith("scene_bev_")
    assert cache_target.name.endswith(".png")


def test_build_with_labels_cache_miss_writes_both(tmp_path: Path):
    from query_scene.scene_bev_builder import _view_params_path

    class TinyBuilder(_DummyBuilder):
        def __init__(self):
            super().__init__(config=SceneBEVConfig(image_size=200))

        def resolve_paths(self, scene_id, data_root):
            mesh = tmp_path / scene_id / "mesh.ply"
            traj = tmp_path / scene_id / "traj.txt"
            intr = tmp_path / scene_id / "intr.txt"
            for p in (mesh, traj, intr):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
            return mesh, traj, intr

        def _render_mesh_with_traj(self, *_a, **_kw):
            return (
                np.full((200, 200, 3), 128, dtype=np.uint8),
                (-1.0, -1.0, 1.0, 1.0),
            )

    builder = TinyBuilder()
    proposals = [
        SceneProposal(proposal_id=3, category="lamp", position_3d=(0.0, 0.0, 0.0), source="mask3d"),
    ]
    out = tmp_path / "scene_y_out.png"
    result = builder.build_with_labels(
        scene_id="scene_y",
        data_root=tmp_path,
        proposals=proposals,
        output_path=out,
    )
    assert result.exists()
    cache_target = builder.cache_path(
        scene_id="scene_y", data_root=tmp_path,
        proposals=proposals, highlight_ids=None,
    )
    assert cache_target.exists(), "cache slot must also be written on miss"
    assert _view_params_path(cache_target).exists(), (
        "view_params sidecar must accompany the cached PNG so highlight "
        "overlays can reuse the perspective camera (Task 18)"
    )
    assert _view_params_path(out).exists(), "sidecar must also sit next to output_path"
