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
        img, proposals, scene_bounds=(-1.0, -1.0, 1.0, 1.0), highlight_ids=None
    )
    assert out_img.shape == (400, 400, 3)
    diff_count = int(np.any(out_img != img, axis=-1).sum())
    assert diff_count > 0


def test_overlay_proposal_labels_highlight_subset_draws_fewer(tmp_path: Path):
    builder = _DummyBuilder(config=SceneBEVConfig(image_size=400))
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    proposals = [
        SceneProposal(proposal_id=1, category="chair", position_3d=(0.5, 0.5, 0.0), source="mask3d"),
        SceneProposal(proposal_id=2, category="chair", position_3d=(0.0, 0.0, 0.0), source="mask3d"),
        SceneProposal(proposal_id=3, category="chair", position_3d=(-0.5, -0.5, 0.0), source="mask3d"),
    ]
    full = builder._overlay_proposal_labels(img.copy(), proposals, (-1, -1, 1, 1), None)
    partial = builder._overlay_proposal_labels(img.copy(), proposals, (-1, -1, 1, 1), [2])
    full_diff = int(np.any(full != img, axis=-1).sum())
    partial_diff = int(np.any(partial != img, axis=-1).sum())
    assert 0 < partial_diff < full_diff


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
