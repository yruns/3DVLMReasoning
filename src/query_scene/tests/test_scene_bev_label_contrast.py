import numpy as np

from agents.catalog import SceneProposal
from query_scene.scene_bev_builder import ScanNetSceneBEVBuilderBase, SceneBEVConfig


class _Stub(ScanNetSceneBEVBuilderBase):
    def resolve_paths(self, scene_id, data_root):
        raise NotImplementedError


def test_label_has_black_outline_around_white_text():
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    proposals = [
        SceneProposal(
            proposal_id=1,
            category="chair",
            position_3d=(0.0, 0.0, 0.0),
            source="mask3d",
            frame_views={},
        )
    ]
    builder = _Stub(config=SceneBEVConfig(image_size=400))
    bounds = (0.0, 0.0, 10.0, 10.0)
    out = builder._overlay_proposal_labels(img, proposals, bounds, highlight_ids=None)
    is_black = out.max(axis=-1) <= 32
    is_white = out.min(axis=-1) >= 248
    adj_right = is_black[:, :-1] & is_white[:, 1:]
    adj_left = is_white[:, :-1] & is_black[:, 1:]
    assert adj_right.any() or adj_left.any(), "expected black/white adjacency from text outline"
