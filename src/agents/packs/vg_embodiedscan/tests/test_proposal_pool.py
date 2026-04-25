"""Adapter: feasibility-module artifacts -> vg_proposal_pool dict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.packs.vg_embodiedscan.proposal_pool import (
    build_vg_proposal_pool,
)


def _write_proposals_json(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"proposals": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_build_vg_proposal_pool_with_visibility(tmp_path: Path) -> None:
    proposals_path = tmp_path / "props.json"
    _write_proposals_json(
        proposals_path,
        [
            {"bbox_3d": [0,0,0,1,1,1,0,0,0], "score": 0.9, "label": "chair", "metadata": {"class_id": 2}},
            {"bbox_3d": [3,0,0,1,1,1,0,0,0], "score": 0.7, "label": "desk", "metadata": {"class_id": 10}},
        ],
    )
    annotated = tmp_path / "ann"
    annotated.mkdir()

    # frustum/depth visibility precomputed externally — pass as dict
    visibility = {
        # frame_id -> [proposal_idx]
        100: [0, 1],
        101: [1],
    }

    pool = build_vg_proposal_pool(
        proposals_jsonl=proposals_path,
        source="vdetr",
        annotated_image_dir=annotated,
        frame_visibility=visibility,
        axis_align_matrix=np.eye(4),
    )
    assert pool["source"] == "vdetr"
    assert pool["annotated_image_dir"] == str(annotated)
    assert {p["id"] for p in pool["proposals"]} == {0, 1}
    assert pool["frame_index"] == {100: [0, 1], 101: [1]}
    assert pool["proposal_index"] == {0: [100], 1: [100, 101]}
    assert len(pool["axis_align_matrix"]) == 4
