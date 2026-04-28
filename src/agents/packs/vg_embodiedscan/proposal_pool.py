"""Build the vg_proposal_pool dict that the runtime adapter expects.

Inputs:
- proposals_jsonl: a JSON file containing
  `{"proposals": [{"id": <int>, "bbox_3d": [...], "score": ..., "label": ...}, ...]}`
  When `id` is present, it is used as the proposal id (e.g. EmbodiedScan
  bbox_id for the GT pool); otherwise the positional index is used.
- source: 'gt' (EmbodiedScan annotation), 'vdetr', or 'conceptgraph'
- annotated_image_dir: directory of pre-rendered set-of-marks frames
- frame_visibility: mapping frame_id -> list of proposal ids visible
  in that frame (must match the ids produced by this builder)
- axis_align_matrix: 4x4 numpy array or None
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np


def build_vg_proposal_pool(
    *,
    proposals_jsonl: Path,
    source: Literal["gt", "vdetr", "conceptgraph"],
    annotated_image_dir: Path,
    frame_visibility: dict[int, list[int]],
    axis_align_matrix: np.ndarray | None,
) -> dict:
    """Read proposals + visibility and produce the `vg_proposal_pool`
    dict expected at `bundle.extra_metadata.vg_proposal_pool`."""
    raw = json.loads(proposals_jsonl.read_text(encoding="utf-8"))
    if "proposals" not in raw:
        raise ValueError(
            f"feasibility proposal JSON must have a top-level 'proposals' key: {proposals_jsonl}"
        )
    proposals_in = raw["proposals"]
    if not isinstance(proposals_in, list):
        raise ValueError(
            f"feasibility proposal JSON 'proposals' must be a list: {proposals_jsonl}"
        )

    proposals_out = []
    for idx, p in enumerate(proposals_in):
        for required_key in ("bbox_3d", "score", "label"):
            if required_key not in p:
                raise ValueError(
                    f"proposal[{idx}].{required_key} is required in {proposals_jsonl}"
                )
        pid = int(p["id"]) if "id" in p else idx
        proposals_out.append(
            {
                "id": pid,
                "bbox_3d_9dof": [float(x) for x in p["bbox_3d"]],
                "category": str(p["label"]),
                "score": float(p["score"]),
            }
        )

    proposal_index: dict[int, list[int]] = defaultdict(list)
    frame_index: dict[int, list[int]] = {}
    for frame_id, prop_ids in frame_visibility.items():
        ids = [int(i) for i in prop_ids]
        frame_index[int(frame_id)] = ids
        for pid in ids:
            proposal_index[pid].append(int(frame_id))

    pool: dict = {
        "source": source,
        "proposals": proposals_out,
        "frame_index": frame_index,
        "proposal_index": dict(proposal_index),
        "annotated_image_dir": str(annotated_image_dir),
    }
    if axis_align_matrix is not None:
        pool["axis_align_matrix"] = axis_align_matrix.tolist()
    return pool


__all__ = ["build_vg_proposal_pool"]
