"""Typed runtime ctx for the EmbodiedScan VG pack."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from agents.core.task_types import Stage2EvidenceBundle

_NEW_VIEW_IDS_RE = re.compile(r"New view IDs: \[([\d,\s]+)\]")


@dataclass(frozen=True)
class ProposalFrameView:
    proposal_id: int
    frame_id: int
    bbox_2d: tuple[int, int, int, int]
    raw_rgb_path: Path
    visibility_weight: float = 1.0


@dataclass(frozen=True)
class Proposal:
    id: int
    bbox_3d_9dof: list[float]
    category: str
    score: float
    frame_views: dict[int, ProposalFrameView] = field(default_factory=dict)


@dataclass
class VgEmbodiedScanCtx:
    proposal_pool_source: Literal["gt", "vdetr", "conceptgraph"]
    proposals: list[Proposal]
    frame_index: dict[int, list[int]]  # frame_id -> [proposal_id]
    proposal_index: dict[int, list[int]]  # proposal_id -> [frame_id]
    annotated_image_dir: Path
    axis_align_matrix: np.ndarray | None = None

    def visible_proposal_ids(self, frame_ids: set[int]) -> set[int]:
        """Return proposal ids visible in any of the supplied frame ids."""
        visible: set[int] = set()
        for frame_id in frame_ids:
            visible.update(self.frame_index.get(int(frame_id), []))
        return visible

    def frame_views_for(
        self, proposal_id: int, frame_ids: set[int]
    ) -> list[ProposalFrameView]:
        """Return stored 2D crop metadata for a proposal in seen frames."""
        proposal = next((p for p in self.proposals if p.id == proposal_id), None)
        if proposal is None:
            return []
        views = [
            view
            for frame_id, view in proposal.frame_views.items()
            if int(frame_id) in frame_ids
        ]
        return sorted(views, key=lambda view: view.visibility_weight, reverse=True)


def cumulative_seen_frame_ids(runtime: object) -> set[int]:
    """Frames visible to the agent so far: bundle keyframes + marked views."""
    frame_ids: set[int] = set()
    bundle = getattr(runtime, "bundle", None)
    for keyframe in getattr(bundle, "keyframes", []) or []:
        frame_id = getattr(keyframe, "frame_id", None)
        if frame_id is not None:
            frame_ids.add(int(frame_id))

    for obs in getattr(runtime, "tool_trace", []) or []:
        if isinstance(obs, dict):
            tool_name = obs.get("tool_name", "")
            response_text = str(obs.get("response_text", ""))
            tool_input = obs.get("tool_input", {}) or {}
        else:
            tool_name = getattr(obs, "tool_name", "")
            response_text = str(getattr(obs, "response_text", ""))
            tool_input = getattr(obs, "tool_input", {}) or {}
        # v9: view_keyframe (any mode) is the canonical frame-injection tool.
        # The pre-v9 stage-1 callback tools were removed and their trace
        # entries are no longer produced.
        if tool_name == "view_keyframe":
            if response_text.startswith("ERROR"):
                continue
            frame_id = tool_input.get("frame_id")
            if frame_id is not None:
                frame_ids.add(int(frame_id))
    return frame_ids


def _parse_frame_views(
    raw: object,
    *,
    proposal_id: int,
) -> dict[int, ProposalFrameView]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        items = []
        for frame_id, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError(
                    f"proposal {proposal_id} frame_views[{frame_id!r}] must be a dict"
                )
            item = dict(value)
            item.setdefault("frame_id", frame_id)
            items.append(item)
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(
            f"proposal {proposal_id}.frame_views must be a dict or list; "
            f"got {type(raw).__name__}"
        )

    parsed: dict[int, ProposalFrameView] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"proposal {proposal_id}.frame_views[{i}] must be a dict; "
                f"got {type(item).__name__}"
            )
        for required_key in ("frame_id", "bbox_2d", "raw_rgb_path"):
            if required_key not in item:
                raise ValueError(
                    f"proposal {proposal_id}.frame_views[{i}].{required_key} is required"
                )
        bbox = item["bbox_2d"]
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            raise ValueError(
                f"proposal {proposal_id}.frame_views[{i}].bbox_2d must be a 4-item list"
            )
        frame_id = int(item["frame_id"])
        parsed[frame_id] = ProposalFrameView(
            proposal_id=int(item.get("proposal_id", proposal_id)),
            frame_id=frame_id,
            bbox_2d=tuple(int(x) for x in bbox),
            raw_rgb_path=Path(str(item["raw_rgb_path"])),
            visibility_weight=float(item.get("visibility_weight", 1.0)),
        )
    return parsed


def build_ctx_from_bundle(bundle: Stage2EvidenceBundle) -> VgEmbodiedScanCtx:
    extra = bundle.extra_metadata or {}
    pool = extra.get("vg_proposal_pool")
    if pool is None:
        raise ValueError("bundle.extra_metadata.vg_proposal_pool is missing")

    source = pool.get("source")
    if source not in ("gt", "vdetr", "conceptgraph"):
        raise ValueError(
            f"proposal_pool_source must be 'gt', 'vdetr', or 'conceptgraph', got {source!r}"
        )

    if "proposals" not in pool:
        raise ValueError("vg_proposal_pool.proposals key is required")
    raw_proposals = pool["proposals"]
    if not isinstance(raw_proposals, list):
        raise ValueError(
            "vg_proposal_pool.proposals must be a list; got "
            f"{type(raw_proposals).__name__}"
        )

    proposals: list[Proposal] = []
    for i, p in enumerate(raw_proposals):
        for required_key in ("id", "bbox_3d_9dof", "category", "score"):
            if required_key not in p:
                raise ValueError(
                    f"vg_proposal_pool.proposals[{i}].{required_key} is required"
                )
        bbox = p["bbox_3d_9dof"]
        if not isinstance(bbox, list) or len(bbox) != 9:
            raise ValueError(
                f"vg_proposal_pool.proposals[{i}].bbox_3d_9dof must be a 9-element list"
            )
        proposals.append(
            Proposal(
                id=int(p["id"]),
                bbox_3d_9dof=[float(x) for x in bbox],
                category=str(p["category"]),
                score=float(p["score"]),
                frame_views=_parse_frame_views(
                    p.get("frame_views"),
                    proposal_id=int(p["id"]),
                ),
            )
        )

    if "frame_index" not in pool:
        raise ValueError("vg_proposal_pool.frame_index key is required")
    if "proposal_index" not in pool:
        raise ValueError("vg_proposal_pool.proposal_index key is required")
    frame_index = {int(k): [int(x) for x in v] for k, v in pool["frame_index"].items()}
    proposal_index = {
        int(k): [int(x) for x in v] for k, v in pool["proposal_index"].items()
    }

    annotated_dir = Path(pool.get("annotated_image_dir") or "")
    if not annotated_dir.exists() or not annotated_dir.is_dir():
        raise ValueError(
            f"annotated_image_dir must exist and be a directory: {annotated_dir}"
        )

    axis_align_matrix: np.ndarray | None = None
    matrix = pool.get("axis_align_matrix")
    if matrix is not None:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(f"axis_align_matrix must be 4x4, got shape {arr.shape}")
        axis_align_matrix = arr

    return VgEmbodiedScanCtx(
        proposal_pool_source=source,
        proposals=proposals,
        frame_index=frame_index,
        proposal_index=proposal_index,
        annotated_image_dir=annotated_dir,
        axis_align_matrix=axis_align_matrix,
    )


__all__ = [
    "Proposal",
    "ProposalFrameView",
    "VgEmbodiedScanCtx",
    "build_ctx_from_bundle",
    "cumulative_seen_frame_ids",
]
