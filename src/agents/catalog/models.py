"""SceneCatalog and supporting models for v9 catalog-first scene exploration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FrameView(BaseModel):
    """One proposal's projection into one frame."""

    model_config = ConfigDict(frozen=False)

    frame_id: int
    bbox_2d: tuple[int, int, int, int]
    raw_rgb_path: str
    visibility_weight: float | None = None


class SceneProposal(BaseModel):
    """Unified scene entity unit (VG Mask3D/V-DETR/GT proposal or QA conceptgraph object)."""

    model_config = ConfigDict(frozen=False)

    proposal_id: int
    category: str
    position_3d: tuple[float, float, float]
    bbox_3d_9dof: (
        tuple[float, float, float, float, float, float, float, float, float] | None
    ) = None
    frame_views: dict[int, FrameView] = Field(default_factory=dict)
    source: Literal["mask3d", "vdetr", "gt", "conceptgraph"]
    enriched_category: str | None = None
    compact_note: str | None = None
    enrichment: dict[str, Any] | None = None


class SceneCatalog(BaseModel):
    """Scene-level catalog shared across an entire Stage-2 session."""

    model_config = ConfigDict(frozen=False)

    scene_id: str
    scene_category: str | None = None
    proposals: list[SceneProposal] = Field(default_factory=list)
    total_frames: int
    frame_id_range: tuple[int, int]
    valid_frame_ids: list[int] = Field(default_factory=list)
    bev_image_path: str
    axis_align_matrix: list[list[float]] | None = None

    def proposals_by_category(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for p in self.proposals:
            out.setdefault(p.category, []).append(p.proposal_id)
        return out

    def proposal_by_id(self, proposal_id: int) -> SceneProposal | None:
        for p in self.proposals:
            if p.proposal_id == proposal_id:
                return p
        return None


__all__ = ["FrameView", "SceneProposal", "SceneCatalog"]
