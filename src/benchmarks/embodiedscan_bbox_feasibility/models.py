from __future__ import annotations

from enum import Enum
from math import isfinite
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


class FailureTag(str, Enum):
    NO_PROPOSAL = "no_proposal"
    COORD_MISMATCH = "coord_mismatch"
    DEGENERATE_BOX = "degenerate_box"
    VISIBILITY_LIMITED = "visibility_limited"
    OVERMERGE = "overmerge"
    FRAGMENTATION = "fragmentation"
    DETECTOR_OOD = "detector_ood"
    MODEL_BLOCKED = "model_blocked"


class EmbodiedScanTarget(BaseModel):
    sample_ids: list[str] = Field(default_factory=list)
    scan_id: str
    scene_id: str
    target_id: int
    target_category: str = ""
    gt_bbox_3d: list[float]

    @field_validator("gt_bbox_3d")
    @classmethod
    def _validate_gt_bbox(cls, value: list[float]) -> list[float]:
        return _normalize_bbox_9dof(value, field_name="gt_bbox_3d")


class ObservationRecord(BaseModel):
    policy: str
    frame_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> dict[str, Any]:
        return _validate_json_safe_metadata(value, field_name="metadata")


class BBox3DProposal(BaseModel):
    bbox_3d: list[float]
    score: float | None = Field(default=None, allow_inf_nan=False)
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox_3d")
    @classmethod
    def _validate_bbox(cls, value: list[float]) -> list[float]:
        return _normalize_bbox_9dof(value, field_name="bbox_3d")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> dict[str, Any]:
        return _validate_json_safe_metadata(value, field_name="metadata")


class ProposalRecord(BaseModel):
    scene_id: str
    scan_id: str
    target_id: int | None = None
    method: str
    input_condition: str
    observation: ObservationRecord | None = None
    proposals: list[BBox3DProposal] = Field(default_factory=list)
    failure_tag: FailureTag | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: Any) -> dict[str, Any]:
        return _validate_json_safe_metadata(value, field_name="metadata")


class TargetScore(BaseModel):
    scan_id: str
    scene_id: str
    target_id: int
    method: str
    input_condition: str
    best_iou: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    best_proposal_index: int | None = Field(default=None, ge=0)
    failure_tag: FailureTag | None = None


class AggregateMetrics(BaseModel):
    method: str
    input_condition: str
    num_targets: int = Field(ge=0)
    mean_best_iou: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    median_best_iou: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    acc_025: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    acc_050: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_proposals_per_record: float = Field(ge=0.0, allow_inf_nan=False)
    non_degenerate_box_ratio: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    failure_counts: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)


def _normalize_bbox_9dof(value: list[float], *, field_name: str) -> list[float]:
    if len(value) < 6:
        raise ValueError(f"{field_name} must contain at least 6 values")
    if len(value) > 9:
        raise ValueError(f"{field_name} must contain at most 9 values")
    out = [float(v) for v in value[:9]]
    if not all(isfinite(v) for v in out):
        raise ValueError(f"{field_name} must contain only finite values")
    while len(out) < 9:
        out.append(0.0)
    return out


def _validate_json_safe_metadata(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    _validate_json_safe_value(value, field_name=field_name)
    return value


def _validate_json_safe_value(value: Any, *, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field_name} must contain only finite floats")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_safe_value(item, field_name=field_name)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} must contain only string keys")
            _validate_json_safe_value(item, field_name=field_name)
        return
    raise ValueError(f"{field_name} must contain only JSON-safe values")
