"""Adapters between query-scene Stage 1 outputs and Stage-2 agent inputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .models import (
    KeyframeEvidence,
    Stage1HypothesisSummary,
    Stage2EvidenceBundle,
)


def _stringify_object_context(obj: object) -> str:
    """Build a compact object summary string for agent consumption."""
    parts: list[str] = []

    summary = getattr(obj, "summary", "")
    if summary:
        parts.append(summary.strip())

    affordance_category = getattr(obj, "affordance_category", "")
    if affordance_category:
        parts.append(f"affordance_category={affordance_category}")

    co_objects = getattr(obj, "co_objects", None) or []
    if co_objects:
        parts.append(f"co_objects={', '.join(str(item) for item in co_objects[:5])}")

    affordances = getattr(obj, "affordances", None) or {}
    if affordances:
        affordance_keys = ", ".join(sorted(str(key) for key in affordances.keys()))
        parts.append(f"affordances={affordance_keys}")

    if not parts:
        category = getattr(obj, "category", "unknown")
        return f"No rich context available for {category}."

    return "; ".join(parts)


def build_object_context(objects: Iterable[object]) -> dict[str, str]:
    """Create a queryable object-context map from stage-1 scene objects."""
    context: dict[str, str] = {}
    for obj in objects:
        key = getattr(obj, "object_tag", "") or getattr(obj, "category", "")
        if not key:
            continue
        context[str(key)] = _stringify_object_context(obj)
    return context


def build_stage2_evidence_bundle(
    keyframe_result: object,
    *,
    scene_id: str = "",
    bev_image_path: str | None = None,
    scene_summary: str = "",
    object_context: dict[str, str] | None = None,
    selector: object | None = None,
) -> Stage2EvidenceBundle:
    """Convert a `KeyframeResult`-like object into a Stage-2 evidence bundle."""
    metadata = dict(getattr(keyframe_result, "metadata", {}) or {})
    hypothesis_output = metadata.get("hypothesis_output", {}) or {}
    hypotheses = hypothesis_output.get("hypotheses", []) or []

    selected_rank = metadata.get("selected_hypothesis_rank")
    selected_hypothesis = None
    for item in hypotheses:
        if item.get("rank") == selected_rank:
            selected_hypothesis = item
            break

    root = ((selected_hypothesis or {}).get("grounding_query") or {}).get("root") or {}
    target_categories = list(root.get("category", []) or [])
    anchor_categories: list[str] = []
    for constraint in root.get("spatial_constraints", []) or []:
        for anchor in constraint.get("anchors", []) or []:
            anchor_categories.extend(anchor.get("category", []) or [])

    bundle_object_context = object_context or build_object_context(
        list(getattr(keyframe_result, "target_objects", []) or [])
        + list(getattr(keyframe_result, "anchor_objects", []) or [])
    )

    keyframes = []
    frame_mappings = metadata.get("frame_mappings", []) or []
    selection_scores = (
        metadata.get("selection_scores")
        or getattr(keyframe_result, "selection_scores", {})
        or {}
    )
    previous_view_id: int | None = None
    for idx, image_path in enumerate(
        getattr(keyframe_result, "keyframe_paths", []) or []
    ):
        mapping = frame_mappings[idx] if idx < len(frame_mappings) else {}
        resolved_view_id = mapping.get(
            "resolved_view_id", mapping.get("requested_view_id")
        )
        note = metadata.get("status", "")
        if (
            selector is not None
            and getattr(selector, "pose_aware_enabled", False)
            and resolved_view_id is not None
        ):
            note = _build_temporal_note(
                selector,
                view_id=int(resolved_view_id),
                order=idx + 1,
                total=len(getattr(keyframe_result, "keyframe_paths", []) or []),
                prev_view_id=previous_view_id,
            )
        keyframes.append(
            KeyframeEvidence(
                keyframe_idx=idx,
                image_path=str(Path(image_path)),
                view_id=resolved_view_id,
                frame_id=mapping.get(
                    "resolved_frame_id", mapping.get("requested_frame_id")
                ),
                score=selection_scores.get(
                    resolved_view_id
                ),
                note=note,
            )
        )
        if resolved_view_id is not None:
            previous_view_id = int(resolved_view_id)

    hypothesis = None
    if selected_hypothesis is not None:
        hypothesis = Stage1HypothesisSummary(
            status=metadata.get("status", ""),
            hypothesis_kind=metadata.get("selected_hypothesis_kind", ""),
            hypothesis_rank=metadata.get("selected_hypothesis_rank"),
            parse_mode=hypothesis_output.get("parse_mode", ""),
            raw_query=getattr(keyframe_result, "query", ""),
            target_categories=target_categories,
            anchor_categories=anchor_categories,
            metadata={
                "version": metadata.get("version"),
                "strategy": metadata.get("strategy"),
            },
        )

    return Stage2EvidenceBundle(
        scene_id=scene_id,
        stage1_query=getattr(keyframe_result, "query", ""),
        keyframes=keyframes,
        bev_image_path=bev_image_path,
        scene_summary=scene_summary,
        object_context=bundle_object_context,
        hypothesis=hypothesis,
        extra_metadata=metadata,
    )


def _build_temporal_note(
    selector: object,
    *,
    view_id: int,
    order: int,
    total: int,
    prev_view_id: int | None,
) -> str:
    parts = [f"order={order}/{total}"]

    dwell_scores = np.asarray(getattr(selector, "dwell_score", []), dtype=np.float64)
    if 0 <= view_id < len(dwell_scores):
        if dwell_scores[view_id] > 0.7:
            parts.append("dwell")
        elif dwell_scores[view_id] < 0.3:
            parts.append("traverse")

    camera_poses = getattr(selector, "camera_poses", [])
    if (
        prev_view_id is not None
        and 0 <= prev_view_id < len(camera_poses)
        and 0 <= view_id < len(camera_poses)
    ):
        prev_forward = -np.asarray(camera_poses[prev_view_id][:3, 2], dtype=np.float64)
        cur_forward = -np.asarray(camera_poses[view_id][:3, 2], dtype=np.float64)
        prev_norm = float(np.linalg.norm(prev_forward))
        cur_norm = float(np.linalg.norm(cur_forward))
        if prev_norm > 1e-8 and cur_norm > 1e-8:
            cos_theta = float(
                np.clip(
                    np.dot(prev_forward / prev_norm, cur_forward / cur_norm),
                    -1.0,
                    1.0,
                )
            )
            dtheta_deg = float(np.degrees(np.arccos(cos_theta)))
            parts.append(f"heading=+{dtheta_deg:.0f}°")

    neighbors: list[int] = []
    for delta in (-2, -1, 1, 2):
        neighbor = view_id + delta
        if 0 <= neighbor < len(camera_poses) and neighbor != prev_view_id:
            neighbors.append(neighbor)
    if neighbors:
        parts.append(f"neighbors={neighbors}")

    return " ".join(parts)
