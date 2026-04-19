"""Stage 1 backend callbacks for Stage 2 agent tools.

This module provides real implementations for:
- request_more_views: retrieve additional keyframes (targeted + explore mode)
- request_crops: generate object-centric crops with red bbox annotations
- switch_or_expand_hypothesis: re-run Stage 1 with a different query
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw

from .models import (
    KeyframeEvidence,
    Stage2EvidenceBundle,
    Stage2ToolResult,
)


# ---------------------------------------------------------------------------
# request_more_views
# ---------------------------------------------------------------------------

def create_more_views_callback(
    keyframe_selector: Any,
    scene_id: str = "",
    max_additional_views: int = 3,
) -> Callable[[Stage2EvidenceBundle, dict[str, Any]], Stage2ToolResult]:
    """Create a callback that retrieves more views from Stage 1 KeyframeSelector.

    Supports two modes:
    - **targeted** (default): find views covering specified objects
    - **explore**: find views maximally different from existing keyframes
    """

    def callback(
        bundle: Stage2EvidenceBundle,
        request: dict[str, Any],
    ) -> Stage2ToolResult:
        request_text = request.get("request_text", "")
        object_terms = request.get("object_terms", [])
        frame_indices = request.get("frame_indices", [])
        mode = request.get("mode", "targeted")
        if mode not in {"targeted", "explore", "temporal_fan"}:
            raise ValueError(f"invalid mode: {mode!r}")

        logger.info(
            "[Stage1Callback] request_more_views: mode={}, text='{}', objects={}",
            mode,
            request_text[:50],
            object_terms,
        )

        existing_view_ids = {
            kf.view_id for kf in bundle.keyframes if kf.view_id is not None
        }

        if mode == "explore":
            new_view_ids = _explore_views(
                keyframe_selector, existing_view_ids, max_additional_views
            )
        elif mode == "temporal_fan":
            raise NotImplementedError("temporal_fan will land in S2 task")
        else:
            new_view_ids = _targeted_views(
                keyframe_selector,
                bundle,
                object_terms,
                existing_view_ids,
                max_additional_views,
                frame_indices,
            )

        if not new_view_ids:
            return Stage2ToolResult(
                response_text=(
                    "No additional views available. "
                    "Current keyframes already provide best coverage."
                ),
            )

        new_keyframes, added_paths = _resolve_views_to_keyframes(
            keyframe_selector, bundle, new_view_ids, request_text
        )

        if not added_paths:
            return Stage2ToolResult(
                response_text="Could not resolve any additional keyframe paths.",
            )

        updated_bundle = bundle.model_copy(
            update={"keyframes": new_keyframes}, deep=True
        )

        return Stage2ToolResult(
            response_text=(
                f"Added {len(added_paths)} new keyframe(s) (mode={mode}). "
                f"New view IDs: {new_view_ids}. "
                f"Total keyframes now: {len(new_keyframes)}."
            ),
            updated_bundle=updated_bundle,
        )

    return callback


def _targeted_views(
    selector: Any,
    bundle: Stage2EvidenceBundle,
    object_terms: list[str],
    existing_view_ids: set[int],
    max_views: int,
    frame_indices: list[int],
) -> list[int]:
    """Find views covering specified objects (hypothesis + object_terms)."""
    pinned = [
        view_id
        for view_id in (int(v) for v in frame_indices)
        if 0 <= view_id < len(selector.camera_poses) and view_id not in existing_view_ids
    ]

    candidate_object_ids: list[int] = []

    # From hypothesis target/anchor categories
    if bundle.hypothesis:
        for cats in [
            bundle.hypothesis.target_categories or [],
            bundle.hypothesis.anchor_categories or [],
        ]:
            for obj in selector.objects:
                obj_cat = getattr(obj, "category", "") or ""
                obj_tag = getattr(obj, "object_tag", "") or ""
                for cat in cats:
                    if cat.lower() in obj_cat.lower() or cat.lower() in obj_tag.lower():
                        if obj.obj_id not in candidate_object_ids:
                            candidate_object_ids.append(obj.obj_id)
                        break

    # From agent-specified object_terms (CLIP fallback via find_objects)
    for term in object_terms:
        matched = selector.find_objects(term, top_k=5)
        for obj in matched:
            if obj.obj_id not in candidate_object_ids:
                candidate_object_ids.append(obj.obj_id)

    if not candidate_object_ids:
        return pinned[:max_views]

    all_views = selector.get_joint_coverage_views(
        candidate_object_ids,
        max_views=len(existing_view_ids) + max_views,
    )
    new_views = [v for v in all_views if v not in existing_view_ids]

    out = pinned[:max_views]
    if len(out) >= max_views:
        return out[:max_views]

    for view_id in new_views:
        if view_id in out:
            continue
        out.append(view_id)
        if len(out) >= max_views:
            break
    return out[:max_views]


def _explore_views(
    selector: Any,
    existing_view_ids: set[int],
    max_views: int,
) -> list[int]:
    """Find views maximally different from existing keyframes.

    Ranks candidate views by inverse Jaccard overlap of their visible
    object sets against the union of objects in existing keyframes.
    """
    # Collect all objects visible in existing keyframes
    existing_objects: set[int] = set()
    for vid in existing_view_ids:
        for obj_id, _score in selector.view_to_objects.get(vid, []):
            existing_objects.add(obj_id)

    # Score each candidate view by novelty
    scored: list[tuple[float, int]] = []
    for vid, obj_scores in selector.view_to_objects.items():
        if vid in existing_view_ids:
            continue
        view_objects = {obj_id for obj_id, _ in obj_scores}
        if not view_objects:
            continue
        overlap = len(view_objects & existing_objects)
        union = len(view_objects | existing_objects)
        novelty = 1.0 - (overlap / union) if union > 0 else 0.0
        scored.append((novelty, vid))

    scored.sort(reverse=True)
    return [vid for _, vid in scored[:max_views]]


def _resolve_views_to_keyframes(
    selector: Any,
    bundle: Stage2EvidenceBundle,
    view_ids: list[int],
    note_prefix: str,
) -> tuple[list[KeyframeEvidence], list[str]]:
    """Resolve view IDs to KeyframeEvidence entries."""
    new_keyframes: list[KeyframeEvidence] = list(bundle.keyframes)
    added_paths: list[str] = []

    for view_id in view_ids:
        frame_id = selector.map_view_to_frame(view_id)
        path, resolved_view_id = selector._resolve_keyframe_path(view_id)

        if path is not None and path.exists():
            new_keyframes.append(
                KeyframeEvidence(
                    keyframe_idx=len(new_keyframes),
                    image_path=str(path),
                    view_id=resolved_view_id,
                    frame_id=frame_id,
                    note=f"Additional view: {note_prefix[:30]}",
                )
            )
            added_paths.append(str(path))
            logger.info(
                "[Stage1Callback] Added keyframe: view_id={}, frame_id={}, path={}",
                resolved_view_id,
                frame_id,
                path.name,
            )
        else:
            logger.warning(
                "[Stage1Callback] Could not resolve path for view_id={}",
                view_id,
            )

    return new_keyframes, added_paths


# ---------------------------------------------------------------------------
# request_crops
# ---------------------------------------------------------------------------

def create_crop_callback(
    keyframe_selector: Any,
    scene_id: str = "",
    crop_scale: float = 2.0,
) -> Callable[[Stage2EvidenceBundle, dict[str, Any]], Stage2ToolResult]:
    """Create a callback that generates annotated, zoomed-in crops around objects.

    For each matched object visible in the keyframes:
    1. Expands the bbox by crop_scale (default 2x)
    2. Draws red bounding boxes for ALL visible objects in the crop region
    3. Saves the annotated crop as a new keyframe

    Args:
        keyframe_selector: KeyframeSelector with visibility index and objects
        scene_id: Scene identifier
        crop_scale: Multiplier for bbox expansion (2.0 = 2x the bbox size)
    """

    def callback(
        bundle: Stage2EvidenceBundle,
        request: dict[str, Any],
    ) -> Stage2ToolResult:
        object_terms = request.get("object_terms", [])
        crop_multiplier = request.get("crop_scale", crop_scale)
        request_text = request.get("request_text", "")

        logger.info(
            "[Stage1Callback] request_crops: text='{}', objects={}, scale={}",
            request_text[:50],
            object_terms,
            crop_multiplier,
        )

        if not object_terms:
            return Stage2ToolResult(
                response_text="No object_terms specified for crop generation.",
            )

        # Find matching scene objects via find_objects (CLIP fallback)
        target_objects = []
        seen_ids: set[int] = set()
        for term in object_terms:
            matched = keyframe_selector.find_objects(term, top_k=3)
            for obj in matched:
                if obj.obj_id not in seen_ids:
                    target_objects.append(obj)
                    seen_ids.add(obj.obj_id)

        if not target_objects:
            return Stage2ToolResult(
                response_text=(
                    f"No scene objects found matching terms: {object_terms}. "
                    "Try different object descriptions."
                ),
            )

        # Get existing keyframe view IDs
        existing_view_ids = {
            kf.view_id for kf in bundle.keyframes if kf.view_id is not None
        }

        # Generate crops: for each target object, find the best existing
        # keyframe view where it's visible and crop around it
        new_keyframes: list[KeyframeEvidence] = list(bundle.keyframes)
        crops_added = 0

        for target_obj in target_objects:
            crop_added = _generate_crop_for_object(
                keyframe_selector,
                target_obj,
                existing_view_ids,
                new_keyframes,
                crop_multiplier,
                scene_id,
            )
            if crop_added:
                crops_added += 1

        if crops_added == 0:
            return Stage2ToolResult(
                response_text=(
                    f"Could not generate crops for objects: {object_terms}. "
                    "The objects may not be visible in current keyframes."
                ),
            )

        updated_bundle = bundle.model_copy(
            update={"keyframes": new_keyframes}, deep=True
        )

        return Stage2ToolResult(
            response_text=(
                f"Generated {crops_added} annotated crop(s) for {object_terms}. "
                f"Each crop is {crop_multiplier}x the object bbox with red boxes "
                f"marking all visible objects. Total keyframes: {len(new_keyframes)}."
            ),
            updated_bundle=updated_bundle,
        )

    return callback


def _generate_crop_for_object(
    selector: Any,
    target_obj: Any,
    existing_view_ids: set[int],
    keyframes: list[KeyframeEvidence],
    crop_scale: float,
    scene_id: str,
) -> bool:
    """Generate an annotated crop for one object in its best visible keyframe.

    Returns True if a crop was successfully generated and appended.
    """
    # Find the best view where this object is visible (prefer existing keyframes)
    best_view: int | None = None
    best_det_idx: int | None = None
    best_score: float = -1.0

    for det_idx, vid in enumerate(target_obj.image_idx):
        if vid not in existing_view_ids:
            continue
        # Check we have a bbox for this detection
        if det_idx >= len(target_obj.xyxy) or target_obj.xyxy[det_idx] is None:
            continue
        xyxy = target_obj.xyxy[det_idx]
        if not hasattr(xyxy, "__len__") or len(xyxy) != 4:
            continue
        # Score by bbox area (larger = better visibility)
        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area > best_score:
            best_score = area
            best_view = vid
            best_det_idx = det_idx

    # If not in existing keyframes, pick the best view overall
    if best_view is None:
        for det_idx, vid in enumerate(target_obj.image_idx):
            if det_idx >= len(target_obj.xyxy) or target_obj.xyxy[det_idx] is None:
                continue
            xyxy = target_obj.xyxy[det_idx]
            if not hasattr(xyxy, "__len__") or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area > best_score:
                best_score = area
                best_view = vid
                best_det_idx = det_idx

    if best_view is None or best_det_idx is None:
        return False

    # Resolve the image path for this view
    path, _ = selector._resolve_keyframe_path(best_view)
    if path is None or not path.exists():
        return False

    # Load image and get target bbox
    img = Image.open(path).convert("RGB")
    img_w, img_h = img.size
    tx1, ty1, tx2, ty2 = target_obj.xyxy[best_det_idx]

    # Expand bbox by crop_scale
    bbox_w, bbox_h = tx2 - tx1, ty2 - ty1
    cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    crop_w = bbox_w * crop_scale
    crop_h = bbox_h * crop_scale
    # Ensure minimum crop size (at least 256px in each dimension)
    crop_w = max(crop_w, 256)
    crop_h = max(crop_h, 256)

    crop_x1 = max(0, int(cx - crop_w / 2))
    crop_y1 = max(0, int(cy - crop_h / 2))
    crop_x2 = min(img_w, int(cx + crop_w / 2))
    crop_y2 = min(img_h, int(cy + crop_h / 2))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return False

    # Draw red bounding boxes for ALL objects visible in this view
    draw = ImageDraw.Draw(img)
    view_objects = selector.view_to_objects.get(best_view, [])
    for obj_id, _score in view_objects:
        obj = _find_object_by_id(selector.objects, obj_id)
        if obj is None:
            continue
        for det_i, vid in enumerate(obj.image_idx):
            if vid != best_view:
                continue
            if det_i >= len(obj.xyxy) or obj.xyxy[det_i] is None:
                continue
            xyxy = obj.xyxy[det_i]
            if not hasattr(xyxy, "__len__") or len(xyxy) != 4:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in xyxy]
            if bx2 <= bx1 or by2 <= by1:
                continue
            # Red for target object, orange for others
            color = "red" if obj.obj_id == target_obj.obj_id else "orange"
            width = 3 if obj.obj_id == target_obj.obj_id else 1
            draw.rectangle([bx1, by1, bx2, by2], outline=color, width=width)
            label = getattr(obj, "category", "") or f"obj_{obj.obj_id}"
            draw.text((bx1, max(0, by1 - 12)), label, fill=color)
            break

    # Crop the annotated image
    cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    # Resize to at least 512px on the short side for VLM readability
    short_side = min(cropped.size)
    if short_side < 512:
        scale = 512 / short_side
        cropped = cropped.resize(
            (int(cropped.size[0] * scale), int(cropped.size[1] * scale)),
            Image.LANCZOS,
        )

    # Save crop
    crop_dir = path.parent / "crops"
    crop_dir.mkdir(exist_ok=True)
    crop_name = f"crop_obj{target_obj.obj_id}_view{best_view}.jpg"
    crop_path = crop_dir / crop_name
    cropped.save(crop_path, quality=90)

    keyframes.append(
        KeyframeEvidence(
            keyframe_idx=len(keyframes),
            image_path=str(crop_path),
            view_id=best_view,
            frame_id=selector.map_view_to_frame(best_view),
            note=(
                f"Crop of {target_obj.category} (obj {target_obj.obj_id}), "
                f"scale={crop_scale}x, annotated with all visible objects"
            ),
        )
    )

    # Track this view so subsequent crops don't duplicate it
    existing_view_ids.add(best_view)

    logger.info(
        "[Stage1Callback] Generated crop: obj={} ({}) view={} size={}x{} -> {}",
        target_obj.obj_id,
        target_obj.category,
        best_view,
        cropped.size[0],
        cropped.size[1],
        crop_path.name,
    )
    return True


def _find_object_by_id(objects: list, obj_id: int) -> Any | None:
    """Find a SceneObject by its obj_id."""
    for obj in objects:
        if obj.obj_id == obj_id:
            return obj
    return None


# ---------------------------------------------------------------------------
# switch_or_expand_hypothesis
# ---------------------------------------------------------------------------

def create_hypothesis_callback(
    keyframe_selector: Any,
    scene_id: str = "",
    max_new_keyframes: int = 3,
) -> Callable[[Stage2EvidenceBundle, dict[str, Any]], Stage2ToolResult]:
    """Create a callback that re-runs Stage 1 with a new query.

    The agent provides a new_query string. This callback:
    1. Runs select_keyframes_v2 with the new query (triggers LLM parser call)
    2. Appends the new keyframes to the existing bundle (incremental)
    3. Returns the updated bundle

    Note: Each invocation triggers a full Stage 1 LLM parser call (~2-5s latency).
    Use sparingly — prefer request_more_views or request_crops first.
    """

    def callback(
        bundle: Stage2EvidenceBundle,
        request: dict[str, Any],
    ) -> Stage2ToolResult:
        new_query = request.get("new_query", "")
        k = request.get("max_keyframes", max_new_keyframes)

        if not new_query:
            return Stage2ToolResult(
                response_text="No new_query specified for hypothesis switching.",
            )

        logger.info(
            "[Stage1Callback] switch_or_expand_hypothesis: query='{}', k={}",
            new_query,
            k,
        )

        result = keyframe_selector.select_keyframes_v2(new_query, k=k)

        if not result.keyframe_paths:
            return Stage2ToolResult(
                response_text=(
                    f"Re-run Stage 1 with query '{new_query}' produced no keyframes. "
                    "The query may not match any objects in the scene."
                ),
            )

        # Deduplicate against existing keyframes
        existing_paths = {kf.image_path for kf in bundle.keyframes}
        new_keyframes: list[KeyframeEvidence] = list(bundle.keyframes)
        added_paths: list[str] = []

        for i, kf_path in enumerate(result.keyframe_paths):
            path_str = str(kf_path)
            if path_str in existing_paths:
                continue
            view_id = result.keyframe_indices[i] if i < len(result.keyframe_indices) else None
            frame_id = (
                keyframe_selector.map_view_to_frame(view_id)
                if view_id is not None
                else None
            )
            new_keyframes.append(
                KeyframeEvidence(
                    keyframe_idx=len(new_keyframes),
                    image_path=path_str,
                    view_id=view_id,
                    frame_id=frame_id,
                    note=f"Re-query: '{new_query[:40]}'",
                )
            )
            added_paths.append(path_str)

        if not added_paths:
            return Stage2ToolResult(
                response_text=(
                    f"Re-query '{new_query}' found keyframes but all overlap "
                    "with existing views. No new evidence added."
                ),
            )

        updated_bundle = bundle.model_copy(
            update={"keyframes": new_keyframes}, deep=True
        )

        return Stage2ToolResult(
            response_text=(
                f"Re-ran Stage 1 with query '{new_query}': added {len(added_paths)} "
                f"new keyframe(s). Total keyframes now: {len(new_keyframes)}."
            ),
            updated_bundle=updated_bundle,
        )

    return callback


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------

class Stage1BackendCallbacks:
    """Convenience class to hold all Stage 1 callbacks together."""

    def __init__(
        self,
        keyframe_selector: Any,
        scene_id: str = "",
        max_additional_views: int = 3,
        crop_scale: float = 2.0,
        max_hypothesis_keyframes: int = 3,
    ) -> None:
        self.keyframe_selector = keyframe_selector
        self.scene_id = scene_id

        self.more_views = create_more_views_callback(
            keyframe_selector,
            scene_id=scene_id,
            max_additional_views=max_additional_views,
        )
        self.crops = create_crop_callback(
            keyframe_selector,
            scene_id=scene_id,
            crop_scale=crop_scale,
        )
        self.hypothesis = create_hypothesis_callback(
            keyframe_selector,
            scene_id=scene_id,
            max_new_keyframes=max_hypothesis_keyframes,
        )
