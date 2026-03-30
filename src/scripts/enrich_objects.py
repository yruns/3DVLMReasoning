#!/usr/bin/env python3
"""Enrich ConceptGraph objects with LLM-generated metadata.

For each object in a scene, selects the best frames from the visibility index,
draws bounding boxes, crops small objects, and sends annotated images to Gemini
to generate accurate category labels, descriptions, and affordances.

Usage:
    # Single scene
    python -m src.scripts.enrich_objects \
        --scene_path data/OpenEQA/scannet/002-scannet-scene0709_00

    # All scenes
    python -m src.scripts.enrich_objects \
        --scannet_root data/OpenEQA/scannet --all

    # Force re-process (ignore existing results)
    python -m src.scripts.enrich_objects \
        --scene_path data/OpenEQA/scannet/002-scannet-scene0709_00 --force
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from loguru import logger
from PIL import Image, ImageDraw

# Reuse existing utilities
from src.scripts.validate_scene_graph import (
    best_bbox_for_view,
    get_process_stride,
    load_scene_graph,
    object_label,
    resolve_view_image_paths,
)
from src.utils.llm_client import GeminiClientPool, _is_rate_limit_error

REQUIRED_ENRICHMENT_FIELDS = {
    "category",
    "description",
    "location",
    "nearby_objects",
    "color",
    "usability",
}

ENRICHMENT_PROMPT_TEMPLATE = """\
You are analyzing an object highlighted with a red bounding box in the \
following {num_images} image(s) from an indoor 3D scan.

The current (possibly noisy) label for this object is: "{current_label}"

Based on what you see in the image(s), provide the following information \
about the highlighted object in JSON format:

{{
  "category": "A specific, accurate category label (e.g. 'office chair', \
'floor lamp', 'kitchen sink'). Be specific and environment-aware. \
Correct any noisy or overly generic labels.",
  "description": "Concise description of the object (max 100 words). \
Include appearance, material, condition, and distinguishing features.",
  "location": "Where this object is in the scene. Describe position \
relative to walls, floors, and major landmarks.",
  "nearby_objects": ["list", "of", "objects", "visible", "near", "this", "one"],
  "color": "Primary color(s) of the object.",
  "usability": "How this object can be used or interacted with. \
Describe functional affordances (max 100 words)."
}}

IMPORTANT:
- Focus ONLY on the object inside the red bounding box.
- For "nearby_objects", list only objects you can directly observe in \
the images, not guesses.
- Return ONLY valid JSON, no markdown wrapping, no extra text."""


# ---------------------------------------------------------------------------
# Image annotation
# ---------------------------------------------------------------------------


def _resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """Resize image to fit within max_size, preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w > h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_h, new_w = max_size, int(w * max_size / h)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _image_to_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Convert PIL image to base64 JPEG string."""
    resized = _resize_image(image, max_size)
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _maybe_crop_to_object(
    image: Image.Image,
    bbox_xyxy: list[int],
    min_ratio: float = 0.15,
    target_ratio: float = 0.20,
) -> tuple[Image.Image, list[int]]:
    """Crop around the object if its bbox is too small relative to the image.

    Returns the (possibly cropped) image and the updated bbox coordinates.
    """
    x1, y1, x2, y2 = bbox_xyxy
    img_w, img_h = image.size
    bbox_longest = max(x2 - x1, y2 - y1)
    img_longest = max(img_w, img_h)

    if bbox_longest / img_longest >= min_ratio:
        return image, bbox_xyxy

    # Calculate crop size so bbox occupies ~target_ratio of cropped image
    desired_crop_size = int(bbox_longest / target_ratio)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = desired_crop_size // 2

    crop_x1 = max(0, cx - half)
    crop_y1 = max(0, cy - half)
    crop_x2 = min(img_w, cx + half)
    crop_y2 = min(img_h, cy + half)

    # Adjust if crop is smaller than desired (near image edge)
    if crop_x2 - crop_x1 < desired_crop_size and crop_x1 == 0:
        crop_x2 = min(img_w, desired_crop_size)
    elif crop_x2 - crop_x1 < desired_crop_size and crop_x2 == img_w:
        crop_x1 = max(0, img_w - desired_crop_size)

    if crop_y2 - crop_y1 < desired_crop_size and crop_y1 == 0:
        crop_y2 = min(img_h, desired_crop_size)
    elif crop_y2 - crop_y1 < desired_crop_size and crop_y2 == img_h:
        crop_y1 = max(0, img_h - desired_crop_size)

    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    new_bbox = [x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1]
    return cropped, new_bbox


def prepare_annotated_image(
    frame_path: Path,
    bbox_xyxy: list[int],
    max_size: int = 1024,
) -> str:
    """Load frame, crop if bbox small, draw red bbox, return base64 data URL."""
    img = Image.open(frame_path).convert("RGB")
    img, bbox = _maybe_crop_to_object(img, bbox_xyxy)

    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline="red", width=4)

    b64 = _image_to_base64(img, max_size)
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def select_best_frames(
    obj_id: int,
    obj: dict[str, Any],
    object_to_views: dict[int, list[tuple[int, float]]],
    view_paths: list[Path],
    num_frames: int = 3,
) -> list[dict[str, Any]]:
    """Select top-N frames for an object by visibility score."""
    views = object_to_views.get(obj_id, [])
    frames: list[dict[str, Any]] = []
    for view_id, score in views:
        if len(frames) >= num_frames:
            break
        view_id_int = int(view_id)
        if view_id_int >= len(view_paths):
            continue
        bbox = best_bbox_for_view(obj, view_id_int)
        if bbox is None:
            continue
        # Skip zero-area bboxes
        bx1, by1, bx2, by2 = bbox
        if (bx2 - bx1) <= 0 or (by2 - by1) <= 0:
            continue
        frames.append(
            {
                "view_id": view_id_int,
                "frame_path": view_paths[view_id_int],
                "frame_name": view_paths[view_id_int].name,
                "score": round(float(score), 4),
                "bbox": bbox,
            }
        )
    return frames


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


def _build_prompt(current_label: str, num_images: int) -> str:
    return ENRICHMENT_PROMPT_TEMPLATE.format(
        current_label=current_label, num_images=num_images
    )


def _parse_llm_response(response_text: str) -> dict[str, Any]:
    """Parse JSON from LLM response.

    Tries in order: raw text as JSON, markdown-wrapped block, then fails.
    """
    stripped = response_text.strip()

    # 1) Try raw text as JSON directly
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 2) Try markdown-wrapped JSON block
    m = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    raise ValueError(f"No valid JSON in response: {response_text[:300]}")


def enrich_single_object(
    pool: GeminiClientPool,
    obj_id: int,
    obj: dict[str, Any],
    frames: list[dict[str, Any]],
    max_retries: int = 10,
) -> dict[str, Any]:
    """Enrich one object with LLM-generated metadata. Retries up to max_retries."""
    label = object_label(obj, obj_id)

    # Prepare annotated images
    image_urls = [prepare_annotated_image(f["frame_path"], f["bbox"]) for f in frames]
    prompt = _build_prompt(label, len(image_urls))

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    message = HumanMessage(content=content)

    last_error = None
    config_idx = None
    for attempt in range(max_retries):
        try:
            client, config_idx = pool.get_next_client(temperature=0.2, timeout=120)
            response = client.invoke([message])
            pool.record_request(config_idx)

            # response.content may be str or list of content blocks
            raw_content = response.content
            if isinstance(raw_content, list):
                raw_content = " ".join(
                    block.get("text", "")
                    for block in raw_content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            enrichment = _parse_llm_response(raw_content)
            missing = REQUIRED_ENRICHMENT_FIELDS - set(enrichment.keys())
            if missing:
                raise ValueError(f"Missing fields: {missing}")

            return {
                "obj_id": obj_id,
                "status": "success",
                "original_label": label,
                "frames_used": [
                    {
                        "view_id": f["view_id"],
                        "frame_name": f["frame_name"],
                        "score": f["score"],
                    }
                    for f in frames
                ],
                "enrichment": enrichment,
                "retry_count": attempt,
            }

        except Exception as e:
            last_error = e
            is_rate = _is_rate_limit_error(e)
            if is_rate:
                if config_idx is not None:
                    pool.record_request(config_idx, rate_limited=True)
                wait = min(5 * 2**attempt, 60)
                logger.debug(
                    f"[Enrich] obj {obj_id} attempt {attempt + 1} rate limited, "
                    f"sleeping {wait}s"
                )
                time.sleep(wait)
            else:
                logger.warning(
                    f"[Enrich] obj {obj_id} attempt {attempt + 1} failed: {e}"
                )
                time.sleep(2)

    error_msg = f"All {max_retries} retries exhausted. Last error: {last_error}"
    logger.error(f"[Enrich] obj {obj_id}: {error_msg}")
    return {
        "obj_id": obj_id,
        "status": "failed",
        "original_label": label,
        "error": error_msg,
        "retry_count": max_retries,
    }


# ---------------------------------------------------------------------------
# Scene-level orchestration
# ---------------------------------------------------------------------------

_save_lock = threading.Lock()


def _load_existing(output_path: Path) -> dict[int, dict[str, Any]]:
    """Load existing enriched_objects.json and return obj_id -> entry map.

    Raises on corrupt files to avoid silently losing progress.
    """
    if not output_path.exists():
        return {}
    data = json.loads(output_path.read_text(encoding="utf-8"))
    return {
        int(entry["obj_id"]): entry
        for entry in data.get("objects", [])
        if entry.get("status") == "success"
    }


def _save_output(
    output_path: Path,
    clip_id: str,
    all_results: list[dict[str, Any]],
    total_objects: int,
) -> None:
    """Atomically write the enriched_objects.json output."""
    enriched = sum(1 for r in all_results if r["status"] == "success")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")
    failed = sum(1 for r in all_results if r["status"] == "failed")

    output = {
        "format_version": "enriched_objects_v1",
        "clip_id": clip_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "gemini-2.5-pro",
        "summary": {
            "total_objects": total_objects,
            "enriched": enriched,
            "skipped": skipped,
            "failed": failed,
        },
        "objects": sorted(all_results, key=lambda r: r["obj_id"]),
    }

    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tmp_path.rename(output_path)


def enrich_scene(
    scene_root: Path,
    max_workers: int = 7,
    max_retries: int = 10,
    force: bool = False,
) -> dict[str, Any]:
    """Enrich all objects in a single scene with parallel LLM calls."""
    cg_path = scene_root / "conceptgraph"
    scene_info_path = cg_path / "scene_info.json"
    clip_id = scene_root.name
    if scene_info_path.exists():
        si = json.loads(scene_info_path.read_text(encoding="utf-8"))
        clip_id = si.get("clip_id", clip_id)

    logger.info(f"[Enrich] Processing scene: {clip_id}")

    objects, visibility, _data, build_info = load_scene_graph(cg_path)
    stride = get_process_stride(build_info)
    # Normalize keys to int (pickle may store string keys)
    raw_o2v = visibility.get("object_to_views", {})
    object_to_views = {int(k): v for k, v in raw_o2v.items()}

    # Determine max_view_id for frame path resolution
    max_view_id = 0
    for views in object_to_views.values():
        for view_id, _ in views:
            max_view_id = max(max_view_id, int(view_id))

    raw_path = scene_root / "raw"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw frames directory not found: {raw_path}. "
            f"Expected RGB frames at {raw_path}/NNNNNN-rgb.png"
        )

    view_paths = resolve_view_image_paths(raw_path, stride, max_view_id)

    output_path = cg_path / "enriched_objects.json"
    existing = _load_existing(output_path) if not force else {}

    if existing:
        logger.info(f"[Enrich] Resuming: {len(existing)} objects already done")

    # Build tasks
    all_results: list[dict[str, Any]] = list(existing.values())
    tasks: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []

    for obj_id, obj in enumerate(objects):
        if obj_id in existing:
            continue
        frames = select_best_frames(obj_id, obj, object_to_views, view_paths)
        if not frames:
            all_results.append(
                {
                    "obj_id": obj_id,
                    "status": "skipped",
                    "original_label": object_label(obj, obj_id),
                    "reason": "no_visible_frames",
                }
            )
            continue
        tasks.append((obj_id, obj, frames))

    if not tasks:
        logger.info(f"[Enrich] {clip_id}: nothing to process")
        _save_output(output_path, clip_id, all_results, len(objects))
        return {"clip_id": clip_id, "processed": 0, "total": len(objects)}

    logger.info(
        f"[Enrich] {clip_id}: processing {len(tasks)} objects "
        f"({len(existing)} existing, {len(objects)} total)"
    )

    pool = GeminiClientPool.get_instance()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                enrich_single_object, pool, obj_id, obj, frames, max_retries
            ): obj_id
            for obj_id, obj, frames in tasks
        }
        done_count = 0
        for future in as_completed(futures):
            result = future.result()
            with _save_lock:
                all_results.append(result)
                _save_output(output_path, clip_id, all_results, len(objects))
            done_count += 1
            status = result["status"]
            oid = result["obj_id"]
            if status == "success":
                cat = result["enrichment"].get("category", "?")
                logger.info(
                    f"[Enrich] {clip_id} obj {oid}: {cat} "
                    f"({done_count}/{len(tasks)})"
                )
            else:
                logger.warning(
                    f"[Enrich] {clip_id} obj {oid}: {status} "
                    f"({done_count}/{len(tasks)})"
                )

    summary = {
        "clip_id": clip_id,
        "processed": len(tasks),
        "total": len(objects),
        "enriched": sum(1 for r in all_results if r["status"] == "success"),
        "failed": sum(1 for r in all_results if r["status"] == "failed"),
    }
    logger.info(f"[Enrich] {clip_id} done: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def discover_scenes(scannet_root: Path) -> list[Path]:
    """Find all scene directories that have conceptgraph data."""
    scenes = []
    for d in sorted(scannet_root.iterdir()):
        if d.is_dir() and (d / "conceptgraph" / "pcd_saves").is_dir():
            scenes.append(d)
    return scenes


def enrich_batch(
    scannet_root: Path,
    clip_ids: list[str] | None = None,
    max_workers: int = 7,
    max_retries: int = 10,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Process multiple scenes sequentially (objects parallel within each)."""
    scenes = discover_scenes(scannet_root)
    if clip_ids:
        clip_set = set(clip_ids)
        scenes = [s for s in scenes if s.name in clip_set]

    logger.info(f"[Enrich] Batch: {len(scenes)} scenes to process")
    summaries = []
    for scene_root in scenes:
        summary = enrich_scene(
            scene_root,
            max_workers=max_workers,
            max_retries=max_retries,
            force=force,
        )
        summaries.append(summary)

    total_enriched = sum(s.get("enriched", 0) for s in summaries)
    total_failed = sum(s.get("failed", 0) for s in summaries)
    logger.info(
        f"[Enrich] Batch complete: {len(summaries)} scenes, "
        f"{total_enriched} enriched, {total_failed} failed"
    )
    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich ConceptGraph objects with LLM-generated metadata."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scene_path",
        type=Path,
        help="Path to a single scene directory (e.g. data/OpenEQA/scannet/<clip_id>)",
    )
    group.add_argument(
        "--scannet_root",
        type=Path,
        help="Root directory containing scene directories for batch mode",
    )
    parser.add_argument(
        "--clip_ids",
        nargs="*",
        help="Specific clip IDs to process (batch mode only)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=7,
        help="Max parallel LLM requests per scene (default: 7)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Max retries per object (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-process, ignoring existing results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scene_path:
        enrich_scene(
            args.scene_path,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            force=args.force,
        )
    else:
        enrich_batch(
            args.scannet_root,
            clip_ids=args.clip_ids,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            force=args.force,
        )


if __name__ == "__main__":
    main()
