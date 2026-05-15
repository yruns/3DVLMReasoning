"""v9.1 mark_frame_with_bbox tool: high-contrast annotated single-frame view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from langchain_core.tools import BaseTool, tool

from agents.catalog import SceneProposal
from agents.runtime.scene_runtime import get_scene_catalog, queue_pending_image
from agents.tools.scene_perception import _gate

BBOX_PALETTE: list[tuple[int, int, int]] = [
    (34, 197, 94),  # green
    (239, 68, 68),  # red
    (59, 130, 246),  # blue
    (234, 179, 8),  # yellow
    (168, 85, 247),  # purple
]
BLACK_OUTLINE_PAD: int = 2


def _bbox_stroke_thickness(img_width: int) -> int:
    return max(5, img_width // 200)


def _draw_palette_bbox(
    img: np.ndarray,
    bbox: tuple[float, float, float, float],
    colour: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    stroke = _bbox_stroke_thickness(img.shape[1])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), stroke + 2 * BLACK_OUTLINE_PAD)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, stroke)


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _filter_visible(
    proposals: list[SceneProposal],
    frame_id: int,
    labels: list[str],
    ids: list[int],
) -> list[SceneProposal]:
    wanted_cat = {_norm_category(c) for c in labels if c}
    wanted_ids = {int(i) for i in ids}
    out: list[SceneProposal] = []
    for p in proposals:
        if frame_id not in p.frame_views:
            continue
        cat_match = bool(wanted_cat) and _norm_category(p.category) in wanted_cat
        id_match = p.proposal_id in wanted_ids
        if cat_match or id_match:
            out.append(p)
    return out


def build_mark_frame_with_bbox_tool(runtime: Any) -> BaseTool:
    @tool
    def mark_frame_with_bbox(
        frame_id: int,
        labels: list[str] | None = None,
        ids: list[int] | None = None,
    ) -> str:
        """Render one first-person frame with high-contrast bboxes for the labels / ids you name.

        Requires at least one of `labels` (category names) or `ids` (proposal ids).
        Detailed usage in 'scene-exploration-playbook'.
        """
        labels_in = [str(c) for c in (labels or []) if c]
        ids_in = [int(i) for i in (ids or [])]
        request = {"frame_id": int(frame_id), "labels": labels_in, "ids": ids_in}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("mark_frame_with_bbox", request, gate)
            return gate
        if not labels_in and not ids_in:
            err = (
                "ERROR: mark_frame_with_bbox requires at least one of {labels, ids}; "
                "to view plain RGB, request the frame via a selector"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        if int(frame_id) not in {int(f) for f in catalog.valid_frame_ids}:
            err = (
                f"ERROR: frame_id={frame_id} not in valid_frame_ids; "
                f"available[:20]={sorted(int(f) for f in catalog.valid_frame_ids)[:20]}"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        proposals_here = [p for p in catalog.proposals if int(frame_id) in p.frame_views]
        visible = _filter_visible(proposals_here, int(frame_id), labels_in, ids_in)
        if not visible:
            err = (
                f"ERROR: no visible proposals matched filters for frame_id={frame_id}; "
                f"filtered_by={{'labels': {labels_in}, 'ids': {ids_in}}}; "
                f"visible_proposals={[p.proposal_id for p in proposals_here]}"
            )
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        from PIL import Image

        raw_path = Path(visible[0].frame_views[int(frame_id)].raw_rgb_path)
        if not raw_path.exists():
            err = f"ERROR: raw RGB image not found: {raw_path}"
            runtime.record("mark_frame_with_bbox", request, err)
            return err
        img = np.asarray(Image.open(raw_path).convert("RGB")).copy()
        for idx, prop in enumerate(visible):
            colour = BBOX_PALETTE[idx % len(BBOX_PALETTE)]
            _draw_palette_bbox(img, prop.frame_views[int(frame_id)].bbox_2d, colour)
        catalog_dir = Path(catalog.bev_image_path).parent
        cache_dir = catalog_dir / "filtered_marks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ids_token = "_".join(str(p.proposal_id) for p in sorted(visible, key=lambda q: q.proposal_id))
        out_path = cache_dir / f"frame_{int(frame_id)}_ids_{ids_token}.png"
        Image.fromarray(img).save(out_path, format="PNG")
        queue_pending_image(runtime, str(out_path))
        left_to_right_pairs = sorted(
            visible,
            key=lambda p: (
                (p.frame_views[int(frame_id)].bbox_2d[0]
                 + p.frame_views[int(frame_id)].bbox_2d[2]) / 2.0,
                p.proposal_id,
            ),
        )
        visible_ids = [p.proposal_id for p in left_to_right_pairs]
        categories = [p.category for p in left_to_right_pairs]
        left_to_right = [f"{p.proposal_id}:{p.category}" for p in left_to_right_pairs]
        boxes_2d = {
            p.proposal_id: [int(round(v)) for v in p.frame_views[int(frame_id)].bbox_2d]
            for p in left_to_right_pairs
        }
        body = (
            f"frame_id={frame_id} mark image at {out_path}; "
            f"filtered_by={{'labels': {labels_in}, 'ids': {ids_in}}}; "
            f"visible_proposals={visible_ids}; "
            f"categories={categories}; "
            f"left_to_right={left_to_right}; "
            f"boxes_2d={boxes_2d}"
        )
        runtime.record("mark_frame_with_bbox", request, body)
        return body

    return mark_frame_with_bbox


__all__ = [
    "BBOX_PALETTE",
    "BLACK_OUTLINE_PAD",
    "build_mark_frame_with_bbox_tool",
]
