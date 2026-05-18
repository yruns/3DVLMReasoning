"""v9.1 mark_frame_with_bbox tool: high-contrast annotated single-frame view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from langchain_core.tools import BaseTool, tool

from agents.catalog import SceneProposal
from agents.runtime.scene_runtime import get_scene_catalog, make_tool_image_ref
from agents.tools.scene_perception import _gate

BBOX_PALETTE: list[tuple[int, int, int]] = [
    (34, 197, 94),  # green
    (239, 68, 68),  # red
    (59, 130, 246),  # blue
    (234, 179, 8),  # yellow
    (168, 85, 247),  # purple
]
BLACK_OUTLINE_PAD: int = 2
LABEL_PADDING_PX: int = 5

# v9.3 user-requested defaults (see commit log):
# - Label font ~20 % larger than the pre-v9.3 `max(0.7, w/1800)` scale.
# - Label always anchored at the top-left of the bbox (inside if it fits,
#   otherwise just above the bbox top, left-edge-aligned).
# - Label background colour matches the bbox stroke colour; text colour
#   picked from black/white by perceived luminance so contrast is preserved.
#
# Kept here as module-level constants so tests can reference them directly.
LABEL_FONT_SCALE_FLOOR: float = 0.84
LABEL_FONT_SCALE_REF_WIDTH: int = 1500  # px width at which scale == 1.0
LABEL_FONT_THICKNESS_DIVISOR: int = 500
LABEL_LUMINANCE_BLACK_TEXT_THRESHOLD: float = 140.0
# Legacy area-ratio threshold from v9.1 (the "label sits inside vs centred"
# heuristic). v9.3 always anchors top-left, so this is unused by the new
# renderer but is kept exported for any historical-trace tooling that read it.
LABEL_AREA_RATIO_THRESHOLD: float = 0.15


@dataclass(frozen=True)
class LabelAnchor:
    """Where a label is drawn.

    ``mode`` values:
      - ``"topleft_inside"``: top-left corner of the label aligns with the
        top-left corner of the bbox; the label sits **inside** the bbox.
        Used when the bbox is large enough to fit the label.
      - ``"topleft_above"``: bbox is too small to fit the label inside;
        the label is placed just **above** the bbox, left-edge-aligned to
        the bbox's left edge. Keeps the label visually associated with
        the bbox in tight scenes.
    """

    mode: str  # 'topleft_inside' | 'topleft_above'
    origin: tuple[int, int]  # cv2.putText origin (x, baseline-y)
    bg_box: tuple[int, int, int, int]  # (x1, y1, x2, y2) for the opaque rect


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


def _label_font_scale(img_width: int) -> float:
    """v9.3 user-tuned label font scale, ~20 % larger than v9.1.

    - 1296 px (ScanNet raw RGB):   scale ≈ 0.86 (was 0.72)
    - 1500 px:                     scale  = 1.00 (was 0.83)
    - 1920 px:                     scale ≈ 1.28 (was 1.07)
    """
    return max(LABEL_FONT_SCALE_FLOOR, img_width / LABEL_FONT_SCALE_REF_WIDTH)


def _label_text_color_for_bg(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text by perceived luminance of the background.

    ITU-R BT.601 luminance: ``0.299 R + 0.587 G + 0.114 B``. Threshold
    chosen so that the v9.3 5-colour palette gets:
      green  (34, 197, 94)  → lum≈136 → white text
      red    (239, 68, 68)  → lum≈119 → white text
      blue   (59, 130, 246) → lum≈121 → white text
      yellow (234, 179, 8)  → lum≈175 → black text
      purple (168, 85, 247) → lum≈128 → white text
    """
    r, g, b = bg
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum > LABEL_LUMINANCE_BLACK_TEXT_THRESHOLD:
        return (0, 0, 0)
    return (255, 255, 255)


def _decide_label_anchor(
    bbox: tuple[float, float, float, float],
    text_w: int,
    text_h: int,
    *,
    img_shape: tuple[int, int] | None = None,
) -> LabelAnchor:
    """v9.3: always anchor at the bbox top-left.

    Default position: inside the bbox, top-left aligned. If the bbox is
    too small to fit the label inside (either dimension), the label is
    placed just ABOVE the bbox top, still left-edge-aligned so the
    visual association with the bbox is preserved.

    ``img_shape``, when provided, clamps the label background to the
    image bounds (so labels on bboxes near the top edge don't go
    negative).
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    pad = LABEL_PADDING_PX
    label_w = text_w + 2 * pad
    label_h = text_h + 2 * pad
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    if label_w <= bbox_w and label_h <= bbox_h:
        bg = (x1, y1, x1 + label_w, y1 + label_h)
        origin = (x1 + pad, y1 + text_h + pad)
        return LabelAnchor(mode="topleft_inside", origin=origin, bg_box=bg)
    # Fallback: place above the bbox top.
    above_top = y1 - label_h
    if img_shape is not None and above_top < 0:
        # If above-top would clip the image, slide the label down to start
        # at row 0; it will overlap the top of the bbox by a few pixels
        # but stays fully visible.
        above_top = 0
    bg = (x1, above_top, x1 + label_w, above_top + label_h)
    origin = (x1 + pad, above_top + text_h + pad)
    return LabelAnchor(mode="topleft_above", origin=origin, bg_box=bg)


def _draw_label(
    img: np.ndarray,
    text: str,
    bbox: tuple[float, float, float, float],
    colour: tuple[int, int, int],
) -> None:
    """v9.3: top-left anchored, ~20 % larger, colour-matched to the bbox.

    The label background is filled with ``colour`` (the bbox stroke
    palette colour); a thin black outline is drawn around the
    background for crisp separation from the underlying frame; the text
    colour is auto-picked (black or white) by luminance.
    """
    font = cv2.FONT_HERSHEY_DUPLEX  # bolder than SIMPLEX at same scale
    scale = _label_font_scale(img.shape[1])
    thickness = max(2, img.shape[1] // LABEL_FONT_THICKNESS_DIVISOR)
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    anchor = _decide_label_anchor(bbox, text_w, text_h, img_shape=img.shape)
    x1, y1, x2, y2 = anchor.bg_box
    # Black halo around the coloured background (gives a sharp edge
    # against any backdrop).
    cv2.rectangle(
        img,
        (x1 - BLACK_OUTLINE_PAD, y1 - BLACK_OUTLINE_PAD),
        (x2 + BLACK_OUTLINE_PAD, y2 + BLACK_OUTLINE_PAD),
        (0, 0, 0),
        -1,
    )
    # Coloured background (matches the bbox stroke palette)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)
    text_color = _label_text_color_for_bg(colour)
    cv2.putText(
        img,
        text,
        anchor.origin,
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


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
        proposals_here = [
            p for p in catalog.proposals if int(frame_id) in p.frame_views
        ]
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
            view = prop.frame_views[int(frame_id)]
            _draw_palette_bbox(img, view.bbox_2d, colour)
            _draw_label(
                img,
                f"#{prop.proposal_id} {prop.category}",
                view.bbox_2d,
                colour=colour,
            )
        catalog_dir = Path(catalog.bev_image_path).parent
        cache_dir = catalog_dir / "filtered_marks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ids_token = "_".join(
            str(p.proposal_id) for p in sorted(visible, key=lambda q: q.proposal_id)
        )
        out_path = cache_dir / f"frame_{int(frame_id)}_ids_{ids_token}.png"
        Image.fromarray(img).save(out_path, format="PNG")
        image_ref = make_tool_image_ref(
            runtime,
            str(out_path),
            metadata={
                "frame_id": int(frame_id),
                "source_tool": "mark_frame_with_bbox",
                "selected_because": f"marked labels={labels_in} ids={ids_in}",
            },
        )
        left_to_right_pairs = sorted(
            visible,
            key=lambda p: (
                (
                    p.frame_views[int(frame_id)].bbox_2d[0]
                    + p.frame_views[int(frame_id)].bbox_2d[2]
                )
                / 2.0,
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
        runtime.record(
            "mark_frame_with_bbox",
            request,
            body,
            image_metadata=[image_ref],
        )
        return body

    return mark_frame_with_bbox


__all__ = [
    "BBOX_PALETTE",
    "BLACK_OUTLINE_PAD",
    "LABEL_AREA_RATIO_THRESHOLD",  # exported for historical-trace tooling; unused by v9.3 renderer
    "LABEL_FONT_SCALE_FLOOR",
    "LABEL_FONT_SCALE_REF_WIDTH",
    "LABEL_FONT_THICKNESS_DIVISOR",
    "LABEL_LUMINANCE_BLACK_TEXT_THRESHOLD",
    "LABEL_PADDING_PX",
    "LabelAnchor",
    "build_mark_frame_with_bbox_tool",
]
