"""EmbodiedScan VG tools. All bodies FAIL-LOUD on missing primary skill."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

PRIMARY_SKILL = "vg-grounding-playbook"
CLIP_VISIBLE_OVERFLOW_K = 3
SPATIAL_RELATIONS = (
    "closest_to",
    "near",
    "next_to",
    "farthest_from",
    "above",
    "below",
    "left_of",
    "right_of",
)
_SPATIAL_RELATION_ALIASES = {
    "closer_to": "closest_to",
    "closest": "closest_to",
    "nearest": "closest_to",
    "nearest_to": "closest_to",
    "nearer_to": "closest_to",
    "farther_from": "farthest_from",
    "further_from": "farthest_from",
    "furthest_from": "farthest_from",
    "furthest": "farthest_from",
    "farthest": "farthest_from",
    "far_from": "farthest_from",
    "near_to": "near",
    "nearby": "near",
    "beside": "next_to",
    "adjacent": "next_to",
    "adjacent_to": "next_to",
}


def _canonical_spatial_relation(relation: str) -> str:
    norm = str(relation or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _SPATIAL_RELATION_ALIASES.get(norm, norm)


def _gate(runtime: Any) -> str | None:
    """Return ERROR string if skill not loaded, else None."""
    if PRIMARY_SKILL not in runtime.skills_loaded:
        return f"ERROR: load_skill({PRIMARY_SKILL!r}) before calling this tool."
    return None


def _marked_frame_geometry(ctx: Any, frame_id: int, visible: list[int]) -> str:
    """Compact 2D mark geometry for a viewed annotated frame."""
    proposal_by_id = {p.id: p for p in ctx.proposals}
    rows: list[tuple[float, int, str, tuple[int, int, int, int]]] = []
    for proposal_id in visible:
        proposal = proposal_by_id.get(proposal_id)
        if proposal is None:
            continue
        view = proposal.frame_views.get(frame_id)
        if view is None:
            continue
        x1, y1, x2, y2 = view.bbox_2d
        center_x = (float(x1) + float(x2)) / 2.0
        rows.append((center_x, proposal_id, proposal.category, (x1, y1, x2, y2)))
    if not rows:
        return ""

    rows.sort(key=lambda item: item[0])
    left_to_right = [
        f"{proposal_id}:{category}@x={center_x:.1f}"
        for center_x, proposal_id, category, _ in rows
    ]
    boxes = {
        proposal_id: [int(x1), int(y1), int(x2), int(y2)]
        for _, proposal_id, _, (x1, y1, x2, y2) in rows
    }
    return f"; left_to_right={left_to_right}; boxes_2d={boxes}"


def _left_to_right_entries(
    ctx: Any, frame_id: int, visible: Sequence[int]
) -> list[str]:
    proposal_by_id = {p.id: p for p in ctx.proposals}
    rows: list[tuple[float, int, str]] = []
    missing_geometry = False
    for order, proposal_id in enumerate(visible):
        proposal = proposal_by_id.get(int(proposal_id))
        if proposal is None:
            continue
        view = proposal.frame_views.get(int(frame_id))
        if view is None:
            missing_geometry = True
            rows.append((float(order), int(proposal_id), proposal.category))
            continue
        rows.append((_box_center_x(view.bbox_2d), int(proposal_id), proposal.category))

    if not rows:
        return []
    if not missing_geometry:
        rows.sort(key=lambda item: item[0])
    return [f"#{proposal_id} {category}" for _, proposal_id, category in rows]


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        items: list[int] = []
        for chunk in value.replace(",", " ").split():
            try:
                items.append(int(chunk))
            except ValueError:
                continue
        return items
    if isinstance(value, dict):
        for key in ("proposal_ids", "ids", "visible_proposal_ids"):
            if key in value:
                return _coerce_int_list(value[key])
        return []
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            items.extend(_coerce_int_list(item))
        return items
    return []


def _dedupe_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _frame_inventory_payload(
    ctx: Any,
    frame_id: int,
    visible: Sequence[int],
) -> dict[str, Any]:
    proposal_by_id = {int(p.id): p for p in ctx.proposals}
    boxes: dict[int, list[int]] = {}
    categories: dict[int, str] = {}
    for proposal_id in visible:
        proposal = proposal_by_id.get(int(proposal_id))
        if proposal is None:
            continue
        categories[int(proposal_id)] = proposal.category
        view = proposal.frame_views.get(int(frame_id))
        if view is not None:
            boxes[int(proposal_id)] = [int(v) for v in view.bbox_2d]
    return {
        "frame_id": int(frame_id),
        "visible_proposal_ids": [int(proposal_id) for proposal_id in visible],
        "left_to_right": _left_to_right_entries(ctx, int(frame_id), visible),
        "categories": categories,
        "boxes_2d": boxes,
    }


def _filter_visible_proposals(
    ctx: Any,
    frame_id: int,
    visible: Sequence[int],
    categories: list[str],
    proposal_ids: list[int],
) -> list[int]:
    if not categories and not proposal_ids:
        return [int(proposal_id) for proposal_id in visible]

    wanted_categories = {_norm_category(category) for category in categories}
    wanted_ids = set(proposal_ids)
    proposal_by_id = {int(p.id): p for p in ctx.proposals}
    matched: list[int] = []
    for proposal_id_raw in visible:
        proposal_id = int(proposal_id_raw)
        proposal = proposal_by_id.get(proposal_id)
        if proposal is None or int(frame_id) not in proposal.frame_views:
            continue
        category_match = (
            bool(wanted_categories)
            and _norm_category(proposal.category) in wanted_categories
        )
        id_match = proposal_id in wanted_ids
        if category_match or id_match:
            matched.append(proposal_id)
    return matched


def _resolve_image_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return Path.cwd() / path


def _mark_color(index: int) -> tuple[int, int, int]:
    palette = [
        (34, 197, 94),
        (239, 68, 68),
        (59, 130, 246),
        (234, 179, 8),
        (168, 85, 247),
        (20, 184, 166),
        (249, 115, 22),
        (236, 72, 153),
    ]
    return palette[index % len(palette)]


def _load_mark_font(image_height: int) -> Any:
    from PIL import ImageFont

    size = max(18, min(34, image_height // 28))
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    return draw.textsize(text, font=font)


def _render_filtered_marked_frame(
    ctx: Any,
    frame_id: int,
    visible: Sequence[int],
) -> Path:
    from PIL import Image, ImageDraw

    proposal_by_id = {int(p.id): p for p in ctx.proposals}
    first_view = None
    for proposal_id in visible:
        proposal = proposal_by_id.get(int(proposal_id))
        if proposal is None:
            continue
        first_view = proposal.frame_views.get(int(frame_id))
        if first_view is not None:
            break
    if first_view is None:
        raise ValueError(f"no 2D geometry for frame_id={frame_id}")

    raw_path = _resolve_image_path(Path(first_view.raw_rgb_path))
    if not raw_path.exists():
        raise FileNotFoundError(f"raw RGB image not found: {raw_path}")

    image = Image.open(raw_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_mark_font(image.height)
    line_width = max(4, image.width // 320)
    label_pad = max(3, line_width)

    for index, proposal_id_raw in enumerate(visible):
        proposal_id = int(proposal_id_raw)
        proposal = proposal_by_id.get(proposal_id)
        if proposal is None:
            continue
        view = proposal.frame_views.get(int(frame_id))
        if view is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in view.bbox_2d]
        color = _mark_color(index)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = f"#{proposal_id} {proposal.category}"
        text_w, text_h = _text_size(draw, label, font)
        label_x1 = max(0, min(x1, image.width - text_w - 2 * label_pad))
        label_y2 = max(text_h + 2 * label_pad, y1)
        label_y1 = max(0, label_y2 - text_h - 2 * label_pad)
        label_x2 = label_x1 + text_w + 2 * label_pad
        draw.rectangle((label_x1, label_y1, label_x2, label_y2), fill=(0, 0, 0))
        draw.text(
            (label_x1 + label_pad, label_y1 + label_pad),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    out_dir = ctx.annotated_image_dir.parent / "filtered_marks"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = "_".join(str(int(proposal_id)) for proposal_id in visible)
    out_path = out_dir / f"frame_{int(frame_id)}_ids_{ids}.png"
    image.save(out_path, format="PNG")
    return out_path


def _box_center_x(box: tuple[int, int, int, int]) -> float:
    x1, _, x2, _ = box
    return (float(x1) + float(x2)) / 2.0


def _coviewed_horizontal_votes(candidate: Any, anchor: Any) -> dict[str, Any]:
    """Return 2D left/right evidence from frames where both proposals appear."""
    shared_frame_ids = sorted(
        set(getattr(candidate, "frame_views", {}) or {})
        & set(getattr(anchor, "frame_views", {}) or {})
    )
    offsets: list[float] = []
    for frame_id in shared_frame_ids:
        candidate_view = candidate.frame_views.get(frame_id)
        anchor_view = anchor.frame_views.get(frame_id)
        if candidate_view is None or anchor_view is None:
            continue
        offsets.append(
            _box_center_x(candidate_view.bbox_2d) - _box_center_x(anchor_view.bbox_2d)
        )
    return {
        "shared_frame_count": len(offsets),
        "mean_2d_center_offset_x": (
            round(sum(offsets) / len(offsets), 6) if offsets else None
        ),
        "left_frame_count": sum(1 for offset in offsets if offset < 0.0),
        "right_frame_count": sum(1 for offset in offsets if offset > 0.0),
    }


def _supporting_frame_count(coview: dict[str, Any], relation: str) -> int:
    if relation == "left_of":
        return int(coview["left_frame_count"])
    if relation == "right_of":
        return int(coview["right_frame_count"])
    return 0


def _contradicting_frame_count(coview: dict[str, Any], relation: str) -> int:
    if relation == "left_of":
        return int(coview["right_frame_count"])
    if relation == "right_of":
        return int(coview["left_frame_count"])
    return 0


def _left_right_bucket(coview: dict[str, Any], *, direction: str) -> int:
    """Sort confirmed relation first, unknown second, contradicted last."""
    relation = "left_of" if direction == "left" else "right_of"
    supporting = _supporting_frame_count(coview, relation)
    contradicting = _contradicting_frame_count(coview, relation)
    if supporting > contradicting and supporting > 0:
        return 0
    if int(coview["shared_frame_count"]) == 0:
        return 1
    return 2


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _coerce_category_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        for key in ("categories", "target_categories", "parser_categories"):
            if key in value:
                return _coerce_category_list(value[key])
        return []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_coerce_category_list(item))
        return items
    return []


def _dedupe_categories(categories: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for category in categories:
        norm = _norm_category(category)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(str(category).strip())
    return out


def build_vg_tools(runtime: Any) -> list[BaseTool]:
    ctx = runtime.task_ctx  # VgEmbodiedScanCtx

    @tool
    def list_frame_proposals(frame_id: int) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        request = {"frame_id": frame_id}
        if gate is not None:
            runtime.record("list_frame_proposals", request, gate)
            return gate
        if frame_id not in ctx.frame_index:
            err = (
                f"ERROR: frame_id={frame_id} not in proposal index; "
                f"available: {sorted(ctx.frame_index.keys())[:20]}"
            )
            runtime.record("list_frame_proposals", request, err)
            return err
        visible = ctx.frame_index[int(frame_id)]
        text = json.dumps(
            _frame_inventory_payload(ctx, int(frame_id), visible),
            ensure_ascii=False,
        )
        runtime.record("list_frame_proposals", request, text)
        return text

    # v9.1: the legacy pack-local marked-frame tool was retired together
    # with the unified frame-injection tool; `agents.tools.mark_frame_with_bbox`
    # is the canonical high-contrast annotated-zoom tool, wired for both VG
    # and QA in DeepAgentsStage2Runtime.build_runtime_tools.

    @tool
    def inspect_proposal(proposal_id: int) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("inspect_proposal", {"proposal_id": proposal_id}, gate)
            return gate
        proposal = next((p for p in ctx.proposals if p.id == proposal_id), None)
        if proposal is None:
            err = (
                f"ERROR: proposal_id={proposal_id} not in pool; "
                f"available count={len(ctx.proposals)}"
            )
            runtime.record("inspect_proposal", {"proposal_id": proposal_id}, err)
            return err
        payload = {
            "proposal_id": proposal.id,
            "category": proposal.category,
            "score": proposal.score,
            "bbox_3d_9dof": list(proposal.bbox_3d_9dof),
            "frames_appeared": ctx.proposal_index.get(proposal_id, []),
            "source": ctx.proposal_pool_source,
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("inspect_proposal", {"proposal_id": proposal_id}, text)
        return text

    @tool
    def compare_proposals_spatial(
        candidate_ids: list[int],
        anchor_id: int,
        relation: str,
    ) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        requested_relation = str(relation or "")
        normalized_requested_relation = (
            requested_relation.strip().lower().replace("-", "_").replace(" ", "_")
        )
        relation = _canonical_spatial_relation(requested_relation)
        request = {
            "candidate_ids": candidate_ids,
            "anchor_id": anchor_id,
            "relation": requested_relation,
        }
        if relation != normalized_requested_relation:
            request["canonical_relation"] = relation
        if gate is not None:
            runtime.record("compare_proposals_spatial", request, gate)
            return gate
        if relation not in SPATIAL_RELATIONS:
            err = f"ERROR: unsupported relation {relation!r}; allowed: " + " | ".join(
                SPATIAL_RELATIONS
            )
            runtime.record("compare_proposals_spatial", request, err)
            return err

        import numpy as np

        anchor = next((p for p in ctx.proposals if p.id == anchor_id), None)
        if anchor is None:
            err = f"ERROR: anchor_id={anchor_id} not in pool"
            runtime.record("compare_proposals_spatial", request, err)
            return err
        candidates = [p for p in ctx.proposals if p.id in set(candidate_ids)]
        missing = sorted(set(candidate_ids) - {p.id for p in candidates})
        if missing:
            err = f"ERROR: candidate ids not in pool: {missing}"
            runtime.record("compare_proposals_spatial", request, err)
            return err

        anchor_center = np.array(anchor.bbox_3d_9dof[:3], dtype=float)
        scored = []
        for p in candidates:
            center = np.array(p.bbox_3d_9dof[:3], dtype=float)
            delta = center - anchor_center
            distance = float(np.linalg.norm(delta))
            horizontal_distance = float(np.linalg.norm(delta[:2]))
            vertical_offset = float(delta[2])
            horizontal_offset = float(delta[0])
            coview = _coviewed_horizontal_votes(p, anchor)
            scored.append(
                (
                    p.id,
                    distance,
                    horizontal_distance,
                    vertical_offset,
                    horizontal_offset,
                    coview,
                )
            )
        if relation == "closest_to":
            scored.sort(key=lambda x: x[1])
        elif relation in ("near", "next_to"):
            scored.sort(key=lambda x: (x[2], x[1]))
        elif relation == "farthest_from":
            scored.sort(key=lambda x: x[1], reverse=True)
        elif relation == "above":
            scored.sort(key=lambda x: (x[3] <= 0.0, -x[3], x[2]))
        elif relation == "below":
            scored.sort(key=lambda x: (x[3] >= 0.0, x[3], x[2]))
        elif relation == "left_of":
            scored.sort(
                key=lambda x: (
                    _left_right_bucket(x[5], direction="left"),
                    -int(x[5]["left_frame_count"]),
                    int(x[5]["right_frame_count"]),
                    (
                        abs(float(x[5]["mean_2d_center_offset_x"]))
                        if x[5]["mean_2d_center_offset_x"] is not None
                        else float("inf")
                    ),
                    x[2],
                )
            )
        else:  # right_of
            scored.sort(
                key=lambda x: (
                    _left_right_bucket(x[5], direction="right"),
                    -int(x[5]["right_frame_count"]),
                    int(x[5]["left_frame_count"]),
                    (
                        abs(float(x[5]["mean_2d_center_offset_x"]))
                        if x[5]["mean_2d_center_offset_x"] is not None
                        else float("inf")
                    ),
                    x[2],
                )
            )
        payload = {
            "anchor_id": anchor_id,
            "relation": relation,
            "requested_relation": requested_relation,
            "ranked_ids": [pid for pid, _, _, _, _, _ in scored],
            "distances": [d for _, d, _, _, _, _ in scored],
            "horizontal_distances": [d for _, _, d, _, _, _ in scored],
            "vertical_offsets": [z for _, _, _, z, _, _ in scored],
            "x_offsets": [x for _, _, _, _, x, _ in scored],
            "shared_frame_counts": [
                int(coview["shared_frame_count"]) for _, _, _, _, _, coview in scored
            ],
            "mean_2d_center_offsets_x": [
                coview["mean_2d_center_offset_x"] for _, _, _, _, _, coview in scored
            ],
            "supporting_frame_counts": [
                _supporting_frame_count(coview, relation)
                for _, _, _, _, _, coview in scored
            ],
            "contradicting_frame_counts": [
                _contradicting_frame_count(coview, relation)
                for _, _, _, _, _, coview in scored
            ],
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("compare_proposals_spatial", request, text)
        return text

    return [
        list_frame_proposals,
        inspect_proposal,
        compare_proposals_spatial,
    ]


__all__ = ["build_vg_tools", "PRIMARY_SKILL"]
