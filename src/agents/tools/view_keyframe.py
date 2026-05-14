"""Unified view_keyframe tool (v9): mode='rgb' | 'marked' | 'auto', with selective filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool, tool

from agents.catalog import SceneProposal
from agents.core.agent_config import Stage2TaskType
from agents.runtime.scene_runtime import get_scene_catalog, queue_pending_image
from agents.tools.scene_perception import _gate


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        out: list[int] = []
        for chunk in value.replace(",", " ").split():
            try:
                out.append(int(chunk))
            except ValueError:
                continue
        return out
    if isinstance(value, (list, tuple, set)):
        out_seq: list[int] = []
        for item in value:
            out_seq.extend(_coerce_int_list(item))
        return out_seq
    return []


def _coerce_category_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item)
        return out
    return []


def _resolve_mode(mode: str, task_type: Stage2TaskType | None) -> Literal["rgb", "marked"]:
    if mode == "rgb":
        return "rgb"
    if mode == "marked":
        return "marked"
    if mode == "auto":
        if task_type == Stage2TaskType.VISUAL_GROUNDING:
            return "marked"
        return "rgb"
    raise ValueError(f"mode must be 'rgb' | 'marked' | 'auto'; got {mode!r}")


def _filter_visible(
    proposals: list[SceneProposal],
    frame_id: int,
    categories: list[str],
    proposal_ids: list[int],
) -> list[SceneProposal]:
    if not categories and not proposal_ids:
        return [p for p in proposals if frame_id in p.frame_views]
    wanted_cat = {_norm_category(c) for c in categories}
    wanted_ids = {int(i) for i in proposal_ids}
    out: list[SceneProposal] = []
    for p in proposals:
        if frame_id not in p.frame_views:
            continue
        cat_match = bool(wanted_cat) and _norm_category(p.category) in wanted_cat
        id_match = p.proposal_id in wanted_ids
        if cat_match or id_match:
            out.append(p)
    return out


def _render_filtered_marked(
    proposals: list[SceneProposal],
    frame_id: int,
    cache_dir: Path,
) -> Path:
    from PIL import Image, ImageDraw

    first = next((p for p in proposals if frame_id in p.frame_views), None)
    if first is None:
        raise ValueError(f"no 2D geometry for frame_id={frame_id}")
    raw_path = Path(first.frame_views[frame_id].raw_rgb_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw RGB image not found: {raw_path}")
    img = Image.open(raw_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    palette = [(34, 197, 94), (239, 68, 68), (59, 130, 246), (234, 179, 8), (168, 85, 247)]
    for idx, p in enumerate(proposals):
        view = p.frame_views[frame_id]
        x1, y1, x2, y2 = view.bbox_2d
        color = palette[idx % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, img.width // 320))
        label = f"#{p.proposal_id} {p.category}"
        try:
            tw, th = draw.textbbox((0, 0), label)[2:]
        except AttributeError:
            tw, th = draw.textsize(label)
        draw.rectangle((x1, max(0, y1 - th - 4), x1 + tw + 4, y1), fill=(0, 0, 0))
        draw.text((x1 + 2, max(0, y1 - th - 2)), label, fill=(255, 255, 255))
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = "_".join(str(p.proposal_id) for p in proposals)
    out = cache_dir / f"frame_{frame_id}_ids_{ids or 'all'}.png"
    img.save(out, format="PNG")
    return out


def _left_to_right_entries(visible: list[SceneProposal], frame_id: int) -> list[str]:
    rows: list[tuple[float, int, str]] = []
    for p in visible:
        view = p.frame_views[frame_id]
        x1, _, x2, _ = view.bbox_2d
        cx = (float(x1) + float(x2)) / 2.0
        rows.append((cx, p.proposal_id, p.category))
    rows.sort(key=lambda item: item[0])
    return [f"#{pid} {cat}" for _, pid, cat in rows]


def _resolve_raw_rgb_path(catalog: Any, frame_id: int) -> Path | None:
    for p in catalog.proposals:
        view = p.frame_views.get(int(frame_id))
        if view is not None and view.raw_rgb_path:
            return Path(view.raw_rgb_path)
    return None


def build_view_keyframe_tool(runtime: Any) -> BaseTool:
    @tool
    def view_keyframe(
        frame_id: int,
        mode: str = "auto",
        categories: list[str] | str | None = None,
        proposal_ids: list[int] | int | str | None = None,
    ) -> str:
        """Inject one first-person frame. mode='auto' uses task_type to pick rgb (QA) or marked (VG)."""
        category_filter = _coerce_category_list(categories)
        id_filter = _coerce_int_list(proposal_ids)
        request = {
            "frame_id": int(frame_id),
            "mode": mode,
            "categories": category_filter,
            "proposal_ids": id_filter,
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("view_keyframe", request, gate)
            return gate
        try:
            resolved_mode = _resolve_mode(mode, runtime.task_type)
        except ValueError as exc:
            err = f"ERROR: {exc}"
            runtime.record("view_keyframe", request, err)
            return err

        catalog = get_scene_catalog(runtime)
        proposals_at_frame = [p for p in catalog.proposals if int(frame_id) in p.frame_views]
        if not proposals_at_frame and resolved_mode == "marked":
            err = (
                f"ERROR: frame_id={frame_id} has no visible proposals in scene_catalog; "
                f"available frames: {sorted({fid for p in catalog.proposals for fid in p.frame_views})[:20]}"
            )
            runtime.record("view_keyframe", request, err)
            return err

        if resolved_mode == "rgb":
            raw_path = _resolve_raw_rgb_path(catalog, int(frame_id))
            if raw_path is None:
                err = (
                    f"ERROR: frame_id={frame_id} has no raw RGB reference; "
                    f"valid_frame_ids[:20]={catalog.valid_frame_ids[:20]}"
                )
                runtime.record("view_keyframe", request, err)
                return err
            queue_pending_image(runtime, str(raw_path))
            body = f"frame_id={frame_id} rgb image at {raw_path}"
            runtime.record("view_keyframe", request, body)
            return body

        # mode == 'marked'
        visible = _filter_visible(proposals_at_frame, int(frame_id), category_filter, id_filter)
        if (category_filter or id_filter) and not visible:
            err = (
                f"ERROR: no visible proposals matched filters for frame_id={frame_id}; "
                f"filtered_by={{'categories': {category_filter}, 'proposal_ids': {id_filter}}}; "
                f"visible_proposals={[p.proposal_id for p in proposals_at_frame]}"
            )
            runtime.record("view_keyframe", request, err)
            return err
        if not visible:
            visible = proposals_at_frame
        annotated_dir = runtime.bundle.extra_metadata.get("annotated_image_dir") if runtime.bundle.extra_metadata else None
        has_filters = bool(category_filter or id_filter)
        if not has_filters and annotated_dir:
            marked_path = Path(annotated_dir) / f"frame_{int(frame_id)}.png"
            if not marked_path.exists():
                err = f"ERROR: annotated image not found: {marked_path}"
                runtime.record("view_keyframe", request, err)
                return err
            prefix = "marked image"
        else:
            cache_dir = Path(catalog.bev_image_path).parent / "filtered_marks"
            try:
                marked_path = _render_filtered_marked(visible, int(frame_id), cache_dir)
            except Exception as exc:  # noqa: BLE001 — fail-loud with explicit type
                err = f"ERROR: marked image render failed for frame_id={frame_id}: {type(exc).__name__}: {exc}"
                runtime.record("view_keyframe", request, err)
                return err
            prefix = "filtered marked image" if has_filters else "marked image"
        queue_pending_image(runtime, str(marked_path))
        ltr = _left_to_right_entries(visible, int(frame_id))
        boxes = {p.proposal_id: list(p.frame_views[int(frame_id)].bbox_2d) for p in visible}
        cat_list = [p.category for p in visible]
        filter_text = ""
        if has_filters:
            filter_text = (
                f" filtered_by={{'categories': {category_filter}, 'proposal_ids': {id_filter}}};"
            )
        body = (
            f"frame_id={frame_id} {prefix} at {marked_path};"
            f"{filter_text} "
            f"visible_proposals={[p.proposal_id for p in visible]}; "
            f"categories={cat_list}; "
            f"left_to_right={ltr}; "
            f"boxes_2d={boxes}"
        )
        runtime.record("view_keyframe", request, body)
        return body

    return view_keyframe


__all__ = ["build_view_keyframe_tool"]
