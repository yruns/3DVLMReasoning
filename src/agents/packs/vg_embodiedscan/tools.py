"""EmbodiedScan VG tools. All bodies FAIL-LOUD on missing primary skill."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.packs.vg_embodiedscan.ctx import cumulative_seen_frame_ids

PRIMARY_SKILL = "vg-grounding-playbook"
CLIP_VISIBLE_OVERFLOW_K = 3


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


def _extract_ranked_categories(
    runtime: Any,
    requested_category: str,
) -> list[tuple[int, list[str]]]:
    """Return parser target categories by rank, with the tool argument as rank-1."""
    ranks: dict[int, list[str]] = {}
    extra = getattr(getattr(runtime, "bundle", None), "extra_metadata", {}) or {}

    raw_ranked = extra.get("parser_categories_by_rank") or extra.get(
        "target_categories_by_rank"
    )
    if isinstance(raw_ranked, dict):
        for rank_raw, value in raw_ranked.items():
            try:
                rank = int(rank_raw)
            except (TypeError, ValueError):
                continue
            ranks.setdefault(rank, []).extend(_coerce_category_list(value))
    elif isinstance(raw_ranked, list):
        for idx, item in enumerate(raw_ranked, start=1):
            rank = idx
            if isinstance(item, dict):
                try:
                    rank = int(item.get("rank", idx))
                except (TypeError, ValueError):
                    rank = idx
            ranks.setdefault(rank, []).extend(_coerce_category_list(item))

    hypothesis_output = extra.get("hypothesis_output") or {}
    for item in hypothesis_output.get("hypotheses", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            rank = int(item.get("rank", len(ranks) + 1))
        except (TypeError, ValueError):
            continue
        root = ((item.get("grounding_query") or {}).get("root")) or {}
        ranks.setdefault(rank, []).extend(_coerce_category_list(root))

    hypothesis = getattr(getattr(runtime, "bundle", None), "hypothesis", None)
    if hypothesis is not None:
        ranks.setdefault(1, []).extend(
            _coerce_category_list(getattr(hypothesis, "target_categories", []))
        )

    requested = str(requested_category).strip()
    if requested:
        ranks.setdefault(1, [])
        ranks[1] = [requested, *ranks[1]]

    ranked = [
        (rank, _dedupe_categories(categories))
        for rank, categories in sorted(ranks.items())
        if 1 <= rank <= 3
    ]
    ranked = [(rank, categories) for rank, categories in ranked if categories]
    if ranked:
        return ranked
    if requested:
        return [(1, [requested])]
    return []


def _label_hits_for_categories(
    ctx: Any,
    categories: list[str],
) -> tuple[list[int], list[dict[str, Any]]]:
    wanted = {_norm_category(category) for category in categories}
    ids: list[int] = []
    seen: set[int] = set()
    for proposal in ctx.proposals:
        if proposal.id in seen:
            continue
        if _norm_category(proposal.category) in wanted:
            seen.add(proposal.id)
            ids.append(proposal.id)
    return ids, [{"proposal_id": pid, "source": "label_exact"} for pid in ids]


def _get_clip_provider(runtime: Any) -> Any:
    provider = getattr(runtime, "clip_visible_provider", None)
    if provider is not None:
        return provider

    from agents.packs.vg_embodiedscan.clip_provider import BatchedClipProvider

    provider = BatchedClipProvider(
        backbone=runtime.clip_visible_backbone,
        cache_dir=runtime.clip_visible_cache_dir,
        batch_size=16,
    )
    runtime.clip_visible_provider = provider
    return provider


def _clip_visible_aug_for_rank(
    *,
    runtime: Any,
    ctx: Any,
    rank: int,
    search_category: str,
    categories: list[str],
    label_hit_ids: list[int],
    seen_frame_ids: set[int],
    visible_ids: set[int],
    max_aug: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from agents.packs.vg_embodiedscan.clip_provider import ClipImageRequest

    excluded_ids = set(label_hit_ids)
    requests: list[ClipImageRequest] = []
    for proposal_id in sorted(visible_ids):
        if proposal_id in excluded_ids:
            continue
        views = ctx.frame_views_for(proposal_id, seen_frame_ids)
        if not views:
            continue
        view = views[0]
        requests.append(
            ClipImageRequest(
                scene_id=str(getattr(runtime.bundle, "scene_id", "")),
                proposal_id=proposal_id,
                frame_id=view.frame_id,
                raw_rgb_path=view.raw_rgb_path,
                bbox_2d=view.bbox_2d,
                visibility_weight=view.visibility_weight,
            )
        )
    if not requests:
        return [], []

    tau = float(getattr(runtime, "clip_visible_tau", 0.18))
    provider = _get_clip_provider(runtime)
    best_by_id: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for category in categories:
        for score in provider.score(category, requests):
            if score.clip_score < tau:
                continue
            current = best_by_id.get(score.proposal_id)
            if current is not None and current["clip_score"] >= score.clip_score:
                continue
            best_by_id[score.proposal_id] = {
                "proposal_id": score.proposal_id,
                "clip_score": round(score.clip_score, 6),
                "rank_used": rank,
                "via_category": category,
                "source_frame_id": str(score.frame_id),
                "source": "clip_visible",
                "cvra_category_source": f"rank{rank}",
            }
        failures.extend(getattr(provider, "last_failures", []) or [])

    proposal_by_id = {int(p.id): p for p in ctx.proposals}
    aug = sorted(
        best_by_id.values(),
        key=lambda item: (-float(item["clip_score"]), int(item["proposal_id"])),
    )
    standard_aug = aug[:max_aug]
    search_norm = _norm_category(search_category)
    overflow_aug: list[dict[str, Any]] = []
    for item in aug[max_aug:]:
        proposal = proposal_by_id.get(int(item["proposal_id"]))
        if proposal is None:
            continue
        if search_norm and _norm_category(proposal.category) == search_norm:
            continue
        overflow_item = dict(item)
        overflow_item["cvra_overflow"] = True
        overflow_aug.append(overflow_item)
        if len(overflow_aug) >= CLIP_VISIBLE_OVERFLOW_K:
            break
    return [*standard_aug, *overflow_aug], _dedupe_clip_failures(failures)


def _dedupe_clip_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    out: list[dict[str, Any]] = []
    for failure in failures:
        key = (
            int(failure.get("proposal_id", -1)),
            int(failure.get("frame_id", -1)),
            str(failure.get("raw_rgb_path", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(failure)
    return out


def _find_proposals_by_category_with_cvra(
    *,
    runtime: Any,
    ctx: Any,
    category: str,
) -> dict[str, Any]:
    ranked_categories = _extract_ranked_categories(runtime, category)
    seen_frame_ids = cumulative_seen_frame_ids(runtime)
    visible_ids = ctx.visible_proposal_ids(seen_frame_ids)
    available_categories = sorted({p.category for p in ctx.proposals if p.category})

    rank_fallback_used = False
    tried_any = False
    last_payload: dict[str, Any] | None = None
    hard_cap = max(1, min(int(getattr(runtime, "clip_visible_k_aug", 8)), 10))

    for rank, categories in ranked_categories:
        tried_any = True
        if rank > 1:
            rank_fallback_used = True
        label_hit_ids, label_hits = _label_hits_for_categories(ctx, categories)
        clip_visible_aug, clip_visible_failures = _clip_visible_aug_for_rank(
            runtime=runtime,
            ctx=ctx,
            rank=rank,
            search_category=category,
            categories=categories,
            label_hit_ids=label_hit_ids,
            seen_frame_ids=seen_frame_ids,
            visible_ids=visible_ids,
            max_aug=hard_cap,
        )
        label_hit_id_set = set(label_hit_ids)
        proposal_ids = [
            *label_hit_ids,
            *[
                int(item["proposal_id"])
                for item in clip_visible_aug
                if int(item["proposal_id"]) not in label_hit_id_set
            ],
        ]
        payload = {
            "category": category,
            "proposal_ids": proposal_ids,
            "available_categories": available_categories,
            "label_hits": label_hits,
            "clip_visible_aug": clip_visible_aug,
            "clip_visible_failures": clip_visible_failures,
            "rank_fallback_used": rank_fallback_used,
            "n_visible_set": len(visible_ids),
            "cvra_exhausted": False,
        }
        last_payload = payload
        if label_hits or clip_visible_aug:
            return payload

    if last_payload is not None:
        last_payload["cvra_exhausted"] = True
        return last_payload

    return {
        "category": category,
        "proposal_ids": [],
        "available_categories": available_categories,
        "label_hits": [],
        "clip_visible_aug": [],
        "clip_visible_failures": [],
        "rank_fallback_used": False,
        "n_visible_set": len(visible_ids),
        "cvra_exhausted": not tried_any,
    }


def build_vg_tools(runtime: Any) -> list[BaseTool]:
    ctx = runtime.task_ctx  # VgEmbodiedScanCtx

    @tool
    def list_keyframes_with_proposals() -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("list_keyframes_with_proposals", {}, gate)
            return gate
        items = []
        for kf in runtime.bundle.keyframes:
            fid = kf.frame_id
            visible = ctx.frame_index.get(fid, []) if fid is not None else []
            items.append(
                {
                    "keyframe_idx": kf.keyframe_idx,
                    "frame_id": fid,
                    "visible_proposal_ids": visible,
                    "n_proposals": len(visible),
                    "annotated_image": str(
                        ctx.annotated_image_dir / f"frame_{fid}.png"
                    ),
                }
            )
        text = json.dumps(items, ensure_ascii=False)
        runtime.record("list_keyframes_with_proposals", {}, text)
        return text

    @tool
    def view_keyframe_marked(frame_id: int) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, gate)
            return gate
        if frame_id not in ctx.frame_index:
            err = (
                f"ERROR: frame_id={frame_id} not in proposal index; "
                f"available: {sorted(ctx.frame_index.keys())[:20]}"
            )
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, err)
            return err
        marked_path = ctx.annotated_image_dir / f"frame_{frame_id}.png"
        if not marked_path.exists():
            err = f"ERROR: annotated image not found: {marked_path}"
            runtime.record("view_keyframe_marked", {"frame_id": frame_id}, err)
            return err
        visible = ctx.frame_index[frame_id]
        # Mark the path as a fresh image to inject into the next user message
        runtime.bundle.extra_metadata = dict(runtime.bundle.extra_metadata or {})
        runtime.bundle.extra_metadata.setdefault("vg_pending_images", []).append(
            str(marked_path)
        )
        runtime.mark_evidence_updated()
        body = (
            f"frame_id={frame_id} marked image at {marked_path}; "
            f"visible_proposals={visible}; "
            f"categories={[next((p.category for p in ctx.proposals if p.id == pid), '?') for pid in visible]}"
            f"{_marked_frame_geometry(ctx, frame_id, visible)}"
        )
        runtime.record("view_keyframe_marked", {"frame_id": frame_id}, body)
        return body

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
    def find_proposals_by_category(category: str) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("find_proposals_by_category", {"category": category}, gate)
            return gate
        if getattr(runtime, "use_clip_visible_aug", False):
            try:
                payload = _find_proposals_by_category_with_cvra(
                    runtime=runtime,
                    ctx=ctx,
                    category=category,
                )
            except Exception as exc:
                err = (
                    "ERROR: clip_visible augmentation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                runtime.record(
                    "find_proposals_by_category", {"category": category}, err
                )
                return err
            text = json.dumps(payload, ensure_ascii=False)
            runtime.record("find_proposals_by_category", {"category": category}, text)
            return text

        ids = [
            p.id
            for p in ctx.proposals
            if p.category.strip().lower() == category.strip().lower()
        ]
        payload = {
            "category": category,
            "proposal_ids": ids,
            "available_categories": sorted(
                {p.category for p in ctx.proposals if p.category}
            ),
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("find_proposals_by_category", {"category": category}, text)
        return text

    @tool
    def compare_proposals_spatial(
        candidate_ids: list[int],
        anchor_id: int,
        relation: str,
    ) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        request = {
            "candidate_ids": candidate_ids,
            "anchor_id": anchor_id,
            "relation": relation,
        }
        if gate is not None:
            runtime.record("compare_proposals_spatial", request, gate)
            return gate
        allowed_relations = (
            "closest_to",
            "near",
            "next_to",
            "farthest_from",
            "above",
            "below",
            "left_of",
            "right_of",
        )
        if relation not in allowed_relations:
            err = f"ERROR: unsupported relation {relation!r}; allowed: " + " | ".join(
                allowed_relations
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
        list_keyframes_with_proposals,
        view_keyframe_marked,
        inspect_proposal,
        find_proposals_by_category,
        compare_proposals_spatial,
    ]


__all__ = ["build_vg_tools", "PRIMARY_SKILL"]
