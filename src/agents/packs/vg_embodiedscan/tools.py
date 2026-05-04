"""EmbodiedScan VG tools. All bodies FAIL-LOUD on missing primary skill."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

PRIMARY_SKILL = "vg-grounding-playbook"


def _gate(runtime: Any) -> str | None:
    """Return ERROR string if skill not loaded, else None."""
    if PRIMARY_SKILL not in runtime.skills_loaded:
        return f"ERROR: load_skill({PRIMARY_SKILL!r}) before calling this tool."
    return None


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
        allowed_relations = ("closest_to", "farthest_from", "above", "below")
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
            scored.append((p.id, distance, horizontal_distance, vertical_offset))
        if relation == "closest_to":
            scored.sort(key=lambda x: x[1])
        elif relation == "farthest_from":
            scored.sort(key=lambda x: x[1], reverse=True)
        elif relation == "above":
            scored.sort(key=lambda x: (x[3] <= 0.0, -x[3], x[2]))
        else:  # below
            scored.sort(key=lambda x: (x[3] >= 0.0, x[3], x[2]))
        payload = {
            "anchor_id": anchor_id,
            "relation": relation,
            "ranked_ids": [pid for pid, _, _, _ in scored],
            "distances": [d for _, d, _, _ in scored],
            "horizontal_distances": [d for _, _, d, _ in scored],
            "vertical_offsets": [z for _, _, _, z in scored],
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("compare_proposals_spatial", request, text)
        return text

    @tool
    def select_among_proposals(
        candidate_ids: list[int],
        description: str,
    ) -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        request = {"candidate_ids": candidate_ids, "description": description}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_among_proposals", request, gate)
            return gate
        if not isinstance(candidate_ids, list) or len(candidate_ids) < 2:
            err = "ERROR: select_among_proposals requires candidate_ids with length >= 2"
            runtime.record("select_among_proposals", request, err)
            return err
        if not isinstance(description, str) or not description.strip():
            err = "ERROR: description must be a non-empty string"
            runtime.record("select_among_proposals", request, err)
            return err

        pool_by_id = {p.id: p for p in ctx.proposals}
        missing = sorted(set(candidate_ids) - set(pool_by_id))
        if missing:
            err = f"ERROR: candidate ids not in pool: {missing}"
            runtime.record("select_among_proposals", request, err)
            return err

        if runtime.vlm_judge is None or runtime.image_to_data_url is None:
            err = (
                "ERROR: select_among_proposals requires VLM hooks; "
                "framework runtime did not attach_vlm_hooks before tool build."
            )
            runtime.record("select_among_proposals", request, err)
            return err

        # For each candidate, pick the most-visible annotated frame.
        # ctx.proposal_index is ordered by the visibility index already.
        candidate_frames: list[tuple[int, int]] = []
        for cid in candidate_ids:
            frames = ctx.proposal_index.get(cid, [])
            if not frames:
                err = (
                    f"ERROR: proposal_id={cid} has no frames in proposal_index; "
                    "the candidate is invisible in all current keyframes."
                )
                runtime.record("select_among_proposals", request, err)
                return err
            best_frame = int(frames[0])
            annotated = ctx.annotated_image_dir / f"frame_{best_frame}.png"
            if not annotated.exists():
                err = (
                    f"ERROR: annotated image missing for frame_{best_frame}: "
                    f"{annotated}"
                )
                runtime.record("select_among_proposals", request, err)
                return err
            candidate_frames.append((cid, best_frame))

        from langchain_core.messages import HumanMessage, SystemMessage

        common_category = pool_by_id[candidate_ids[0]].category
        system_text = (
            "You are a multi-distractor visual grounding referee. The user "
            "describes ONE specific object from a Mask3D candidate pool. "
            "Several same-category candidates are listed below; each is "
            "shown in a marked frame where its proposal_id is drawn as a "
            "labeled bounding box. Pick the single proposal_id that best "
            "matches the user's description. "
            'Output STRICT JSON: {"selected_proposal_id": <int>, '
            '"reasoning": "<one short sentence citing the visual cue>"}. '
            "The id MUST be one of the candidate_ids listed."
        )

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Description: {description!r}\n"
                    f"Candidate proposal_ids: {candidate_ids}\n"
                    f"Common category: {common_category!r}\n\n"
                    "For each candidate, the next image is its marked frame; "
                    "find the labeled box matching that proposal_id and apply "
                    "the description's spatial cues to disambiguate."
                ),
            }
        ]
        for cid, frame_id in candidate_frames:
            annotated = ctx.annotated_image_dir / f"frame_{frame_id}.png"
            user_content.append(
                {
                    "type": "text",
                    "text": f"Candidate proposal_id={cid} (frame_{frame_id}):",
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": runtime.image_to_data_url(annotated)},
                }
            )
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Reply with STRICT JSON only. selected_proposal_id MUST be "
                    f"one of: {candidate_ids}."
                ),
            }
        )

        try:
            raw = runtime.vlm_judge(
                [
                    SystemMessage(content=system_text),
                    HumanMessage(content=user_content),
                ]
            )
        except Exception as exc:  # noqa: BLE001 — FAIL-LOUD via tool error string
            err = f"ERROR: VLM call failed: {type(exc).__name__}: {exc}"
            runtime.record("select_among_proposals", request, err)
            return err

        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            chosen_id = int(parsed["selected_proposal_id"])
            reasoning = str(parsed.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            err = (
                f"ERROR: could not parse VLM JSON ({type(exc).__name__}: {exc}); "
                f"raw={raw[:240]!r}"
            )
            runtime.record("select_among_proposals", request, err)
            return err

        if chosen_id not in candidate_ids:
            err = (
                f"ERROR: VLM chose {chosen_id} not in candidate_ids "
                f"{candidate_ids}; raw={raw[:240]!r}"
            )
            runtime.record("select_among_proposals", request, err)
            return err

        payload = {
            "selected_proposal_id": chosen_id,
            "reasoning": reasoning,
            "candidate_ids": candidate_ids,
        }
        text_out = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_among_proposals", request, text_out)
        return text_out

    return [
        list_keyframes_with_proposals,
        view_keyframe_marked,
        inspect_proposal,
        find_proposals_by_category,
        compare_proposals_spatial,
        select_among_proposals,
    ]


__all__ = ["build_vg_tools", "PRIMARY_SKILL"]
