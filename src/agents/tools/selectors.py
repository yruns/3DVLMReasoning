"""v9 selectors A-F. Return text-only JSON with frame_id + visible_proposal_ids + camera pose."""

from __future__ import annotations

import json
import math
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.catalog import SceneCatalog
from agents.runtime.scene_runtime import get_scene_catalog
from agents.tools.scene_perception import _gate


def _frame_to_proposals(catalog: SceneCatalog) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for p in catalog.proposals:
        for fid in p.frame_views.keys():
            out.setdefault(int(fid), []).append(p.proposal_id)
    return out


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _camera_pose(runtime: Any, frame_id: int) -> tuple[list[float] | None, float | None]:
    traj = (runtime.bundle.extra_metadata or {}).get("camera_trajectory_xy_yaw") or {}
    entry = traj.get(int(frame_id)) or traj.get(str(frame_id))
    if entry is None:
        return None, None
    return [float(entry[0]), float(entry[1])], float(entry[2])


def _strip_hidden(
    visible: list[int],
    catalog: SceneCatalog,
    hidden_categories: list[str],
) -> list[int]:
    if not hidden_categories:
        return visible
    hidden = {_norm_category(c) for c in hidden_categories}
    cat_by_id = {p.proposal_id: _norm_category(p.category) for p in catalog.proposals}
    return [pid for pid in visible if cat_by_id.get(pid) not in hidden]


def _build_frame_payload(
    runtime: Any,
    catalog: SceneCatalog,
    frame_id: int,
    selected_because: str,
    hidden_categories: list[str],
) -> dict:
    bev_xy, yaw = _camera_pose(runtime, frame_id)
    visible = _frame_to_proposals(catalog).get(int(frame_id), [])
    visible = _strip_hidden(visible, catalog, hidden_categories)
    return {
        "frame_id": int(frame_id),
        "visible_proposal_ids": sorted(visible),
        "bev_xy": bev_xy,
        "camera_yaw": yaw,
        "selected_because": selected_because,
    }


def build_selector_tools(runtime: Any) -> list[BaseTool]:
    @tool
    def select_by_text(
        query: str,
        k: int = 3,
        hidden_categories: list[str] | None = None,
    ) -> str:
        """Selector A. Detailed usage in 'scene-exploration-playbook'."""
        request = {
            "query": query,
            "k": int(k),
            "hidden_categories": list(hidden_categories or []),
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_text", request, gate)
            return gate
        selector = getattr(runtime, "keyframe_selector", None)
        if selector is None:
            err = "ERROR: runtime.keyframe_selector is None; cannot run Stage-1 text retrieval"
            runtime.record("select_by_text", request, err)
            return err
        try:
            result = selector.select_keyframes_v2(
                query=str(query),
                k=int(k),
                hidden_categories=list(hidden_categories or []),
                use_visual_context=False,
            )
        except Exception as exc:  # noqa: BLE001 — fail-loud
            err = f"ERROR: Stage-1 parse/exec failed: {type(exc).__name__}: {exc}"
            runtime.record("select_by_text", request, err)
            return err

        catalog = get_scene_catalog(runtime)
        frames = [
            _build_frame_payload(
                runtime,
                catalog,
                int(fid),
                selected_because=f"select_by_text(query={query!r})",
                hidden_categories=list(hidden_categories or []),
            )
            for fid in (result.keyframe_indices or [])
        ]
        summary = ""
        hyp = (result.metadata or {}).get("hypothesis_output")
        if isinstance(hyp, dict) and hyp.get("hypotheses"):
            first = hyp["hypotheses"][0]
            root = (first.get("grounding_query") or {}).get("root") or {}
            summary = f"target={root.get('category')!r} kind={first.get('kind', 'direct')}"
        payload = {"hypothesis_summary": summary, "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_text", request, text)
        return text

    tools = [select_by_text]

    @tool
    def select_by_hypothesis(
        hypothesis: Any,
        k: int = 3,
        hidden_categories: list[str] | None = None,
    ) -> str:
        """Selector B. Detailed usage in 'scene-exploration-playbook'."""
        request = {
            "hypothesis": hypothesis,
            "k": int(k),
            "hidden_categories": list(hidden_categories or []),
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_hypothesis", request, gate)
            return gate
        if not isinstance(hypothesis, dict) or "hypotheses" not in hypothesis:
            err = "ERROR: hypothesis must be a dict with 'hypotheses' list (HypothesisOutputV1 shape)"
            runtime.record("select_by_hypothesis", request, err)
            return err
        selector = getattr(runtime, "keyframe_selector", None)
        if selector is None or not hasattr(selector, "execute_hypothesis_dict"):
            err = "ERROR: runtime.keyframe_selector does not support execute_hypothesis_dict"
            runtime.record("select_by_hypothesis", request, err)
            return err
        try:
            result = selector.execute_hypothesis_dict(
                hypothesis_dict=hypothesis,
                k=int(k),
                hidden_categories=list(hidden_categories or []),
            )
        except Exception as exc:  # noqa: BLE001
            err = f"ERROR: hypothesis execution failed: {type(exc).__name__}: {exc}"
            runtime.record("select_by_hypothesis", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because="select_by_hypothesis",
                hidden_categories=list(hidden_categories or []),
            )
            for fid in (result.get("keyframe_indices") or [])
        ]
        payload = {
            "hypothesis_summary": str(result.get("summary") or ""),
            "frames": frames,
        }
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_hypothesis", request, text)
        return text

    tools.append(select_by_hypothesis)

    @tool
    def select_by_frame_neighbor(
        anchor_frame_id: int,
        mode: str = "temporal",
        k: int = 3,
    ) -> str:
        """Selector C. Detailed usage in 'scene-exploration-playbook'."""
        request = {"anchor_frame_id": int(anchor_frame_id), "mode": mode, "k": int(k)}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_frame_neighbor", request, gate)
            return gate
        if mode not in ("temporal", "viewpoint_diverse"):
            err = f"ERROR: mode must be 'temporal' or 'viewpoint_diverse'; got {mode!r}"
            runtime.record("select_by_frame_neighbor", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        valid = sorted(int(f) for f in catalog.valid_frame_ids)
        if int(anchor_frame_id) not in valid:
            err = (
                f"ERROR: anchor_frame_id={anchor_frame_id} not in valid_frame_ids; "
                f"available[:20]={valid[:20]}"
            )
            runtime.record("select_by_frame_neighbor", request, err)
            return err
        others = [f for f in valid if f != int(anchor_frame_id)]
        if mode == "temporal":
            others.sort(key=lambda f: (abs(f - int(anchor_frame_id)), f))
            chosen = others[: int(k)]
        else:
            anchor_pose, anchor_yaw = _camera_pose(runtime, int(anchor_frame_id))
            ranked: list[tuple[float, float, int]] = []
            for f in others:
                pose, yaw = _camera_pose(runtime, f)
                if anchor_pose is None or pose is None or anchor_yaw is None or yaw is None:
                    distance = float(abs(f - int(anchor_frame_id)))
                    yaw_delta = 0.0
                else:
                    distance = math.hypot(pose[0] - anchor_pose[0], pose[1] - anchor_pose[1])
                    yaw_delta = abs(yaw - anchor_yaw)
                ranked.append((distance - yaw_delta, -yaw_delta, f))
            ranked.sort(key=lambda item: (item[0], item[1]))
            chosen = [t[2] for t in ranked[: int(k)]]
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because=f"select_by_frame_neighbor(anchor={anchor_frame_id}, mode={mode!r})",
                hidden_categories=[],
            )
            for fid in chosen
        ]
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_frame_neighbor", request, text)
        return text

    tools.append(select_by_frame_neighbor)

    @tool
    def select_by_proposal(
        proposal_ids: list[int],
        require_all: bool = False,
        k: int = 3,
    ) -> str:
        """Selector D. Detailed usage in 'scene-exploration-playbook'."""
        request = {
            "proposal_ids": list(proposal_ids or []),
            "require_all": bool(require_all),
            "k": int(k),
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_proposal", request, gate)
            return gate
        ids = [int(i) for i in (proposal_ids or [])]
        if not ids:
            err = "ERROR: proposal_ids must be a non-empty list of ints"
            runtime.record("select_by_proposal", request, err)
            return err
        catalog = get_scene_catalog(runtime)
        proposal_by_id = {p.proposal_id: p for p in catalog.proposals}
        missing = [pid for pid in ids if pid not in proposal_by_id]
        if missing:
            err = (
                f"ERROR: proposal_id(s) not in catalog: {missing}; "
                f"available count={len(proposal_by_id)}"
            )
            runtime.record("select_by_proposal", request, err)
            return err
        sets = [set(proposal_by_id[pid].frame_views.keys()) for pid in ids]
        if not sets:
            frames_set: set[int] = set()
        elif require_all:
            frames_set = set.intersection(*sets)
        else:
            frames_set = set().union(*sets)
        chosen = sorted(int(f) for f in frames_set)[: int(k)]
        frames = [
            _build_frame_payload(
                runtime, catalog, int(fid),
                selected_because=(
                    f"select_by_proposal(proposal_ids={ids}, require_all={bool(require_all)})"
                ),
                hidden_categories=[],
            )
            for fid in chosen
        ]
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_proposal", request, text)
        return text

    tools.append(select_by_proposal)
    return tools


__all__ = ["build_selector_tools"]
