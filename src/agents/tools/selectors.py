"""v9 selectors A-F. Return text-only JSON with frame_id + visible_proposal_ids + camera pose."""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from agents.catalog import SceneCatalog
from agents.runtime.scene_runtime import get_scene_catalog, make_tool_image_ref_if_new
from agents.tools.scene_perception import _gate

_STAGE1_TEXT_CONCURRENCY_ENV = "STAGE1_TEXT_RETRIEVAL_MAX_CONCURRENCY"
_STAGE1_TEXT_LOCK_DIR_ENV = "STAGE1_TEXT_RETRIEVAL_LOCK_DIR"


@contextmanager
def _stage1_text_retrieval_slot():
    raw_limit = os.environ.get(_STAGE1_TEXT_CONCURRENCY_ENV, "").strip()
    if not raw_limit:
        yield
        return
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 0
    if limit <= 0:
        yield
        return

    lock_root = Path(
        os.environ.get(_STAGE1_TEXT_LOCK_DIR_ENV)
        or tempfile.gettempdir()
    ) / "3dvlmreasoning_stage1_text_retrieval"
    lock_root.mkdir(parents=True, exist_ok=True)

    if limit == 1:
        lock_file = (lock_root / "slot_0.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
        return

    while True:
        for slot_idx in range(limit):
            lock_file = (lock_root / f"slot_{slot_idx}.lock").open(
                "a+",
                encoding="utf-8",
            )
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock_file.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
            return
        time.sleep(0.1)


def _frame_to_proposals(catalog: SceneCatalog) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for p in catalog.proposals:
        for fid in p.frame_views.keys():
            out.setdefault(int(fid), []).append(p.proposal_id)
    return out


def _norm_category(category: str) -> str:
    return " ".join(str(category).strip().lower().split())


def _camera_pose(
    runtime: Any, frame_id: int
) -> tuple[list[float] | None, float | None]:
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


def _resolve_raw_rgb_path(catalog: SceneCatalog, frame_id: int) -> Path | None:
    for p in catalog.proposals:
        view = p.frame_views.get(int(frame_id))
        if view is not None and view.raw_rgb_path:
            return Path(view.raw_rgb_path)
    return None


def build_selector_tools(runtime: Any) -> list[BaseTool]:
    flag_enabled = bool(getattr(runtime, "enable_stage1_text_retrieval", True))
    selector_available = getattr(runtime, "text_frame_selector", None) is not None
    selector_factory = getattr(runtime, "text_frame_selector_factory", None)
    selector_factory_available = callable(selector_factory)
    text_retrieval_enabled = flag_enabled and (
        selector_available or selector_factory_available
    )
    if flag_enabled and not selector_available and not selector_factory_available:
        from loguru import logger

        logger.warning(
            "build_selector_tools: enable_stage1_text_retrieval=True but "
            "runtime has neither text_frame_selector nor "
            "text_frame_selector_factory; dropping select_by_text from the "
            "tool list to keep the agent's tool surface consistent with the "
            "runtime. This should have been caught at "
            "Stage2DeepResearchAgent construction; if it wasn't, look for a "
            "caller that bypasses BaseStage2Runtime.__init__ (e.g., a test "
            "fixture that directly mutates Stage2RuntimeState)."
        )

    @tool
    def select_by_text(
        query: str,
        k: int = 3,
        hidden_categories: list[str] | None = None,
    ) -> str:
        """Selector A — Stage-1 language→frame retrieval.

        Best for queries naming a rare or unique target category
        ("where is the kitchen counter") or when the BEV does not give
        you a confident catalog shortlist. Returns ≤3 first-person RGB
        frames each with `visible_proposal_ids` + pose; also surfaces an
        `image_path` and `already_seen` flag.

        Known limitations from the NR3D Stage-1 audit
        (docs/benchmark/nr3d/v9_1_select_by_text_audit_20260516.md):
        returns `frames: []` (`parse=no_evidence`) on ~32 % of NR3D
        queries — mostly anchor-spatial phrasing ("X next to Y", "facing
        X", "right of Y"), explicit negation, or over-specific
        sub-categories. If the response has `frames: []`, do NOT retry
        the same query with a larger `k`; switch to `select_by_proposal`
        / `select_by_region` on the candidate IDs visible in the BEV.

        For NR3D/VG, omit `hidden_categories` by default. Do not hide
        support/anchor/context categories such as wall, floor, door, bed,
        or desk; Stage-1 validates the whole parsed expression, so masking
        anchors can raise "Masked category leak detected" and remove useful
        evidence. If that happens, retry once with `hidden_categories=[]`.

        Detailed usage in 'scene-exploration-playbook'.
        """
        hidden_in = list(hidden_categories or [])
        request = {
            "query": query,
            "k": int(k),
            "hidden_categories": hidden_in,
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_text", request, gate)
            return gate
        # v9.4 cadence experiment: when the force-to-error flag is set, short
        # circuit before touching the selector. This intentionally reproduces
        # the v9.1_fix bug-state behaviour (Stage-1 always ERRORs while the
        # text-first playbook + system prompt remain loaded) so the agent's
        # documented fallback chain acts as a forcing function for the
        # deliberation cadence isolated in
        # docs/benchmark/nr3d/v9_1_fix_vs_v9_3_audit30_20260517.md.
        if getattr(runtime, "force_stage1_text_retrieval_to_error", False):
            err = (
                "ERROR: Stage-1 text retrieval is force-disabled "
                "(force_stage1_text_retrieval_to_error=True). Fall back to "
                "the catalog-first chain documented in the "
                "scene-exploration-playbook (BEV + list_scene_proposals + "
                "select_by_proposal + per-candidate mark_frame_with_bbox)."
            )
            runtime.record("select_by_text", request, err)
            return err
        selector = getattr(runtime, "text_frame_selector", None)
        if selector is None:
            factory = getattr(runtime, "text_frame_selector_factory", None)
            if not callable(factory):
                err = (
                    "ERROR: runtime.text_frame_selector is None and "
                    "runtime.text_frame_selector_factory is not callable; "
                    "cannot run Stage-1 text retrieval"
                )
                runtime.record("select_by_text", request, err)
                return err
            try:
                selector = factory()
            except Exception as exc:  # noqa: BLE001 — fail-loud through tool trace
                err = (
                    "ERROR: Stage-1 selector init failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                runtime.record("select_by_text", request, err)
                return err
            if selector is None:
                err = (
                    "ERROR: Stage-1 selector init failed: "
                    "text_frame_selector_factory returned None"
                )
                runtime.record("select_by_text", request, err)
                return err
            runtime.text_frame_selector = selector
        k_in = int(k)
        capped = min(k_in, 3)
        k_warning = "" if k_in == capped else f" (k capped at 3 from {k_in})"

        def _select_with_hidden(hidden: list[str]) -> Any:
            with _stage1_text_retrieval_slot():
                return selector.select_keyframes_v2(
                    query=str(query),
                    k=capped,
                    hidden_categories=hidden,
                    use_visual_context=False,
                    viewpoint_aware=True,
                )

        effective_hidden_categories = hidden_in
        retry_metadata: dict[str, Any] | None = None
        try:
            result = _select_with_hidden(hidden_in)
        except Exception as exc:  # noqa: BLE001 — fail-loud
            first_err = f"{type(exc).__name__}: {exc}"
            if "Masked category leak detected" in str(exc) and hidden_in:
                try:
                    result = _select_with_hidden([])
                except Exception as retry_exc:  # noqa: BLE001 — fail-loud
                    retry_err = f"{type(retry_exc).__name__}: {retry_exc}"
                    err = (
                        "ERROR: Stage-1 parse/exec failed after masked-category "
                        f"retry: first error: {first_err}; retry error: {retry_err}"
                    )
                    runtime.record("select_by_text", request, err)
                    return err
                effective_hidden_categories = []
                retry_metadata = {
                    "original_hidden_categories": hidden_in,
                    "error": first_err,
                    "retried_with_hidden_categories": [],
                }
            else:
                err = f"ERROR: Stage-1 parse/exec failed: {first_err}"
                runtime.record("select_by_text", request, err)
                return err

        catalog = get_scene_catalog(runtime)
        frames: list[dict] = []
        image_metadata: list[dict] = []
        for fid in (result.keyframe_indices or [])[:capped]:
            base = _build_frame_payload(
                runtime,
                catalog,
                int(fid),
                selected_because=f"select_by_text(query={query!r}){k_warning}",
                hidden_categories=effective_hidden_categories,
            )
            image_path = _resolve_raw_rgb_path(catalog, int(fid))
            base["image_path"] = str(image_path) if image_path else None
            image_ref = make_tool_image_ref_if_new(
                runtime,
                base["image_path"] or "",
                metadata={
                    "frame_id": int(fid),
                    "source_tool": "select_by_text",
                    "selected_because": base["selected_because"],
                },
            )
            if image_ref is not None:
                image_metadata.append(image_ref)
            base["already_seen"] = (image_ref is None) and (
                base["image_path"] in runtime.seen_image_paths
            )
            frames.append(base)

        summary = ""
        hyp = (result.metadata or {}).get("hypothesis_output")
        if isinstance(hyp, dict) and hyp.get("hypotheses"):
            first = hyp["hypotheses"][0]
            root = (first.get("grounding_query") or {}).get("root") or {}
            summary = (
                f"target={root.get('category')!r} kind={first.get('kind', 'direct')}"
            )
        payload = {"hypothesis_summary": summary + k_warning, "frames": frames}
        if retry_metadata is not None:
            payload["masked_category_retry"] = retry_metadata
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_text", request, text, image_metadata=image_metadata)
        return text

    tools: list[BaseTool] = []
    if text_retrieval_enabled:
        tools.append(select_by_text)

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
        k_in = int(k)
        capped = min(k_in, 3)
        k_warning = "" if k_in == capped else f" (k capped at 3 from {k_in})"
        others = [f for f in valid if f != int(anchor_frame_id)]
        if mode == "temporal":
            others.sort(key=lambda f: (abs(f - int(anchor_frame_id)), f))
            chosen = others[:capped]
        else:
            anchor_pose, anchor_yaw = _camera_pose(runtime, int(anchor_frame_id))
            ranked: list[tuple[float, float, int]] = []
            for f in others:
                pose, yaw = _camera_pose(runtime, f)
                if (
                    anchor_pose is None
                    or pose is None
                    or anchor_yaw is None
                    or yaw is None
                ):
                    distance = float(abs(f - int(anchor_frame_id)))
                    yaw_delta = 0.0
                else:
                    distance = math.hypot(
                        pose[0] - anchor_pose[0], pose[1] - anchor_pose[1]
                    )
                    yaw_delta = abs(yaw - anchor_yaw)
                ranked.append((distance - yaw_delta, -yaw_delta, f))
            ranked.sort(key=lambda item: (item[0], item[1]))
            chosen = [t[2] for t in ranked[:capped]]
        frames: list[dict] = []
        image_metadata: list[dict] = []
        for fid in chosen:
            base = _build_frame_payload(
                runtime,
                catalog,
                int(fid),
                selected_because=(
                    f"select_by_frame_neighbor(anchor={anchor_frame_id}, mode={mode!r}){k_warning}"
                ),
                hidden_categories=[],
            )
            image_path = _resolve_raw_rgb_path(catalog, int(fid))
            base["image_path"] = str(image_path) if image_path else None
            image_ref = make_tool_image_ref_if_new(
                runtime,
                base["image_path"] or "",
                metadata={
                    "frame_id": int(fid),
                    "source_tool": "select_by_frame_neighbor",
                    "selected_because": base["selected_because"],
                },
            )
            if image_ref is not None:
                image_metadata.append(image_ref)
            base["already_seen"] = (image_ref is None) and (
                base["image_path"] in runtime.seen_image_paths
            )
            frames.append(base)
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record(
            "select_by_frame_neighbor",
            request,
            text,
            image_metadata=image_metadata,
        )
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
        k_in = int(k)
        capped = min(k_in, 3)
        k_warning = "" if k_in == capped else f" (k capped at 3 from {k_in})"
        sets = [set(proposal_by_id[pid].frame_views.keys()) for pid in ids]
        if not sets:
            frames_set: set[int] = set()
        elif require_all:
            frames_set = set.intersection(*sets)
        else:
            frames_set = set().union(*sets)
        chosen = sorted(int(f) for f in frames_set)[:capped]
        frames: list[dict] = []
        image_metadata: list[dict] = []
        for fid in chosen:
            base = _build_frame_payload(
                runtime,
                catalog,
                int(fid),
                selected_because=(
                    f"select_by_proposal(proposal_ids={ids}, require_all={bool(require_all)}){k_warning}"
                ),
                hidden_categories=[],
            )
            image_path = _resolve_raw_rgb_path(catalog, int(fid))
            base["image_path"] = str(image_path) if image_path else None
            image_ref = make_tool_image_ref_if_new(
                runtime,
                base["image_path"] or "",
                metadata={
                    "frame_id": int(fid),
                    "source_tool": "select_by_proposal",
                    "selected_because": base["selected_because"],
                },
            )
            if image_ref is not None:
                image_metadata.append(image_ref)
            base["already_seen"] = (image_ref is None) and (
                base["image_path"] in runtime.seen_image_paths
            )
            frames.append(base)
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record(
            "select_by_proposal", request, text, image_metadata=image_metadata
        )
        return text

    tools.append(select_by_proposal)

    @tool
    def select_by_region(
        region: list[float],
        region_type: str = "bev_2d",
        k: int = 3,
    ) -> str:
        """Selector E. Detailed usage in 'scene-exploration-playbook'."""
        request = {
            "region": [float(v) for v in (region or [])],
            "region_type": region_type,
            "k": int(k),
        }
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_region", request, gate)
            return gate
        if region_type not in ("bev_2d", "bbox_3d"):
            err = (
                f"ERROR: region_type must be 'bev_2d' or 'bbox_3d'; got {region_type!r}"
            )
            runtime.record("select_by_region", request, err)
            return err
        k_in = int(k)
        capped = min(k_in, 3)
        k_warning = "" if k_in == capped else f" (k capped at 3 from {k_in})"
        if region_type == "bev_2d":
            if len(region) != 4:
                err = "ERROR: bev_2d region must be [xmin, ymin, xmax, ymax]"
                runtime.record("select_by_region", request, err)
                return err
            xmin, ymin, xmax, ymax = (float(v) for v in region)
            catalog = get_scene_catalog(runtime)
            chosen: list[int] = []
            for fid in sorted(int(f) for f in catalog.valid_frame_ids):
                pose, _ = _camera_pose(runtime, fid)
                if pose is None:
                    continue
                if xmin <= pose[0] <= xmax and ymin <= pose[1] <= ymax:
                    chosen.append(fid)
                    if len(chosen) >= capped:
                        break
        else:
            if len(region) != 6:
                err = (
                    "ERROR: bbox_3d region must be [xmin, ymin, zmin, xmax, ymax, zmax]"
                )
                runtime.record("select_by_region", request, err)
                return err
            xmin, ymin, zmin, xmax, ymax, zmax = (float(v) for v in region)
            catalog = get_scene_catalog(runtime)
            inside_proposals = [
                p.proposal_id
                for p in catalog.proposals
                if xmin <= p.position_3d[0] <= xmax
                and ymin <= p.position_3d[1] <= ymax
                and zmin <= p.position_3d[2] <= zmax
            ]
            frame_to_props = _frame_to_proposals(catalog)
            chosen_set: set[int] = set()
            for fid, props in frame_to_props.items():
                if any(pid in inside_proposals for pid in props):
                    chosen_set.add(int(fid))
            chosen = sorted(chosen_set)[:capped]
        frames: list[dict] = []
        image_metadata: list[dict] = []
        for fid in chosen:
            base = _build_frame_payload(
                runtime,
                catalog,
                int(fid),
                selected_because=f"select_by_region(type={region_type!r}){k_warning}",
                hidden_categories=[],
            )
            image_path = _resolve_raw_rgb_path(catalog, int(fid))
            base["image_path"] = str(image_path) if image_path else None
            image_ref = make_tool_image_ref_if_new(
                runtime,
                base["image_path"] or "",
                metadata={
                    "frame_id": int(fid),
                    "source_tool": "select_by_region",
                    "selected_because": base["selected_because"],
                },
            )
            if image_ref is not None:
                image_metadata.append(image_ref)
            base["already_seen"] = (image_ref is None) and (
                base["image_path"] in runtime.seen_image_paths
            )
            frames.append(base)
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record("select_by_region", request, text, image_metadata=image_metadata)
        return text

    tools.append(select_by_region)

    def _seen_frame_ids_from_runtime() -> set[int]:
        seen: set[int] = set()
        for entry in list(getattr(runtime, "tool_trace", []) or []):
            tool_input = getattr(entry, "tool_input", {}) or {}
            if getattr(entry, "tool_name", "") in (
                "mark_frame_with_bbox",
                "select_by_text",
                "select_by_proposal",
                "select_by_region",
                "select_by_frame_neighbor",
                "select_by_coverage",
            ):
                fid = tool_input.get("frame_id")
                if isinstance(fid, int):
                    seen.add(fid)
        return seen

    @tool
    def select_by_coverage(
        method: str = "obj_iou",
        k: int = 3,
        seen_frame_ids: list[int] | None = None,
    ) -> str:
        """Selector F. Detailed usage in 'scene-exploration-playbook'."""
        request = {"method": method, "k": int(k), "seen_frame_ids": seen_frame_ids}
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("select_by_coverage", request, gate)
            return gate
        if method not in ("obj_iou", "pose_depth"):
            err = f"ERROR: method must be 'obj_iou' or 'pose_depth'; got {method!r}"
            runtime.record("select_by_coverage", request, err)
            return err
        k_in = int(k)
        capped = min(k_in, 3)
        k_warning = "" if k_in == capped else f" (k capped at 3 from {k_in})"
        catalog = get_scene_catalog(runtime)
        valid = sorted(int(f) for f in catalog.valid_frame_ids)
        seen_set: set[int]
        if seen_frame_ids is None:
            seen_set = _seen_frame_ids_from_runtime()
        else:
            seen_set = {int(f) for f in seen_frame_ids}
        unseen = [f for f in valid if f not in seen_set]
        if not unseen:
            payload = {"hypothesis_summary": "all frames already seen", "frames": []}
            text = json.dumps(payload, ensure_ascii=False)
            runtime.record("select_by_coverage", request, text)
            return text
        if method == "obj_iou":
            frame_to_props = _frame_to_proposals(catalog)
            if not seen_set:
                chosen = unseen[:capped]
            else:
                seen_union: set[int] = set()
                for s in seen_set:
                    seen_union.update(frame_to_props.get(int(s), []))

                def jaccard_distance(fid: int) -> float:
                    a = set(frame_to_props.get(int(fid), []))
                    if not a and not seen_union:
                        return 0.0
                    union = a | seen_union
                    inter = a & seen_union
                    return 1.0 - (len(inter) / len(union)) if union else 0.0

                unseen.sort(key=lambda f: (-jaccard_distance(f), f))
                chosen = unseen[:capped]
        else:  # pose_depth
            if not seen_set:
                chosen = unseen[:capped]
            else:
                centroid_x = 0.0
                centroid_y = 0.0
                count = 0
                for s in seen_set:
                    pose, _ = _camera_pose(runtime, int(s))
                    if pose is None:
                        continue
                    centroid_x += pose[0]
                    centroid_y += pose[1]
                    count += 1
                if count == 0:
                    chosen = unseen[:capped]
                else:
                    centroid_x /= count
                    centroid_y /= count

                    def dist(fid: int) -> float:
                        pose, _ = _camera_pose(runtime, int(fid))
                        if pose is None:
                            return float("inf")
                        return -math.hypot(pose[0] - centroid_x, pose[1] - centroid_y)

                    unseen.sort(key=lambda f: (dist(f), f))
                    chosen = unseen[:capped]
        frames: list[dict] = []
        image_metadata: list[dict] = []
        for fid in chosen:
            base = _build_frame_payload(
                runtime,
                catalog,
                int(fid),
                selected_because=f"select_by_coverage(method={method!r}){k_warning}",
                hidden_categories=[],
            )
            image_path = _resolve_raw_rgb_path(catalog, int(fid))
            base["image_path"] = str(image_path) if image_path else None
            image_ref = make_tool_image_ref_if_new(
                runtime,
                base["image_path"] or "",
                metadata={
                    "frame_id": int(fid),
                    "source_tool": "select_by_coverage",
                    "selected_because": base["selected_because"],
                },
            )
            if image_ref is not None:
                image_metadata.append(image_ref)
            base["already_seen"] = (image_ref is None) and (
                base["image_path"] in runtime.seen_image_paths
            )
            frames.append(base)
        payload = {"hypothesis_summary": "", "frames": frames}
        text = json.dumps(payload, ensure_ascii=False)
        runtime.record(
            "select_by_coverage", request, text, image_metadata=image_metadata
        )
        return text

    tools.append(select_by_coverage)
    return tools


__all__ = ["build_selector_tools"]
