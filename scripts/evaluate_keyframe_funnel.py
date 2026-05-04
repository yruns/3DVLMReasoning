"""ScanRefer keyframe-selection funnel evaluator.

For each sample in a frozen fold (e.g. random100), measures the
calibration funnel of the Stage 1 -> Stage 2 keyframe pipeline:

    100 samples
    └── F0 (optional, --run-parser): the LLM query parser maps the
            description to a target category that fuzzy-matches the
            pack-prep GT category (i.e. the parser stage is correct
            independently of retrieval).
        └── F1: Phase 8 has target_id visible somewhere
            └── F2: pack-prep initial KFs cover at least one frame where
                    the Phase 8 GT target_id is visible
                └── F3: pack-prep initial KFs cover at least one Mask3D
                        candidate whose label fuzzy-matches the GT category
                    └── F4: cumulative KFs (initial + view_keyframe_marked
                            + callback-added) cover Phase 8 target_id
                        └── F5: agent submitted a proposal with IoU >= 0.25
                                (Phase 8 GT bbox, the same eval the pack
                                 ran under)

The funnel pinpoints where samples fall off:

- F1 fail: scene's Phase 8 visibility never sees the target -> data issue.
- F2 fail: pack-prep KF picker missed the target frame -> Stage 1 KF logic.
- F3 fail: same-category Mask3D candidate is absent from initial KFs ->
  proposal-pool / Mask3D-density / annotated-PNG mismatch.
- F4 fail: Stage 2 callbacks failed to recover -> ReAct loop ineffective.
- F5 fail: the agent saw the right frames but picked the wrong proposal
  -> picking error.

Inputs:
  --sample-ids     frozen fold JSON (random100_sample_ids.json)
  --pack-name      pack used at runtime (pack_scanrefer_v3_iterative,
                   pack_scanrefer_v3p_iterative, ...)
  --data-root      ScanRefer scannet root with the pack samples
  --phase8-data-root NR3D scannet root with Phase 8 visibility indices
  --per-sample-dir directory of Stage 2 per-sample checkpoints with
                   tool_trace, e.g.
                   tmp/v3p3_vertical_spatial_random100_eval/per_sample/<pack>/
  --output         JSON file to write the per-sample + summary report

The script does NOT call any LLM; everything is deterministic from
disk artifacts and the Phase 8 visibility / Mask3D-CG indices.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

VIS_REL = Path("conceptgraph/indices/visibility_index.pkl")
MASK3D_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz")

_NEW_VIEW_IDS_RE = re.compile(r"New view IDs: \[([\d,\s]+)\]")


def _safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def _digest8(sample_id: str) -> str:
    """Match how the runner names its checkpoint files."""
    safe = _safe_sample_id(sample_id)
    return hashlib.sha1(safe.encode("utf-8")).hexdigest()[:12]


def _category_tokens(text: str) -> set[str]:
    raw = (text or "").replace("_", " ").lower().strip()
    if not raw:
        return set()
    stop = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to"}
    return {t for t in raw.split() if t and t not in stop}


def _category_match(label: str, target_categories: Iterable[str]) -> bool:
    label_l = (label or "").replace("_", " ").lower().strip()
    if not label_l:
        return False
    label_toks = _category_tokens(label)
    for cat in target_categories:
        if not cat:
            continue
        cat_l = cat.replace("_", " ").lower().strip()
        if not cat_l or cat_l in {"unknow", "unknown"}:
            continue
        if label_l == cat_l:
            return True
        if label_toks & _category_tokens(cat):
            return True
    return False


def _load_phase8_visibility(scene_root: Path) -> dict[int, list[tuple[int, float]]]:
    p = scene_root / VIS_REL
    if not p.exists():
        raise FileNotFoundError(f"Phase 8 visibility missing: {p}")
    with open(p, "rb") as f:
        payload = pickle.load(f)
    raw = payload.get("object_to_views") or {}
    return {int(k): [(int(e[0]), float(e[1])) for e in v] for k, v in raw.items()}


def _load_mask3d_pool(scene_root: Path) -> tuple[list[dict], dict[int, list[int]]]:
    p = scene_root / MASK3D_PCD_REL
    if not p.exists():
        raise FileNotFoundError(f"Mask3D pkl missing: {p}")
    with gzip.open(p, "rb") as f:
        payload = pickle.load(f)
    raw_objects = payload["objects"]
    proposals: list[dict] = []
    for obj_id, obj in enumerate(raw_objects):
        names = obj.get("class_name") or []
        if not names:
            continue
        label = Counter(names).most_common(1)[0][0]
        proposals.append({"id": obj_id, "label": label})

    vis_p = scene_root / VIS_REL
    if not vis_p.exists():
        raise FileNotFoundError(f"Mask3D visibility missing: {vis_p}")
    return proposals, {}


def _load_mask3d_visibility_view_to_objects(
    scene_root: Path,
) -> dict[int, list[tuple[int, float]]]:
    """The pack-prep writes Mask3D-aligned visibility under
    data/scanrefer/scannet/<scene>/conceptgraph/indices/visibility_index.pkl."""
    p = scene_root / VIS_REL
    if not p.exists():
        raise FileNotFoundError(f"Mask3D visibility missing: {p}")
    with open(p, "rb") as f:
        payload = pickle.load(f)
    raw = payload.get("view_to_objects") or {}
    return {int(k): [(int(e[0]), float(e[1])) for e in v] for k, v in raw.items()}


def _bbox_iou_axis_aligned(a: list[float], b: list[float]) -> float:
    """3D IoU between two 9-DOF bboxes treated as axis-aligned (cx,cy,cz,dx,dy,dz,…)."""
    ac = np.asarray(a[:3], dtype=float)
    ad = np.asarray(a[3:6], dtype=float)
    bc = np.asarray(b[:3], dtype=float)
    bd = np.asarray(b[3:6], dtype=float)
    a_min, a_max = ac - ad / 2.0, ac + ad / 2.0
    b_min, b_max = bc - bd / 2.0, bc + bd / 2.0
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(0.0, inter_max - inter_min).prod()
    vol_a = float(np.prod(ad))
    vol_b = float(np.prod(bd))
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


def _extract_cumulative_frame_ids(
    initial_frame_ids: list[int], tool_trace: list[dict]
) -> tuple[list[int], dict[str, int]]:
    """Return (sorted_unique_frame_ids, breakdown_by_source)."""
    seen: set[int] = set(initial_frame_ids)
    by_source = {
        "initial": len(seen),
        "view_keyframe_marked": 0,
        "request_more_views": 0,
        "switch_or_expand_hypothesis": 0,
    }
    for t in tool_trace or []:
        name = t.get("tool_name")
        if name == "view_keyframe_marked":
            inp = t.get("tool_input") or {}
            fid = inp.get("frame_id")
            if isinstance(fid, int) and fid not in seen:
                seen.add(fid)
                by_source["view_keyframe_marked"] += 1
        elif name in ("request_more_views", "switch_or_expand_hypothesis"):
            resp = t.get("response_text") or ""
            for m in _NEW_VIEW_IDS_RE.finditer(resp):
                ids_str = m.group(1)
                for chunk in ids_str.split(","):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    try:
                        fid = int(chunk)
                    except ValueError:
                        continue
                    if fid not in seen:
                        seen.add(fid)
                        by_source[name] += 1
    return sorted(seen), by_source


def _read_pack_sample(data_root: Path, pack_name: str, sample_id: str) -> dict:
    parts = sample_id.split("/")
    scene_id = parts[-1].split("::")[0]
    safe = _safe_sample_id(sample_id)
    p = data_root / scene_id / pack_name / "samples" / f"{safe}.json"
    if not p.exists():
        raise FileNotFoundError(f"pack sample missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_cvra_telemetry(
    tool_trace: list[dict],
    cvra_addressable_for_sample: dict | None = None,
) -> dict:
    """Walk a per-sample tool_trace and extract CVRA telemetry from
    every find_proposals_by_category call.

    Returns a dict with:
      - cvra_aug_count: int — total clip_visible_aug entries across all calls
      - cvra_overflow_count: int — entries with cvra_overflow=True
      - cvra_aug_pids: list[int] — sorted union of clip_visible_aug pids
      - cvra_label_hit_pids: list[int] — sorted union of label_hits pids
      - cvra_find_calls: int — number of find_proposals_by_category calls
      - cvra_exhausted: bool — any call surfaced cvra_exhausted=True
      - clip_visible_aug_included_gt_overlap: bool | None — True iff
        cvra_addressable_for_sample is provided AND any of its
        gt_overlap_pids appears in cvra_aug_pids ∪ cvra_label_hit_pids.
        None when addressable record is missing.

    The telemetry is computed independently of CVRA being enabled at
    runtime: a trace from a CVRA-off run will yield zeros across the
    board (no `clip_visible_aug` payload). This makes the field safe
    to emit on every funnel run.
    """
    aug_pids: set[int] = set()
    label_pids: set[int] = set()
    aug_count = 0
    overflow_count = 0
    find_calls = 0
    cvra_exhausted = False
    for t in tool_trace or []:
        if t.get("tool_name") != "find_proposals_by_category":
            continue
        find_calls += 1
        try:
            payload = json.loads(t.get("response_text") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        for entry in payload.get("clip_visible_aug") or []:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("proposal_id")
            if isinstance(pid, int):
                aug_pids.add(pid)
            aug_count += 1
            if entry.get("cvra_overflow") is True:
                overflow_count += 1
        for entry in payload.get("label_hits") or []:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("proposal_id")
            if isinstance(pid, int):
                label_pids.add(pid)
        if payload.get("cvra_exhausted") is True:
            cvra_exhausted = True

    included: bool | None = None
    if cvra_addressable_for_sample is not None:
        gt_pids = set(cvra_addressable_for_sample.get("gt_overlap_pids") or [])
        if gt_pids:
            included = bool(gt_pids & (aug_pids | label_pids))

    return {
        "cvra_find_calls": find_calls,
        "cvra_aug_count": aug_count,
        "cvra_overflow_count": overflow_count,
        "cvra_aug_pids": sorted(aug_pids),
        "cvra_label_hit_pids": sorted(label_pids),
        "cvra_exhausted": cvra_exhausted,
        "clip_visible_aug_included_gt_overlap": included,
    }


def _load_cvra_addressable_audit(path: Path | None) -> dict[str, dict]:
    """Load a sidecar audit JSON mapping sample_id -> {gt_overlap_pids: [..]}.

    Used to compute clip_visible_aug_included_gt_overlap per sample
    (spec § L PASS-gate retrieval-recall metric). Format:

        {
          "sample_id": {
            "gt_overlap_pids": [int, ...],   # proposal ids whose 3D bbox
                                              # IoU >= 0.25 with GT bbox
            ...
          },
          ...
        }

    The format is deliberately minimal so this can be derived from any
    upstream addressability audit (currently
    tmp/cvra_addressable_audit_v2.md / .json sidecar). When path is
    None, the funnel emits None for the included field on every sample.
    """
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"CVRA addressable audit JSON missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"CVRA addressable audit JSON must be a dict[sample_id -> ...]: {path}"
        )
    return raw


# ---------------------------------------------------------------------
# F0 — query-parser fidelity (optional, --run-parser)
# ---------------------------------------------------------------------


def _scene_categories_from_proposals(proposals: list[dict]) -> list[str]:
    """Mirror the pack-prep convention: scene_categories = sorted set of
    Mask3D-CG proposal labels (the parser's allowed vocab)."""
    return sorted({str(p.get("label", "")).lower().strip() for p in proposals if p.get("label")})


def _hypothesis_output_to_dict(output: Any) -> dict:
    """Normalize HypothesisOutputV1 (or any pydantic model) to a dict."""
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return dict(output)  # last resort


def _extract_rank1_categories(output: Any) -> list[str]:
    """Pull the rank-1 hypothesis's top-level categories. Returns [] on
    any unexpected shape (so the caller can record a parse-fail case)."""
    try:
        ordered = output.ordered_hypotheses()
        if not ordered:
            return []
        rank1 = ordered[0]
        cats = list(rank1.grounding_query.root.categories or [])
        return [str(c) for c in cats if isinstance(c, str)]
    except Exception:
        return []


def _load_parser_cache(path: Path) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _save_parser_cache(path: Path, cache: dict[str, dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_query_with_cache(
    *,
    sample_id: str,
    query: str,
    parser: Any,
    cache: dict[str, dict],
    parser_model: str,
    scene_categories: list[str],
) -> dict:
    """Parse `query` (with caching keyed by sample_id) and return a
    dict with categories, raw output, and runtime in ms. Cache hit is
    a 3-key match (sample_id, query, parser_model). Errors are recorded.
    """
    import time

    hit = cache.get(sample_id)
    if (
        isinstance(hit, dict)
        and hit.get("query") == query
        and hit.get("parser_model") == parser_model
        and "categories" in hit
    ):
        return hit

    t0 = time.perf_counter()
    try:
        output = parser.parse(query)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        rank1_categories = _extract_rank1_categories(output)
        all_cats: list[str] = []
        seen: set[str] = set()
        try:
            for h in output.ordered_hypotheses():
                for c in h.grounding_query.get_all_categories():
                    if isinstance(c, str) and c not in seen:
                        seen.add(c)
                        all_cats.append(c)
        except Exception:
            all_cats = list(rank1_categories)
        entry = {
            "query": query,
            "parser_model": parser_model,
            "scene_categories_count": len(scene_categories),
            "categories": rank1_categories,
            "all_categories": all_cats,
            "raw_hypothesis_output": _hypothesis_output_to_dict(output),
            "parser_runtime_ms": runtime_ms,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — record error per-sample
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        entry = {
            "query": query,
            "parser_model": parser_model,
            "scene_categories_count": len(scene_categories),
            "categories": [],
            "all_categories": [],
            "raw_hypothesis_output": None,
            "parser_runtime_ms": runtime_ms,
            "error": f"{type(exc).__name__}: {exc}",
        }

    cache[sample_id] = entry
    return entry


def _parse_batch_concurrent(
    *,
    requests: list[dict],
    parser_by_scene: dict[str, Any],
    cache: dict[str, dict],
    parser_model: str,
    parser_cache_path: Path | None,
    max_workers: int,
) -> None:
    """Parse a batch of (sample_id, scene_id, query, scene_categories)
    requests in parallel against the gemini pool. The QueryParser
    instances are pre-built per scene; we reuse them across samples.

    Mutates `cache` in-place. Flushes cache every 10 completions.
    Cache hits are skipped (so re-runs after a crash are cheap).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from loguru import logger as _flog

    pending = []
    for r in requests:
        hit = cache.get(r["sample_id"])
        if (
            isinstance(hit, dict)
            and hit.get("query") == r["query"]
            and hit.get("parser_model") == parser_model
            and "categories" in hit
        ):
            continue
        pending.append(r)
    if not pending:
        _flog.info("[funnel] parser cache covers all {} samples; nothing to parse", len(requests))
        return

    _flog.info(
        "[funnel] parsing {}/{} sample queries (max_workers={}, model={})",
        len(pending), len(requests), max_workers, parser_model,
    )

    def _do(req: dict) -> tuple[str, dict]:
        parser = parser_by_scene[req["scene_id"]]
        entry = _parse_query_with_cache(
            sample_id=req["sample_id"],
            query=req["query"],
            parser=parser,
            cache={},  # per-call cache; result is then merged into shared cache below
            parser_model=parser_model,
            scene_categories=req["scene_categories"],
        )
        return req["sample_id"], entry

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do, r) for r in pending]
        for fut in as_completed(futures):
            sample_id, entry = fut.result()
            cache[sample_id] = entry
            completed += 1
            if parser_cache_path is not None and (completed % 10 == 0):
                _save_parser_cache(parser_cache_path, cache)
            if completed % 20 == 0:
                _flog.info(
                    "[funnel] parser progress: {}/{} done", completed, len(pending)
                )
    if parser_cache_path is not None:
        _save_parser_cache(parser_cache_path, cache)


def _read_per_sample_checkpoint(
    per_sample_dir: Path, sample_id: str
) -> dict | None:
    safe = _safe_sample_id(sample_id)
    digest = _digest8(sample_id)
    candidate = per_sample_dir / f"{safe}_{digest}.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    # fall back: any file starting with the safe prefix
    matches = sorted(per_sample_dir.glob(f"{safe}_*.json"))
    if matches:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    return None


def evaluate(
    *,
    sample_ids_path: Path,
    pack_name: str,
    data_root: Path,
    phase8_data_root: Path,
    per_sample_dir: Path,
    output_path: Path,
    run_parser: bool = False,
    parser_cache_path: Path | None = None,
    parser_model: str = "gemini-2.5-pro",
    parser_max_workers: int = 8,
    quiet_parser_logging: bool = True,
    cvra_addressable_audit_path: Path | None = None,
) -> dict:
    fold = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if not isinstance(fold, list):
        raise ValueError(f"fold JSON must be a list: {sample_ids_path}")

    # cache scene-level data
    phase8_obj_to_views: dict[str, dict[int, list[tuple[int, float]]]] = {}
    mask3d_proposals: dict[str, list[dict]] = {}
    mask3d_view_to_objects: dict[
        str, dict[int, list[tuple[int, float]]]
    ] = {}
    # F0 parser state (only populated when run_parser=True)
    parser_cache: dict[str, dict] = (
        _load_parser_cache(parser_cache_path) if run_parser else {}
    )
    parser_by_scene: dict[str, Any] = {}

    # CVRA addressable audit sidecar (optional). Maps sample_id ->
    # {gt_overlap_pids: [int]} so we can compute the spec § L
    # `clip_visible_aug_included_gt_overlap` PASS-gate metric per sample.
    # When path is None or sample missing from the audit, the field is None.
    cvra_addressable_audit = _load_cvra_addressable_audit(
        cvra_addressable_audit_path
    )

    # Optional pre-pass: parse all queries concurrently against the gemini
    # pool, then look up results in the per-sample loop below. This avoids
    # the multi-minute serial latency that QueryParser.parse() incurs per
    # call when run sequentially.
    if run_parser:
        from loguru import logger as _flog
        if quiet_parser_logging:
            # QueryParser uses loguru INFO heavily; suppress unless user
            # sets LOGURU_LEVEL explicitly.
            _flog.disable("query_scene.query_parser")

        # Pre-load Mask3D pool per scene so we have scene_categories ready
        # for the parser pre-pass.
        scene_ids = sorted({row["scene_id"] for row in fold})
        for sid in scene_ids:
            if sid not in mask3d_proposals:
                proposals_, _ = _load_mask3d_pool(data_root / sid)
                mask3d_proposals[sid] = proposals_
                mask3d_view_to_objects[sid] = (
                    _load_mask3d_visibility_view_to_objects(data_root / sid)
                )

        from query_scene.query_parser import QueryParser
        for sid in scene_ids:
            scene_cats = _scene_categories_from_proposals(mask3d_proposals[sid])
            parser_by_scene[sid] = QueryParser(
                llm_model=parser_model,
                scene_categories=scene_cats,
            )

        # Build parse requests
        parse_requests: list[dict] = []
        for row in fold:
            sample_id = row["sample_id"]
            scene_id = row["scene_id"]
            try:
                pack_sample_ = _read_pack_sample(data_root, pack_name, sample_id)
            except FileNotFoundError:
                continue
            query_text = pack_sample_.get("query") or ""
            if not query_text.strip():
                continue
            parse_requests.append(
                {
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "query": query_text,
                    "scene_categories": _scene_categories_from_proposals(
                        mask3d_proposals[scene_id]
                    ),
                }
            )

        _parse_batch_concurrent(
            requests=parse_requests,
            parser_by_scene=parser_by_scene,
            cache=parser_cache,
            parser_model=parser_model,
            parser_cache_path=parser_cache_path,
            max_workers=parser_max_workers,
        )

    per_sample: list[dict] = []
    for row in fold:
        sample_id = row["sample_id"]
        scene_id = row["scene_id"]
        target_id = int(row["target_id"])
        target_category = str(row.get("category") or "")

        # Phase 8 visibility for the GT target object
        if scene_id not in phase8_obj_to_views:
            phase8_obj_to_views[scene_id] = _load_phase8_visibility(
                phase8_data_root / scene_id
            )
        phase8_views_for_target = phase8_obj_to_views[scene_id].get(target_id, [])
        phase8_visible_frames = {f for f, _ in phase8_views_for_target}
        f1_phase8_has_visible = bool(phase8_visible_frames)

        # Pack-prep sample artifact -> initial keyframes
        pack_sample = _read_pack_sample(data_root, pack_name, sample_id)
        initial_keyframes = pack_sample.get("keyframes") or []
        initial_frame_ids = [int(k["frame_id"]) for k in initial_keyframes]

        # F2: any initial KF visible in Phase 8 for the target id
        f2_initial_kf_covers_phase8_target = bool(
            phase8_visible_frames & set(initial_frame_ids)
        )

        # Mask3D pool / visibility for category-level coverage
        if scene_id not in mask3d_proposals:
            proposals, _ = _load_mask3d_pool(
                data_root / scene_id  # scanrefer scannet root has the Mask3D pkl
            )
            mask3d_proposals[scene_id] = proposals
            mask3d_view_to_objects[scene_id] = (
                _load_mask3d_visibility_view_to_objects(data_root / scene_id)
            )
        proposals = mask3d_proposals[scene_id]
        view_to_objects = mask3d_view_to_objects[scene_id]
        mask3d_pool_has_match = any(
            _category_match(p["label"], [target_category]) for p in proposals
        )

        # F3: at least one initial KF carries a Mask3D candidate of GT category
        labels_by_id = {p["id"]: p["label"] for p in proposals}
        f3_initial_kf_carries_mask3d_match = False
        for fid in initial_frame_ids:
            entries = view_to_objects.get(fid, [])
            for oid, _w in entries:
                if _category_match(labels_by_id.get(int(oid), ""), [target_category]):
                    f3_initial_kf_carries_mask3d_match = True
                    break
            if f3_initial_kf_carries_mask3d_match:
                break

        # Cumulative keyframes after Stage 2 callbacks
        per_chk = _read_per_sample_checkpoint(per_sample_dir, sample_id)
        tool_trace = (per_chk or {}).get("tool_trace") or []
        cumulative_frame_ids, kf_breakdown = _extract_cumulative_frame_ids(
            initial_frame_ids, tool_trace
        )
        f4_cumulative_covers_phase8_target = bool(
            phase8_visible_frames & set(cumulative_frame_ids)
        )

        # Cumulative same-category Mask3D coverage (whether the agent
        # ever had a same-category mark on screen).
        cumulative_mask3d_match = False
        for fid in cumulative_frame_ids:
            for oid, _w in view_to_objects.get(fid, []):
                if _category_match(labels_by_id.get(int(oid), ""), [target_category]):
                    cumulative_mask3d_match = True
                    break
            if cumulative_mask3d_match:
                break

        # F5: agent picked correctly under whatever evaluator the per-sample
        # was scored with (Phase 8 GT-CG; the same eval the pack used).
        agent_iou = float((per_chk or {}).get("iou", 0.0)) if per_chk else 0.0
        f5_agent_correct_25 = agent_iou >= 0.25
        f5_agent_correct_50 = agent_iou >= 0.50

        # F0 (optional) — query-parser fidelity vs GT category. The
        # parse pre-pass populated parser_cache; we only read here.
        f0_parser_match: bool | None = None
        parser_categories: list[str] = []
        parser_runtime_ms: float | None = None
        parser_error: str | None = None
        if run_parser:
            entry = parser_cache.get(sample_id)
            if isinstance(entry, dict):
                parser_categories = list(entry.get("categories") or [])
                parser_runtime_ms = float(entry.get("parser_runtime_ms") or 0.0)
                parser_error = entry.get("error")
                if parser_error:
                    f0_parser_match = None  # parse failed; do not count as no
                else:
                    f0_parser_match = bool(
                        _category_match(target_category, parser_categories)
                    )

        # CVRA telemetry per spec § L. Always extracted (zero-cost when
        # `clip_visible_aug` is absent, e.g. CVRA-off runs or pre-CVRA
        # packs). When a CVRA addressable audit sidecar is provided,
        # `clip_visible_aug_included_gt_overlap` is computed; otherwise None.
        cvra = _extract_cvra_telemetry(
            tool_trace,
            cvra_addressable_for_sample=cvra_addressable_audit.get(sample_id),
        )

        per_sample.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "target_id": target_id,
                "target_category": target_category,
                "is_unique": pack_sample.get("is_unique"),
                # raw inputs to the funnel
                "n_phase8_visible_frames": len(phase8_visible_frames),
                "initial_frame_ids": initial_frame_ids,
                "n_cumulative_frame_ids": len(cumulative_frame_ids),
                "kf_breakdown": kf_breakdown,
                "mask3d_pool_size": len(proposals),
                "mask3d_pool_has_category_match": mask3d_pool_has_match,
                "agent_iou": agent_iou,
                "agent_selected_object_id": (
                    (per_chk or {}).get("selected_object_id") if per_chk else None
                ),
                # boolean funnel checkpoints
                "F1_phase8_has_visible_frames": f1_phase8_has_visible,
                "F2_initial_kf_covers_phase8_target": f2_initial_kf_covers_phase8_target,
                "F3_initial_kf_carries_mask3d_category_match": (
                    f3_initial_kf_carries_mask3d_match
                ),
                "F4_cumulative_kf_covers_phase8_target": (
                    f4_cumulative_covers_phase8_target
                ),
                "F4b_cumulative_mask3d_category_match": cumulative_mask3d_match,
                "F5a_agent_iou_ge_025": f5_agent_correct_25,
                "F5b_agent_iou_ge_050": f5_agent_correct_50,
                "checkpoint_present": per_chk is not None,
                # F0 (only populated when run_parser=True; otherwise None)
                "F0_parser_target_matches_gt_category": f0_parser_match,
                "parser_categories": parser_categories,
                "parser_runtime_ms": parser_runtime_ms,
                "parser_error": parser_error,
                # CVRA telemetry (spec § L). Always emitted; zero when CVRA off.
                "cvra_find_calls": cvra["cvra_find_calls"],
                "cvra_aug_count": cvra["cvra_aug_count"],
                "cvra_overflow_count": cvra["cvra_overflow_count"],
                "cvra_aug_pids": cvra["cvra_aug_pids"],
                "cvra_label_hit_pids": cvra["cvra_label_hit_pids"],
                "cvra_exhausted": cvra["cvra_exhausted"],
                "clip_visible_aug_included_gt_overlap": (
                    cvra["clip_visible_aug_included_gt_overlap"]
                ),
            }
        )

    # final cache flush
    if run_parser and parser_cache_path is not None:
        _save_parser_cache(parser_cache_path, parser_cache)

    n = len(per_sample)
    summary = {
        "n_total": n,
        "n_F1_phase8_has_visible_frames": sum(
            r["F1_phase8_has_visible_frames"] for r in per_sample
        ),
        "n_F2_initial_kf_covers_phase8_target": sum(
            r["F2_initial_kf_covers_phase8_target"] for r in per_sample
        ),
        "n_F3_initial_kf_carries_mask3d_category_match": sum(
            r["F3_initial_kf_carries_mask3d_category_match"] for r in per_sample
        ),
        "n_F4_cumulative_kf_covers_phase8_target": sum(
            r["F4_cumulative_kf_covers_phase8_target"] for r in per_sample
        ),
        "n_F4b_cumulative_mask3d_category_match": sum(
            r["F4b_cumulative_mask3d_category_match"] for r in per_sample
        ),
        "n_F5a_agent_iou_ge_025": sum(r["F5a_agent_iou_ge_025"] for r in per_sample),
        "n_F5b_agent_iou_ge_050": sum(r["F5b_agent_iou_ge_050"] for r in per_sample),
        "n_mask3d_pool_has_category_match": sum(
            r["mask3d_pool_has_category_match"] for r in per_sample
        ),
        "n_checkpoint_present": sum(r["checkpoint_present"] for r in per_sample),
    }
    if run_parser:
        f0_yes = sum(
            1 for r in per_sample if r.get("F0_parser_target_matches_gt_category") is True
        )
        f0_evaluated = sum(
            1 for r in per_sample if r.get("F0_parser_target_matches_gt_category") is not None
        )
        f0_errors = sum(1 for r in per_sample if r.get("parser_error"))
        summary["n_F0_parser_target_matches_gt_category"] = f0_yes
        summary["n_F0_parser_evaluated"] = f0_evaluated
        summary["n_F0_parser_errors"] = f0_errors

    # CVRA aggregate telemetry (always emitted)
    summary["cvra_total_aug_emissions"] = sum(
        r["cvra_aug_count"] for r in per_sample
    )
    summary["cvra_total_overflow_emissions"] = sum(
        r["cvra_overflow_count"] for r in per_sample
    )
    summary["cvra_total_find_calls"] = sum(r["cvra_find_calls"] for r in per_sample)
    summary["n_samples_with_cvra_aug"] = sum(
        1 for r in per_sample if r["cvra_aug_count"] > 0
    )
    summary["n_samples_with_cvra_exhausted"] = sum(
        1 for r in per_sample if r.get("cvra_exhausted")
    )
    if cvra_addressable_audit:
        n_evaluated = sum(
            1
            for r in per_sample
            if r.get("clip_visible_aug_included_gt_overlap") is not None
        )
        n_included = sum(
            1
            for r in per_sample
            if r.get("clip_visible_aug_included_gt_overlap") is True
        )
        summary["n_clip_visible_aug_included_gt_overlap"] = n_included
        summary["n_clip_visible_aug_evaluated"] = n_evaluated

    summary["pct"] = {k: f"{(v / n) * 100:.1f}%" for k, v in summary.items() if k.startswith("n_")}

    output = {
        "fold": str(sample_ids_path),
        "pack_name": pack_name,
        "data_root": str(data_root),
        "phase8_data_root": str(phase8_data_root),
        "per_sample_dir": str(per_sample_dir),
        "run_parser": run_parser,
        "parser_model": parser_model if run_parser else None,
        "parser_cache_path": str(parser_cache_path) if (run_parser and parser_cache_path) else None,
        "cvra_addressable_audit_path": (
            str(cvra_addressable_audit_path) if cvra_addressable_audit_path else None
        ),
        "summary": summary,
        "per_sample": per_sample,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def _print_summary(report: dict) -> None:
    s = report["summary"]
    n = s["n_total"]
    print(f"\nFunnel report (n={n}, pack={report['pack_name']})")
    print(f"  per_sample_dir: {report['per_sample_dir']}")
    rows: list[tuple[str, int]] = []
    if report.get("run_parser"):
        rows.append(
            (
                "F0: parser_target_matches_gt_category",
                s.get("n_F0_parser_target_matches_gt_category", 0),
            )
        )
    rows.extend(
        [
            ("F1: Phase 8 has visible frames for target_id", s["n_F1_phase8_has_visible_frames"]),
            ("F2: initial KFs cover Phase 8 target frame", s["n_F2_initial_kf_covers_phase8_target"]),
            ("F3: initial KFs carry Mask3D candidate of GT category", s["n_F3_initial_kf_carries_mask3d_category_match"]),
            ("F4: cumulative KFs cover Phase 8 target frame", s["n_F4_cumulative_kf_covers_phase8_target"]),
            ("F4b: cumulative KFs carry Mask3D candidate of GT category", s["n_F4b_cumulative_mask3d_category_match"]),
            ("F5a: agent IoU >= 0.25 (Phase 8 eval)", s["n_F5a_agent_iou_ge_025"]),
            ("F5b: agent IoU >= 0.50 (Phase 8 eval)", s["n_F5b_agent_iou_ge_050"]),
            ("(ref) Mask3D pool has same-category candidate", s["n_mask3d_pool_has_category_match"]),
            ("(ref) per-sample checkpoint present", s["n_checkpoint_present"]),
        ]
    )
    for label, k in rows:
        print(f"  {label:<60s} {k:3d} / {n:<3d} ({k / n * 100:5.1f}%)")
    if report.get("run_parser"):
        f0_err = s.get("n_F0_parser_errors", 0)
        if f0_err:
            print(f"  (parser errors: {f0_err} / {n})")
        print(f"  (parser model: {report.get('parser_model')})")
        print(f"  (parser cache: {report.get('parser_cache_path')})")

    # CVRA telemetry block (always emitted; values 0 when CVRA off)
    cvra_total_aug = s.get("cvra_total_aug_emissions", 0)
    cvra_total_overflow = s.get("cvra_total_overflow_emissions", 0)
    n_with_aug = s.get("n_samples_with_cvra_aug", 0)
    n_exhausted = s.get("n_samples_with_cvra_exhausted", 0)
    if cvra_total_aug or report.get("cvra_addressable_audit_path"):
        print()
        print("  CVRA telemetry (find_proposals_by_category):")
        print(
            f"    samples with any clip_visible_aug:        "
            f"{n_with_aug:3d} / {n:<3d} ({n_with_aug / n * 100:5.1f}%)"
        )
        print(
            f"    total clip_visible_aug emissions:         {cvra_total_aug:5d}"
        )
        print(
            f"    total cvra_overflow=True emissions:       {cvra_total_overflow:5d}"
        )
        print(
            f"    samples with cvra_exhausted=True:         "
            f"{n_exhausted:3d} / {n:<3d} ({n_exhausted / n * 100:5.1f}%)"
        )
        if report.get("cvra_addressable_audit_path"):
            n_eval = s.get("n_clip_visible_aug_evaluated", 0)
            n_inc = s.get("n_clip_visible_aug_included_gt_overlap", 0)
            denom = n_eval if n_eval else 1
            print(
                f"    clip_visible_aug_included_gt_overlap:    "
                f" {n_inc:3d} / {n_eval:<3d} ({n_inc / denom * 100:5.1f}%)"
            )
            print(
                f"    (audit: {report['cvra_addressable_audit_path']})"
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-ids", required=True, type=Path)
    p.add_argument(
        "--pack-name",
        required=True,
        help="Pack used at runtime, e.g. pack_scanrefer_v3p_iterative.",
    )
    p.add_argument("--data-root", default=Path("data/scanrefer/scannet"), type=Path)
    p.add_argument(
        "--phase8-data-root", default=Path("data/nr3d/scannet"), type=Path
    )
    p.add_argument(
        "--per-sample-dir",
        required=True,
        type=Path,
        help="Directory of per-sample checkpoints, e.g. "
        "tmp/<run>/per_sample/<pack-name>/",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--run-parser",
        action="store_true",
        default=False,
        help="If set, run the LLM query parser per sample to compute F0 "
        "(parser-target ↔ GT-category match). Default off; F0 is None when off.",
    )
    p.add_argument(
        "--parser-cache",
        type=Path,
        default=None,
        help="Path to parser-cache JSON. Defaults to sibling of --output: "
        "<output_stem>_parser_cache.json. Re-runs reuse cached parses.",
    )
    p.add_argument(
        "--parser-model",
        default="gemini-2.5-pro",
        help="LLM model for the F0 parser. Default gemini-2.5-pro (pool-enabled).",
    )
    p.add_argument(
        "--parser-max-workers",
        type=int,
        default=8,
        help="Concurrent worker count for the F0 parse pre-pass (gemini "
        "pool is auto-enabled). Default 8; 16 is safe per pool docs.",
    )
    p.add_argument(
        "--parser-verbose",
        action="store_true",
        default=False,
        help="If set, do NOT silence QueryParser's INFO logging "
        "(default: suppress to keep funnel output tight).",
    )
    p.add_argument(
        "--cvra-addressable-audit",
        type=Path,
        default=None,
        help="Optional sidecar JSON mapping sample_id -> "
        "{gt_overlap_pids: [int]} so the funnel can compute "
        "clip_visible_aug_included_gt_overlap (spec § L PASS metric). "
        "When omitted, the field is None per sample.",
    )
    args = p.parse_args()
    parser_cache = args.parser_cache
    if args.run_parser and parser_cache is None:
        parser_cache = args.output.with_name(args.output.stem + "_parser_cache.json")
    report = evaluate(
        sample_ids_path=args.sample_ids,
        pack_name=args.pack_name,
        data_root=args.data_root,
        phase8_data_root=args.phase8_data_root,
        per_sample_dir=args.per_sample_dir,
        output_path=args.output,
        run_parser=args.run_parser,
        parser_cache_path=parser_cache,
        parser_model=args.parser_model,
        parser_max_workers=args.parser_max_workers,
        quiet_parser_logging=not args.parser_verbose,
        cvra_addressable_audit_path=args.cvra_addressable_audit,
    )
    _print_summary(report)


if __name__ == "__main__":
    main()


__all__ = [
    "evaluate",
    "_extract_cumulative_frame_ids",
    "_extract_cvra_telemetry",
    "_load_cvra_addressable_audit",
    "_category_match",
    "_category_tokens",
    "_extract_rank1_categories",
    "_parse_query_with_cache",
    "_load_parser_cache",
    "_save_parser_cache",
]
