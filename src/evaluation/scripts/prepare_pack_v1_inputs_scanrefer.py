"""Prepare offline pack inputs for ScanRefer VG runs from Mask3D-CG output.

Mirrors prepare_pack_v1_inputs_nr3d.py with two pkl sources:
- Proposal pool: Mask3D-CG pkl (full_pcd_mask3d_axisaligned.pkl.gz)
- GT bbox lookup: Phase 8 GT-CG pkl (full_pcd_gt_axisaligned_post.pkl.gz)
  via ScanRefVGDataset (carries gt_bbox_3d on each sample).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.catalog import from_vg_proposal_pool
from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)
from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    project_visible_bbox_3d_to_2d,
)
from benchmarks.scanrefer_loader import (
    ScanRefVGDataset,
    ScanRefVGSample,
)
from evaluation.scripts.prepare_pack_v1_inputs import (
    load_image_size,
    validate_bbox_9dof,
    validate_matrix_4x4,
)
from query_scene.scene_bev_builder import ScanReferScanNetBEVBuilder

MASK3D_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz")
VIS_REL = Path("conceptgraph/indices/visibility_index.pkl")


@dataclass(frozen=True)
class SampleRequest:
    sample_id: str
    scene_id: str
    target_id: int
    ann_id: str
    category: str


@dataclass(frozen=True)
class SceneFrame:
    frame_id: int
    raw_frame_id: int
    rgb_path: Path
    extrinsic_world_to_cam: np.ndarray


@dataclass(frozen=True)
class Mask3dVisibility:
    object_to_views: dict[int, list[tuple[int, float]]]
    view_to_objects: dict[int, list[tuple[int, float]]]


@dataclass(frozen=True)
class SceneArtifacts:
    scene_dir: Path
    proposals_jsonl: Path
    visibility_json: Path
    annotated_dir: Path
    frame_visibility: dict[int, list[int]]
    proposal_ids: list[int]
    proposal_labels: dict[int, str]
    mask3d_visibility: Mask3dVisibility
    scene_categories: list[str]


def parse_sample_id(sample_id: str) -> tuple[str, int, str]:
    parts = sample_id.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Expected ScanRefer sample_id format '<scan_id>::<target_id>::<ann_id>', "
            f"got {sample_id!r}"
        )
    scan_id, target_text, ann_id = parts
    if not scan_id or not ann_id:
        raise ValueError(f"Invalid ScanRefer sample_id={sample_id!r}")
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"Invalid target_id in sample_id={sample_id!r}") from exc
    scene = scan_id.split("/")[-1]
    return scene, target_id, ann_id


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="ScanRefer scannet root, e.g. data/scanrefer/scannet",
    )
    parser.add_argument(
        "--phase8-data-root",
        default=Path("data/nr3d/scannet"),
        type=Path,
        help="Phase 8 GT-CG root for GT bbox lookup",
    )
    parser.add_argument(
        "--scanrefer-root",
        default=Path("data/scanrefer"),
        type=Path,
        help="Root containing raw/ScanRefer_filtered_*.json",
    )
    parser.add_argument(
        "--raw-frames-root",
        default=Path("data/nr3d/scannet"),
        type=Path,
        help="Root with <scene>/raw/ frames; ScanRefer "
        "scenes are a superset of NR3D scenes so this is shared.",
    )
    parser.add_argument("--pack-name", default="pack_scanrefer_v1")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--keyframe-mode",
        default="gt_target",
        choices=["gt_target", "query_driven", "mask3d_query_driven"],
        help=(
            "How to pick the 3-5 initial RGB keyframes per query. "
            "'gt_target' (v1/v2 default): top-5 frames where the GT "
            "target_id is most visible (a GT view oracle). "
            "'query_driven' (v3): KeyframeSelector.select_keyframes_v2(query, k=3) "
            "via Phase 8 GT visibility — same hypothesis-driven entry OpenEQA uses. "
            "'mask3d_query_driven' (v3.1): parse query → categories → score frames "
            "by Mask3D-CG visibility weight of same-category candidates → top-3. "
            "Aligned with the visibility index used to render annotated PNGs."
        ),
    )
    parser.add_argument(
        "--keyframe-llm-model",
        default="gemini-2.5-pro",
        help="LLM for hypothesis parsing in query_driven mode (ignored otherwise).",
    )
    return parser.parse_args()


def prepare_pack_v1_inputs_scanrefer(
    *,
    sample_ids_path: Path,
    data_root: Path,
    pack_name: str = "pack_scanrefer_v1",
    split: str = "val",
    scanrefer_root: Path = Path("data/scanrefer"),
    phase8_data_root: Path = Path("data/nr3d/scannet"),
    raw_frames_root: Path = Path("data/nr3d/scannet"),
    max_samples: int | None = None,
    keyframe_mode: str = "gt_target",
    keyframe_llm_model: str = "gemini-2.5-pro",
) -> list[Path]:
    if keyframe_mode not in ("gt_target", "query_driven", "mask3d_query_driven"):
        raise ValueError(
            "keyframe_mode must be 'gt_target', 'query_driven', or "
            f"'mask3d_query_driven', got {keyframe_mode!r}"
        )

    requests = load_sample_requests(sample_ids_path)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        requests = requests[:max_samples]

    requested_sids = {r.sample_id for r in requests}
    ds = ScanRefVGDataset.from_path(
        data_root=scanrefer_root,
        phase8_data_root=phase8_data_root,
        split=split,
        sample_ids=requested_sids,
    )
    sample_lookup: dict[str, ScanRefVGSample] = {s.sample_id: s for s in ds}

    scene_artifacts: dict[str, SceneArtifacts] = {}
    selector_cache: dict[str, Any] = {}  # scene_id -> KeyframeSelector (lazy)
    parser_cache: dict[str, Any] = {}  # scene_id -> QueryParser (lazy)
    fallback_count = 0
    written: list[Path] = []
    for request in requests:
        sample = sample_lookup.get(request.sample_id)
        if sample is None:
            raise ValueError(f"No ScanRefer sample for sample_id={request.sample_id!r}")
        if request.scene_id not in scene_artifacts:
            scene_artifacts[request.scene_id] = prepare_scene_artifacts(
                scene_id=request.scene_id,
                data_root=data_root,
                raw_frames_root=raw_frames_root,
                pack_name=pack_name,
            )
        keyframes, used_fallback = _select_keyframes(
            request=request,
            sample=sample,
            raw_frames_root=raw_frames_root,
            phase8_data_root=phase8_data_root,
            scene_artifacts=scene_artifacts[request.scene_id],
            keyframe_mode=keyframe_mode,
            keyframe_llm_model=keyframe_llm_model,
            selector_cache=selector_cache,
            parser_cache=parser_cache,
        )
        if used_fallback:
            fallback_count += 1
        written.append(
            write_sample_artifact(
                request=request,
                sample=sample,
                data_root=data_root,
                raw_frames_root=raw_frames_root,
                scene_artifacts=scene_artifacts[request.scene_id],
                keyframes=keyframes,
                keyframe_mode=keyframe_mode,
            )
        )
    if keyframe_mode in ("query_driven", "mask3d_query_driven"):
        from loguru import logger

        logger.info(
            "[prepare_pack_v1_inputs_scanrefer] {} mode: "
            "{}/{} samples used Mask3D-density fallback ({:.2%})",
            keyframe_mode,
            fallback_count,
            len(requests),
            fallback_count / max(len(requests), 1),
        )
    return written


def _select_keyframes(
    *,
    request: SampleRequest,
    sample: ScanRefVGSample,
    raw_frames_root: Path,
    phase8_data_root: Path,
    scene_artifacts: SceneArtifacts,
    keyframe_mode: str,
    keyframe_llm_model: str,
    selector_cache: dict[str, Any],
    parser_cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Dispatch keyframe selection by mode. Returns (keyframes, used_fallback)."""
    if keyframe_mode == "gt_target":
        kfs = select_keyframes_from_phase8_target(
            scene_id=request.scene_id,
            target_id=request.target_id,
            raw_frames_root=phase8_data_root,
            k=5,
        )
        return kfs, False

    if keyframe_mode == "mask3d_query_driven":
        parser = parser_cache.get(request.scene_id)
        if parser is None:
            from loguru import logger

            from query_scene.query_parser import QueryParser

            logger.info(
                "[prepare_pack_v1_inputs_scanrefer] building QueryParser for {} "
                "(LLM={}, n_categories={})",
                request.scene_id,
                keyframe_llm_model,
                len(scene_artifacts.scene_categories),
            )
            parser = QueryParser(
                llm_model=keyframe_llm_model,
                scene_categories=scene_artifacts.scene_categories,
            )
            parser_cache[request.scene_id] = parser
        kfs = select_keyframes_mask3d_query_driven(
            scene_id=request.scene_id,
            query=sample.query,
            scene_artifacts=scene_artifacts,
            raw_frames_root=raw_frames_root,
            query_parser=parser,
            k=3,
        )
        if kfs:
            return kfs, False
        return (
            _fallback_top5_by_mask3d_density(
                scene_id=request.scene_id,
                scene_artifacts=scene_artifacts,
                raw_frames_root=raw_frames_root,
                k=3,
            ),
            True,
        )

    # query_driven path
    selector = selector_cache.get(request.scene_id)
    if selector is None:
        from loguru import logger

        from query_scene.keyframe_selector import KeyframeSelector

        scene_cg_root = phase8_data_root / request.scene_id / "conceptgraph"
        logger.info(
            "[prepare_pack_v1_inputs_scanrefer] building Stage1 KeyframeSelector "
            "for {} (LLM={})",
            request.scene_id,
            keyframe_llm_model,
        )
        # Phase-8 visibility indices for the ScanRefer/NR3D shared scenes are
        # saved at stride=1; keep the selector stride aligned so view IDs map
        # back to existing raw frames.
        selector = KeyframeSelector.from_scene_path(
            str(scene_cg_root),
            stride=1,
            llm_model=keyframe_llm_model,
        )
        selector_cache[request.scene_id] = selector
    kfs = select_keyframes_query_driven(
        selector=selector,
        scene_id=request.scene_id,
        query=sample.query,
        raw_frames_root=raw_frames_root,
        k=3,  # OpenEQA-aligned: Stage 1 returns 1-3 KFs; agent calls switch_or_expand_hypothesis to refresh
    )
    if kfs:
        return kfs, False
    # Fallback: top-3 frames by Mask3D candidate density (proposal-aware, query-blind)
    return (
        _fallback_top5_by_mask3d_density(
            scene_id=request.scene_id,
            scene_artifacts=scene_artifacts,
            raw_frames_root=raw_frames_root,
            k=3,
        ),
        True,
    )


def prepare_scene_artifacts(
    *,
    scene_id: str,
    data_root: Path,
    raw_frames_root: Path,
    pack_name: str = "pack_scanrefer_v1",
) -> SceneArtifacts:
    scene_root = data_root / scene_id
    objects = load_mask3d_objects(scene_root)
    proposals = build_proposals_from_mask3d_objects(objects=objects, scene_id=scene_id)
    if not proposals:
        raise ValueError(f"scene has no Mask3D objects: {scene_id}")
    visibility = load_mask3d_visibility_index(scene_root)
    frame_visibility = {
        frame_id: [obj_id for obj_id, _score in entries]
        for frame_id, entries in visibility.view_to_objects.items()
    }
    proposal_ids = [int(p["id"]) for p in proposals]
    proposal_labels = {
        int(p["id"]): str(p.get("label", "")).lower().strip() for p in proposals
    }
    scene_categories = sorted({lab for lab in proposal_labels.values() if lab})
    valid_ids = set(proposal_ids)
    for frame_id, ids in frame_visibility.items():
        unknown = sorted(set(ids) - valid_ids)
        if unknown:
            raise ValueError(
                f"{scene_id} visibility frame {frame_id} unknown ids: {unknown}"
            )

    scene_dir = data_root / scene_id / pack_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    raw_scene_root = raw_frames_root / scene_id
    frames = scene_frames(raw_scene_root, sorted(frame_visibility))
    if not frames:
        raise ValueError(f"scene has no visible frames: {scene_id}")
    frame_by_id = {f.frame_id: f for f in frames}
    intrinsic = scene_intrinsic(raw_scene_root)
    image_size = load_image_size(frames[0].rgb_path)
    proposal_frame_views = compute_proposal_frame_views(
        proposal_by_id={int(p["id"]): p for p in proposals},
        frame_visibility=frame_visibility,
        frame_by_id=frame_by_id,
        intrinsic=intrinsic,
        image_size=image_size,
        view_to_objects=visibility.view_to_objects,
        raw_frames_root=raw_frames_root,
        scene_id=scene_id,
    )
    proposals_for_jsonl = [
        {**p, "frame_views": proposal_frame_views.get(int(p["id"]), {})}
        for p in proposals
    ]

    (scene_dir / "proposals.jsonl").write_text(
        json.dumps(
            {
                "source": "conceptgraph",
                "scene_id": scene_id,
                "axis_align_matrix": None,
                "proposals": proposals_for_jsonl,
                "proposal_provenance": "mask3d",
                "cvra_metadata_emitted": "cvra" in pack_name.lower(),
                "frame_views_emitted": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (scene_dir / "visibility.json").write_text(
        json.dumps(
            {str(k): v for k, v in sorted(frame_visibility.items())},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    annotated_dir = scene_dir / "annotated"
    render_annotated_frames(
        proposal_by_id={int(p["id"]): p for p in proposals},
        frame_visibility=frame_visibility,
        frame_by_id=frame_by_id,
        intrinsic=intrinsic,
        image_size=image_size,
        annotated_dir=annotated_dir,
    )
    return SceneArtifacts(
        scene_dir=scene_dir,
        proposals_jsonl=scene_dir / "proposals.jsonl",
        visibility_json=scene_dir / "visibility.json",
        annotated_dir=annotated_dir,
        frame_visibility=frame_visibility,
        proposal_ids=proposal_ids,
        proposal_labels=proposal_labels,
        mask3d_visibility=visibility,
        scene_categories=scene_categories,
    )


def build_proposals_from_mask3d_objects(
    *,
    objects: list[dict[str, Any]],
    scene_id: str,
) -> list[dict[str, Any]]:
    """Convert Mask3D-CG object list to proposals.jsonl shape."""
    proposals: list[dict[str, Any]] = []
    for obj_id, obj in enumerate(objects):
        names = obj.get("class_name")
        if not (
            isinstance(names, list)
            and names
            and all(isinstance(n, str) and n for n in names)
        ):
            continue
        if "bbox_np" not in obj:
            raise ValueError(f"{scene_id}.objects[{obj_id}] missing bbox_np")
        corners = np.asarray(obj["bbox_np"], dtype=np.float64)
        if corners.shape != (8, 3):
            raise ValueError(
                f"{scene_id}::{obj_id} bbox_np shape {corners.shape}, expected (8,3)"
            )
        mn = corners.min(axis=0)
        mx = corners.max(axis=0)
        bbox_9dof = [
            *((mn + mx) / 2.0).tolist(),  # cx, cy, cz
            *(mx - mn).tolist(),  # dx, dy, dz
            0.0,
            0.0,
            0.0,  # Euler=0
        ]
        label = Counter(names).most_common(1)[0][0]
        ids = obj.get("class_id") or [-1]
        label_idx = int(Counter(int(v) for v in ids).most_common(1)[0][0])
        proposals.append(
            {
                "id": obj_id,
                "bbox_3d": bbox_9dof,
                "score": 1.0,  # uniform per Decision 4
                "label": label,
                "label_idx": label_idx,
            }
        )
    return proposals


def write_sample_artifact(
    *,
    request: SampleRequest,
    sample: ScanRefVGSample,
    data_root: Path,
    raw_frames_root: Path,
    scene_artifacts: SceneArtifacts,
    keyframes: list[dict[str, Any]] | None = None,
    keyframe_mode: str = "gt_target",
) -> Path:
    if keyframes is None:
        keyframes = select_keyframes_from_phase8_target(
            scene_id=request.scene_id,
            target_id=request.target_id,
            raw_frames_root=raw_frames_root,
            k=5,
        )
    gt_bbox = validate_bbox_9dof(
        sample.gt_bbox_3d, f"{request.sample_id}.gt_bbox_3d_9dof"
    )
    artifacts_v9 = write_v9_scene_artifacts_scanrefer(
        scene_id=request.scene_id,
        data_root=data_root,
        pack_name=scene_artifacts.scene_dir.name,
        proposals_jsonl=scene_artifacts.proposals_jsonl,
        scene_category=None,
        valid_frame_ids=sorted(scene_artifacts.frame_visibility.keys()),
    )
    payload = {
        "sample_id": request.sample_id,
        "scene_id": request.scene_id,
        "target_id": request.target_id,
        "ann_id": request.ann_id,
        "category": request.category or sample.target,
        "is_unique": bool(sample.is_unique),
        "query": sample.query,
        "gt_bbox_3d_9dof": gt_bbox,
        "scene_artifacts_dir": str(scene_artifacts.scene_dir),
        "source": "conceptgraph",
        "proposal_provenance": "mask3d",
        "keyframe_mode": keyframe_mode,
        "scene_catalog_path": artifacts_v9["scene_catalog_path"],
        "bev_image_path": artifacts_v9["bev_image_path"],
        "camera_trajectory_path": artifacts_v9["camera_trajectory_path"],
    }
    path = sample_artifact_path(
        data_root, request, pack_name=scene_artifacts.scene_dir.name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def select_keyframes_from_phase8_target(
    *,
    scene_id: str,
    target_id: int,
    raw_frames_root: Path,
    k: int = 5,
) -> list[dict[str, Any]]:
    """v1/v2 GT-view-oracle path. Pick top-k frames where the Phase 8
    GT target is most visible.

    Uses the Phase 8 visibility index at
    data/nr3d/scannet/<scene>/conceptgraph/indices/visibility_index.pkl
    (note: NOT the Mask3D-CG visibility — keyframe choice is GT-driven so
    the agent sees frames where the target object is actually present).
    Carries a GT view oracle: see v3 query_driven mode for the apples-to-
    apples Camp-A path.
    """
    phase8_vis_path = raw_frames_root / scene_id / VIS_REL
    if not phase8_vis_path.exists():
        raise FileNotFoundError(f"Phase 8 visibility missing: {phase8_vis_path}")
    with open(phase8_vis_path, "rb") as f:
        payload = pickle.load(f)
    obj_to_views = payload.get("object_to_views") or {}
    views = obj_to_views.get(int(target_id))
    if not views:
        raise ValueError(
            f"no Phase 8-visible frames for target_id={target_id} in {scene_id}"
        )
    keyframes = []
    for kfi, (frame_id, _score) in enumerate(views[:k]):
        keyframes.append(
            {
                "keyframe_idx": kfi,
                "image_path": str(
                    _resolve_raw_rgb_path(raw_frames_root / scene_id, int(frame_id))
                ),
                "frame_id": int(frame_id),
            }
        )
    return keyframes


def select_keyframes_query_driven(
    *,
    selector: Any,
    scene_id: str,
    query: str,
    raw_frames_root: Path,
    k: int = 5,
) -> list[dict[str, Any]]:
    """v3 query-driven path. Calls KeyframeSelector.select_keyframes_v2
    with hypothesis-driven retrieval (the same entry OpenEQA pilot uses)
    and returns frame_ids resolved to raw RGB paths.

    `use_visual_context=False` because ScanNet `_vh_clean.ply` mesh isn't
    available locally for BEV synthesis — text-only hypothesis parse.
    Returns [] when Stage 1 finds no executable hypothesis (caller should
    fall back via _fallback_top5_by_mask3d_density).
    """
    from loguru import logger

    res = selector.select_keyframes_v2(
        query=query,
        k=k,
        use_visual_context=False,
    )
    if not res.keyframe_indices:
        logger.warning(
            "[select_keyframes_query_driven] empty Stage1 result for {} "
            "query={!r} (target={!r})",
            scene_id,
            query,
            res.target_term,
        )
        return []
    keyframes = []
    for kfi, frame_id in enumerate(res.keyframe_indices[:k]):
        keyframes.append(
            {
                "keyframe_idx": kfi,
                "image_path": str(
                    _resolve_raw_rgb_path(raw_frames_root / scene_id, int(frame_id))
                ),
                "frame_id": int(frame_id),
            }
        )
    return keyframes


# Stopwords stripped from category tokens before fuzzy matching against
# Mask3D ScanNet200 labels. Keep narrow — the goal is to drop articles, not
# prune useful nouns ("table" should still match "coffee table").
_CATEGORY_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "in",
        "on",
        "at",
        "to",
    }
)


def _normalize_category_tokens(category: str) -> set[str]:
    """Lowercase, split on whitespace + underscores, drop short stopwords.

    Returns a set of meaningful tokens used for fuzzy match against
    Mask3D-CG ScanNet200 labels.
    """
    raw = category.replace("_", " ").lower().strip()
    if not raw:
        return set()
    tokens = {t for t in raw.split() if t and t not in _CATEGORY_STOPWORDS}
    return tokens


def _matching_proposal_ids(
    target_categories: list[str],
    proposal_labels: dict[int, str],
) -> set[int]:
    """Fuzzy-match LLM-parsed categories against Mask3D-CG label vocab.

    A proposal matches when:
    - its full lowercased label equals one of the categories (after _ -> space), or
    - its label tokens share at least one non-stopword token with a category.

    This handles common ScanNet200 vs LLM-vocab gaps such as
    "office chair" vs "chair", "trash can" vs "trash_can", and
    "coffee table" vs "table".
    """
    cat_tokens_per_cat: list[tuple[str, set[str]]] = []
    for cat in target_categories:
        if not isinstance(cat, str):
            continue
        flat = cat.replace("_", " ").lower().strip()
        if flat in {"", "unknow", "unknown"}:
            continue
        toks = _normalize_category_tokens(cat)
        if toks:
            cat_tokens_per_cat.append((flat, toks))
    if not cat_tokens_per_cat:
        return set()

    matching: set[int] = set()
    for pid, label in proposal_labels.items():
        if not label:
            continue
        label_l = label.replace("_", " ").lower().strip()
        label_toks = _normalize_category_tokens(label)
        for cat_flat, cat_toks in cat_tokens_per_cat:
            if label_l == cat_flat:
                matching.add(pid)
                break
            if cat_toks & label_toks:
                matching.add(pid)
                break
    return matching


def select_keyframes_mask3d_query_driven(
    *,
    scene_id: str,
    query: str,
    scene_artifacts: SceneArtifacts,
    raw_frames_root: Path,
    query_parser: Any,
    k: int = 3,
) -> list[dict[str, Any]]:
    """v3.1 Mask3D-driven Stage 1 path.

    Pipeline:
        1. Parse `query` -> HypothesisOutputV1 via the cached QueryParser.
        2. Collect all category strings (target + anchors) across hypotheses.
        3. Fuzzy-match against Mask3D-CG proposal labels for this scene.
        4. Score each frame by sum of visibility weights of matching candidates.
        5. Return top-k frames as keyframes.

    Returns [] when the parser yields no usable categories or no Mask3D
    candidate matches the categories. The caller should fall back to
    Mask3D-density top-k in that case.

    The Mask3D-CG visibility index used here is the SAME index the
    annotated-PNG renderer consumes, so initial keyframes are guaranteed
    to contain at least one same-category mark when this path succeeds.
    """
    from loguru import logger

    try:
        output = query_parser.parse(query)
    except Exception as exc:
        logger.warning(
            "[select_keyframes_mask3d_query_driven] parse failed for {} "
            "query={!r} err={}",
            scene_id,
            query,
            exc,
        )
        return []

    target_categories: list[str] = []
    seen: set[str] = set()
    for hypothesis in output.ordered_hypotheses():
        for cat in hypothesis.grounding_query.get_all_categories():
            if not cat:
                continue
            key = cat.replace("_", " ").lower().strip()
            if key in seen:
                continue
            seen.add(key)
            target_categories.append(cat)
    if not target_categories:
        logger.warning(
            "[select_keyframes_mask3d_query_driven] no categories parsed "
            "for {} query={!r}",
            scene_id,
            query,
        )
        return []

    matching_pids = _matching_proposal_ids(
        target_categories,
        scene_artifacts.proposal_labels,
    )
    if not matching_pids:
        logger.warning(
            "[select_keyframes_mask3d_query_driven] no Mask3D candidates match "
            "categories={} for {} (query={!r})",
            target_categories,
            scene_id,
            query,
        )
        return []

    frame_scores: dict[int, float] = {}
    for view_id, entries in scene_artifacts.mask3d_visibility.view_to_objects.items():
        score = sum(
            float(weight) for (oid, weight) in entries if int(oid) in matching_pids
        )
        if score > 0.0:
            frame_scores[int(view_id)] = score
    if not frame_scores:
        logger.warning(
            "[select_keyframes_mask3d_query_driven] matching candidates {} "
            "have no Mask3D-visible frames in {}",
            sorted(matching_pids),
            scene_id,
        )
        return []

    top = sorted(frame_scores.items(), key=lambda kv: -kv[1])[:k]
    keyframes = []
    for kfi, (frame_id, _) in enumerate(top):
        keyframes.append(
            {
                "keyframe_idx": kfi,
                "image_path": str(
                    _resolve_raw_rgb_path(
                        raw_frames_root / scene_id,
                        int(frame_id),
                    )
                ),
                "frame_id": int(frame_id),
            }
        )
    logger.info(
        "[select_keyframes_mask3d_query_driven] {} q={!r} cats={} "
        "matching_pids={} -> {} frames (top weights {})",
        scene_id,
        query,
        target_categories,
        len(matching_pids),
        len(top),
        [round(s, 2) for _, s in top],
    )
    return keyframes


def _fallback_top5_by_mask3d_density(
    *,
    scene_id: str,
    scene_artifacts: SceneArtifacts,
    raw_frames_root: Path,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Empty-Stage1 fallback: top-k frames ranked by number of Mask3D
    candidates visible. Query-blind but proposal-aware — preserves a
    visual prior without leaking GT.
    """
    ranked = sorted(
        scene_artifacts.frame_visibility.items(),
        key=lambda kv: -len(kv[1]),
    )[:k]
    if not ranked:
        raise ValueError(f"no Mask3D-visible frames for fallback in {scene_id}")
    keyframes = []
    for kfi, (frame_id, _) in enumerate(ranked):
        keyframes.append(
            {
                "keyframe_idx": kfi,
                "image_path": str(
                    _resolve_raw_rgb_path(raw_frames_root / scene_id, int(frame_id))
                ),
                "frame_id": int(frame_id),
            }
        )
    return keyframes


def compute_proposal_frame_views(
    *,
    proposal_by_id: dict[int, dict[str, Any]],
    frame_visibility: dict[int, list[int]],
    frame_by_id: dict[int, SceneFrame],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    view_to_objects: dict[int, list[tuple[int, float]]],
    raw_frames_root: Path,
    scene_id: str,
) -> dict[int, dict[str, dict[str, Any]]]:
    """For each (proposal, frame) where the proposal is visible, project
    the proposal's 9-DOF bbox to a 2D rect, look up the visibility weight
    from the Mask3D-CG ``view_to_objects`` index, and resolve the raw
    RGB path. Returns a per-proposal dict suitable for embedding into
    ``proposals.jsonl`` as ``proposal["frame_views"]``.

    Schema per the M2a canonical spec § H + the M2b worker contract:

        {
          <proposal_id_int>: {
            <frame_id_str>: {
              "bbox_2d": [x1, y1, x2, y2]      # int pixels, x1<=x2, y1<=y2
              "raw_rgb_path": "<project-root-relative path>"
              "visibility_weight": <float from view_to_objects>
            }
          }
        }

    Frames where the projection collapses (returns None — out-of-frustum
    or zero-area) are omitted. Proposals with no visible frame after
    projection get an empty dict.
    """
    weight_lookup: dict[tuple[int, int], float] = {}
    for fid_, entries in view_to_objects.items():
        for oid_, score_ in entries:
            weight_lookup[(int(oid_), int(fid_))] = float(score_)

    raw_scene_root = raw_frames_root / scene_id
    out: dict[int, dict[str, dict[str, Any]]] = {
        int(pid): {} for pid in proposal_by_id
    }
    for frame_id, visible_ids in frame_visibility.items():
        if frame_id not in frame_by_id:
            raise ValueError(f"visibility references missing frame_id={frame_id}")
        frame = frame_by_id[frame_id]
        for prop_id in visible_ids:
            prop = proposal_by_id.get(int(prop_id))
            if prop is None:
                raise ValueError(f"unknown proposal_id={prop_id}")
            rect = project_visible_bbox_3d_to_2d(
                prop["bbox_3d"],
                intrinsic,
                frame.extrinsic_world_to_cam,
                image_size,
            )
            if rect is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in rect)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if x2 == x1 or y2 == y1:
                continue
            raw_rgb = _resolve_raw_rgb_path(raw_scene_root, int(frame_id))
            visibility_weight = weight_lookup.get((int(prop_id), int(frame_id)))
            entry: dict[str, Any] = {
                "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                "raw_rgb_path": str(raw_rgb),
            }
            if visibility_weight is not None:
                entry["visibility_weight"] = float(visibility_weight)
            out[int(prop_id)][str(int(frame_id))] = entry
    return out


def render_annotated_frames(
    *,
    proposal_by_id: dict[int, dict[str, Any]],
    frame_visibility: dict[int, list[int]],
    frame_by_id: dict[int, SceneFrame],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    annotated_dir: Path,
) -> None:
    for frame_id, visible_ids in frame_visibility.items():
        if frame_id not in frame_by_id:
            raise ValueError(f"visibility references missing frame_id={frame_id}")
        frame = frame_by_id[frame_id]
        marks = []
        for prop_id in visible_ids:
            prop = proposal_by_id.get(int(prop_id))
            if prop is None:
                raise ValueError(f"unknown proposal_id={prop_id}")
            rect = project_visible_bbox_3d_to_2d(
                prop["bbox_3d"],
                intrinsic,
                frame.extrinsic_world_to_cam,
                image_size,
            )
            if rect is None:
                continue
            marks.append(
                {
                    "proposal_id": int(prop_id),
                    "label": prop["label"],
                    "bbox_2d": rect,
                }
            )
        render_marked_keyframe(
            rgb_path=frame.rgb_path,
            out_path=annotated_dir / f"frame_{frame_id}.png",
            marks=marks,
        )


def scene_frames(scene_root: Path, frame_ids: Sequence[int]) -> list[SceneFrame]:
    frames: list[SceneFrame] = []
    for frame_id in frame_ids:
        raw_id = _raw_frame_id_for_view(scene_root, int(frame_id))
        pose_path = scene_root / "raw" / f"{raw_id:06d}.txt"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing pose: {pose_path}")
        cam_to_world = validate_matrix_4x4(
            np.loadtxt(pose_path), field_name=str(pose_path)
        )
        frames.append(
            SceneFrame(
                frame_id=int(frame_id),
                raw_frame_id=raw_id,
                rgb_path=_resolve_raw_rgb_path(scene_root, int(frame_id)),
                extrinsic_world_to_cam=np.linalg.inv(cam_to_world),
            )
        )
    return frames


def scene_intrinsic(scene_root: Path) -> np.ndarray:
    p = scene_root / "raw" / "intrinsic_color.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing intrinsic: {p}")
    mat = np.asarray(np.loadtxt(p), dtype=float)
    if mat.shape == (4, 4):
        return mat[:3, :3]
    if mat.shape == (3, 3):
        return mat
    raise ValueError(f"intrinsic must be 3x3 or 4x4, got {mat.shape}: {p}")


def _resolve_raw_rgb_path(scene_root: Path, frame_id: int) -> Path:
    raw_id = _raw_frame_id_for_view(scene_root, frame_id)
    p = scene_root / "raw" / f"{raw_id:06d}-rgb.png"
    if not p.exists():
        raise FileNotFoundError(f"Missing RGB: {p}")
    return p


def _raw_frame_id_for_view(scene_root: Path, frame_id: int) -> int:
    info_p = scene_root / "raw" / "scene_info.json"
    info = json.loads(info_p.read_text(encoding="utf-8"))
    kept = info.get("kept_frame_ids")
    if not isinstance(kept, list):
        raise ValueError(f"{info_p} missing kept_frame_ids")
    if frame_id < 0 or frame_id >= len(kept):
        raise ValueError(
            f"{scene_root.name} frame_id={frame_id} out of range ({len(kept)} kept)"
        )
    return int(kept[frame_id])


def load_mask3d_objects(scene_root: Path) -> list[dict[str, Any]]:
    p = scene_root / MASK3D_PCD_REL
    if not p.exists():
        raise FileNotFoundError(f"Missing Mask3D-CG pkl: {p}")
    with gzip.open(p, "rb") as f:
        return pickle.load(f)["objects"]


def load_mask3d_visibility_index(scene_root: Path) -> Mask3dVisibility:
    p = scene_root / VIS_REL
    if not p.exists():
        raise FileNotFoundError(f"Missing Mask3D visibility: {p}")
    with open(p, "rb") as f:
        payload = pickle.load(f)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("use_depth") is False:
        raise ValueError(
            f"{p} is projection-only. ScanRefer view_to_objects/object_to_views "
            "must include depth-occlusion checks."
        )
    return Mask3dVisibility(
        object_to_views=_coerce_visibility(payload.get("object_to_views"), p),
        view_to_objects=_coerce_visibility(payload.get("view_to_objects"), p),
    )


def _coerce_visibility(
    raw: dict[Any, Any] | None, path: Path
) -> dict[int, list[tuple[int, float]]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} missing visibility map")
    out: dict[int, list[tuple[int, float]]] = {}
    for k, entries in raw.items():
        out[int(k)] = [(int(e[0]), float(e[1])) for e in entries]
    return out


def load_sample_requests(path: Path) -> list[SampleRequest]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"sample ids JSON must be a list: {path}")
    out: list[SampleRequest] = []
    for i, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"row {i} must be object: {row!r}")
        sid = row.get("sample_id")
        if not isinstance(sid, str):
            raise ValueError(f"row {i} missing sample_id")
        scene, tid, ann_id = parse_sample_id(sid)
        out.append(
            SampleRequest(
                sample_id=sid,
                scene_id=row.get("scene_id") or scene,
                target_id=int(row.get("target_id", tid)),
                ann_id=str(row.get("ann_id", ann_id)),
                category=str(row.get("category") or ""),
            )
        )
    return out


def sample_artifact_path(
    data_root: Path, request: SampleRequest, *, pack_name: str = "pack_scanrefer_v1"
) -> Path:
    return (
        data_root
        / request.scene_id
        / pack_name
        / "samples"
        / f"{safe_sample_id(request.sample_id)}.json"
    )


def _render_v9_bev_scanrefer(
    *,
    scene_id: str,
    data_root: Path,
    proposals,
    output_path: Path,
    highlight_ids: list[int] | None,
) -> Path:
    """v9 BEV renderer for ScanRefer. Tests can monkey-patch this."""
    builder = ScanReferScanNetBEVBuilder()
    return builder.build_with_labels(
        scene_id=scene_id,
        data_root=data_root,
        proposals=proposals,
        output_path=output_path,
        highlight_ids=highlight_ids,
    )


def _build_camera_trajectory_scanrefer(scene_dir: Path) -> dict[int, list[float]]:
    """Read conceptgraph/traj.txt and emit {frame_id: [x, y, yaw]} for v9."""
    traj_path = scene_dir / "conceptgraph" / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"traj.txt missing for scene {scene_dir.name}: {traj_path}"
        )
    raw = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)
    out: dict[int, list[float]] = {}
    for i, pose in enumerate(raw):
        x = float(pose[0, 3])
        y = float(pose[1, 3])
        forward = -pose[:3, 2]
        yaw = float(math.atan2(forward[1], forward[0]))
        out[i] = [x, y, yaw]
    return out


def write_v9_scene_artifacts_scanrefer(
    *,
    scene_id: str,
    data_root: Path,
    pack_name: str,
    proposals_jsonl: Path,
    scene_category: str | None,
    valid_frame_ids: list[int],
) -> dict[str, str]:
    """Emit BEV png + scene_catalog.json + camera trajectory for v9 catalog-first prep."""
    scene_dir = data_root / scene_id
    pack_dir = scene_dir / pack_name
    bev_dir = pack_dir / "bev"
    bev_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = pack_dir / "scene_catalog.json"
    traj_out_path = pack_dir / "camera_trajectory.json"
    raw_pool = json.loads(proposals_jsonl.read_text())
    raw_pool.setdefault("source", "mask3d")
    raw_pool.setdefault("frame_index", {})
    raw_pool.setdefault("proposal_index", {})
    raw_pool.setdefault("annotated_image_dir", str(pack_dir / "annotated"))
    bev_path = bev_dir / "scene_bev_scanrefer.png"
    catalog = from_vg_proposal_pool(
        pool=raw_pool,
        scene_id=scene_id,
        bev_image_path=str(bev_path),
        scene_category=scene_category,
        axis_align_matrix=raw_pool.get("axis_align_matrix"),
        valid_frame_ids=list(valid_frame_ids),
    )
    _render_v9_bev_scanrefer(
        scene_id=scene_id,
        data_root=data_root,
        proposals=catalog.proposals,
        output_path=bev_path,
        highlight_ids=None,
    )
    catalog_path.write_text(
        json.dumps(catalog.model_dump(), ensure_ascii=False, indent=2)
    )
    traj = _build_camera_trajectory_scanrefer(scene_dir)
    traj_out_path.write_text(json.dumps({str(k): v for k, v in traj.items()}))
    return {
        "bev_image_path": str(bev_path),
        "scene_catalog_path": str(catalog_path),
        "camera_trajectory_path": str(traj_out_path),
    }


def main() -> None:
    args = parse_args()
    written = prepare_pack_v1_inputs_scanrefer(
        sample_ids_path=args.sample_ids,
        data_root=args.data_root,
        pack_name=args.pack_name,
        split=args.split,
        scanrefer_root=args.scanrefer_root,
        phase8_data_root=args.phase8_data_root,
        raw_frames_root=args.raw_frames_root,
        max_samples=args.max_samples,
        keyframe_mode=args.keyframe_mode,
        keyframe_llm_model=args.keyframe_llm_model,
    )
    print(
        f"wrote {len(written)} sample artifacts under "
        f"{args.data_root}/<scene>/{args.pack_name}/  "
        f"(keyframe_mode={args.keyframe_mode})"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "Mask3dVisibility",
    "SampleRequest",
    "SceneArtifacts",
    "build_proposals_from_mask3d_objects",
    "compute_proposal_frame_views",
    "load_sample_requests",
    "parse_sample_id",
    "prepare_pack_v1_inputs_scanrefer",
    "safe_sample_id",
    "select_keyframes_mask3d_query_driven",
    "_matching_proposal_ids",
    "_normalize_category_tokens",
]
