"""Prepare offline pack inputs for NR3D VG runs from Phase 8 GT-CG output."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agents.adapters.nr3d_adapter import Nr3dVGAdapter
from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)
from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    project_depth_visible_points_to_2d,
)
from benchmarks.nr3d_loader import Nr3dVGSample, _phase8_corners_to_9dof
from evaluation.scripts.prepare_pack_v1_inputs import (
    load_image_size,
    normalize_prepared_keyframes,
    validate_bbox_9dof,
    validate_matrix_4x4,
)
from query_scene.lightweight_conceptgraph import (
    load_conceptgraph_objects,
    write_lightweight_conceptgraph_cache,
)

PHASE8_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz")
PHASE8_VIS_REL = Path("conceptgraph/indices/visibility_index.pkl")


@dataclass(frozen=True)
class SampleRequest:
    sample_id: str
    scene_id: str
    target_id: int
    category: str


@dataclass(frozen=True)
class SceneFrame:
    frame_id: int
    raw_frame_id: int
    rgb_path: Path
    extrinsic_world_to_cam: np.ndarray


@dataclass(frozen=True)
class Phase8Visibility:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help=(
            "Phase 8 NR3D ScanNet root containing <scene>/conceptgraph and "
            "<scene>/raw. Outputs land under <data_root>/<scene>/<pack_name>/."
        ),
    )
    parser.add_argument("--pack-name", default="pack_nr3d_v1")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--nr3d-root",
        type=Path,
        default=None,
        help="NR3D root containing raw/nr3d.csv. Defaults to data-root parent.",
    )
    parser.add_argument(
        "--keyframe-mode",
        default="gt_target",
        choices=["gt_target", "query_driven"],
        help=(
            "How to pick initial RGB keyframes. 'gt_target' preserves the "
            "legacy top-5 GT-target-visible behavior; 'query_driven' uses "
            "Stage 1 query-driven keyframe selection with a density fallback."
        ),
    )
    parser.add_argument(
        "--keyframe-llm-model",
        default="gemini-2.5-pro",
        help="LLM for query_driven keyframe selection.",
    )
    parser.add_argument(
        "--max-selector-cache-size",
        type=int,
        default=1,
        help=(
            "Maximum number of per-scene KeyframeSelector instances to keep in "
            "memory. Full NR3D prep is scene-heavy; keep this small to avoid "
            "retaining point clouds/features for completed scenes."
        ),
    )
    parser.add_argument(
        "--max-scene-artifact-cache-size",
        type=int,
        default=1,
        help=(
            "Maximum number of per-scene prepared-artifact handles to keep in "
            "memory. Requests are grouped by scene before processing, so 1 is "
            "normally sufficient."
        ),
    )
    parser.add_argument(
        "--build-lightweight-cache-only",
        action="store_true",
        default=False,
        help=(
            "Only build stripped ConceptGraph object caches for requested scenes. "
            "Run this with low concurrency before high-worker prep."
        ),
    )
    parser.add_argument(
        "--overwrite-lightweight-cache",
        action="store_true",
        default=False,
        help="Rewrite existing stripped ConceptGraph object caches.",
    )
    parser.add_argument(
        "--ensure-lightweight-cache",
        "--require-lightweight-cache",
        dest="ensure_lightweight_cache",
        action="store_true",
        default=False,
        help=(
            "Ensure stripped ConceptGraph object caches before loading objects. "
            "Missing caches are built under a cross-process lock, so final "
            "sample results do not fail because a cache was absent."
        ),
    )
    return parser.parse_args()


def prepare_pack_v1_inputs_nr3d(
    *,
    sample_ids_path: Path,
    data_root: Path,
    pack_name: str = "pack_nr3d_v1",
    split: str = "test",
    max_samples: int | None = None,
    nr3d_root: Path | None = None,
    keyframe_mode: str = "gt_target",
    keyframe_llm_model: str = "gemini-2.5-pro",
    max_selector_cache_size: int = 1,
    max_scene_artifact_cache_size: int = 1,
    ensure_lightweight_cache: bool = False,
) -> list[Path]:
    if keyframe_mode not in ("gt_target", "query_driven"):
        raise ValueError(
            "keyframe_mode must be 'gt_target' or 'query_driven', "
            f"got {keyframe_mode!r}"
        )
    requests = load_sample_requests(sample_ids_path)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        requests = requests[:max_samples]
    if max_selector_cache_size <= 0:
        raise ValueError("max_selector_cache_size must be positive")
    if max_scene_artifact_cache_size <= 0:
        raise ValueError("max_scene_artifact_cache_size must be positive")

    # Full NR3D contains many scenes. Grouping prevents alternation between
    # scenes from defeating the one-scene LRU caches below.
    requests = sorted(
        requests, key=lambda request: (request.scene_id, request.sample_id)
    )

    nr3d_root = nr3d_root or data_root.parent
    _adapter, sample_lookup = load_sample_lookup(
        nr3d_root=nr3d_root,
        phase8_data_root=data_root,
        split=split,
        sample_ids={request.sample_id for request in requests},
    )

    scene_artifacts: OrderedDict[str, SceneArtifacts] = OrderedDict()
    selector_cache: OrderedDict[str, Any] = OrderedDict()
    written_samples: list[Path] = []
    for request in requests:
        sample = sample_lookup.get(request.sample_id)
        if sample is None:
            raise ValueError(
                f"No NR3D sample for sample_id={request.sample_id!r} "
                f"(split={split})"
            )
        if request.scene_id not in scene_artifacts:
            scene_artifacts[request.scene_id] = prepare_scene_artifacts(
                scene_id=request.scene_id,
                data_root=data_root,
                pack_name=pack_name,
                ensure_lightweight_cache=ensure_lightweight_cache,
            )
            evict_lru_cache(scene_artifacts, max_scene_artifact_cache_size)
        else:
            scene_artifacts.move_to_end(request.scene_id)
        written_samples.append(
            write_sample_artifact(
                request=request,
                sample=sample,
                data_root=data_root,
                scene_artifacts=scene_artifacts[request.scene_id],
                keyframe_mode=keyframe_mode,
                keyframe_llm_model=keyframe_llm_model,
                selector_cache=selector_cache,
                max_selector_cache_size=max_selector_cache_size,
                ensure_lightweight_cache=ensure_lightweight_cache,
            )
        )
    return written_samples


def build_lightweight_caches_for_sample_ids(
    *,
    sample_ids_path: Path,
    data_root: Path,
    max_samples: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    requests = load_sample_requests(sample_ids_path)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        requests = requests[:max_samples]

    paths: list[Path] = []
    for scene_id in sorted({request.scene_id for request in requests}):
        paths.append(
            write_lightweight_conceptgraph_cache(
                data_root / scene_id / PHASE8_PCD_REL,
                overwrite=overwrite,
            )
        )
    return paths


def evict_lru_cache(cache: OrderedDict[str, Any], max_size: int) -> None:
    evicted = False
    while len(cache) > max_size:
        cache.popitem(last=False)
        evicted = True
    if evicted:
        gc.collect()


def load_sample_lookup(
    *,
    nr3d_root: Path,
    phase8_data_root: Path,
    split: str,
    sample_ids: set[str] | None = None,
) -> tuple[Nr3dVGAdapter, dict[str, Nr3dVGSample]]:
    adapter = Nr3dVGAdapter.from_phase8_test(
        data_root=nr3d_root,
        phase8_data_root=phase8_data_root,
        scene_data_root=phase8_data_root,
    )
    samples = adapter.load_samples(split=split, sample_ids=sample_ids)
    lookup: dict[str, Nr3dVGSample] = {}
    for sample in samples:
        if not isinstance(sample, Nr3dVGSample):
            continue
        if sample.sample_id in lookup:
            raise ValueError(f"Duplicate NR3D sample_id loaded: {sample.sample_id}")
        lookup[sample.sample_id] = sample
    return adapter, lookup


def load_sample_requests(sample_ids_path: Path) -> list[SampleRequest]:
    raw = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"sample ids JSON must be a list: {sample_ids_path}")
    requests: list[SampleRequest] = []
    for row_index, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"sample ids row {row_index} must be an object: {row!r}")
        sample_id = _required_nonempty_str(row, "sample_id", row_index)
        parsed_scene, parsed_target, _assignment = parse_sample_id(sample_id)
        scene_id = str(row.get("scene_id") or parsed_scene).split("/")[-1]
        raw_target_id = row.get("target_id", parsed_target)
        try:
            target_id = int(raw_target_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sample ids row {row_index} has invalid target_id: {row!r}"
            ) from exc
        if scene_id != parsed_scene.split("/")[-1]:
            raise ValueError(
                f"sample ids row {row_index} scene_id={scene_id!r} does not "
                f"match sample_id={sample_id!r}"
            )
        if target_id != parsed_target:
            raise ValueError(
                f"sample ids row {row_index} target_id={target_id} does not "
                f"match sample_id={sample_id!r}"
            )
        requests.append(
            SampleRequest(
                sample_id=sample_id,
                scene_id=scene_id,
                target_id=target_id,
                category=str(row.get("category") or "").strip(),
            )
        )
    if not requests:
        raise ValueError(f"sample ids JSON is empty: {sample_ids_path}")
    return requests


def prepare_scene_artifacts(
    *,
    scene_id: str,
    data_root: Path,
    pack_name: str = "pack_nr3d_v1",
    ensure_lightweight_cache: bool = False,
) -> SceneArtifacts:
    scene_root = data_root / scene_id
    objects = load_phase8_objects(
        scene_root,
        ensure_lightweight_cache=ensure_lightweight_cache,
        prefer_lightweight=False,
    )
    proposals = build_proposals_from_phase8_objects(objects=objects, scene_id=scene_id)
    if not proposals:
        raise ValueError(f"scene has no Phase 8 objects: {scene_id}")
    visibility = load_phase8_visibility_index(scene_root)
    visible_mark_ids = {
        int(proposal["id"])
        for proposal in proposals
        if not bool(proposal.get("is_background", False))
    }
    frame_visibility = {}
    for frame_id, entries in visibility.view_to_objects.items():
        ids = [obj_id for obj_id, _score in entries if obj_id in visible_mark_ids]
        if ids:
            frame_visibility[frame_id] = ids
    proposal_ids = [int(proposal["id"]) for proposal in proposals]
    valid_ids = set(proposal_ids)
    for frame_id, ids in frame_visibility.items():
        unknown = sorted(set(ids) - valid_ids)
        if unknown:
            raise ValueError(
                f"{scene_id} visibility frame {frame_id} references unknown "
                f"object id(s): {unknown}"
            )

    scene_dir = data_root / scene_id / pack_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    proposals_jsonl = scene_dir / "proposals.jsonl"
    proposals_jsonl.write_text(
        json.dumps(
            {
                "source": "gt",
                "scene_id": scene_id,
                "axis_align_matrix": None,
                "proposals": proposals,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    visibility_json = scene_dir / "visibility.json"
    visibility_json.write_text(
        json.dumps(
            {str(k): v for k, v in sorted(frame_visibility.items())},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    frames = scene_frames(scene_root, sorted(frame_visibility))
    if not frames:
        raise ValueError(f"scene has no visible frames: {scene_id}")
    frame_by_id = {frame.frame_id: frame for frame in frames}
    intrinsic = scene_intrinsic(scene_root)
    image_size = load_image_size(frames[0].rgb_path)
    annotated_dir = scene_dir / "annotated"
    render_annotated_frames(
        proposal_by_id={int(p["id"]): p for p in proposals},
        proposal_points_by_id={
            int(obj_id): np.asarray(objects[int(obj_id)]["pcd_np"], dtype=np.float64)
            for obj_id in proposal_ids
        },
        frame_visibility=frame_visibility,
        frame_by_id=frame_by_id,
        intrinsic=intrinsic,
        image_size=image_size,
        annotated_dir=annotated_dir,
    )

    return SceneArtifacts(
        scene_dir=scene_dir,
        proposals_jsonl=proposals_jsonl,
        visibility_json=visibility_json,
        annotated_dir=annotated_dir,
        frame_visibility=frame_visibility,
        proposal_ids=proposal_ids,
    )


def build_proposals_from_phase8_objects(
    *,
    objects: list[dict[str, Any]],
    scene_id: str,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for object_id, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(
                f"{scene_id}.objects[{object_id}] must be an object, got {type(obj).__name__}"
            )
        if "bbox_np" not in obj:
            raise ValueError(f"{scene_id}.objects[{object_id}] missing bbox_np")
        corners = np.asarray(obj["bbox_np"], dtype=np.float64)
        if corners.shape != (8, 3):
            raise ValueError(
                f"{scene_id}::{object_id} bbox_np shape is {corners.shape}, "
                "expected (8,3)"
            )
        names = obj.get("class_name")
        if not (
            isinstance(names, list)
            and names
            and all(isinstance(n, str) and n for n in names)
        ):
            # Phase 8 producer occasionally leaks a background-only object into
            # ``objects`` with empty class_name/class_id/n_points (observed once
            # in 130 scenes: scene0496_00.objects[28], is_background=1,
            # num_detections=0). Such entries are not valid candidates — they
            # carry no semantic label the agent can reason about — so we drop
            # them from the proposal pool. The object_id space stays anchored
            # to the source pkl index, so any direct GT lookup by id still
            # resolves through the loader.
            continue
        label = _dominant_label(obj, scene_id=scene_id, object_id=object_id)
        label_idx = _dominant_label_idx(obj, scene_id=scene_id, object_id=object_id)
        proposals.append(
            {
                "id": object_id,
                "bbox_3d": _phase8_corners_to_9dof(
                    corners,
                    field_name=f"{scene_id}::{object_id}.bbox_np",
                ),
                "score": 1.0,
                "label": label,
                "label_idx": label_idx,
                "is_background": bool(int(obj.get("is_background") or 0)),
            }
        )
    return proposals


def write_sample_artifact(
    *,
    request: SampleRequest,
    sample: Nr3dVGSample,
    data_root: Path,
    scene_artifacts: SceneArtifacts,
    keyframe_mode: str = "gt_target",
    keyframe_llm_model: str = "gemini-2.5-pro",
    selector_cache: OrderedDict[str, Any] | None = None,
    max_selector_cache_size: int = 1,
    ensure_lightweight_cache: bool = False,
) -> Path:
    visibility = load_phase8_visibility_index(data_root / request.scene_id)
    query = getattr(sample, "query", "") or getattr(sample, "text", "")
    if not query:
        raise ValueError(f"Missing query for {request.sample_id}")
    if keyframe_mode == "gt_target":
        keyframes = select_keyframes_for_sample(
            scene_root=data_root / request.scene_id,
            target_id=request.target_id,
            visibility=visibility,
            k=5,
        )
        uses_gt_target = True
        used_fallback = False
    elif keyframe_mode == "query_driven":
        from query_scene.keyframe_selector import KeyframeSelector

        if selector_cache is None:
            selector_cache = OrderedDict()
        selector = selector_cache.get(request.scene_id)
        if selector is None:
            selector = KeyframeSelector.from_scene_path(
                str(data_root / request.scene_id / "conceptgraph"),
                stride=1,
                llm_model=keyframe_llm_model,
                ensure_lightweight_pcd=ensure_lightweight_cache,
            )
            selector_cache[request.scene_id] = selector
            evict_lru_cache(selector_cache, max_selector_cache_size)
        else:
            selector_cache.move_to_end(request.scene_id)
        keyframes, used_fallback = select_keyframes_query_driven(
            selector=selector,
            scene_id=request.scene_id,
            query=query,
            raw_frames_root=data_root,
            k=3,
            fallback_visibility=visibility,
        )
        uses_gt_target = False
    else:
        raise ValueError(
            "keyframe_mode must be 'gt_target' or 'query_driven', "
            f"got {keyframe_mode!r}"
        )
    normalized_keyframes = normalize_prepared_keyframes(
        keyframes,
        scene_artifacts.annotated_dir,
    )
    gt_bbox = validate_bbox_9dof(
        getattr(sample, "gt_bbox_3d", None),
        f"{request.sample_id}.gt_bbox_3d_9dof",
    )
    payload = {
        "sample_id": request.sample_id,
        "scene_id": request.scene_id,
        "target_id": request.target_id,
        "category": request.category or getattr(sample, "target", ""),
        "query": query,
        "gt_bbox_3d_9dof": gt_bbox,
        "scene_artifacts_dir": str(scene_artifacts.scene_dir),
        "source": "gt",
        "keyframe_mode": keyframe_mode,
        "keyframe_selection_uses_gt_target": uses_gt_target,
        "keyframe_selection_used_fallback": used_fallback,
        "keyframes": normalized_keyframes,
    }
    path = sample_artifact_path(
        data_root,
        request,
        pack_name=scene_artifacts.scene_dir.name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def select_keyframes_for_sample(
    *,
    scene_root: Path,
    target_id: int,
    visibility: Phase8Visibility,
    k: int = 5,
) -> list[dict[str, Any]]:
    views = visibility.object_to_views.get(int(target_id))
    if not views:
        raise ValueError(
            f"no visible frames for target_id={target_id} in {scene_root.name}"
        )
    chosen = views[:k]
    keyframes: list[dict[str, Any]] = []
    for keyframe_idx, (frame_id, _score) in enumerate(chosen):
        keyframes.append(
            {
                "keyframe_idx": keyframe_idx,
                "image_path": str(resolve_raw_rgb_path(scene_root, int(frame_id))),
                "frame_id": int(frame_id),
            }
        )
    return keyframes


def _keyframes_from_frame_ids(
    scene_root: Path,
    frame_ids: Sequence[int],
    k: int,
) -> list[dict[str, Any]]:
    keyframes: list[dict[str, Any]] = []
    for keyframe_idx, frame_id in enumerate(frame_ids[:k]):
        keyframes.append(
            {
                "keyframe_idx": keyframe_idx,
                "image_path": str(resolve_raw_rgb_path(scene_root, int(frame_id))),
                "frame_id": int(frame_id),
            }
        )
    return keyframes


def select_keyframes_by_scene_density(
    *,
    scene_root: Path,
    visibility: Phase8Visibility,
    k: int = 3,
) -> list[dict[str, Any]]:
    frame_ids = [
        frame_id
        for frame_id, _entries in sorted(
            visibility.view_to_objects.items(),
            key=lambda item: (-len(item[1]), int(item[0])),
        )
    ]
    return _keyframes_from_frame_ids(scene_root, frame_ids, k)


def select_keyframes_query_driven(
    *,
    selector: Any,
    scene_id: str,
    query: str,
    raw_frames_root: Path,
    k: int = 3,
    fallback_visibility: Phase8Visibility | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    result = selector.select_keyframes_v2(
        query=query,
        k=k,
        use_visual_context=False,
    )
    keyframe_indices = list(getattr(result, "keyframe_indices", []) or [])
    scene_root = raw_frames_root / scene_id
    if keyframe_indices:
        return _keyframes_from_frame_ids(scene_root, keyframe_indices, k), False
    if fallback_visibility is not None:
        return (
            select_keyframes_by_scene_density(
                scene_root=scene_root,
                visibility=fallback_visibility,
                k=k,
            ),
            True,
        )
    return [], True


def render_annotated_frames(
    *,
    proposal_by_id: dict[int, dict[str, Any]],
    proposal_points_by_id: dict[int, np.ndarray],
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
        for proposal_id in visible_ids:
            proposal = proposal_by_id.get(int(proposal_id))
            if proposal is None:
                raise ValueError(
                    f"visibility refers to unknown proposal_id={proposal_id}"
                )
            points = proposal_points_by_id.get(int(proposal_id))
            if points is None:
                raise ValueError(f"missing pcd_np for proposal_id={proposal_id}")
            depth_path = frame.rgb_path.with_name(f"{frame.raw_frame_id:06d}-depth.png")
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth image: {depth_path}")
            depth_map = np.asarray(Image.open(depth_path))
            rect = project_depth_visible_points_to_2d(
                points,
                intrinsic,
                frame.extrinsic_world_to_cam,
                depth_map,
                image_size=image_size,
            )
            if rect is None:
                continue
            marks.append(
                {
                    "proposal_id": int(proposal_id),
                    "label": proposal["label"],
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
        raw_frame_id = raw_frame_id_for_view(scene_root, int(frame_id))
        pose_path = scene_root / "raw" / f"{raw_frame_id:06d}.txt"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing Phase 8 pose file: {pose_path}")
        cam_to_world = validate_matrix_4x4(
            np.loadtxt(pose_path),
            field_name=str(pose_path),
        )
        frames.append(
            SceneFrame(
                frame_id=int(frame_id),
                raw_frame_id=raw_frame_id,
                rgb_path=resolve_raw_rgb_path(scene_root, int(frame_id)),
                extrinsic_world_to_cam=np.linalg.inv(cam_to_world),
            )
        )
    return frames


def scene_intrinsic(scene_root: Path) -> np.ndarray:
    path = scene_root / "raw" / "intrinsic_color.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 8 intrinsic file: {path}")
    mat = np.asarray(np.loadtxt(path), dtype=float)
    if mat.shape == (4, 4):
        return mat[:3, :3]
    if mat.shape == (3, 3):
        return mat
    raise ValueError(f"intrinsic matrix must be 3x3 or 4x4, got {mat.shape}: {path}")


def resolve_raw_rgb_path(scene_root: Path, frame_id: int) -> Path:
    raw_frame_id = raw_frame_id_for_view(scene_root, frame_id)
    path = scene_root / "raw" / f"{raw_frame_id:06d}-rgb.png"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 8 RGB frame: {path}")
    return path


def raw_frame_id_for_view(scene_root: Path, frame_id: int) -> int:
    info = load_raw_scene_info(scene_root)
    kept = info.get("kept_frame_ids")
    if not isinstance(kept, list) or not all(isinstance(v, int) for v in kept):
        raise ValueError(f"{scene_root}/raw/scene_info.json missing kept_frame_ids")
    if frame_id < 0 or frame_id >= len(kept):
        raise ValueError(
            f"{scene_root.name} frame_id={frame_id} out of range for "
            f"{len(kept)} kept frames"
        )
    return int(kept[frame_id])


def load_raw_scene_info(scene_root: Path) -> dict[str, Any]:
    path = scene_root / "raw" / "scene_info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 8 raw scene_info: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Phase 8 raw scene_info must be an object: {path}")
    return payload


def load_phase8_objects(
    scene_root: Path,
    *,
    ensure_lightweight_cache: bool = False,
    prefer_lightweight: bool = True,
) -> list[dict[str, Any]]:
    pkl_path = scene_root / PHASE8_PCD_REL
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing Phase 8 GT-CG pkl: {pkl_path}")
    return load_conceptgraph_objects(
        pkl_path,
        prefer_lightweight=prefer_lightweight,
        ensure_lightweight=ensure_lightweight_cache,
    )


def load_phase8_visibility_index(scene_root: Path) -> Phase8Visibility:
    vis_path = scene_root / PHASE8_VIS_REL
    if not vis_path.exists():
        raise FileNotFoundError(f"Missing Phase 8 visibility index: {vis_path}")
    with open(vis_path, "rb") as f:
        payload = pickle.load(f)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("use_depth") is not True:
        raise ValueError(
            f"{vis_path} is not depth-aware. NR3D view_to_objects/object_to_views "
            "must include depth-occlusion checks; rebuild the scene visibility with "
            "src/scripts/nr3d_gt_conceptgraph.py build-scenes --use-depth."
        )
    if int(metadata.get("num_projection_fallback_objects") or 0) > 0:
        raise ValueError(
            f"{vis_path} contains projection fallback mappings. NR3D visibility "
            "must not synthesize object-frame mappings without depth support."
        )
    raw_object_to_views = payload.get("object_to_views")
    raw_view_to_objects = payload.get("view_to_objects")
    if not isinstance(raw_object_to_views, dict) or not isinstance(
        raw_view_to_objects,
        dict,
    ):
        raise ValueError(f"{vis_path} must contain object_to_views and view_to_objects")
    return Phase8Visibility(
        object_to_views=_coerce_visibility_mapping(raw_object_to_views, vis_path),
        view_to_objects=_coerce_visibility_mapping(raw_view_to_objects, vis_path),
    )


def sample_artifact_path(
    data_root: Path,
    request: SampleRequest,
    *,
    pack_name: str = "pack_nr3d_v1",
) -> Path:
    return (
        data_root
        / request.scene_id
        / pack_name
        / "samples"
        / f"{safe_sample_id(request.sample_id)}.json"
    )


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def parse_sample_id(sample_id: str) -> tuple[str, int, str]:
    parts = sample_id.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Expected NR3D sample_id '<scene>::<target_id>::<assignment>', "
            f"got {sample_id!r}"
        )
    scene_id, target_text, assignment_id = parts
    if not scene_id or not assignment_id:
        raise ValueError(f"Invalid NR3D sample_id={sample_id!r}")
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"Invalid target_id in sample_id={sample_id!r}") from exc
    return scene_id, target_id, assignment_id


def _coerce_visibility_mapping(
    raw: dict[Any, Any],
    path: Path,
) -> dict[int, list[tuple[int, float]]]:
    out: dict[int, list[tuple[int, float]]] = {}
    for raw_key, raw_entries in raw.items():
        key = int(raw_key)
        if not isinstance(raw_entries, list):
            raise ValueError(f"{path} visibility[{raw_key!r}] must be a list")
        entries: list[tuple[int, float]] = []
        for entry in raw_entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"{path} visibility entry must be (id, score): {entry!r}"
                )
            entries.append((int(entry[0]), float(entry[1])))
        out[key] = entries
    return out


def _dominant_label(obj: dict[str, Any], *, scene_id: str, object_id: int) -> str:
    names = obj.get("class_name")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError(f"{scene_id}.objects[{object_id}].class_name must be strings")
    return Counter(names).most_common(1)[0][0]


def _dominant_label_idx(obj: dict[str, Any], *, scene_id: str, object_id: int) -> int:
    ids = obj.get("class_id")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{scene_id}.objects[{object_id}].class_id must be a list")
    return int(Counter(int(value) for value in ids).most_common(1)[0][0])


def _required_nonempty_str(row: dict[str, Any], key: str, row_index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"sample ids row {row_index}.{key} must be a non-empty string: {row!r}"
        )
    return value.strip()


def main() -> None:
    args = parse_args()
    if args.build_lightweight_cache_only:
        written_caches = build_lightweight_caches_for_sample_ids(
            sample_ids_path=args.sample_ids,
            data_root=args.data_root,
            max_samples=args.max_samples,
            overwrite=args.overwrite_lightweight_cache,
        )
        print(f"wrote {len(written_caches)} lightweight ConceptGraph cache(s)")
        return

    written = prepare_pack_v1_inputs_nr3d(
        sample_ids_path=args.sample_ids,
        data_root=args.data_root,
        pack_name=args.pack_name,
        split=args.split,
        max_samples=args.max_samples,
        nr3d_root=args.nr3d_root,
        keyframe_mode=args.keyframe_mode,
        keyframe_llm_model=args.keyframe_llm_model,
        max_selector_cache_size=args.max_selector_cache_size,
        max_scene_artifact_cache_size=args.max_scene_artifact_cache_size,
        ensure_lightweight_cache=args.ensure_lightweight_cache,
    )
    print(
        f"wrote {len(written)} sample artifacts under "
        f"{args.data_root}/<scene>/{args.pack_name}/"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "Phase8Visibility",
    "SampleRequest",
    "SceneArtifacts",
    "SceneFrame",
    "build_proposals_from_phase8_objects",
    "build_lightweight_caches_for_sample_ids",
    "load_phase8_visibility_index",
    "load_sample_lookup",
    "load_sample_requests",
    "prepare_pack_v1_inputs_nr3d",
    "prepare_scene_artifacts",
    "sample_artifact_path",
    "safe_sample_id",
    "select_keyframes_by_scene_density",
    "select_keyframes_for_sample",
    "select_keyframes_query_driven",
    "write_sample_artifact",
]
