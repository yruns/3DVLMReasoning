"""Prepare offline pack inputs for NR3D VG runs from Phase 8 GT-CG output."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import textwrap
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from agents.adapters.nr3d_adapter import Nr3dVGAdapter
from agents.catalog import from_vg_proposal_pool
from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)
from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    project_depth_visible_points_to_2d,
)
from benchmarks.nr3d_loader import Nr3dVGSample, _phase8_corners_to_9dof
from evaluation.scripts.prepare_pack_v1_inputs import (
    load_image_size,
    validate_bbox_9dof,
    validate_matrix_4x4,
)
from query_scene.lightweight_conceptgraph import (
    load_conceptgraph_objects,
    write_lightweight_conceptgraph_cache,
)
from query_scene.scene_bev_builder import Nr3dScanNetBEVBuilder

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
    parser.add_argument(
        "--require-enrichment",
        action="store_true",
        default=False,
        help=(
            "Fail if conceptgraph/enriched_objects.json is missing. Use this "
            "for enriched NR3D packs where proposal notes are part of the "
            "benchmark contract."
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
    max_scene_artifact_cache_size: int = 1,
    ensure_lightweight_cache: bool = False,
    require_enrichment: bool = False,
) -> list[Path]:
    requests = load_sample_requests(sample_ids_path)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        requests = requests[:max_samples]
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
                require_enrichment=require_enrichment,
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
    require_enrichment: bool = False,
) -> SceneArtifacts:
    scene_root = data_root / scene_id
    objects = load_phase8_objects(
        scene_root,
        ensure_lightweight_cache=ensure_lightweight_cache,
        prefer_lightweight=False,
    )
    proposals = build_proposals_from_phase8_objects(objects=objects, scene_id=scene_id)
    apply_enrichment_to_proposals(
        proposals,
        load_enrichment_by_object_id(
            scene_root / "conceptgraph" / "enriched_objects.json",
            required=require_enrichment,
        ),
    )
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
    proposal_frame_views = render_annotated_frames(
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
    proposals_jsonl = scene_dir / "proposals.jsonl"
    proposals_jsonl.write_text(
        json.dumps(
            {
                "source": "gt",
                "scene_id": scene_id,
                "axis_align_matrix": None,
                "proposals": [
                    {
                        **proposal,
                        "frame_views": proposal_frame_views.get(
                            int(proposal["id"]), {}
                        ),
                    }
                    for proposal in proposals
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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


def load_enrichment_by_object_id(
    path: Path,
    *,
    required: bool = False,
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        if not required:
            return {}
        raise FileNotFoundError(
            f"Missing NR3D object enrichment: {path}. "
            "Run `python -m src.scripts.enrich_objects` for this scene before "
            "preparing an enriched NR3D pack."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{path} must contain an objects list")
    out: dict[int, dict[str, Any]] = {}
    for row in objects:
        if not isinstance(row, dict) or row.get("status") != "success":
            continue
        obj_id = row.get("obj_id")
        enrichment = row.get("enrichment")
        if obj_id is None or not isinstance(enrichment, dict):
            continue
        out[int(obj_id)] = enrichment
    return out


def apply_enrichment_to_proposals(
    proposals: list[dict[str, Any]],
    enrichment_by_id: dict[int, dict[str, Any]],
) -> None:
    for proposal in proposals:
        raw_proposal_id = proposal.get("id", proposal.get("proposal_id"))
        if raw_proposal_id is None:
            raise ValueError(f"proposal is missing id/proposal_id: {proposal}")
        proposal_id = int(raw_proposal_id)
        enrichment = enrichment_by_id.get(proposal_id)
        if enrichment is None:
            continue
        label = str(proposal.get("label") or proposal.get("category") or "").strip()
        enriched_category = _merged_enriched_category(
            base_category=label,
            raw_enriched_category=enrichment.get("category"),
        )
        if enriched_category:
            proposal["enriched_category"] = enriched_category
        compact_note = _compact_enrichment_note(enrichment)
        if compact_note:
            proposal["compact_note"] = compact_note
        proposal["enrichment"] = enrichment


def _merged_enriched_category(
    *,
    base_category: str,
    raw_enriched_category: Any,
) -> str | None:
    enriched = _clean_text(raw_enriched_category)
    base = _clean_text(base_category)
    if not enriched:
        return None
    if not base:
        return enriched
    enriched_norm = enriched.lower()
    base_norm = base.lower()
    if enriched_norm == base_norm:
        return enriched
    if base_norm in enriched_norm or enriched_norm in base_norm:
        return enriched
    return f"{enriched}/{base}"


def _compact_enrichment_note(enrichment: dict[str, Any], max_chars: int = 260) -> str:
    parts = [
        _first_sentence(enrichment.get("description")),
        _first_sentence(enrichment.get("location")),
        _first_sentence(enrichment.get("usability")),
    ]
    text = " ".join(part for part in parts if part)
    return textwrap.shorten(text, width=max_chars, placeholder="...") if text else ""


def _first_sentence(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    for marker in (". ", "? ", "! "):
        if marker in text:
            return text.split(marker, 1)[0].strip() + marker.strip()
    return text


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def write_sample_artifact(
    *,
    request: SampleRequest,
    sample: Nr3dVGSample,
    data_root: Path,
    scene_artifacts: SceneArtifacts,
) -> Path:
    query = getattr(sample, "query", "") or getattr(sample, "text", "")
    if not query:
        raise ValueError(f"Missing query for {request.sample_id}")
    gt_bbox = validate_bbox_9dof(
        getattr(sample, "gt_bbox_3d", None),
        f"{request.sample_id}.gt_bbox_3d_9dof",
    )
    artifacts_v9 = write_v9_scene_artifacts(
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
        "category": request.category or getattr(sample, "target", ""),
        "query": query,
        "gt_bbox_3d_9dof": gt_bbox,
        "scene_artifacts_dir": str(scene_artifacts.scene_dir),
        "source": "gt",
        "scene_catalog_path": artifacts_v9["scene_catalog_path"],
        "bev_image_path": artifacts_v9["bev_image_path"],
        "camera_trajectory_path": artifacts_v9["camera_trajectory_path"],
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


def render_annotated_frames(
    *,
    proposal_by_id: dict[int, dict[str, Any]],
    proposal_points_by_id: dict[int, np.ndarray],
    frame_visibility: dict[int, list[int]],
    frame_by_id: dict[int, SceneFrame],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    annotated_dir: Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    frame_views: dict[int, dict[str, dict[str, Any]]] = {
        int(pid): {} for pid in proposal_by_id
    }
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
            bbox_2d = [int(v) for v in rect]
            frame_views[int(proposal_id)][str(int(frame_id))] = {
                "bbox_2d": bbox_2d,
                "raw_rgb_path": str(frame.rgb_path),
            }
            marks.append(
                {
                    "proposal_id": int(proposal_id),
                    "label": proposal["label"],
                    "bbox_2d": tuple(bbox_2d),
                }
            )
        render_marked_keyframe(
            rgb_path=frame.rgb_path,
            out_path=annotated_dir / f"frame_{frame_id}.png",
            marks=marks,
        )
    return frame_views


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


def _render_v9_bev(
    *,
    scene_id: str,
    data_root: Path,
    proposals,
    output_path: Path,
    highlight_ids: list[int] | None,
) -> Path:
    """v9 BEV renderer. Extracted as a separate function so tests can monkeypatch it."""
    builder = Nr3dScanNetBEVBuilder()
    return builder.build_with_labels(
        scene_id=scene_id,
        data_root=data_root,
        proposals=proposals,
        output_path=output_path,
        highlight_ids=highlight_ids,
    )


def _build_camera_trajectory(scene_dir: Path) -> dict[int, list[float]]:
    """Read traj.txt and emit {frame_id: [x, y, yaw]} for v9.

    Prepared scenes ship traj.txt either under ``conceptgraph/`` or ``raw/``;
    tolerate both, matching ScanNetSceneBEVBuilderBase._resolve_traj.
    """
    candidates = [
        scene_dir / "conceptgraph" / "traj.txt",
        scene_dir / "raw" / "traj.txt",
    ]
    traj_path = next((p for p in candidates if p.exists()), None)
    if traj_path is None:
        raise FileNotFoundError(
            f"traj.txt missing for scene {scene_dir.name}: tried "
            f"{', '.join(str(p) for p in candidates)}"
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


def write_v9_scene_artifacts(
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
    apply_enrichment_to_proposals(
        raw_pool.get("proposals", []) or [],
        load_enrichment_by_object_id(
            scene_dir / "conceptgraph" / "enriched_objects.json"
        ),
    )
    raw_pool.setdefault("source", "mask3d")
    raw_pool.setdefault("frame_index", {})
    raw_pool.setdefault("proposal_index", {})
    raw_pool.setdefault("annotated_image_dir", str(pack_dir / "annotated"))
    bev_path = bev_dir / "scene_bev_nr3d.png"
    catalog = from_vg_proposal_pool(
        pool=raw_pool,
        scene_id=scene_id,
        bev_image_path=str(bev_path),
        scene_category=scene_category,
        axis_align_matrix=raw_pool.get("axis_align_matrix"),
        valid_frame_ids=list(valid_frame_ids),
    )
    _render_v9_bev(
        scene_id=scene_id,
        data_root=data_root,
        proposals=catalog.proposals,
        output_path=bev_path,
        highlight_ids=None,
    )
    catalog_path.write_text(
        json.dumps(catalog.model_dump(), ensure_ascii=False, indent=2)
    )
    traj = _build_camera_trajectory(scene_dir)
    traj_out_path.write_text(json.dumps({str(k): v for k, v in traj.items()}))
    return {
        "bev_image_path": str(bev_path),
        "scene_catalog_path": str(catalog_path),
        "camera_trajectory_path": str(traj_out_path),
    }


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
        max_scene_artifact_cache_size=args.max_scene_artifact_cache_size,
        ensure_lightweight_cache=args.ensure_lightweight_cache,
        require_enrichment=args.require_enrichment,
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
    "write_sample_artifact",
]
