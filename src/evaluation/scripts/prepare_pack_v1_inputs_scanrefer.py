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
import pickle
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)
from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    project_bbox_3d_to_2d,
)
from benchmarks.scanrefer_loader import (
    ScanRefVGDataset,
    ScanRefVGSample,
)
from evaluation.scripts.prepare_pack_v1_inputs import (
    load_image_size,
    normalize_prepared_keyframes,
    validate_bbox_9dof,
    validate_matrix_4x4,
)

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
    parser.add_argument("--data-root", required=True, type=Path,
                        help="ScanRefer scannet root, e.g. data/scanrefer/scannet")
    parser.add_argument("--phase8-data-root", default=Path("data/nr3d/scannet"),
                        type=Path, help="Phase 8 GT-CG root for GT bbox lookup")
    parser.add_argument("--scanrefer-root", default=Path("data/scanrefer"),
                        type=Path, help="Root containing raw/ScanRefer_filtered_*.json")
    parser.add_argument("--raw-frames-root", default=Path("data/nr3d/scannet"),
                        type=Path, help="Root with <scene>/raw/ frames; ScanRefer "
                        "scenes are a superset of NR3D scenes so this is shared.")
    parser.add_argument("--pack-name", default="pack_scanrefer_v1")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=None)
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
) -> list[Path]:
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
    written: list[Path] = []
    for request in requests:
        sample = sample_lookup.get(request.sample_id)
        if sample is None:
            raise ValueError(
                f"No ScanRefer sample for sample_id={request.sample_id!r}"
            )
        if request.scene_id not in scene_artifacts:
            scene_artifacts[request.scene_id] = prepare_scene_artifacts(
                scene_id=request.scene_id,
                data_root=data_root,
                raw_frames_root=raw_frames_root,
                pack_name=pack_name,
            )
        written.append(
            write_sample_artifact(
                request=request,
                sample=sample,
                data_root=data_root,
                raw_frames_root=raw_frames_root,
                scene_artifacts=scene_artifacts[request.scene_id],
            )
        )
    return written


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
    valid_ids = set(proposal_ids)
    for frame_id, ids in frame_visibility.items():
        unknown = sorted(set(ids) - valid_ids)
        if unknown:
            raise ValueError(
                f"{scene_id} visibility frame {frame_id} unknown ids: {unknown}"
            )

    scene_dir = data_root / scene_id / pack_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "proposals.jsonl").write_text(
        json.dumps({
            "source": "mask3d",
            "scene_id": scene_id,
            "axis_align_matrix": None,
            "proposals": proposals,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (scene_dir / "visibility.json").write_text(
        json.dumps(
            {str(k): v for k, v in sorted(frame_visibility.items())},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    raw_scene_root = raw_frames_root / scene_id
    frames = scene_frames(raw_scene_root, sorted(frame_visibility))
    if not frames:
        raise ValueError(f"scene has no visible frames: {scene_id}")
    frame_by_id = {f.frame_id: f for f in frames}
    intrinsic = scene_intrinsic(raw_scene_root)
    image_size = load_image_size(frames[0].rgb_path)
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
    )


def build_proposals_from_mask3d_objects(
    *, objects: list[dict[str, Any]], scene_id: str,
) -> list[dict[str, Any]]:
    """Convert Mask3D-CG object list to proposals.jsonl shape."""
    proposals: list[dict[str, Any]] = []
    for obj_id, obj in enumerate(objects):
        names = obj.get("class_name")
        if not (isinstance(names, list) and names and all(
            isinstance(n, str) and n for n in names
        )):
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
            *((mn + mx) / 2.0).tolist(),    # cx, cy, cz
            *(mx - mn).tolist(),            # dx, dy, dz
            0.0, 0.0, 0.0,                  # Euler=0
        ]
        label = Counter(names).most_common(1)[0][0]
        ids = obj.get("class_id") or [-1]
        label_idx = int(Counter(int(v) for v in ids).most_common(1)[0][0])
        proposals.append({
            "id": obj_id,
            "bbox_3d": bbox_9dof,
            "score": 1.0,                   # uniform per Decision 4
            "label": label,
            "label_idx": label_idx,
        })
    return proposals


def write_sample_artifact(
    *, request: SampleRequest, sample: ScanRefVGSample,
    data_root: Path, raw_frames_root: Path, scene_artifacts: SceneArtifacts,
) -> Path:
    keyframes = select_keyframes_from_phase8_target(
        scene_id=request.scene_id,
        target_id=request.target_id,
        raw_frames_root=raw_frames_root,
        k=5,
    )
    normalized = normalize_prepared_keyframes(keyframes, scene_artifacts.annotated_dir)
    gt_bbox = validate_bbox_9dof(sample.gt_bbox_3d, f"{request.sample_id}.gt_bbox_3d_9dof")
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
        "source": "mask3d",
        "keyframes": normalized,
    }
    path = sample_artifact_path(data_root, request,
                                pack_name=scene_artifacts.scene_dir.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def select_keyframes_from_phase8_target(
    *, scene_id: str, target_id: int,
    raw_frames_root: Path, k: int = 5,
) -> list[dict[str, Any]]:
    """Pick top-k frames where the Phase 8 GT target is most visible.

    Uses the Phase 8 visibility index at
    data/nr3d/scannet/<scene>/conceptgraph/indices/visibility_index.pkl
    (note: NOT the Mask3D-CG visibility — keyframe choice is GT-driven so
    the agent sees frames where the target object is actually present).
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
        keyframes.append({
            "keyframe_idx": kfi,
            "image_path": str(_resolve_raw_rgb_path(
                raw_frames_root / scene_id, int(frame_id)
            )),
            "frame_id": int(frame_id),
        })
    return keyframes


def render_annotated_frames(
    *, proposal_by_id: dict[int, dict[str, Any]],
    frame_visibility: dict[int, list[int]],
    frame_by_id: dict[int, SceneFrame],
    intrinsic: np.ndarray, image_size: tuple[int, int],
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
            rect = project_bbox_3d_to_2d(
                prop["bbox_3d"], intrinsic, frame.extrinsic_world_to_cam, image_size,
            )
            if rect is None:
                continue
            marks.append({
                "proposal_id": int(prop_id),
                "label": prop["label"],
                "bbox_2d": rect,
            })
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
        cam_to_world = validate_matrix_4x4(np.loadtxt(pose_path), field_name=str(pose_path))
        frames.append(SceneFrame(
            frame_id=int(frame_id), raw_frame_id=raw_id,
            rgb_path=_resolve_raw_rgb_path(scene_root, int(frame_id)),
            extrinsic_world_to_cam=np.linalg.inv(cam_to_world),
        ))
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
    return Mask3dVisibility(
        object_to_views=_coerce_visibility(payload.get("object_to_views"), p),
        view_to_objects=_coerce_visibility(payload.get("view_to_objects"), p),
    )


def _coerce_visibility(raw: dict[Any, Any] | None,
                       path: Path) -> dict[int, list[tuple[int, float]]]:
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
        out.append(SampleRequest(
            sample_id=sid,
            scene_id=row.get("scene_id") or scene,
            target_id=int(row.get("target_id", tid)),
            ann_id=str(row.get("ann_id", ann_id)),
            category=str(row.get("category") or ""),
        ))
    return out


def sample_artifact_path(data_root: Path, request: SampleRequest, *,
                         pack_name: str = "pack_scanrefer_v1") -> Path:
    return (data_root / request.scene_id / pack_name / "samples"
            / f"{safe_sample_id(request.sample_id)}.json")


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
    )
    print(f"wrote {len(written)} sample artifacts under "
          f"{args.data_root}/<scene>/{args.pack_name}/")


if __name__ == "__main__":
    main()


__all__ = [
    "SampleRequest",
    "SceneArtifacts",
    "build_proposals_from_mask3d_objects",
    "load_sample_requests",
    "parse_sample_id",
    "prepare_pack_v1_inputs_scanrefer",
    "safe_sample_id",
]
