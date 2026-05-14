"""Prepare detector-backed pack inputs for EmbodiedScan VG runs.

Per-scene layout (output):

    <data_root>/<scene_id>/<pack_name>/
        proposals.jsonl          # V-DETR proposals normalized to pack schema
        visibility.json          # frame_id -> [detector proposal id, ...]
        annotated/frame_<id>.png # set-of-marks render
        samples/<target_id>.json # per-question bundle

The detector proposal IDs are local to the detector pool and deliberately do
not reuse EmbodiedScan GT bbox IDs.
"""

# NOTE: helpers below duplicate select_keyframes_for_sample / scene_intrinsic /
# load_image_size / etc. from prepare_pack_v1_inputs.py. Tracked for refactor
# into _pack_prep_common.py - do not let them drift.

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from benchmarks.embodiedscan_bbox_feasibility.models import ProposalRecord
from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)
from benchmarks.embodiedscan_bbox_feasibility.vdetr import class_name_from_id
from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    project_bbox_3d_to_2d,
    project_visible_bbox_3d_to_2d,
)


@dataclass(frozen=True)
class SampleRequest:
    sample_id: str
    scene_id: str
    target_id: int
    category: str


@dataclass(frozen=True)
class DetectorVGSample:
    sample_id: str
    scene_id: str
    scan_id: str
    target_id: int
    target: str
    query: str
    gt_bbox_3d: list[float]

    @property
    def text(self) -> str:
        return self.query


@dataclass(frozen=True)
class SceneFrame:
    frame_id: int
    rgb_path: Path
    extrinsic_world_to_cam: np.ndarray


@dataclass(frozen=True)
class DetectorSceneArtifacts:
    scene_dir: Path
    proposals_jsonl: Path
    visibility_json: Path
    annotated_dir: Path
    frame_visibility: dict[int, list[int]]
    proposals: list[dict[str, Any]]


class _EmbodiedScanDatasetView:
    def __init__(
        self,
        *,
        scene_index: dict[str, dict[str, Any]],
        label_to_name: dict[int, str],
    ) -> None:
        self._scene_index = scene_index
        self.label_to_name = label_to_name

    def get_scene_info(self, scan_id: str) -> dict[str, Any]:
        if scan_id not in self._scene_index:
            raise KeyError(f"scan_id {scan_id!r} not in scene index")
        return self._scene_index[scan_id]


class _EmbodiedScanAdapterView:
    def __init__(self, dataset: _EmbodiedScanDatasetView) -> None:
        self.dataset = dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector-records",
        required=True,
        type=Path,
        help="Path to detector_records.jsonl produced by run-detector.",
    )
    parser.add_argument(
        "--infos-pkl",
        type=Path,
        default=None,
        help="EmbodiedScan infos PKL. Defaults to <data-root>/embodiedscan_infos_<split>.pkl.",
    )
    parser.add_argument(
        "--vg-json",
        type=Path,
        default=None,
        help="EmbodiedScan VG JSON. Defaults to <data-root>/embodiedscan_<split>_vg.json.",
    )
    parser.add_argument(
        "--sample-ids",
        type=Path,
        default=None,
        help="Optional JSON list of sample objects or '<scene_id>::<target_id>' strings.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="EmbodiedScan data root. Outputs land under <data-root>/<scene>/<pack-name>/.",
    )
    parser.add_argument("--pack-name", default="pack_vdetr")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--visibility-min-area", type=float, default=32.0)
    parser.add_argument("--max-proposals-per-scene", type=int, default=256)
    return parser.parse_args()


def prepare_detector_pack_inputs(
    *,
    detector_records_path: Path,
    data_root: Path,
    sample_ids_path: Path | None = None,
    infos_pkl: Path | None = None,
    vg_json: Path | None = None,
    split: str = "val",
    pack_name: str = "pack_vdetr",
    visibility_min_area: float = 32.0,
    max_proposals_per_scene: int = 256,
) -> list[Path]:
    if visibility_min_area < 0:
        raise ValueError("visibility_min_area must be non-negative")
    if max_proposals_per_scene <= 0:
        raise ValueError("max_proposals_per_scene must be positive")
    if not pack_name:
        raise ValueError("pack_name must be non-empty")

    records_by_scene = load_detector_records(detector_records_path)
    requested_keys: set[tuple[str, int]] | None = None
    requests: list[SampleRequest] | None = None
    if sample_ids_path is not None:
        requests = load_sample_requests(sample_ids_path)
        requested_keys = {(request.scene_id, request.target_id) for request in requests}
    adapter, sample_lookup = load_sample_lookup(
        data_root,
        split,
        infos_pkl=infos_pkl,
        vg_json=vg_json,
        requested_keys=requested_keys,
        detector_record_scene_ids=set(records_by_scene),
    )
    if requests is None:
        requests = resolve_sample_requests(sample_ids_path, sample_lookup)

    scene_artifacts: dict[str, DetectorSceneArtifacts] = {}
    written_samples: list[Path] = []
    for request in requests:
        sample = sample_lookup.get((request.scene_id, request.target_id))
        if sample is None:
            raise ValueError(
                f"No EmbodiedScan VG sample for scene_id={request.scene_id!r}, "
                f"target_id={request.target_id} (split={split})"
            )
        record = records_by_scene.get(request.scene_id)
        if record is None:
            raise ValueError(
                f"No detector record for scene_id={request.scene_id!r} in "
                f"{detector_records_path}"
            )
        if request.scene_id not in scene_artifacts:
            scene_artifacts[request.scene_id] = prepare_detector_scene_artifacts(
                scene_id=request.scene_id,
                sample=sample,
                adapter=adapter,
                data_root=data_root,
                pack_name=pack_name,
                detector_record=record,
                visibility_min_area=visibility_min_area,
                max_proposals_per_scene=max_proposals_per_scene,
            )
        written_samples.append(
            write_sample_artifact(
                request=request,
                sample=sample,
                adapter=adapter,
                data_root=data_root,
                scene_artifacts=scene_artifacts[request.scene_id],
            )
        )

    if not written_samples:
        raise ValueError("No detector pack sample artifacts were written")
    return written_samples


def load_detector_records(path: Path) -> dict[str, ProposalRecord]:
    if not path.exists():
        raise FileNotFoundError(f"detector records not found: {path}")

    records_by_scene: dict[str, ProposalRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = ProposalRecord.model_validate(json.loads(line))
            if not record.scene_id:
                raise ValueError(f"{path}:{line_number} scene_id is empty")
            if record.failure_tag is not None:
                raise ValueError(
                    f"{path}:{line_number} detector record for {record.scene_id} "
                    f"has failure_tag={record.failure_tag}"
                )
            if not record.proposals:
                raise ValueError(
                    f"{path}:{line_number} detector record for {record.scene_id} "
                    "has no proposals"
                )
            if record.scene_id in records_by_scene:
                raise ValueError(
                    f"{path}:{line_number} duplicate detector record for "
                    f"scene_id={record.scene_id!r}; expected one scene-level record"
                )
            records_by_scene[record.scene_id] = record
    if not records_by_scene:
        raise ValueError(f"detector records JSONL is empty: {path}")
    return records_by_scene


def load_sample_lookup(
    data_root: Path,
    split: str,
    *,
    infos_pkl: Path | None = None,
    vg_json: Path | None = None,
    requested_keys: set[tuple[str, int]] | None = None,
    detector_record_scene_ids: set[str] | None = None,
) -> tuple[_EmbodiedScanAdapterView, dict[tuple[str, int], DetectorVGSample]]:
    infos_path = infos_pkl or data_root / f"embodiedscan_infos_{split}.pkl"
    vg_path = vg_json or data_root / f"embodiedscan_{split}_vg.json"
    if not infos_path.exists():
        raise FileNotFoundError(f"EmbodiedScan infos PKL not found: {infos_path}")
    if not vg_path.exists():
        raise FileNotFoundError(f"EmbodiedScan VG JSON not found: {vg_path}")

    with infos_path.open("rb") as handle:
        pkl_data = pickle.load(handle)
    categories = pkl_data["metainfo"]["categories"]
    label_to_name = {int(label): str(name) for name, label in categories.items()}
    scene_index = {str(scene["sample_idx"]): scene for scene in pkl_data["data_list"]}
    dataset = _EmbodiedScanDatasetView(
        scene_index=scene_index,
        label_to_name=label_to_name,
    )

    vg_entries = json.loads(vg_path.read_text(encoding="utf-8"))
    if not isinstance(vg_entries, list):
        raise ValueError(f"EmbodiedScan VG JSON must contain a list: {vg_path}")

    bbox_lookup = {
        scan_id: _build_bbox_dict(scene_info.get("instances") or [])
        for scan_id, scene_info in scene_index.items()
    }
    lookup: dict[tuple[str, int], DetectorVGSample] = {}
    for idx, entry in enumerate(vg_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"VG entry {idx} must be an object: {entry!r}")
        scan_id = str(entry.get("scan_id") or "")
        if not scan_id:
            if detector_record_scene_ids is not None:
                continue
            raise ValueError(f"VG row {idx} is missing required scan_id")
        scene_id = scan_id.split("/")[-1]
        # This is not a fallback: rows for scenes absent from detector_records
        # cannot be materialized by this run, so strict validation starts only
        # at the detector-record scene boundary.
        if (
            detector_record_scene_ids is not None
            and scene_id not in detector_record_scene_ids
        ):
            continue
        try:
            target_id = int(entry["target_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"VG row {idx} has invalid target_id") from exc
        if requested_keys is not None and (scene_id, target_id) not in requested_keys:
            continue
        if scan_id not in scene_index:
            raise ValueError(
                f"VG row {idx} references scan_id={scan_id!r} not present in "
                f"{infos_path}"
            )
        gt_bbox = bbox_lookup[scan_id].get(target_id)
        if gt_bbox is None:
            raise ValueError(
                f"VG row {idx} references target_id={target_id} in "
                f"scan_id={scan_id!r} without a unique GT bbox in {infos_path}"
            )
        sample = DetectorVGSample(
            sample_id=f"es_vg_{split}_{idx}",
            scene_id=scene_id,
            scan_id=scan_id,
            target_id=target_id,
            target=str(entry.get("target") or ""),
            query=str(entry.get("text") or ""),
            gt_bbox_3d=gt_bbox,
        )
        lookup.setdefault((scene_id, target_id), sample)
    if not lookup:
        raise ValueError(
            f"No valid EmbodiedScan VG samples loaded from {vg_path} and {infos_path}"
        )
    return _EmbodiedScanAdapterView(dataset), lookup


def resolve_sample_requests(
    sample_ids_path: Path | None,
    sample_lookup: dict[tuple[str, int], DetectorVGSample],
) -> list[SampleRequest]:
    if sample_ids_path is None:
        return [
            SampleRequest(
                sample_id=sample.sample_id,
                scene_id=sample.scene_id,
                target_id=int(sample.target_id),
                category=sample.target,
            )
            for sample in sample_lookup.values()
        ]
    return load_sample_requests(sample_ids_path)


def load_sample_requests(sample_ids_path: Path) -> list[SampleRequest]:
    raw = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"sample ids JSON must be a list: {sample_ids_path}")
    requests: list[SampleRequest] = []
    for row_index, row in enumerate(raw, start=1):
        if isinstance(row, str):
            sample_id = row.strip()
            scene_id, target_id = parse_scene_target_sample_id(
                sample_id,
                field_name=f"sample ids row {row_index}",
            )
            requests.append(
                SampleRequest(
                    sample_id=sample_id,
                    scene_id=scene_id,
                    target_id=target_id,
                    category="",
                )
            )
            continue
        if not isinstance(row, dict):
            raise ValueError(f"sample ids row {row_index} must be an object: {row!r}")
        scene_id = _required_nonempty_str(row, "scene_id", row_index)
        raw_target_id = row.get("target_id")
        try:
            target_id = int(raw_target_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sample ids row {row_index} has invalid target_id: {row!r}"
            ) from exc
        category = str(row.get("category") or "").strip()
        sample_id = str(row.get("sample_id") or f"{scene_id}::{target_id}").strip()
        if not sample_id:
            raise ValueError(
                f"sample ids row {row_index} has invalid sample_id: {row!r}"
            )
        requests.append(
            SampleRequest(
                sample_id=sample_id,
                scene_id=scene_id,
                target_id=target_id,
                category=category,
            )
        )
    if not requests:
        raise ValueError(f"sample ids JSON is empty: {sample_ids_path}")
    return requests


def parse_scene_target_sample_id(
    sample_id: str,
    *,
    field_name: str,
) -> tuple[str, int]:
    if "::" not in sample_id:
        raise ValueError(
            f"{field_name} must be '<scene_id>::<target_id>', got {sample_id!r}"
        )
    scene_id, target_text = sample_id.split("::", 1)
    if not scene_id:
        raise ValueError(f"{field_name} has empty scene_id: {sample_id!r}")
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"{field_name} has invalid target_id: {sample_id!r}") from exc
    return scene_id, target_id


def prepare_detector_scene_artifacts(
    *,
    scene_id: str,
    sample: Any,
    adapter: Any,
    data_root: Path,
    pack_name: str,
    detector_record: ProposalRecord,
    visibility_min_area: float,
    max_proposals_per_scene: int,
) -> DetectorSceneArtifacts:
    scene_info = load_scene_info(adapter, sample)
    intrinsic = scene_intrinsic(scene_info)
    frames = scene_frames(scene_info, data_root)
    if not frames:
        raise ValueError(f"scene has no frames: {scene_id}")

    proposals = build_proposals_from_detector_record(
        detector_record,
        max_proposals_per_scene=max_proposals_per_scene,
    )
    if not proposals:
        raise ValueError(f"scene has no detector proposals: {scene_id}")

    scene_dir = data_root / scene_id / pack_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    axis_align_matrix = scene_info.get("axis_align_matrix")
    axis_align_arr = (
        validate_matrix_4x4(axis_align_matrix, field_name="axis_align_matrix")
        if axis_align_matrix is not None
        else None
    )

    frame_visibility = derive_detector_visibility(
        proposals=proposals,
        frames=frames,
        intrinsic=intrinsic,
        visibility_min_area=visibility_min_area,
        axis_align_matrix=axis_align_arr,
    )
    visibility_json = scene_dir / "visibility.json"
    visibility_json.write_text(
        json.dumps(frame_visibility, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    annotated_dir = scene_dir / "annotated"
    proposal_frame_views = render_detector_annotated_frames(
        proposal_by_id={int(p["id"]): p for p in proposals},
        frame_visibility=frame_visibility,
        frame_by_id={frame.frame_id: frame for frame in frames},
        intrinsic=intrinsic,
        annotated_dir=annotated_dir,
        visibility_min_area=visibility_min_area,
        axis_align_matrix=axis_align_arr,
    )
    proposals_jsonl = scene_dir / "proposals.jsonl"
    proposals_jsonl.write_text(
        json.dumps(
            {
                "source": "vdetr",
                "scene_id": scene_id,
                "axis_align_matrix": (
                    axis_align_arr.tolist() if axis_align_arr is not None else None
                ),
                "proposals": [
                    {
                        **proposal,
                        "frame_views": proposal_frame_views.get(
                            int(proposal["id"]), {}
                        ),
                    }
                    for proposal in proposals
                ],
                "frame_views_emitted": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return DetectorSceneArtifacts(
        scene_dir=scene_dir,
        proposals_jsonl=proposals_jsonl,
        visibility_json=visibility_json,
        annotated_dir=annotated_dir,
        frame_visibility=frame_visibility,
        proposals=proposals,
    )


def build_proposals_from_detector_record(
    record: ProposalRecord,
    *,
    max_proposals_per_scene: int,
) -> list[dict[str, Any]]:
    if not record.proposals:
        raise ValueError(f"detector record for {record.scene_id} has no proposals")

    scored: list[tuple[int, Any, float]] = []
    for input_idx, proposal in enumerate(record.proposals):
        if proposal.score is None:
            raise ValueError(
                f"{record.scene_id}.proposals[{input_idx}] missing required score"
            )
        scored.append((input_idx, proposal, float(proposal.score)))

    sorted_items = sorted(scored, key=lambda item: (-item[2], item[0]))
    out: list[dict[str, Any]] = []
    for output_id, (input_idx, proposal, score) in enumerate(
        sorted_items[:max_proposals_per_scene]
    ):
        metadata = dict(proposal.metadata)
        if "class_id" not in metadata:
            raise ValueError(
                f"{record.scene_id}.proposals[{input_idx}].metadata.class_id "
                "is required"
            )
        class_id = int(metadata["class_id"])
        clean_metadata = {
            str(key): value for key, value in metadata.items() if key != "label"
        }
        clean_metadata["class_id"] = class_id
        detector = clean_metadata.get("detector")
        if not isinstance(detector, str) or not detector.strip():
            raise ValueError(
                f"proposal[{input_idx}] in scene {record.scene_id} is missing "
                "required metadata.detector"
            )
        clean_metadata["detector"] = detector.strip()
        clean_metadata["raw_corners"] = validate_raw_corners_metadata(
            clean_metadata.get("raw_corners"),
            scene_id=record.scene_id,
            proposal_index=input_idx,
        )
        out.append(
            {
                "id": output_id,
                "bbox_3d": validate_bbox_9dof(
                    proposal.bbox_3d,
                    f"{record.scene_id}.proposals[{input_idx}].bbox_3d",
                ),
                "score": score,
                "label": class_name_from_id(class_id),
                "source": "vdetr",
                "metadata": clean_metadata,
            }
        )
    return out


def validate_raw_corners_metadata(
    raw: Any,
    *,
    scene_id: str,
    proposal_index: int,
) -> list[list[float]]:
    if not isinstance(raw, list) or len(raw) != 8:
        raise ValueError(
            f"proposal[{proposal_index}] in scene {scene_id} has invalid "
            "metadata.raw_corners"
        )
    for corner in raw:
        if not isinstance(corner, list) or len(corner) != 3:
            raise ValueError(
                f"proposal[{proposal_index}] in scene {scene_id} has invalid "
                "metadata.raw_corners"
            )
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"proposal[{proposal_index}] in scene {scene_id} has invalid "
            "metadata.raw_corners"
        ) from exc
    if arr.shape != (8, 3) or not np.isfinite(arr).all():
        raise ValueError(
            f"proposal[{proposal_index}] in scene {scene_id} has invalid "
            "metadata.raw_corners"
        )
    return raw


def derive_detector_visibility(
    *,
    proposals: list[dict[str, Any]],
    frames: list[SceneFrame],
    intrinsic: np.ndarray,
    visibility_min_area: float,
    axis_align_matrix: np.ndarray | None,
) -> dict[int, list[int]]:
    frame_visibility: dict[int, list[int]] = {}
    for frame in frames:
        visible: list[int] = []
        image_size = load_image_size(frame.rgb_path)
        extrinsic = detector_extrinsic_for_frame(frame, axis_align_matrix)
        for proposal in proposals:
            rect = visible_projected_rect(
                proposal["bbox_3d"],
                intrinsic,
                extrinsic,
                image_size,
                visibility_min_area,
            )
            # Depth-aware occlusion is intentionally left to a later pass.
            if rect is not None:
                visible.append(int(proposal["id"]))
        frame_visibility[int(frame.frame_id)] = visible
    return frame_visibility


def render_detector_annotated_frames(
    *,
    proposal_by_id: dict[int, dict[str, Any]],
    frame_visibility: dict[int, list[int]],
    frame_by_id: dict[int, SceneFrame],
    intrinsic: np.ndarray,
    annotated_dir: Path,
    visibility_min_area: float,
    axis_align_matrix: np.ndarray | None,
) -> dict[int, dict[str, dict[str, Any]]]:
    frame_views: dict[int, dict[str, dict[str, Any]]] = {
        int(pid): {} for pid in proposal_by_id
    }
    for frame_id, visible_ids in frame_visibility.items():
        if frame_id not in frame_by_id:
            raise ValueError(f"visibility references missing frame_id={frame_id}")
        frame = frame_by_id[frame_id]
        image_size = load_image_size(frame.rgb_path)
        extrinsic = detector_extrinsic_for_frame(frame, axis_align_matrix)
        marks = []
        for proposal_id in visible_ids:
            proposal = proposal_by_id.get(int(proposal_id))
            if proposal is None:
                raise ValueError(
                    f"visibility refers to unknown proposal_id={proposal_id}"
                )
            rect = visible_projected_rect(
                proposal["bbox_3d"],
                intrinsic,
                extrinsic,
                image_size,
                visibility_min_area,
            )
            if rect is None:
                raise ValueError(
                    f"visible proposal_id={proposal_id} no longer projects in "
                    f"frame_id={frame_id}"
                )
            marks.append(
                {
                    "proposal_id": int(proposal_id),
                    "label": proposal["label"],
                    "bbox_2d": rect,
                }
            )
            frame_views[int(proposal_id)][str(int(frame_id))] = {
                "bbox_2d": [int(v) for v in rect],
                "raw_rgb_path": str(frame.rgb_path),
            }
        render_marked_keyframe(
            rgb_path=frame.rgb_path,
            out_path=annotated_dir / f"frame_{frame_id}.png",
            marks=marks,
        )
    return frame_views


def visible_projected_rect(
    bbox_9dof: list[float],
    intrinsic: np.ndarray,
    extrinsic_world_to_cam: np.ndarray,
    image_size: tuple[int, int],
    visibility_min_area: float,
) -> tuple[int, int, int, int] | None:
    rect = project_visible_bbox_3d_to_2d(
        bbox_9dof,
        intrinsic,
        extrinsic_world_to_cam,
        image_size,
    )
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
    if area < visibility_min_area:
        return None
    return rect


def detector_extrinsic_for_frame(
    frame: SceneFrame,
    axis_align_matrix: np.ndarray | None,
) -> np.ndarray:
    if axis_align_matrix is None:
        return frame.extrinsic_world_to_cam
    aligned_to_world = np.linalg.inv(axis_align_matrix)
    return frame.extrinsic_world_to_cam @ aligned_to_world


def write_sample_artifact(
    *,
    request: SampleRequest,
    sample: Any,
    adapter: Any,
    data_root: Path,
    scene_artifacts: DetectorSceneArtifacts,
) -> Path:
    keyframes = select_keyframes_for_sample(sample, adapter, data_root)
    if not keyframes:
        raise ValueError(
            f"keyframe selection returned no frames for {request.sample_id}"
        )
    normalized_keyframes = normalize_prepared_keyframes(
        keyframes,
        scene_artifacts.annotated_dir,
    )
    payload = {
        "sample_id": request.sample_id,
        "scene_id": request.scene_id,
        "target_id": request.target_id,
        "category": request.category or getattr(sample, "target", ""),
        "query": getattr(sample, "query", "") or getattr(sample, "text", ""),
        "gt_bbox_3d_9dof": validate_gt_bbox(
            getattr(sample, "gt_bbox_3d", None),
            request.sample_id,
        ),
        "scene_artifacts_dir": str(scene_artifacts.scene_dir),
        "source": "vdetr",
        "keyframes": normalized_keyframes,
        "proposals": scene_artifacts.proposals,
    }
    if not payload["query"]:
        raise ValueError(f"Missing query for {request.sample_id}")

    samples_dir = scene_artifacts.scene_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_path = samples_dir / f"{request.target_id}.json"
    sample_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sample_path


def select_keyframes_for_sample(
    sample: Any,
    adapter: Any,
    embodiedscan_data_root: Path,
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Pick up to k GT-visible, projectable frames for the target."""
    scene_info = adapter.dataset.get_scene_info(sample.scan_id)
    if not scene_info:
        raise ValueError(f"scene_info missing for scan_id={sample.scan_id}")
    images = scene_info.get("images") or []
    if not isinstance(images, list) or not images:
        raise ValueError(f"scene_info.images missing or empty for {sample.scan_id}")

    target_instance_idx = unique_instance_index_for_bbox_id(
        scene_info.get("instances") or [],
        int(sample.target_id),
        field_name=f"{sample.scan_id}.target_id={sample.target_id}",
    )

    visible: list[tuple[int, dict[str, Any]]] = []
    for default_id, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        visible_instance_indices = {
            int(x) for x in (image.get("visible_instance_ids") or [])
        }
        if target_instance_idx in visible_instance_indices:
            frame_id = int(image.get("frame_id", image.get("frame_idx", default_id)))
            visible.append((frame_id, image))
    if not visible:
        raise ValueError(
            f"no visible frames for target_id={sample.target_id} in {sample.scan_id}"
        )

    sample_name = getattr(
        sample,
        "sample_id",
        f"{getattr(sample, 'scene_id', sample.scan_id)}::{sample.target_id}",
    )
    gt_bbox = validate_gt_bbox(getattr(sample, "gt_bbox_3d", None), sample_name)
    intrinsic = scene_intrinsic(scene_info)
    axis_align_matrix = scene_info.get("axis_align_matrix")
    aligned_to_world: np.ndarray | None = None
    if axis_align_matrix is not None:
        aligned_to_world = np.linalg.inv(
            validate_matrix_4x4(axis_align_matrix, field_name="axis_align_matrix")
        )

    projectable: list[tuple[int, dict[str, Any], Path]] = []
    for frame_id, image in visible:
        rgb_path = resolve_rgb_path(image, embodiedscan_data_root, frame_pos=frame_id)
        extrinsic = image_world_to_cam(image)
        if aligned_to_world is not None:
            extrinsic = extrinsic @ aligned_to_world
        rect = project_bbox_3d_to_2d(
            gt_bbox,
            intrinsic,
            extrinsic,
            load_image_size(rgb_path),
        )
        if rect is not None:
            projectable.append((frame_id, image, rgb_path))
    if not projectable:
        raise ValueError(
            f"no projectable visible frames for target_id={sample.target_id} "
            f"in {sample.scan_id}"
        )

    if len(projectable) <= k:
        chosen = projectable
    else:
        step = len(projectable) / k
        chosen = [
            projectable[min(int(i * step), len(projectable) - 1)] for i in range(k)
        ]

    keyframes: list[dict[str, Any]] = []
    for keyframe_idx, (frame_id, _image, rgb_path) in enumerate(chosen):
        keyframes.append(
            {
                "keyframe_idx": keyframe_idx,
                "image_path": str(rgb_path),
                "frame_id": frame_id,
            }
        )
    return keyframes


def load_scene_info(adapter: Any, sample: Any) -> dict[str, Any]:
    try:
        scene_info = adapter.dataset.get_scene_info(sample.scan_id)
    except KeyError as exc:
        raise FileNotFoundError(f"Missing scene info for {sample.scan_id}") from exc
    if not isinstance(scene_info, dict):
        raise FileNotFoundError(f"Missing scene info for {sample.scan_id}")
    return scene_info


def scene_intrinsic(scene_info: dict[str, Any]) -> np.ndarray:
    raw = None
    for key in ("cam2img", "intrinsic", "depth_cam2img"):
        v = scene_info.get(key)
        if v is not None:
            raw = v
            break
    if raw is None:
        images = scene_info.get("images") or []
        if images:
            for key in ("cam2img", "intrinsic"):
                v = images[0].get(key)
                if v is not None:
                    raw = v
                    break
    if raw is None:
        raise ValueError("scene_info missing camera intrinsic matrix")
    mat = np.asarray(raw, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 3 or mat.shape[1] < 3:
        raise ValueError(f"intrinsic matrix must be at least 3x3, got {mat.shape}")
    return mat[:3, :3]


def scene_frames(scene_info: dict[str, Any], data_root: Path) -> list[SceneFrame]:
    images = scene_info.get("images")
    if not isinstance(images, list):
        raise ValueError("scene_info.images must be a list")
    frames: list[SceneFrame] = []
    for default_id, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"scene_info image entry must be an object: {image!r}")
        frame_id = int(image.get("frame_id", image.get("frame_idx", default_id)))
        frames.append(
            SceneFrame(
                frame_id=frame_id,
                rgb_path=resolve_rgb_path(image, data_root, frame_pos=frame_id),
                extrinsic_world_to_cam=image_world_to_cam(image),
            )
        )
    return frames


def resolve_rgb_path(
    image: dict[str, Any],
    data_root: Path,
    *,
    frame_pos: int | None = None,
) -> Path:
    raw = None
    for key in ("img_path", "image_path", "rgb_path", "path"):
        v = image.get(key)
        if v is not None:
            raw = v
            break
    if raw is None:
        raise ValueError(f"scene frame missing RGB path: {image!r}")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidate = data_root / path
    if candidate.exists():
        return candidate
    if frame_pos is not None:
        parts = path.parts
        scene_id = parts[-2] if len(parts) >= 2 else None
        if scene_id:
            for ext in ("jpg", "png"):
                local = data_root / scene_id / "raw" / f"{frame_pos:06d}-rgb.{ext}"
                if local.exists():
                    return local
    return candidate


def image_world_to_cam(image: dict[str, Any]) -> np.ndarray:
    for key in ("world_to_cam", "extrinsic_world_to_cam", "extrinsic"):
        if key in image:
            return validate_matrix_4x4(image[key], field_name=key)
    for key in ("cam2global", "cam2world", "pose"):
        if key in image:
            return np.linalg.inv(validate_matrix_4x4(image[key], field_name=key))
    raise ValueError(f"scene frame missing camera transform: {image!r}")


def validate_matrix_4x4(raw: Any, *, field_name: str) -> np.ndarray:
    mat = np.asarray(raw, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"{field_name} must have shape (4, 4), got {mat.shape}")
    if not np.isfinite(mat).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return mat


def validate_bbox_9dof(raw: Any, field_name: str) -> list[float]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of 9 floats")
    if len(raw) != 9:
        raise ValueError(f"{field_name} must have 9 floats, got {len(raw)}")
    values = [float(value) for value in raw]
    if not np.isfinite(values).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return values


def validate_gt_bbox(raw: Any, sample_id: str) -> list[float]:
    return validate_bbox_9dof(raw, f"{sample_id}.gt_bbox_3d_9dof")


def load_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def normalize_prepared_keyframes(
    keyframes: Sequence[dict[str, Any]],
    annotated_dir: Path,
) -> list[dict[str, Any]]:
    out = []
    for idx, keyframe in enumerate(keyframes):
        frame_id = int(keyframe["frame_id"])
        out.append(
            {
                "keyframe_idx": int(keyframe.get("keyframe_idx", idx)),
                "image_path": str(Path(str(keyframe["image_path"]))),
                "frame_id": frame_id,
            }
        )
    return out


def unique_instance_index_for_bbox_id(
    instances: list[dict[str, Any]],
    bbox_id: int,
    *,
    field_name: str,
) -> int:
    matches = [
        idx
        for idx, inst in enumerate(instances)
        if isinstance(inst, dict) and int(inst.get("bbox_id", -1)) == int(bbox_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{field_name} must map to exactly one instance; matched {len(matches)}"
        )
    return matches[0]


def _build_bbox_dict(instances: list[dict[str, Any]]) -> dict[int, list[float]]:
    counts: dict[int, int] = {}
    for inst in instances:
        bbox_id = int(inst["bbox_id"])
        counts[bbox_id] = counts.get(bbox_id, 0) + 1
    return {
        int(inst["bbox_id"]): validate_bbox_9dof(
            inst["bbox_3d"],
            f"instances[{idx}].bbox_3d",
        )
        for idx, inst in enumerate(instances)
        if counts[int(inst["bbox_id"])] == 1
    }


def _required_nonempty_str(row: dict[str, Any], key: str, row_index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"sample ids row {row_index}.{key} must be a non-empty string: {row!r}"
        )
    return value.strip()


def main() -> None:
    args = parse_args()
    written = prepare_detector_pack_inputs(
        detector_records_path=args.detector_records,
        infos_pkl=args.infos_pkl,
        vg_json=args.vg_json,
        sample_ids_path=args.sample_ids,
        data_root=args.data_root,
        split=args.split,
        pack_name=args.pack_name,
        visibility_min_area=args.visibility_min_area,
        max_proposals_per_scene=args.max_proposals_per_scene,
    )
    print(
        f"wrote {len(written)} sample artifacts under "
        f"{args.data_root}/<scene>/{args.pack_name}/"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DetectorSceneArtifacts",
    "DetectorVGSample",
    "SampleRequest",
    "SceneFrame",
    "build_proposals_from_detector_record",
    "derive_detector_visibility",
    "load_detector_records",
    "load_sample_lookup",
    "load_sample_requests",
    "prepare_detector_pack_inputs",
    "prepare_detector_scene_artifacts",
    "select_keyframes_for_sample",
    "write_sample_artifact",
]
