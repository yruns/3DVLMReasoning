"""Prepare offline pack-v1 inputs for EmbodiedScan VG side-by-side runs."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from agents.adapters.embodiedscan_adapter import EmbodiedScanVGAdapter
from benchmarks.embodiedscan_bbox_feasibility.render_marks import (
    render_marked_keyframe,
)
from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    build_frame_visibility,
    project_bbox_3d_to_2d,
)
from benchmarks.embodiedscan_loader import EmbodiedScanVGSample

SourceName = Literal["vdetr", "conceptgraph"]


@dataclass(frozen=True)
class SampleRequest:
    sample_id: str
    scene_id: str
    target_id: int
    category: str


@dataclass(frozen=True)
class SceneFrame:
    frame_id: int
    rgb_path: Path
    extrinsic_world_to_cam: np.ndarray


@dataclass(frozen=True)
class SceneArtifacts:
    scene_dir: Path
    proposals_jsonl: Path
    visibility_json: Path
    annotated_dir: Path
    frame_visibility: dict[int, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--vdetr-proposals-dir", required=True, type=Path)
    parser.add_argument("--embodiedscan-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", choices=["vdetr", "conceptgraph"], default="vdetr")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_sample_lookup(
    data_root: Path,
) -> tuple[EmbodiedScanVGAdapter, dict[tuple[str, int], EmbodiedScanVGSample]]:
    adapter = EmbodiedScanVGAdapter(data_root=data_root, scene_data_root=data_root)
    samples = adapter.load_samples(split="val", source_filter=None)
    lookup: dict[tuple[str, int], EmbodiedScanVGSample] = {}
    for sample in samples:
        if not isinstance(sample, EmbodiedScanVGSample):
            continue
        key = (sample.scene_id, int(sample.target_id))
        lookup.setdefault(key, sample)
    return adapter, lookup


def select_keyframes_for_sample(
    sample: EmbodiedScanVGSample,
    adapter: EmbodiedScanVGAdapter,
    embodiedscan_data_root: Path,
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    from query_scene.keyframe_selector import KeyframeSelector

    scene_path = adapter.get_scene_path(sample) / "conceptgraph"
    selector = KeyframeSelector.from_scene_path(
        str(scene_path),
        llm_model="gemini-2.5-pro",
        stride=1,
    )
    result = selector.select_keyframes_v2(
        sample.query,
        k=k,
        use_visual_context=False,
    )
    return normalize_keyframes(result)


def normalize_keyframes(stage1_result: Any) -> list[dict[str, Any]]:
    if isinstance(stage1_result, Sequence) and not isinstance(
        stage1_result,
        (str, bytes),
    ):
        return [
            _normalize_keyframe_item(item, idx)
            for idx, item in enumerate(stage1_result)
        ]

    metadata = getattr(stage1_result, "metadata", {}) or {}
    frame_mappings = metadata.get("frame_mappings") or []
    if frame_mappings:
        keyframes = []
        for idx, mapping in enumerate(frame_mappings):
            path = mapping.get("path")
            if path is None:
                continue
            frame_id = mapping.get(
                "resolved_frame_id",
                mapping.get("requested_frame_id"),
            )
            keyframes.append(
                {
                    "keyframe_idx": int(idx),
                    "image_path": str(path),
                    "frame_id": int(frame_id),
                }
            )
        return keyframes

    indices = list(getattr(stage1_result, "keyframe_indices", []) or [])
    paths = list(getattr(stage1_result, "keyframe_paths", []) or [])
    keyframes = []
    for idx, path in enumerate(paths):
        frame_id = indices[idx] if idx < len(indices) else idx
        keyframes.append(
            {
                "keyframe_idx": int(idx),
                "image_path": str(path),
                "frame_id": int(frame_id),
            }
        )
    return keyframes


def prepare_pack_v1_inputs(
    *,
    sample_ids_path: Path,
    vdetr_proposals_dir: Path,
    embodiedscan_data_root: Path,
    output_dir: Path,
    source: SourceName = "vdetr",
    max_samples: int | None = None,
) -> list[Path]:
    requests = load_sample_requests(sample_ids_path)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        requests = requests[:max_samples]

    adapter, sample_lookup = load_sample_lookup(embodiedscan_data_root)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    scene_artifacts: dict[str, SceneArtifacts] = {}
    written_samples: list[Path] = []
    for request in requests:
        sample = sample_lookup.get((request.scene_id, request.target_id))
        if sample is None:
            raise ValueError(
                f"No EmbodiedScan VG sample for scene_id={request.scene_id!r}, "
                f"target_id={request.target_id}"
            )

        if request.scene_id not in scene_artifacts:
            scene_artifacts[request.scene_id] = prepare_scene_artifacts(
                scene_id=request.scene_id,
                sample=sample,
                adapter=adapter,
                embodiedscan_data_root=embodiedscan_data_root,
                vdetr_proposals_dir=vdetr_proposals_dir,
                output_dir=output_dir,
            )

        sample_json = write_sample_artifact(
            request=request,
            sample=sample,
            adapter=adapter,
            embodiedscan_data_root=embodiedscan_data_root,
            scene_artifacts=scene_artifacts[request.scene_id],
            output_dir=output_dir,
            source=source,
        )
        written_samples.append(sample_json)
    return written_samples


def load_sample_requests(sample_ids_path: Path) -> list[SampleRequest]:
    raw = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"sample ids JSON must be a list: {sample_ids_path}")
    requests = []
    for row_index, row in enumerate(raw, start=1):
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
            raise ValueError(f"sample ids row {row_index} has invalid sample_id: {row!r}")
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


def prepare_scene_artifacts(
    *,
    scene_id: str,
    sample: EmbodiedScanVGSample,
    adapter: EmbodiedScanVGAdapter,
    embodiedscan_data_root: Path,
    vdetr_proposals_dir: Path,
    output_dir: Path,
) -> SceneArtifacts:
    scene_info = load_scene_info(adapter, sample)
    intrinsic = scene_intrinsic(scene_info)
    frames = scene_frames(scene_info, embodiedscan_data_root)
    if not frames:
        raise ValueError(f"scene has no frames: {scene_id}")

    predictions_path = vdetr_proposals_dir / scene_id / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing V-DETR predictions: {predictions_path}")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    proposals = validate_predictions(predictions, predictions_path)

    scene_dir = output_dir / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    proposals_jsonl = scene_dir / "proposals.jsonl"
    proposals_jsonl.write_text(
        json.dumps({"proposals": proposals}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    image_size = load_image_size(frames[0].rgb_path)
    extrinsics = {frame.frame_id: frame.extrinsic_world_to_cam for frame in frames}
    frame_visibility = build_frame_visibility(
        proposals=proposals,
        intrinsic=intrinsic,
        extrinsics_per_frame=extrinsics,
        image_size=image_size,
    )
    validate_visibility_frames(frame_visibility, {frame.frame_id for frame in frames})

    visibility_json = scene_dir / "visibility.json"
    visibility_json.write_text(
        json.dumps(frame_visibility, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    annotated_dir = scene_dir / "annotated"
    frame_by_id = {frame.frame_id: frame for frame in frames}
    render_annotated_frames(
        proposals=proposals,
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
    )


def write_sample_artifact(
    *,
    request: SampleRequest,
    sample: EmbodiedScanVGSample,
    adapter: EmbodiedScanVGAdapter,
    embodiedscan_data_root: Path,
    scene_artifacts: SceneArtifacts,
    output_dir: Path,
    source: SourceName,
) -> Path:
    keyframes = select_keyframes_for_sample(sample, adapter, embodiedscan_data_root)
    if not keyframes:
        raise ValueError(f"Stage 1 returned no keyframes for {request.sample_id}")
    normalized_keyframes = normalize_prepared_keyframes(
        keyframes,
        scene_artifacts.annotated_dir,
    )
    gt_bbox = validate_gt_bbox(getattr(sample, "gt_bbox_3d", None), request.sample_id)
    payload = {
        "sample_id": request.sample_id,
        "scene_id": request.scene_id,
        "target_id": request.target_id,
        "category": request.category or getattr(sample, "target", ""),
        "query": getattr(sample, "query", "") or getattr(sample, "text", ""),
        "gt_bbox_3d_9dof": gt_bbox,
        "scene_artifacts_dir": str(scene_artifacts.scene_dir),
        "source": source,
        "keyframes": normalized_keyframes,
    }
    if not payload["query"]:
        raise ValueError(f"Missing query for {request.sample_id}")

    sample_path = output_dir / "samples" / f"{request.scene_id}__{request.target_id}.json"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sample_path


def load_scene_info(adapter: EmbodiedScanVGAdapter, sample: EmbodiedScanVGSample) -> dict:
    try:
        scene_info = adapter.dataset.get_scene_info(sample.scan_id)
    except KeyError as exc:
        raise FileNotFoundError(f"Missing scene info for {sample.scan_id}") from exc
    if not isinstance(scene_info, dict):
        raise FileNotFoundError(f"Missing scene info for {sample.scan_id}")
    return scene_info


def scene_intrinsic(scene_info: dict) -> np.ndarray:
    raw = (
        scene_info.get("cam2img")
        or scene_info.get("intrinsic")
        or scene_info.get("depth_cam2img")
    )
    if raw is None:
        images = scene_info.get("images") or []
        if images:
            raw = images[0].get("cam2img") or images[0].get("intrinsic")
    if raw is None:
        raise ValueError("scene_info missing camera intrinsic matrix")
    mat = np.asarray(raw, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 3 or mat.shape[1] < 3:
        raise ValueError(f"intrinsic matrix must be at least 3x3, got {mat.shape}")
    return mat[:3, :3]


def scene_frames(scene_info: dict, data_root: Path) -> list[SceneFrame]:
    images = scene_info.get("images")
    if not isinstance(images, list):
        raise ValueError("scene_info.images must be a list")
    frames = []
    for default_id, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(f"scene_info image entry must be an object: {image!r}")
        frame_id = int(image.get("frame_id", image.get("frame_idx", default_id)))
        frames.append(
            SceneFrame(
                frame_id=frame_id,
                rgb_path=resolve_rgb_path(image, data_root),
                extrinsic_world_to_cam=image_world_to_cam(image),
            )
        )
    return frames


def resolve_rgb_path(image: dict, data_root: Path) -> Path:
    raw = (
        image.get("img_path")
        or image.get("image_path")
        or image.get("rgb_path")
        or image.get("path")
    )
    if raw is None:
        raise ValueError(f"scene frame missing RGB path: {image!r}")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return data_root / path


def image_world_to_cam(image: dict) -> np.ndarray:
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


def validate_predictions(predictions: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(predictions, dict):
        raise ValueError(f"predictions JSON must be an object: {path}")
    proposals = predictions.get("proposals")
    if not isinstance(proposals, list):
        raise ValueError(f"predictions.proposals must be a list: {path}")
    out = []
    for idx, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            raise ValueError(f"proposal[{idx}] must be an object in {path}")
        bbox = proposal.get("bbox_3d") or proposal.get("bbox_3d_9dof")
        bbox_9dof = validate_bbox_9dof(bbox, f"proposal[{idx}].bbox_3d")
        if "score" not in proposal:
            raise ValueError(f"proposal[{idx}].score missing in {path}")
        if "label" not in proposal:
            raise ValueError(f"proposal[{idx}].label missing in {path}")
        out.append(
            {
                **proposal,
                "bbox_3d": bbox_9dof,
                "score": float(proposal["score"]),
                "label": str(proposal["label"]),
            }
        )
    return out


def validate_bbox_9dof(raw: Any, field_name: str) -> list[float]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of 9 floats")
    if len(raw) != 9:
        raise ValueError(f"{field_name} must have 9 floats, got {len(raw)}")
    values = [float(value) for value in raw]
    if not np.isfinite(values).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return values


def load_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def validate_visibility_frames(
    frame_visibility: dict[int, list[int]],
    known_frame_ids: set[int],
) -> None:
    missing = set(frame_visibility) - known_frame_ids
    if missing:
        raise ValueError(f"visibility contains unknown frame ids: {sorted(missing)}")


def render_annotated_frames(
    *,
    proposals: list[dict[str, Any]],
    frame_visibility: dict[int, list[int]],
    frame_by_id: dict[int, SceneFrame],
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    annotated_dir: Path,
) -> None:
    for frame_id, proposal_indices in frame_visibility.items():
        if frame_id not in frame_by_id:
            raise ValueError(f"visibility references missing frame_id={frame_id}")
        frame = frame_by_id[frame_id]
        marks = []
        for proposal_idx in proposal_indices:
            proposal = proposals[int(proposal_idx)]
            rect = project_bbox_3d_to_2d(
                proposal["bbox_3d"],
                intrinsic,
                frame.extrinsic_world_to_cam,
                image_size,
            )
            if rect is None:
                continue
            marks.append(
                {
                    "proposal_id": int(proposal_idx),
                    "label": proposal["label"],
                    "bbox_2d": rect,
                }
            )
        render_marked_keyframe(
            rgb_path=frame.rgb_path,
            out_path=annotated_dir / f"frame_{frame_id}.png",
            marks=marks,
        )


def normalize_prepared_keyframes(
    keyframes: Sequence[dict[str, Any]],
    annotated_dir: Path,
) -> list[dict[str, Any]]:
    out = []
    for idx, keyframe in enumerate(keyframes):
        frame_id = int(keyframe["frame_id"])
        annotated_path = annotated_dir / f"frame_{frame_id}.png"
        image_path = (
            annotated_path
            if annotated_path.exists()
            else Path(str(keyframe["image_path"]))
        )
        out.append(
            {
                "keyframe_idx": int(keyframe.get("keyframe_idx", idx)),
                "image_path": str(image_path),
                "frame_id": frame_id,
            }
        )
    return out


def validate_gt_bbox(raw: Any, sample_id: str) -> list[float]:
    return validate_bbox_9dof(raw, f"{sample_id}.gt_bbox_3d_9dof")


def _normalize_keyframe_item(item: Any, idx: int) -> dict[str, Any]:
    if isinstance(item, dict):
        frame_id = item.get("frame_id", item.get("view_id", idx))
        path = item.get("image_path") or item.get("path")
        if path is None:
            raise ValueError(f"keyframe item missing image_path: {item!r}")
        return {
            "keyframe_idx": int(item.get("keyframe_idx", idx)),
            "image_path": str(path),
            "frame_id": int(frame_id),
        }
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return {
            "keyframe_idx": int(item[0]),
            "image_path": str(item[1]),
            "frame_id": int(item[2]),
        }
    raise ValueError(f"unsupported keyframe item: {item!r}")


def _required_nonempty_str(row: dict[str, Any], key: str, row_index: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"sample ids row {row_index} missing {key}: {row!r}")
    return value


def main() -> None:
    args = parse_args()
    prepare_pack_v1_inputs(
        sample_ids_path=args.sample_ids,
        vdetr_proposals_dir=args.vdetr_proposals_dir,
        embodiedscan_data_root=args.embodiedscan_data_root,
        output_dir=args.output_dir,
        source=args.source,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
