"""Evaluate detector proposal-pool recall for EmbodiedScan VG packs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.embodiedscan_eval import compute_oriented_iou_3d


@dataclass(frozen=True)
class Proposal:
    id: int
    bbox_3d: list[float]
    label: str


@dataclass(frozen=True)
class ScenePack:
    pack_dir: Path
    proposals: list[Proposal]
    visibility: dict[int, list[int]] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detector-pool oracle recall on EmbodiedScan packs."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pack-name", required=True)
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--use-keyframe-pool",
        default=True,
        type=_parse_bool,
        choices=[True, False],
        help="true: use proposals visible in sample keyframes; false: use all scene proposals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = eval_proposal_pool_recall(
        data_root=args.data_root,
        pack_name=args.pack_name,
        sample_ids_path=args.sample_ids,
        output_json=args.output_json,
        use_keyframe_pool=args.use_keyframe_pool,
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


def eval_proposal_pool_recall(
    *,
    data_root: Path,
    pack_name: str,
    sample_ids_path: Path,
    output_json: Path,
    use_keyframe_pool: bool = True,
) -> dict[str, Any]:
    if not pack_name:
        raise ValueError("pack_name must be non-empty")

    sample_ids = load_sample_ids(sample_ids_path)
    scene_cache: dict[str, ScenePack] = {}
    per_sample: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        scene_id, target_id = parse_sample_id(sample_id)
        pack = scene_cache.get(scene_id)
        if pack is None:
            pack = load_scene_pack(
                data_root=data_root,
                scene_id=scene_id,
                pack_name=pack_name,
                require_visibility=use_keyframe_pool,
            )
            scene_cache[scene_id] = pack
        sample_payload = load_sample_payload(pack.pack_dir, sample_id, target_id)
        gt_bbox = validate_bbox_9dof(
            sample_payload.get("gt_bbox_3d_9dof"),
            f"{pack.pack_dir}/samples/{target_id}.json gt_bbox_3d_9dof",
        )
        pool = proposal_pool_for_sample(
            sample_payload=sample_payload,
            sample_id=sample_id,
            pack=pack,
            use_keyframe_pool=use_keyframe_pool,
        )
        per_sample.append(
            evaluate_sample(
                sample_id=sample_id,
                pack_name=pack_name,
                gt_bbox=gt_bbox,
                gt_bbox_label_id=sample_gt_bbox_label_id(sample_payload, sample_id),
                proposals=pool,
            )
        )

    aggregate = aggregate_results(
        per_sample,
        pack_name=pack_name,
        use_keyframe_pool=use_keyframe_pool,
    )
    result = {"aggregate": aggregate, "samples": per_sample}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def load_sample_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"sample ids JSON not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"sample ids JSON must be a list: {path}")

    sample_ids: list[str] = []
    for index, row in enumerate(raw, start=1):
        if isinstance(row, str):
            sample_id = row.strip()
        elif isinstance(row, dict) and isinstance(row.get("sample_id"), str):
            sample_id = row["sample_id"].strip()
        else:
            raise ValueError(
                f"sample ids row {index} must be '<scene>::<target>' or object with sample_id"
            )
        parse_sample_id(sample_id)
        sample_ids.append(sample_id)

    if not sample_ids:
        raise ValueError(f"sample ids JSON is empty: {path}")
    return sample_ids


def parse_sample_id(sample_id: str) -> tuple[str, int]:
    if "::" not in sample_id:
        raise ValueError(f"sample_id must be '<scene_id>::<target_id>': {sample_id!r}")
    scene_id, target_text = sample_id.split("::", 1)
    if not scene_id:
        raise ValueError(f"sample_id has empty scene_id: {sample_id!r}")
    try:
        target_id = int(target_text)
    except ValueError as exc:
        raise ValueError(f"sample_id has invalid target_id: {sample_id!r}") from exc
    return scene_id, target_id


def load_scene_pack(
    *,
    data_root: Path,
    scene_id: str,
    pack_name: str,
    require_visibility: bool,
) -> ScenePack:
    pack_dir = data_root / scene_id / pack_name
    if not pack_dir.exists():
        raise FileNotFoundError(f"pack directory not found: {pack_dir}")
    if not pack_dir.is_dir():
        raise NotADirectoryError(f"pack path is not a directory: {pack_dir}")

    proposals = load_proposals(pack_dir / "proposals.jsonl")
    visibility = (
        load_visibility(pack_dir / "visibility.json", proposals)
        if require_visibility
        else None
    )
    return ScenePack(pack_dir=pack_dir, proposals=proposals, visibility=visibility)


def load_proposals(path: Path) -> list[Proposal]:
    if not path.exists():
        raise FileNotFoundError(f"proposals file not found: {path}")
    docs = load_json_documents(path)
    if len(docs) == 1 and isinstance(docs[0], dict) and "proposals" in docs[0]:
        raw_proposals = docs[0]["proposals"]
        if not isinstance(raw_proposals, list):
            raise ValueError(f"{path} top-level 'proposals' must be a list")
    else:
        raw_proposals = docs

    proposals: list[Proposal] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(raw_proposals):
        proposal = validate_proposal(raw, path, index)
        if proposal.id in seen_ids:
            raise ValueError(f"{path} has duplicate proposal id {proposal.id}")
        seen_ids.add(proposal.id)
        proposals.append(proposal)
    return proposals


def load_json_documents(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    docs: list[Any] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        obj, index = decoder.raw_decode(text, index)
        docs.append(obj)
    if not docs:
        raise ValueError(f"JSON document is empty: {path}")
    return docs


def validate_proposal(raw: Any, path: Path, index: int) -> Proposal:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} proposal[{index}] must be an object")
    for key in ("id", "bbox_3d", "score", "label"):
        if key not in raw:
            raise ValueError(f"{path} proposal[{index}].{key} is required")
    try:
        proposal_id = int(raw["id"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} proposal[{index}].id must be an integer") from exc
    try:
        score = float(raw["score"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} proposal[{index}].score must be numeric") from exc
    if not math.isfinite(score):
        raise ValueError(f"{path} proposal[{index}].score must be finite")
    label = str(raw["label"]).strip()
    if not label:
        raise ValueError(f"{path} proposal[{index}].label must be non-empty")
    return Proposal(
        id=proposal_id,
        bbox_3d=validate_bbox_9dof(raw["bbox_3d"], f"{path} proposal[{index}].bbox_3d"),
        label=label,
    )


def load_visibility(path: Path, proposals: list[Proposal]) -> dict[int, list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"visibility file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"visibility JSON must be an object: {path}")
    known_ids = {proposal.id for proposal in proposals}
    visibility: dict[int, list[int]] = {}
    for raw_frame_id, raw_ids in raw.items():
        try:
            frame_id = int(raw_frame_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} frame id must be an integer: {raw_frame_id!r}"
            ) from exc
        if not isinstance(raw_ids, list):
            raise ValueError(f"{path} frame {frame_id} visibility must be a list")
        ids: list[int] = []
        for raw_id in raw_ids:
            try:
                proposal_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} frame {frame_id} proposal id must be an integer"
                ) from exc
            if proposal_id not in known_ids:
                raise ValueError(
                    f"{path} frame {frame_id} references unknown proposal id {proposal_id}"
                )
            ids.append(proposal_id)
        visibility[frame_id] = ids
    return visibility


def load_sample_payload(
    pack_dir: Path, sample_id: str, target_id: int
) -> dict[str, Any]:
    sample_path = pack_dir / "samples" / f"{target_id}.json"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"sample artifact not found for {sample_id}: {sample_path}"
        )
    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"sample artifact must be an object: {sample_path}")
    found_sample_id = raw.get("sample_id")
    if found_sample_id is not None and str(found_sample_id) != sample_id:
        raise ValueError(
            f"{sample_path} sample_id mismatch: expected {sample_id}, got {found_sample_id}"
        )
    return raw


def proposal_pool_for_sample(
    *,
    sample_payload: dict[str, Any],
    sample_id: str,
    pack: ScenePack,
    use_keyframe_pool: bool,
) -> list[Proposal]:
    if not use_keyframe_pool:
        return pack.proposals
    if pack.visibility is None:
        raise ValueError(
            f"visibility is required when use_keyframe_pool=true: {pack.pack_dir}"
        )

    frame_ids = sample_keyframe_ids(sample_payload, sample_id)
    visible_ids: set[int] = set()
    for frame_id in frame_ids:
        if frame_id not in pack.visibility:
            raise ValueError(
                f"{pack.pack_dir}/visibility.json missing keyframe frame_id={frame_id} "
                f"for sample {sample_id}"
            )
        visible_ids.update(pack.visibility[frame_id])
    return [proposal for proposal in pack.proposals if proposal.id in visible_ids]


def sample_keyframe_ids(sample_payload: dict[str, Any], sample_id: str) -> list[int]:
    raw_keyframes = sample_payload.get("keyframes")
    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        raise ValueError(f"sample {sample_id} must contain a non-empty keyframes list")
    frame_ids: list[int] = []
    for index, keyframe in enumerate(raw_keyframes):
        if not isinstance(keyframe, dict) or "frame_id" not in keyframe:
            raise ValueError(
                f"sample {sample_id} keyframes[{index}].frame_id is required"
            )
        try:
            frame_ids.append(int(keyframe["frame_id"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sample {sample_id} keyframes[{index}].frame_id must be an integer"
            ) from exc
    return frame_ids


def sample_gt_bbox_label_id(sample_payload: dict[str, Any], sample_id: str) -> int:
    if "gt_bbox_label_id" in sample_payload:
        raw_id = sample_payload["gt_bbox_label_id"]
    elif "target_id" in sample_payload:
        raw_id = sample_payload["target_id"]
    else:
        raise ValueError(f"sample {sample_id} missing target_id / gt_bbox_label_id")
    try:
        return int(raw_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample {sample_id} has invalid gt bbox label id") from exc


def evaluate_sample(
    *,
    sample_id: str,
    pack_name: str,
    gt_bbox: list[float],
    gt_bbox_label_id: int,
    proposals: list[Proposal],
) -> dict[str, Any]:
    best_iou = -1.0
    best: Proposal | None = None
    for proposal in proposals:
        iou = compute_oriented_iou_3d(proposal.bbox_3d, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best = proposal
    if best is None:
        best_iou = 0.0

    return {
        "sample_id": sample_id,
        "pack_name": pack_name,
        "n_proposals_in_pool": len(proposals),
        "max_iou_3d": float(best_iou),
        "best_proposal_id": best.id if best is not None else None,
        "best_proposal_label": best.label if best is not None else None,
        "gt_bbox_label_id": gt_bbox_label_id,
        "hit_at_0.25": bool(best_iou >= 0.25),
        "hit_at_0.50": bool(best_iou >= 0.50),
    }


def aggregate_results(
    samples: list[dict[str, Any]],
    *,
    pack_name: str,
    use_keyframe_pool: bool,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot aggregate an empty sample list")
    ious = [float(sample["max_iou_3d"]) for sample in samples]
    proposal_counts = [int(sample["n_proposals_in_pool"]) for sample in samples]
    return {
        "pack_name": pack_name,
        "n_samples": len(samples),
        "use_keyframe_pool": use_keyframe_pool,
        "recall_at_0.25": sum(bool(sample["hit_at_0.25"]) for sample in samples)
        / len(samples),
        "recall_at_0.50": sum(bool(sample["hit_at_0.50"]) for sample in samples)
        / len(samples),
        "mean_max_iou_3d": statistics.fmean(ious),
        "median_max_iou_3d": statistics.median(ious),
        "n_proposals_per_sample": {
            "min": min(proposal_counts),
            "median": statistics.median(proposal_counts),
            "max": max(proposal_counts),
        },
    }


def validate_bbox_9dof(raw: Any, field_name: str) -> list[float]:
    if not isinstance(raw, list) or len(raw) != 9:
        raise ValueError(f"{field_name} must be a 9-element list")
    try:
        bbox = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"{field_name} must contain only finite values")
    return bbox


def _parse_bool(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("--use-keyframe-pool must be true or false")


__all__ = [
    "aggregate_results",
    "eval_proposal_pool_recall",
    "evaluate_sample",
    "load_proposals",
    "load_sample_ids",
    "load_scene_pack",
    "main",
]


if __name__ == "__main__":
    main()
