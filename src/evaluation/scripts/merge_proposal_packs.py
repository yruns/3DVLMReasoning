"""Merge EmbodiedScan detector proposal packs into a union proposal pool."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.scripts.eval_proposal_pool_recall import (
    load_json_documents,
    load_sample_ids,
    parse_sample_id,
    validate_bbox_9dof,
)


@dataclass(frozen=True)
class SourcePack:
    pack_name: str
    source_pool: str


@dataclass(frozen=True)
class SourceScenePack:
    source: SourcePack
    pack_dir: Path
    proposals: list[dict[str, Any]]
    proposal_ids: set[int]
    visibility: dict[int, list[int]]
    axis_align_matrix: Any


@dataclass(frozen=True)
class MergeCandidate:
    source_pool: str
    original_id: int
    score: float
    proposal: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge detector proposal packs into a single union pack."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--source-packs",
        required=True,
        action="append",
        nargs="+",
        help="Source pack names. Can be passed once with many names or repeated.",
    )
    parser.add_argument("--output-pack", required=True)
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--max-proposals-per-scene", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_packs = [pack for group in args.source_packs for pack in group]
    summary = merge_proposal_packs(
        data_root=args.data_root,
        source_packs=source_packs,
        output_pack=args.output_pack,
        sample_ids_path=args.sample_ids,
        max_proposals_per_scene=args.max_proposals_per_scene,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def merge_proposal_packs(
    *,
    data_root: Path,
    source_packs: list[str],
    output_pack: str,
    sample_ids_path: Path,
    max_proposals_per_scene: int = 2000,
) -> dict[str, Any]:
    if max_proposals_per_scene <= 0:
        raise ValueError("max_proposals_per_scene must be positive")
    if not output_pack:
        raise ValueError("output_pack must be non-empty")

    sources = normalize_source_packs(source_packs)
    sample_ids = load_sample_ids(sample_ids_path)
    targets_by_scene = group_targets_by_scene(sample_ids)
    output_source = source_pool_name(output_pack)

    scene_summaries: list[dict[str, Any]] = []
    sample_count = 0
    proposal_counts: list[int] = []
    for scene_id, target_ids in targets_by_scene.items():
        scene_summary = merge_scene_pack(
            data_root=data_root,
            scene_id=scene_id,
            target_ids=target_ids,
            sources=sources,
            output_pack=output_pack,
            output_source=output_source,
            max_proposals_per_scene=max_proposals_per_scene,
        )
        scene_summaries.append(scene_summary)
        sample_count += int(scene_summary["n_samples"])
        proposal_counts.append(int(scene_summary["n_proposals"]))

    summary = {
        "output_pack": output_pack,
        "source_packs": [source.pack_name for source in sources],
        "n_scenes": len(scene_summaries),
        "n_samples": sample_count,
        "n_proposals_per_scene": {
            "min": min(proposal_counts),
            "median": statistics.median(proposal_counts),
            "max": max(proposal_counts),
        },
        "scenes": scene_summaries,
    }
    return summary


def normalize_source_packs(source_packs: list[str]) -> list[SourcePack]:
    if len(source_packs) < 1:
        raise ValueError("at least one source pack is required")
    normalized: list[SourcePack] = []
    seen_packs: set[str] = set()
    seen_sources: set[str] = set()
    for pack_name in source_packs:
        clean_pack = str(pack_name).strip()
        if not clean_pack:
            raise ValueError("source pack names must be non-empty")
        source = source_pool_name(clean_pack)
        if clean_pack in seen_packs:
            raise ValueError(f"duplicate source pack: {clean_pack}")
        if source in seen_sources:
            raise ValueError(f"duplicate source pool name: {source}")
        seen_packs.add(clean_pack)
        seen_sources.add(source)
        normalized.append(SourcePack(pack_name=clean_pack, source_pool=source))
    return normalized


def source_pool_name(pack_name: str) -> str:
    source_pool = pack_name[5:] if pack_name.startswith("pack_") else pack_name
    source_pool = source_pool.strip()
    if not source_pool:
        raise ValueError(f"cannot derive source pool name from {pack_name!r}")
    return source_pool


def group_targets_by_scene(sample_ids: list[str]) -> dict[str, list[int]]:
    targets_by_scene: dict[str, list[int]] = {}
    for sample_id in sample_ids:
        scene_id, target_id = parse_sample_id(sample_id)
        targets_by_scene.setdefault(scene_id, []).append(target_id)
    return targets_by_scene


def merge_scene_pack(
    *,
    data_root: Path,
    scene_id: str,
    target_ids: list[int],
    sources: list[SourcePack],
    output_pack: str,
    output_source: str,
    max_proposals_per_scene: int,
) -> dict[str, Any]:
    source_scene_packs = [
        load_source_scene_pack(data_root, scene_id, source) for source in sources
    ]
    merged_proposals, id_map = merge_scene_proposals(
        source_scene_packs,
        max_proposals_per_scene=max_proposals_per_scene,
    )
    merged_visibility = merge_scene_visibility(source_scene_packs, id_map)
    axis_align_matrix = merged_axis_align_matrix(source_scene_packs)

    output_dir = data_root / scene_id / output_pack
    write_merged_scene_files(
        output_dir=output_dir,
        scene_id=scene_id,
        output_source=output_source,
        source_scene_packs=source_scene_packs,
        merged_proposals=merged_proposals,
        merged_visibility=merged_visibility,
        axis_align_matrix=axis_align_matrix,
    )
    for target_id in target_ids:
        write_merged_sample(
            output_dir=output_dir,
            source_scene_packs=source_scene_packs,
            scene_id=scene_id,
            target_id=target_id,
            output_source=output_source,
            merged_proposals=merged_proposals,
        )
    return {
        "scene_id": scene_id,
        "n_samples": len(target_ids),
        "n_proposals": len(merged_proposals),
        "n_visibility_frames": len(merged_visibility),
        "output_dir": str(output_dir),
    }


def load_source_scene_pack(
    data_root: Path, scene_id: str, source: SourcePack
) -> SourceScenePack:
    pack_dir = data_root / scene_id / source.pack_name
    if not pack_dir.exists():
        raise FileNotFoundError(f"source pack directory not found: {pack_dir}")
    if not pack_dir.is_dir():
        raise NotADirectoryError(f"source pack path is not a directory: {pack_dir}")

    raw_pack = load_proposals_payload(pack_dir / "proposals.jsonl")
    raw_proposals = raw_pack.get("proposals")
    if not isinstance(raw_proposals, list):
        raise ValueError(
            f"{pack_dir}/proposals.jsonl top-level proposals must be a list"
        )
    proposals = [
        validate_source_proposal(
            raw,
            source=source,
            path=pack_dir / "proposals.jsonl",
            index=index,
        )
        for index, raw in enumerate(raw_proposals)
    ]
    proposal_ids = {int(proposal["id"]) for proposal in proposals}
    if len(proposal_ids) != len(proposals):
        raise ValueError(f"{pack_dir}/proposals.jsonl contains duplicate proposal ids")
    visibility = load_source_visibility(
        pack_dir / "visibility.json",
        proposal_ids=proposal_ids,
        source=source,
    )
    return SourceScenePack(
        source=source,
        pack_dir=pack_dir,
        proposals=proposals,
        proposal_ids=proposal_ids,
        visibility=visibility,
        axis_align_matrix=raw_pack.get("axis_align_matrix"),
    )


def load_proposals_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"proposals file not found: {path}")
    docs = load_json_documents(path)
    if len(docs) != 1 or not isinstance(docs[0], dict) or "proposals" not in docs[0]:
        raise ValueError(
            f"{path} must be one JSON object with a top-level proposals list"
        )
    return docs[0]


def validate_source_proposal(
    raw: Any,
    *,
    source: SourcePack,
    path: Path,
    index: int,
) -> dict[str, Any]:
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
    metadata = raw.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} proposal[{index}].metadata must be an object")
    clean_metadata = dict(metadata)
    clean_metadata["source_pool"] = source.source_pool
    clean_metadata["original_pool_id"] = proposal_id
    return {
        "id": proposal_id,
        "bbox_3d": validate_bbox_9dof(
            raw["bbox_3d"], f"{path} proposal[{index}].bbox_3d"
        ),
        "score": score,
        "label": label,
        "source": source.source_pool,
        "metadata": clean_metadata,
    }


def load_source_visibility(
    path: Path,
    *,
    proposal_ids: set[int],
    source: SourcePack,
) -> dict[int, list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"visibility file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"visibility JSON must be an object: {path}")
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
            if proposal_id not in proposal_ids:
                raise ValueError(
                    f"{path} frame {frame_id} references unknown "
                    f"{source.pack_name} proposal id {proposal_id}"
                )
            ids.append(proposal_id)
        visibility[frame_id] = ids
    return visibility


def merge_scene_proposals(
    source_scene_packs: list[SourceScenePack],
    *,
    max_proposals_per_scene: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], int]]:
    candidates: list[MergeCandidate] = []
    for source_pack in source_scene_packs:
        for proposal in source_pack.proposals:
            candidates.append(
                MergeCandidate(
                    source_pool=source_pack.source.source_pool,
                    original_id=int(proposal["id"]),
                    score=float(proposal["score"]),
                    proposal=proposal,
                )
            )

    candidates.sort(key=lambda item: (-item.score, item.source_pool, item.original_id))
    kept = candidates[:max_proposals_per_scene]
    merged_proposals: list[dict[str, Any]] = []
    id_map: dict[tuple[str, int], int] = {}
    for new_id, candidate in enumerate(kept):
        output_proposal = dict(candidate.proposal)
        output_proposal["id"] = new_id
        output_proposal["metadata"] = dict(candidate.proposal["metadata"])
        merged_proposals.append(output_proposal)
        id_map[(candidate.source_pool, candidate.original_id)] = new_id
    if not merged_proposals:
        raise ValueError("merged proposal pool is empty")
    return merged_proposals, id_map


def merge_scene_visibility(
    source_scene_packs: list[SourceScenePack],
    id_map: dict[tuple[str, int], int],
) -> dict[int, list[int]]:
    frame_ids = sorted(
        {
            frame_id
            for source_scene_pack in source_scene_packs
            for frame_id in source_scene_pack.visibility
        }
    )
    merged_visibility: dict[int, list[int]] = {}
    for frame_id in frame_ids:
        merged_ids: set[int] = set()
        for source_scene_pack in source_scene_packs:
            visible_ids = source_scene_pack.visibility.get(frame_id, [])
            for original_id in visible_ids:
                mapped = id_map.get((source_scene_pack.source.source_pool, original_id))
                if mapped is not None:
                    merged_ids.add(mapped)
        merged_visibility[frame_id] = sorted(merged_ids)
    return merged_visibility


def merged_axis_align_matrix(source_scene_packs: list[SourceScenePack]) -> Any:
    axis_align_matrix = source_scene_packs[0].axis_align_matrix
    for source_scene_pack in source_scene_packs[1:]:
        if source_scene_pack.axis_align_matrix != axis_align_matrix:
            raise ValueError(
                "source packs disagree on axis_align_matrix for "
                f"{source_scene_packs[0].pack_dir.parent.name}"
            )
    return axis_align_matrix


def write_merged_scene_files(
    *,
    output_dir: Path,
    scene_id: str,
    output_source: str,
    source_scene_packs: list[SourceScenePack],
    merged_proposals: list[dict[str, Any]],
    merged_visibility: dict[int, list[int]],
    axis_align_matrix: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    if annotated_dir.exists():
        shutil.rmtree(annotated_dir)
    annotated_dir.mkdir()

    (output_dir / "proposals.jsonl").write_text(
        json.dumps(
            {
                "source": output_source,
                "scene_id": scene_id,
                "source_packs": [
                    source_scene_pack.source.pack_name
                    for source_scene_pack in source_scene_packs
                ],
                "axis_align_matrix": axis_align_matrix,
                "proposals": merged_proposals,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "visibility.json").write_text(
        json.dumps(
            {str(frame_id): ids for frame_id, ids in merged_visibility.items()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "Annotated frames are intentionally not rendered for merged recall-only packs.\n",
        encoding="utf-8",
    )


def write_merged_sample(
    *,
    output_dir: Path,
    source_scene_packs: list[SourceScenePack],
    scene_id: str,
    target_id: int,
    output_source: str,
    merged_proposals: list[dict[str, Any]],
) -> None:
    sample_id = f"{scene_id}::{target_id}"
    source_payloads = [
        load_source_sample(source_scene_pack.pack_dir, target_id, sample_id)
        for source_scene_pack in source_scene_packs
    ]
    validate_matching_gt(source_payloads, sample_id)
    payload = dict(source_payloads[0])
    payload["scene_artifacts_dir"] = str(output_dir)
    payload["source"] = output_source
    payload["proposals"] = merged_proposals
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / f"{target_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_source_sample(
    pack_dir: Path, target_id: int, sample_id: str
) -> dict[str, Any]:
    sample_path = pack_dir / "samples" / f"{target_id}.json"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"source sample not found for {sample_id}: {sample_path}"
        )
    raw = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"source sample must be an object: {sample_path}")
    found_sample_id = raw.get("sample_id")
    if found_sample_id is not None and str(found_sample_id) != sample_id:
        raise ValueError(
            f"{sample_path} sample_id mismatch: expected {sample_id}, got {found_sample_id}"
        )
    if "keyframes" not in raw:
        raise ValueError(f"{sample_path} missing keyframes")
    validate_bbox_9dof(
        raw.get("gt_bbox_3d_9dof"),
        f"{sample_path} gt_bbox_3d_9dof",
    )
    return raw


def validate_matching_gt(payloads: list[dict[str, Any]], sample_id: str) -> None:
    gt_bbox = payloads[0]["gt_bbox_3d_9dof"]
    for payload in payloads[1:]:
        if payload.get("gt_bbox_3d_9dof") != gt_bbox:
            raise ValueError(
                f"source packs disagree on gt_bbox_3d_9dof for {sample_id}"
            )


__all__ = [
    "merge_proposal_packs",
    "merge_scene_pack",
    "normalize_source_packs",
]


if __name__ == "__main__":
    main()
