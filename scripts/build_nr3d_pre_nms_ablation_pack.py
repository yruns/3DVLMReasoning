#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a matched NR3D pack whose keyframes are the pre-NMS top-k "
            "candidates recorded by a frame-NMS pack."
        )
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--source-pack", required=True)
    parser.add_argument("--dest-pack", required=True)
    parser.add_argument("--max-keyframes", type=int, default=3)
    parser.add_argument(
        "--asset-mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="How to populate shared proposals/visibility/annotated assets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing dest-pack shared assets and sample JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_keyframes <= 0:
        raise ValueError("--max-keyframes must be positive")
    if args.source_pack == args.dest_pack:
        raise ValueError("--source-pack and --dest-pack must differ")

    sample_ids = load_sample_ids(args.sample_ids)
    stats = build_pack(
        data_root=args.data_root,
        sample_ids=sample_ids,
        source_pack=args.source_pack,
        dest_pack=args.dest_pack,
        max_keyframes=args.max_keyframes,
        asset_mode=args.asset_mode,
        force=args.force,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


def load_sample_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"sample ids file must contain a JSON list: {path}")
    sample_ids: list[str] = []
    for item in payload:
        if isinstance(item, str):
            sample_id = item
        elif isinstance(item, dict) and isinstance(item.get("sample_id"), str):
            sample_id = item["sample_id"]
        else:
            raise ValueError(f"Invalid sample id entry in {path}: {item!r}")
        sample_ids.append(sample_id)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"sample ids file contains duplicates: {path}")
    return sample_ids


def build_pack(
    *,
    data_root: Path,
    sample_ids: list[str],
    source_pack: str,
    dest_pack: str,
    max_keyframes: int,
    asset_mode: str,
    force: bool,
) -> dict[str, Any]:
    scenes: set[str] = set()
    changed_topk = 0
    suppressed_total = 0
    relaxed_backfill_total = 0
    fallback_count = 0
    for sample_id in sample_ids:
        scene_id = scene_id_from_sample_id(sample_id)
        scenes.add(scene_id)
        src_pack_dir = data_root / scene_id / source_pack
        dst_pack_dir = data_root / scene_id / dest_pack
        ensure_shared_assets(
            src_pack_dir=src_pack_dir,
            dst_pack_dir=dst_pack_dir,
            asset_mode=asset_mode,
            force=force,
        )

        src_sample = src_pack_dir / "samples" / f"{safe_sample_id(sample_id)}.json"
        if not src_sample.exists():
            raise FileNotFoundError(src_sample)
        payload = json.loads(src_sample.read_text(encoding="utf-8"))
        if payload.get("sample_id") != sample_id:
            raise ValueError(
                f"{src_sample} has sample_id={payload.get('sample_id')!r}, "
                f"expected {sample_id!r}"
            )
        metadata = payload.get("keyframe_selection_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{src_sample} missing keyframe_selection_metadata")
        frame_nms = metadata.get("frame_nms")
        if not isinstance(frame_nms, dict):
            raise ValueError(f"{src_sample} missing keyframe_selection_metadata.frame_nms")
        pre_nms = [int(v) for v in frame_nms.get("pre_nms_keyframe_indices", [])]
        if len(pre_nms) < max_keyframes:
            raise ValueError(
                f"{src_sample} has only {len(pre_nms)} pre-NMS candidates; "
                f"need {max_keyframes}"
            )
        selected_pre_nms = pre_nms[:max_keyframes]
        original_keyframes = [
            int(kf["frame_id"]) for kf in payload.get("keyframes", [])
        ]
        if selected_pre_nms != original_keyframes[:max_keyframes]:
            changed_topk += 1
        suppressed_total += len(frame_nms.get("suppressed", []))
        relaxed_backfill_total += len(frame_nms.get("relaxed_backfill", []))
        if payload.get("keyframe_selection_used_fallback"):
            fallback_count += 1

        dst_payload = copy.deepcopy(payload)
        dst_payload["scene_artifacts_dir"] = str(dst_pack_dir)
        dst_payload["keyframes"] = [
            {
                "keyframe_idx": idx,
                "image_path": str(dst_pack_dir / "annotated" / f"frame_{view_id}.png"),
                "frame_id": view_id,
            }
            for idx, view_id in enumerate(selected_pre_nms)
        ]
        dst_payload.setdefault("keyframe_selection_metadata", {})[
            "frame_nms_ablation"
        ] = {
            "mode": "pre_nms_topk",
            "max_keyframes": max_keyframes,
            "source_pack": source_pack,
            "source_sample": str(src_sample),
            "selected_keyframe_indices": selected_pre_nms,
            "original_nms_selected": [
                int(v) for v in frame_nms.get("selected", [])
            ],
        }

        dst_samples_dir = dst_pack_dir / "samples"
        dst_samples_dir.mkdir(parents=True, exist_ok=True)
        dst_sample = dst_samples_dir / src_sample.name
        if dst_sample.exists() and not force:
            raise FileExistsError(f"{dst_sample} already exists; pass --force")
        dst_sample.write_text(
            json.dumps(dst_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "sample_count": len(sample_ids),
        "scene_count": len(scenes),
        "source_pack": source_pack,
        "dest_pack": dest_pack,
        "max_keyframes": max_keyframes,
        "asset_mode": asset_mode,
        "changed_topk": changed_topk,
        "suppressed_total": suppressed_total,
        "relaxed_backfill_total": relaxed_backfill_total,
        "fallback_count": fallback_count,
    }


def ensure_shared_assets(
    *,
    src_pack_dir: Path,
    dst_pack_dir: Path,
    asset_mode: str,
    force: bool,
) -> None:
    if not src_pack_dir.exists():
        raise FileNotFoundError(src_pack_dir)
    dst_pack_dir.mkdir(parents=True, exist_ok=True)
    for name in ("proposals.jsonl", "visibility.json", "annotated"):
        src = src_pack_dir / name
        dst = dst_pack_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            if not force:
                continue
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if asset_mode == "copy":
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def scene_id_from_sample_id(sample_id: str) -> str:
    parts = sample_id.split("::")
    if len(parts) != 3:
        raise ValueError(f"Invalid NR3D sample_id={sample_id!r}")
    scan_id = parts[0]
    scene_id = scan_id.split("/")[-1]
    if not scene_id:
        raise ValueError(f"Invalid NR3D sample_id={sample_id!r}")
    return scene_id


def safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


if __name__ == "__main__":
    main()
