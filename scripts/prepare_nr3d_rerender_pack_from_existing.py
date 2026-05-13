"""Rerender an NR3D pack while preserving existing sample keyframe choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.scripts.prepare_pack_v1_inputs_nr3d import (
    load_sample_requests,
    prepare_scene_artifacts,
    sample_artifact_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--source-pack", required=True)
    parser.add_argument("--out-pack", required=True)
    parser.add_argument(
        "--ensure-lightweight-cache",
        action="store_true",
        default=False,
        help="Build/use lightweight ConceptGraph object caches when needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests = load_sample_requests(args.sample_ids)
    scenes = sorted({request.scene_id for request in requests})

    scene_dirs: dict[str, Path] = {}
    for scene_id in scenes:
        artifacts = prepare_scene_artifacts(
            scene_id=scene_id,
            data_root=args.data_root,
            pack_name=args.out_pack,
            ensure_lightweight_cache=args.ensure_lightweight_cache,
        )
        scene_dirs[scene_id] = artifacts.scene_dir

    written = []
    for request in requests:
        source_path = sample_artifact_path(
            args.data_root,
            request,
            pack_name=args.source_pack,
        )
        if not source_path.exists():
            raise FileNotFoundError(f"missing source sample: {source_path}")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("sample_id") != request.sample_id:
            raise ValueError(
                f"{source_path} sample_id={payload.get('sample_id')!r}; "
                f"expected {request.sample_id!r}"
            )

        scene_dir = scene_dirs[request.scene_id]
        payload["scene_artifacts_dir"] = str(scene_dir)
        for keyframe in payload.get("keyframes", []):
            frame_id = int(keyframe["frame_id"])
            image_path = scene_dir / "annotated" / f"frame_{frame_id}.png"
            if not image_path.exists():
                raise FileNotFoundError(
                    f"missing rerendered keyframe for {request.sample_id}: {image_path}"
                )
            keyframe["image_path"] = str(image_path)

        out_path = sample_artifact_path(
            args.data_root,
            request,
            pack_name=args.out_pack,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(str(out_path))

    print(
        json.dumps(
            {
                "sample_ids": str(args.sample_ids),
                "data_root": str(args.data_root),
                "source_pack": args.source_pack,
                "out_pack": args.out_pack,
                "scenes": len(scenes),
                "samples": len(written),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
