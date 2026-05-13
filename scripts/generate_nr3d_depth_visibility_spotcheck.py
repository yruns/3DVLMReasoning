#!/usr/bin/env python3
"""Render spot-check frames from depth-aware NR3D visibility indices."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from benchmarks.embodiedscan_bbox_feasibility.render_marks import render_marked_keyframe
from benchmarks.embodiedscan_bbox_feasibility.visibility_index import (
    project_depth_visible_points_to_2d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/nr3d/scannet"))
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=["scene0474_00", "scene0030_00", "scene0153_00", "scene0435_00"],
    )
    parser.add_argument("--frames-per-scene", type=int, default=3)
    parser.add_argument("--max-marks", type=int, default=18)
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("docs/benchmark/nr3d/depth_visibility_spotcheck_20260513.html"),
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("docs/benchmark/nr3d/assets/depth_visibility_spotcheck_20260513"),
    )
    parser.add_argument(
        "--rebuild-report",
        type=Path,
        default=Path("tmp/nr3d_depth_visibility_rebuild_20260513.json"),
    )
    return parser.parse_args()


def _load_objects(scene_root: Path) -> list[dict[str, Any]]:
    p = scene_root / "conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz"
    with gzip.open(p, "rb") as f:
        return pickle.load(f)["objects"]


def _object_label(obj: dict[str, Any]) -> str:
    names = obj.get("class_name")
    if isinstance(names, list) and names:
        return str(names[0])
    if isinstance(names, str):
        return names
    return "object"


def _coerce_visibility(raw: dict[Any, Any]) -> dict[int, list[tuple[int, float]]]:
    return {
        int(frame_id): [(int(item[0]), float(item[1])) for item in entries]
        for frame_id, entries in raw.items()
    }


def _load_visibility(scene_root: Path) -> tuple[dict[int, list[tuple[int, float]]], dict[str, Any]]:
    p = scene_root / "conceptgraph/indices/visibility_index.pkl"
    with p.open("rb") as f:
        payload = pickle.load(f)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("use_depth") is not True:
        raise ValueError(f"{p} is not depth-aware")
    return _coerce_visibility(payload["view_to_objects"]), metadata


def _render_scene(
    *,
    data_root: Path,
    scene_id: str,
    frames_per_scene: int,
    max_marks: int,
    asset_dir: Path,
) -> list[dict[str, Any]]:
    scene_root = data_root / scene_id
    objects = _load_objects(scene_root)
    view_to_objects, metadata = _load_visibility(scene_root)
    visible_mark_ids = {
        obj_id
        for obj_id, obj in enumerate(objects)
        if not bool(int(obj.get("is_background") or 0))
    }
    raw_dir = scene_root / "raw"
    intrinsic = np.loadtxt(raw_dir / "intrinsic_color.txt").astype(float)[:3, :3]
    rgb_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9]-rgb.png"))
    pose_paths = sorted(raw_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].txt"))
    frame_ids = [
        frame_id
        for frame_id, _entries in sorted(
            (
                (frame_id, [entry for entry in entries if entry[0] in visible_mark_ids])
                for frame_id, entries in view_to_objects.items()
            ),
            key=lambda item: (-len(item[1]), item[0]),
        )[:frames_per_scene]
        if _entries
    ]
    rows: list[dict[str, Any]] = []
    for rank, frame_id in enumerate(frame_ids):
        rgb_path = rgb_paths[frame_id]
        pose_path = pose_paths[frame_id]
        depth_path = raw_dir / f"{rgb_path.name[:6]}-depth.png"
        raw_frame_id = rgb_path.name[:6]
        width, height = Image.open(rgb_path).size
        depth_map = np.asarray(Image.open(depth_path))
        world_to_cam = np.linalg.inv(np.loadtxt(pose_path).astype(float))
        marks = []
        visible_entries = [
            entry for entry in view_to_objects[frame_id] if entry[0] in visible_mark_ids
        ]
        for obj_id, score in visible_entries[:max_marks]:
            obj = objects[obj_id]
            rect = project_depth_visible_points_to_2d(
                np.asarray(obj["pcd_np"], dtype=float),
                intrinsic,
                world_to_cam,
                depth_map,
                image_size=(width, height),
            )
            if rect is None:
                continue
            marks.append(
                {
                    "proposal_id": obj_id,
                    "label": _object_label(obj),
                    "bbox_2d": rect,
                    "score": score,
                }
            )
        out_name = f"{scene_id}_view_{frame_id:04d}_raw_{raw_frame_id}_depth_visible.jpg"
        out_path = asset_dir / out_name
        render_marked_keyframe(rgb_path=rgb_path, out_path=out_path, marks=marks)
        rows.append(
            {
                "scene_id": scene_id,
                "frame_id": frame_id,
                "raw_frame_id": raw_frame_id,
                "image": out_path,
                "visible_count": len(visible_entries),
                "drawn_count": len(marks),
                "metadata": metadata,
                "marks": marks,
                "rank": rank,
            }
        )
    return rows


def _load_rebuild_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_html(rows: list[dict[str, Any]], out_html: Path, report: dict[str, Any]) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>NR3D Depth Visibility Spotcheck</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f5;color:#1f2933}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}",
        ".card{background:white;border:1px solid #d8d8d0;border-radius:6px;padding:12px}",
        "img{width:100%;height:auto;border:1px solid #ddd}",
        "code{background:#eee;padding:1px 4px;border-radius:3px}",
        ".meta{font-size:13px;color:#4b5563;line-height:1.45}",
        "</style></head><body>",
        "<h1>NR3D Depth Visibility Spotcheck</h1>",
        "<p>Frames below are rendered from raw RGB plus the rebuilt "
        "<code>metadata.use_depth=true</code> visibility indices. Only objects "
        "present in the current depth-aware <code>view_to_objects</code> entry "
        "for that frame are drawn; structural background objects marked by "
        "<code>is_background=1</code> are suppressed. Each colored box is a "
        "candidate proposal's visible 2D image region and is labeled directly "
        "as <code>#proposal_id category</code>.</p>",
    ]
    if report:
        parts.append(
            "<p class='meta'>Rebuild summary: "
            f"{html.escape(str(report.get('num_scenes')))} scenes, "
            f"old mappings={html.escape(str(report.get('total_old_mappings')))}, "
            f"new mappings={html.escape(str(report.get('total_new_mappings')))}."
            "</p>"
        )
    parts.append("<div class='grid'>")
    for row in rows:
        rel_image = row["image"].relative_to(out_html.parent)
        labels = ", ".join(
            f"{mark['proposal_id']}:{mark['label']}" for mark in row["marks"][:8]
        )
        parts.extend(
            [
                "<div class='card'>",
                f"<h2>{html.escape(row['scene_id'])} / view {row['frame_id']} "
                f"/ raw {html.escape(str(row['raw_frame_id']))}</h2>",
                f"<img src='{html.escape(rel_image.as_posix())}' loading='lazy'>",
                "<p class='meta'>"
                f"visible objects in index: {row['visible_count']}<br>"
                f"drawn marks: {row['drawn_count']}<br>"
                f"use_depth: {html.escape(str(row['metadata'].get('use_depth')))}<br>"
                f"visibility_kind: {html.escape(str(row['metadata'].get('visibility_kind')))}<br>"
                f"top labels: {html.escape(labels)}"
                "</p>",
                "</div>",
            ]
        )
    parts.extend(["</div></body></html>"])
    out_html.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scene_id in args.scenes:
        rows.extend(
            _render_scene(
                data_root=args.data_root,
                scene_id=scene_id,
                frames_per_scene=args.frames_per_scene,
                max_marks=args.max_marks,
                asset_dir=args.asset_dir,
            )
        )
    _write_html(rows, args.out_html, _load_rebuild_report(args.rebuild_report))
    print(f"wrote {args.out_html}")
    print(f"assets {args.asset_dir}")
    print(f"frames {len(rows)}")


if __name__ == "__main__":
    main()
