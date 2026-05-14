#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.scripts.prepare_pack_v1_inputs_nr3d import (
    SampleRequest,
    load_sample_requests,
    raw_frame_id_for_view,
    sample_artifact_path,
    scene_intrinsic,
)
from query_scene.frustum import frustum_overlap_l1, frustum_overlap_l2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pack-name", required=True)
    parser.add_argument(
        "--out-html",
        type=Path,
        default=Path("docs/benchmark/nr3d/frame_nms_spotcheck.html"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Defaults to <out-html parent>/assets/<out-html stem>.",
    )
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument(
        "--frustum-method",
        default="l1",
        choices=["l1", "l2"],
        help="Method used for the displayed overlap matrix.",
    )
    parser.add_argument(
        "--asset-max-width",
        type=int,
        default=900,
        help="Downscale copied frame assets to this maximum width.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=84,
        help="JPEG quality for generated HTML assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    assets_dir = args.assets_dir or (
        args.out_html.parent / "assets" / args.out_html.stem
    )
    rows = collect_rows(
        sample_ids_path=args.sample_ids,
        data_root=args.data_root,
        pack_name=args.pack_name,
        max_cases=args.max_cases,
        assets_dir=assets_dir,
        html_dir=args.out_html.parent,
        frustum_method=args.frustum_method,
        asset_max_width=args.asset_max_width,
        jpeg_quality=args.jpeg_quality,
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(
        build_html(
            rows=rows,
            pack_name=args.pack_name,
            frustum_method=args.frustum_method,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out_html}")


def collect_rows(
    *,
    sample_ids_path: Path,
    data_root: Path,
    pack_name: str,
    max_cases: int,
    assets_dir: Path,
    html_dir: Path,
    frustum_method: str,
    asset_max_width: int,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[tuple[int, int, str, str], SampleRequest, dict[str, Any]]] = []
    for request in load_sample_requests(sample_ids_path):
        artifact_path = sample_artifact_path(data_root, request, pack_name=pack_name)
        if not artifact_path.exists():
            continue
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        frame_nms = (
            payload.get("keyframe_selection_metadata", {}).get("frame_nms", {})
        )
        if not frame_nms:
            continue
        sort_key = (
            -len(frame_nms.get("suppressed", [])),
            -len(frame_nms.get("relaxed_backfill", [])),
            request.scene_id,
            request.sample_id,
        )
        candidates.append((sort_key, request, payload))

    candidates.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    for _sort_key, request, payload in candidates[:max_cases]:
        rows.append(
            build_row(
                request=request,
                payload=payload,
                data_root=data_root,
                assets_dir=assets_dir,
                html_dir=html_dir,
                frustum_method=frustum_method,
                asset_max_width=asset_max_width,
                jpeg_quality=jpeg_quality,
            )
        )
    return rows


def build_row(
    *,
    request: SampleRequest,
    payload: dict[str, Any],
    data_root: Path,
    assets_dir: Path,
    html_dir: Path,
    frustum_method: str,
    asset_max_width: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    scene_id = request.scene_id
    scene_root = data_root / scene_id
    metadata = payload.get("keyframe_selection_metadata", {})
    frame_nms = metadata.get("frame_nms", {})
    pre_views = [int(v) for v in frame_nms.get("pre_nms_keyframe_indices", [])]
    selected = [int(v) for v in frame_nms.get("selected", [])]
    strict_selected = [int(v) for v in frame_nms.get("strict_selected", [])]
    suppressed = frame_nms.get("suppressed", [])
    relaxed_backfill = frame_nms.get("relaxed_backfill", [])

    view_ids = _unique(pre_views + selected)
    image_entries = []
    status_by_view = {}
    for view_id in selected:
        status_by_view[view_id] = "kept"
    for item in suppressed:
        status_by_view[int(item["view_id"])] = (
            f"suppressed by {int(item['suppressed_by'])}, "
            f"overlap={float(item['overlap']):.3f}"
        )
    for item in relaxed_backfill:
        status_by_view[int(item["view_id"])] = (
            f"relaxed backfill, max_overlap={float(item['max_overlap']):.3f}"
        )

    for view_id in view_ids:
        image_entries.append(
            {
                "view_id": view_id,
                "raw_frame_id": raw_frame_id_for_view(scene_root, view_id),
                "status": status_by_view.get(view_id, "candidate"),
                "src": copy_frame_asset(
                    scene_artifacts_dir=Path(payload["scene_artifacts_dir"]),
                    scene_root=scene_root,
                    view_id=view_id,
                    sample_id=request.sample_id,
                    assets_dir=assets_dir,
                    html_dir=html_dir,
                    asset_max_width=asset_max_width,
                    jpeg_quality=jpeg_quality,
                ),
            }
        )

    overlap_matrix = compute_overlap_matrix(
        scene_root=scene_root,
        view_ids=view_ids,
        frustum_method=frustum_method,
    )

    return {
        "sample_id": request.sample_id,
        "scene_id": scene_id,
        "target_id": request.target_id,
        "category": payload.get("category"),
        "query": payload.get("query"),
        "threshold": frame_nms.get("overlap_threshold"),
        "pre_views": pre_views,
        "selected": selected,
        "strict_selected": strict_selected,
        "suppressed": suppressed,
        "relaxed_backfill": relaxed_backfill,
        "images": image_entries,
        "overlap_matrix": overlap_matrix,
    }


def copy_frame_asset(
    *,
    scene_artifacts_dir: Path,
    scene_root: Path,
    view_id: int,
    sample_id: str,
    assets_dir: Path,
    html_dir: Path,
    asset_max_width: int,
    jpeg_quality: int,
) -> str:
    annotated = scene_artifacts_dir / "annotated" / f"frame_{view_id}.png"
    if annotated.exists():
        source = annotated
    else:
        raw_frame_id = raw_frame_id_for_view(scene_root, view_id)
        source = scene_root / "raw" / f"{raw_frame_id:06d}-rgb.png"
        if not source.exists():
            raise FileNotFoundError(source)

    assets_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{sample_id}:{view_id}:{source}".encode()).hexdigest()[:8]
    out = assets_dir / f"{_safe_id(sample_id)}_view_{view_id:04d}_{digest}.jpg"
    image = Image.open(source).convert("RGB")
    if asset_max_width <= 0:
        raise ValueError("--asset-max-width must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if image.width > asset_max_width:
        new_height = max(1, round(image.height * asset_max_width / image.width))
        image = image.resize((asset_max_width, new_height), Image.Resampling.LANCZOS)
    image.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
    return out.relative_to(html_dir).as_posix()


def compute_overlap_matrix(
    *,
    scene_root: Path,
    view_ids: list[int],
    frustum_method: str,
) -> dict[int, dict[int, float]]:
    if frustum_method not in {"l1", "l2"}:
        raise ValueError(f"unsupported frustum method: {frustum_method}")
    if not view_ids:
        return {}
    k = scene_intrinsic(scene_root)
    first_rgb = scene_root / "raw" / f"{raw_frame_id_for_view(scene_root, view_ids[0]):06d}-rgb.png"
    image_size = Image.open(first_rgb).size
    poses = {
        view_id: load_cam_to_world_pose(scene_root, view_id)
        for view_id in view_ids
    }
    depths = {}
    if frustum_method == "l2":
        depths = {
            view_id: load_depth(scene_root, view_id)
            for view_id in view_ids
        }

    matrix: dict[int, dict[int, float]] = {}
    for view_a in view_ids:
        matrix[view_a] = {}
        for view_b in view_ids:
            if view_a == view_b:
                matrix[view_a][view_b] = 1.0
                continue
            if frustum_method == "l1":
                overlap = max(
                    frustum_overlap_l1(poses[view_a], poses[view_b], k, image_size),
                    frustum_overlap_l1(poses[view_b], poses[view_a], k, image_size),
                )
            else:
                overlap = max(
                    frustum_overlap_l2(
                        depths[view_a],
                        poses[view_a],
                        poses[view_b],
                        k,
                        image_size,
                    ),
                    frustum_overlap_l2(
                        depths[view_b],
                        poses[view_b],
                        poses[view_a],
                        k,
                        image_size,
                    ),
                )
            matrix[view_a][view_b] = float(overlap)
    return matrix


def load_cam_to_world_pose(scene_root: Path, view_id: int) -> np.ndarray:
    raw_frame_id = raw_frame_id_for_view(scene_root, view_id)
    pose_path = scene_root / "raw" / f"{raw_frame_id:06d}.txt"
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    pose = np.asarray(np.loadtxt(pose_path), dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"pose must be 4x4: {pose_path}")
    return pose


def load_depth(scene_root: Path, view_id: int) -> np.ndarray:
    raw_frame_id = raw_frame_id_for_view(scene_root, view_id)
    depth_path = scene_root / "raw" / f"{raw_frame_id:06d}-depth.png"
    if not depth_path.exists():
        raise FileNotFoundError(depth_path)
    return np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0


def build_html(
    *,
    rows: list[dict[str, Any]],
    pack_name: str,
    frustum_method: str,
) -> str:
    body = "\n".join(render_row(row) for row in rows)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NR3D Frame NMS Spotcheck</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
h1 {{ margin-bottom: 4px; }}
.meta {{ color: #475569; margin-bottom: 24px; }}
.case {{ border: 1px solid #cbd5e1; background: white; border-radius: 6px; padding: 16px; margin-bottom: 24px; }}
.case h2 {{ margin: 0 0 8px; font-size: 18px; }}
.query {{ font-size: 15px; margin: 8px 0 12px; }}
.summary {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 14px; }}
.pill {{ border: 1px solid #cbd5e1; border-radius: 999px; padding: 4px 9px; background: #f8fafc; font-size: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
.frame {{ border: 1px solid #d1d5db; border-radius: 6px; overflow: hidden; background: #ffffff; }}
.frame img {{ width: 100%; display: block; }}
.caption {{ padding: 8px; font-size: 12px; line-height: 1.4; }}
.kept {{ border-color: #15803d; box-shadow: 0 0 0 2px rgba(21, 128, 61, 0.18); }}
.suppressed {{ border-color: #b91c1c; box-shadow: 0 0 0 2px rgba(185, 28, 28, 0.18); }}
.backfill {{ border-color: #b45309; box-shadow: 0 0 0 2px rgba(180, 83, 9, 0.18); }}
table {{ border-collapse: collapse; margin-top: 14px; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 5px 7px; text-align: right; }}
th {{ background: #f1f5f9; }}
.high {{ background: #fee2e2; }}
.mid {{ background: #fef3c7; }}
.low {{ background: #dcfce7; }}
code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>NR3D Frame NMS Spotcheck</h1>
<div class="meta">pack=<code>{esc(pack_name)}</code>, displayed overlap method=<code>{esc(frustum_method)}</code></div>
{body if body else "<p>No frame NMS metadata found.</p>"}
</body>
</html>
"""


def render_row(row: dict[str, Any]) -> str:
    images = "\n".join(render_frame(entry) for entry in row["images"])
    matrix = render_matrix(row["overlap_matrix"])
    return f"""
<section class="case">
  <h2>{esc(row["scene_id"])} target #{esc(row["target_id"])} {esc(row["category"])}</h2>
  <div class="query">{esc(row["query"])}</div>
  <div class="summary">
    <span class="pill">threshold={esc(row["threshold"])}</span>
    <span class="pill">pre={esc(row["pre_views"])}</span>
    <span class="pill">selected={esc(row["selected"])}</span>
    <span class="pill">strict={esc(row["strict_selected"])}</span>
    <span class="pill">suppressed={esc(len(row["suppressed"]))}</span>
    <span class="pill">relaxed_backfill={esc(len(row["relaxed_backfill"]))}</span>
  </div>
  <div class="grid">{images}</div>
  {matrix}
</section>
"""


def render_frame(entry: dict[str, Any]) -> str:
    status = str(entry["status"])
    if status.startswith("suppressed"):
        cls = "frame suppressed"
    elif status.startswith("relaxed"):
        cls = "frame backfill"
    elif status == "kept":
        cls = "frame kept"
    else:
        cls = "frame"
    return f"""
<div class="{cls}">
  <img src="{esc(entry["src"])}" loading="lazy">
  <div class="caption"><b>view {esc(entry["view_id"])}</b> / raw {esc(entry["raw_frame_id"])}<br>{esc(status)}</div>
</div>
"""


def render_matrix(matrix: dict[int, dict[int, float]]) -> str:
    if not matrix:
        return ""
    view_ids = list(matrix.keys())
    head = "".join(f"<th>{view_id}</th>" for view_id in view_ids)
    rows = []
    for view_a in view_ids:
        cells = []
        for view_b in view_ids:
            value = matrix[view_a][view_b]
            if value >= 0.75:
                cls = "high"
            elif value >= 0.5:
                cls = "mid"
            else:
                cls = "low"
            cells.append(f'<td class="{cls}">{value:.3f}</td>')
        rows.append(f"<tr><th>{view_a}</th>{''.join(cells)}</tr>")
    return f"<table><tr><th></th>{head}</tr>{''.join(rows)}</table>"


def _unique(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _safe_id(value: str) -> str:
    return value.replace("/", "__").replace("::", "__")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
