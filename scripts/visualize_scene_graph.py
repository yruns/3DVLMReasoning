"""Visualize 3D scene graph: colored point clouds + bidirectional mapping.

Outputs:
  1. <scene>/f2_sn200/vis/scene_objects.ply — all objects in one PLY, colored by instance
  2. <scene>/f2_sn200/vis/object_mapping.html — interactive HTML showing obj↔frame mapping

Usage:
    python scripts/visualize_scene_graph.py \
        --scene_path ~/Datasets/OpenEQA/scannet/002-scannet-scene0709_00/f2_sn200
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

# Distinct colors for up to 200 objects (HSV-based)
def _generate_colors(n: int) -> np.ndarray:
    """Generate N visually distinct RGB colors."""
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360  # golden angle
        sat = 0.7 + (i % 3) * 0.1
        val = 0.8 + (i % 2) * 0.15
        # HSV to RGB
        c = val * sat
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = val - c
        if hue < 60:
            r, g, b = c, x, 0
        elif hue < 120:
            r, g, b = x, c, 0
        elif hue < 180:
            r, g, b = 0, c, x
        elif hue < 240:
            r, g, b = 0, x, c
        elif hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors.append([r + m, g + m, b + m])
    return np.array(colors)


def export_colored_ply(objects: list[dict], labels: list[str], output_path: Path) -> None:
    """Export all objects as a single colored PLY file."""
    all_points = []
    all_colors = []
    colors = _generate_colors(len(objects))

    for i, obj in enumerate(objects):
        pts = np.array(obj.get('pcd_np', []))
        if len(pts) == 0:
            continue

        # Use original RGB colors or instance colors
        obj_colors = np.array(obj.get('pcd_color_np', []))
        if len(obj_colors) != len(pts):
            # Use instance color
            obj_colors = np.tile(colors[i], (len(pts), 1))

        all_points.append(pts)
        all_colors.append(obj_colors)

    if not all_points:
        print("No points to export!")
        return

    points = np.vstack(all_points)
    colors_arr = np.vstack(all_colors)

    # Clamp colors to [0, 1]
    colors_arr = np.clip(colors_arr, 0, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write PLY (instance-colored version)
    inst_path = output_path.with_name("scene_objects_instance_color.ply")
    _write_ply(points, (colors[np.repeat(range(len(objects)),
        [len(np.array(o.get('pcd_np', []))) for o in objects])] * 255).astype(np.uint8)
        if len(all_points) > 0 else np.zeros((0, 3), dtype=np.uint8),
        inst_path)

    # Write PLY (original RGB colors)
    rgb_path = output_path.with_name("scene_objects_rgb.ply")
    _write_ply(points, (colors_arr * 255).astype(np.uint8), rgb_path)

    print(f"  Instance-colored PLY: {inst_path} ({len(points)} points)")
    print(f"  RGB-colored PLY: {rgb_path} ({len(points)} points)")


def _write_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Write a PLY file with colored points."""
    n = len(points)
    header = f"""ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(header)
        for i in range(n):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")


def export_instance_ply(objects: list[dict], labels: list[str], output_path: Path) -> None:
    """Export each object with a distinct instance color into one PLY."""
    all_points = []
    all_colors = []
    inst_colors = _generate_colors(len(objects))

    for i, obj in enumerate(objects):
        pts = np.array(obj.get('pcd_np', []))
        if len(pts) == 0:
            continue
        color = (inst_colors[i] * 255).astype(np.uint8)
        all_points.append(pts)
        all_colors.append(np.tile(color, (len(pts), 1)))

    if not all_points:
        print("No points!")
        return

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(points, colors, output_path)
    print(f"  Instance PLY: {output_path} ({len(points)} points, {len(objects)} objects)")


def export_mapping_html(
    objects: list[dict],
    labels: list[str],
    vis_index: dict,
    scene_path: Path,
    output_path: Path,
) -> None:
    """Generate interactive HTML showing obj↔frame bidirectional mapping."""
    obj_to_views = vis_index.get('object_to_views', {})
    view_to_objects = vis_index.get('view_to_objects', {})

    # Build object info
    obj_info = []
    for i, obj in enumerate(objects):
        n_pts = len(np.array(obj.get('pcd_np', [])))
        n_dets = obj.get('num_detections', 0)
        views = obj_to_views.get(i, [])
        top_views = [(vid, f"{score:.2f}") for vid, score in views[:10]]
        obj_info.append({
            'id': i,
            'label': labels[i],
            'n_points': n_pts,
            'n_detections': n_dets,
            'n_views': len(views),
            'top_views': top_views,
        })

    # Build view info
    view_info = {}
    for vid, objs in sorted(view_to_objects.items()):
        view_info[vid] = [{'id': oid, 'label': labels[oid], 'score': f"{s:.2f}"}
                          for oid, s in objs[:20]]

    # Find raw image dir for frame thumbnails
    raw_dir = scene_path.parent / "raw"
    if not raw_dir.exists():
        raw_dir = scene_path

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Scene Graph Mapping</title>
<style>
body {{ font-family: monospace; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }}
h2 {{ color: #0f3460; background: #16213e; padding: 8px; border-radius: 4px; }}
.container {{ display: flex; gap: 20px; }}
.panel {{ flex: 1; max-height: 80vh; overflow-y: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
td, th {{ border: 1px solid #333; padding: 4px 8px; text-align: left; }}
th {{ background: #16213e; position: sticky; top: 0; }}
tr:hover {{ background: #16213e; cursor: pointer; }}
.selected {{ background: #0f3460 !important; }}
.badge {{ background: #e94560; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px; }}
.frame-list {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }}
.frame-tag {{ background: #16213e; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
#detail {{ background: #16213e; padding: 15px; border-radius: 8px; margin-top: 15px; min-height: 100px; }}
</style></head>
<body>
<h1>Scene Graph: {scene_path.parent.name}</h1>
<p>{len(objects)} objects, {len(view_to_objects)} views with detections</p>

<div class="container">
<div class="panel">
<h2>Objects → Views</h2>
<table id="obj-table">
<tr><th>ID</th><th>Label</th><th>Points</th><th>Views</th></tr>
"""
    for o in obj_info:
        html += f'<tr onclick="showObj({o["id"]})" id="obj-{o["id"]}"><td>{o["id"]}</td><td>{o["label"]}</td><td>{o["n_points"]}</td><td><span class="badge">{o["n_views"]}</span></td></tr>\n'

    html += """</table></div>
<div class="panel">
<h2>Views → Objects</h2>
<table id="view-table">
<tr><th>Frame</th><th>Objects</th></tr>
"""
    for vid in sorted(view_to_objects.keys()):
        n = len(view_to_objects[vid])
        html += f'<tr onclick="showView({vid})" id="view-{vid}"><td>frame {vid*5:06d}</td><td><span class="badge">{n}</span></td></tr>\n'

    html += f"""</table></div></div>

<div id="detail">Click an object or frame to see details.</div>

<script>
const objData = {json.dumps(obj_info)};
const viewData = {json.dumps(view_info)};

function showObj(id) {{
    document.querySelectorAll('#obj-table tr').forEach(r => r.classList.remove('selected'));
    document.getElementById('obj-'+id)?.classList.add('selected');
    const o = objData[id];
    let html = `<h3>Object ${{id}}: ${{o.label}}</h3>`;
    html += `<p>${{o.n_points}} 3D points, ${{o.n_detections}} detections, visible in ${{o.n_views}} views</p>`;
    html += `<div class="frame-list">`;
    o.top_views.forEach(([vid, score]) => {{
        html += `<span class="frame-tag" onclick="showView(${{vid}})">frame ${{String(vid*5).padStart(6,'0')}} (${{score}})</span>`;
    }});
    html += `</div>`;
    document.getElementById('detail').innerHTML = html;
}}

function showView(vid) {{
    document.querySelectorAll('#view-table tr').forEach(r => r.classList.remove('selected'));
    document.getElementById('view-'+vid)?.classList.add('selected');
    const objs = viewData[vid] || [];
    let html = `<h3>Frame ${{String(vid*5).padStart(6,'0')}}</h3>`;
    html += `<p>${{objs.length}} objects visible</p>`;
    html += `<table><tr><th>ID</th><th>Label</th><th>Score</th></tr>`;
    objs.forEach(o => {{
        html += `<tr onclick="showObj(${{o.id}})"><td>${{o.id}}</td><td>${{o.label}}</td><td>${{o.score}}</td></tr>`;
    }});
    html += `</table>`;
    document.getElementById('detail').innerHTML = html;
}}
</script>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"  Mapping HTML: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D scene graph")
    parser.add_argument("--scene_path", type=str, required=True)
    args = parser.parse_args()

    scene_path = Path(args.scene_path)
    vis_dir = scene_path / "vis"

    # Load scene graph
    import glob
    pcd_files = glob.glob(str(scene_path / "pcd_saves" / "*_post.pkl.gz"))
    if not pcd_files:
        raise FileNotFoundError(f"No *_post.pkl.gz in {scene_path / 'pcd_saves'}")
    pcd_file = pcd_files[0]

    print(f"Loading {pcd_file}...")
    data = pickle.load(gzip.open(pcd_file))
    objects = data['objects']
    labels = [Counter(obj['class_name']).most_common(1)[0][0] for obj in objects]
    print(f"  {len(objects)} objects")

    # Export PLY
    print("Generating PLY files...")
    export_instance_ply(objects, labels, vis_dir / "scene_objects_instance_color.ply")

    # Export RGB-colored PLY
    all_pts, all_cols = [], []
    for obj in objects:
        pts = np.array(obj.get('pcd_np', []))
        cols = np.array(obj.get('pcd_color_np', []))
        if len(pts) > 0 and len(cols) == len(pts):
            all_pts.append(pts)
            all_cols.append(cols)
    if all_pts:
        pts = np.vstack(all_pts)
        cols = np.vstack(all_cols)
        cols = np.clip(cols, 0, 1)
        _write_ply(pts, (cols * 255).astype(np.uint8), vis_dir / "scene_objects_rgb.ply")
        print(f"  RGB PLY: {vis_dir / 'scene_objects_rgb.ply'} ({len(pts)} points)")

    # Export per-object label list
    label_info = []
    for i, (obj, label) in enumerate(zip(objects, labels)):
        n_pts = len(np.array(obj.get('pcd_np', [])))
        label_info.append({'id': i, 'label': label, 'n_points': n_pts, 'n_detections': obj.get('num_detections', 0)})
    label_path = vis_dir / "object_labels.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(label_info, indent=2))
    print(f"  Labels JSON: {label_path}")

    # Load visibility index and generate mapping HTML
    vis_index_path = scene_path / "indices" / "visibility_index.pkl"
    if vis_index_path.exists():
        print("Generating mapping visualization...")
        vis_index = pickle.load(open(vis_index_path, 'rb'))
        export_mapping_html(objects, labels, vis_index, scene_path, vis_dir / "mapping.html")
    else:
        print(f"  Skipping mapping HTML (no visibility index at {vis_index_path})")

    print(f"\nDone! All outputs in {vis_dir}/")


if __name__ == "__main__":
    main()
