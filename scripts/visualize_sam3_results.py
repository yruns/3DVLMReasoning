"""Comprehensive visualization for SAM 3 scene graph results.

Outputs per scene (in <scene>/conceptgraph/sam3_vis/):
  1. scene_objects_instance.ply  — all objects, instance-colored
  2. scene_objects_rgb.ply       — all objects, original RGB
  3. object_labels.json          — per-object metadata
  4. mapping.html                — interactive obj↔frame bidirectional mapping
  5. frames/frame_XXXXXX.jpg     — sampled frames with all detection overlays
  6. objects/<id>_<label>/       — per-object folder with representative frames

Usage:
    python scripts/visualize_sam3_results.py --all
    python scripts/visualize_sam3_results.py --scene 002-scannet-scene0709_00
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_ROOT = Path.home() / "Datasets" / "OpenEQA" / "scannet"
SCENES = [
    "002-scannet-scene0709_00",
    "003-scannet-scene0762_00",
    "012-scannet-scene0785_00",
    "013-scannet-scene0720_00",
    "014-scannet-scene0714_00",
]
PCD_GLOB = "*sam3*_post.pkl.gz"
VIS_INDEX_NAME = "visibility_index.pkl"
N_SAMPLE_FRAMES = 8       # number of frames to visualize with detection overlays
N_OBJ_FRAMES = 3          # representative frames per object


def _generate_colors(n: int) -> np.ndarray:
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360
        sat = 0.7 + (i % 3) * 0.1
        val = 0.8 + (i % 2) * 0.15
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


def _write_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    n = len(points)
    header = f"ply\nformat ascii 1.0\nelement vertex {n}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")


# ---------------------------------------------------------------------------
# 1. PLY export
# ---------------------------------------------------------------------------

def export_ply(objects, labels, vis_dir):
    inst_colors = _generate_colors(len(objects))
    all_pts_inst, all_cols_inst = [], []
    all_pts_rgb, all_cols_rgb = [], []

    for i, obj in enumerate(objects):
        pts = np.array(obj.get("pcd_np", []))
        if len(pts) == 0:
            continue
        # Instance color
        color_inst = (inst_colors[i] * 255).astype(np.uint8)
        all_pts_inst.append(pts)
        all_cols_inst.append(np.tile(color_inst, (len(pts), 1)))
        # RGB color
        cols = np.array(obj.get("pcd_color_np", []))
        if len(cols) == len(pts):
            all_pts_rgb.append(pts)
            all_cols_rgb.append(np.clip(cols, 0, 1))

    if all_pts_inst:
        pts = np.vstack(all_pts_inst)
        cols = np.vstack(all_cols_inst)
        _write_ply(pts, cols, vis_dir / "scene_objects_instance.ply")
        print(f"    Instance PLY: {len(pts)} points, {len(objects)} objects")

    if all_pts_rgb:
        pts = np.vstack(all_pts_rgb)
        cols = (np.vstack(all_cols_rgb) * 255).astype(np.uint8)
        _write_ply(pts, cols, vis_dir / "scene_objects_rgb.ply")
        print(f"    RGB PLY: {len(pts)} points")


# ---------------------------------------------------------------------------
# 2. Object labels JSON
# ---------------------------------------------------------------------------

def export_labels(objects, labels, vis_dir):
    info = []
    for i, (obj, label) in enumerate(zip(objects, labels)):
        n_pts = len(np.array(obj.get("pcd_np", [])))
        info.append({
            "id": i, "label": label, "n_points": n_pts,
            "n_detections": obj.get("num_detections", 0),
        })
    path = vis_dir / "object_labels.json"
    path.write_text(json.dumps(info, indent=2))
    print(f"    Labels: {len(info)} objects → {path.name}")
    return info


# ---------------------------------------------------------------------------
# 3. Interactive HTML mapping
# ---------------------------------------------------------------------------

def export_mapping_html(objects, labels, vis_index, scene_name, vis_dir):
    obj_to_views = vis_index.get("object_to_views", {})
    view_to_objects = vis_index.get("view_to_objects", {})

    obj_info = []
    for i, obj in enumerate(objects):
        n_pts = len(np.array(obj.get("pcd_np", [])))
        views = obj_to_views.get(i, [])
        top_views = [(vid, f"{score:.2f}") for vid, score in views[:10]]
        obj_info.append({
            "id": i, "label": labels[i], "n_points": n_pts,
            "n_detections": obj.get("num_detections", 0),
            "n_views": len(views), "top_views": top_views,
        })

    view_info = {}
    for vid, objs in sorted(view_to_objects.items()):
        view_info[str(vid)] = [
            {"id": oid, "label": labels[oid], "score": f"{s:.2f}"}
            for oid, s in objs[:30]
        ]

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SAM 3 Scene Graph: {scene_name}</title>
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
.frame-tag {{ background: #16213e; padding: 2px 6px; border-radius: 3px; font-size: 11px; cursor: pointer; }}
.frame-tag:hover {{ background: #0f3460; }}
#detail {{ background: #16213e; padding: 15px; border-radius: 8px; margin-top: 15px; min-height: 100px; }}
.thumb {{ max-width: 300px; max-height: 200px; margin: 5px; border: 2px solid #333; border-radius: 4px; }}
</style></head>
<body>
<h1>SAM 3 Scene Graph: {scene_name}</h1>
<p>{len(objects)} objects, {len(view_to_objects)} views with detections</p>

<div class="container">
<div class="panel">
<h2>Objects → Views</h2>
<table id="obj-table">
<tr><th>ID</th><th>Label</th><th>Points</th><th>Dets</th><th>Views</th></tr>
"""
    for o in obj_info:
        html += (f'<tr onclick="showObj({o["id"]})" id="obj-{o["id"]}">'
                 f'<td>{o["id"]}</td><td>{o["label"]}</td><td>{o["n_points"]}</td>'
                 f'<td>{o["n_detections"]}</td>'
                 f'<td><span class="badge">{o["n_views"]}</span></td></tr>\n')

    html += """</table></div>
<div class="panel">
<h2>Views → Objects</h2>
<table id="view-table">
<tr><th>Frame</th><th>Objects</th></tr>
"""
    for vid in sorted(view_to_objects.keys()):
        n = len(view_to_objects[vid])
        fid = vid * 5
        html += (f'<tr onclick="showView({vid})" id="view-{vid}">'
                 f'<td>frame {fid:06d}</td>'
                 f'<td><span class="badge">{n}</span></td></tr>\n')

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
    // Show object folder link
    const safeName = `${{String(id).padStart(3,'0')}}_${{o.label.replace(/ /g,'_')}}`;
    html += `<p>Object folder: <a href="objects/${{safeName}}/" style="color:#e94560">objects/${{safeName}}/</a></p>`;
    html += `<div class="frame-list">`;
    o.top_views.forEach(([vid, score]) => {{
        const fid = String(vid*5).padStart(6,'0');
        html += `<span class="frame-tag" onclick="showView(${{vid}})">frame ${{fid}} (${{score}})</span>`;
    }});
    html += `</div>`;
    // Show representative images
    html += `<div style="margin-top:10px">`;
    o.top_views.slice(0,3).forEach(([vid, score]) => {{
        const fid = String(vid*5).padStart(6,'0');
        html += `<img class="thumb" src="objects/${{safeName}}/${{fid}}.jpg" onerror="this.style.display='none'">`;
    }});
    html += `</div>`;
    document.getElementById('detail').innerHTML = html;
}}

function showView(vid) {{
    document.querySelectorAll('#view-table tr').forEach(r => r.classList.remove('selected'));
    document.getElementById('view-'+vid)?.classList.add('selected');
    const objs = viewData[vid] || [];
    const fid = String(vid*5).padStart(6,'0');
    let html = `<h3>Frame ${{fid}}</h3>`;
    html += `<p>${{objs.length}} objects visible</p>`;
    html += `<img class="thumb" src="frames/frame_${{fid}}.jpg" style="max-width:600px;max-height:400px" onerror="this.style.display='none'">`;
    html += `<table><tr><th>ID</th><th>Label</th><th>Score</th></tr>`;
    objs.forEach(o => {{
        html += `<tr onclick="showObj(${{o.id}})" style="cursor:pointer"><td>${{o.id}}</td><td>${{o.label}}</td><td>${{o.score}}</td></tr>`;
    }});
    html += `</table>`;
    document.getElementById('detail').innerHTML = html;
}}
</script>
</body></html>"""

    path = vis_dir / "mapping.html"
    path.write_text(html)
    print(f"    Mapping HTML: {path.name}")


# ---------------------------------------------------------------------------
# 4. Per-frame detection overlay images
# ---------------------------------------------------------------------------

def export_frame_overlays(det_dir, raw_dir, vis_dir, n_frames=N_SAMPLE_FRAMES):
    frames_dir = vis_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    det_files = sorted(det_dir.glob("*.pkl.gz"))
    if not det_files:
        print("    No detection files found!")
        return

    # Pick evenly spaced frames
    step = max(1, len(det_files) // n_frames)
    indices = list(range(0, len(det_files), step))[:n_frames]

    np.random.seed(42)
    palette = [(int(np.random.randint(50, 255)), int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255))) for _ in range(300)]

    for idx in indices:
        f = det_files[idx]
        d = pickle.load(gzip.open(f))
        stem = f.stem.replace(".pkl", "")
        img_path = raw_dir / f"{stem}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        xyxy = d["xyxy"]
        classes = d["classes"]
        class_ids = d["class_id"]
        confs = d["confidence"]

        drawn = 0
        for i in range(len(xyxy)):
            label = classes[class_ids[i]]
            if label in ("wall", "floor", "ceiling"):
                continue
            x1, y1, x2, y2 = map(int, xyxy[i])
            conf = confs[i]
            color = palette[class_ids[i] % len(palette)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            drawn += 1

        out = frames_dir / f"frame_{stem.split('-')[0]}.jpg"
        cv2.imwrite(str(out), img)
        print(f"      Frame {stem}: {drawn} objects → {out.name}")

    print(f"    Frames: {len(indices)} annotated frames → frames/")


# ---------------------------------------------------------------------------
# 5. Per-object representative frames
# ---------------------------------------------------------------------------

def export_object_galleries(objects, labels, vis_index, raw_dir, det_dir, vis_dir):
    obj_to_views = vis_index.get("object_to_views", {})
    objects_dir = vis_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    # Build detection lookup: frame_stem → detections
    det_lookup = {}
    for f in sorted(det_dir.glob("*.pkl.gz")):
        stem = f.stem.replace(".pkl", "")
        det_lookup[stem] = f

    np.random.seed(42)
    palette = [(int(np.random.randint(50, 255)), int(np.random.randint(50, 255)),
                int(np.random.randint(50, 255))) for _ in range(300)]

    for obj_id, obj in enumerate(objects):
        label = labels[obj_id]
        safe_name = f"{obj_id:03d}_{label.replace(' ', '_')}"
        obj_dir = objects_dir / safe_name
        obj_dir.mkdir(exist_ok=True)

        views = obj_to_views.get(obj_id, [])
        if not views:
            continue

        # Pick top N representative views by score
        top_views = views[:N_OBJ_FRAMES]

        for vid, score in top_views:
            fid = vid * 5
            frame_stem = f"{fid:06d}-rgb"
            img_path = raw_dir / f"{frame_stem}.png"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))

            # Try to find and highlight this object's detection in this frame
            if frame_stem in det_lookup:
                d = pickle.load(gzip.open(det_lookup[frame_stem]))
                classes = d["classes"]
                class_ids = d["class_id"]
                xyxy = d["xyxy"]
                confs = d["confidence"]

                # Draw all detections lightly, highlight matching ones
                for i in range(len(xyxy)):
                    det_label = classes[class_ids[i]]
                    if det_label in ("wall", "floor", "ceiling"):
                        continue
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    if det_label == label:
                        # Highlight: thick box + label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        text = f"{det_label} {confs[i]:.2f}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
                        cv2.putText(img, text, (x1 + 3, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    else:
                        # Dim other detections
                        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)

            # Add header
            header_text = f"Obj {obj_id}: {label} | Frame {fid:06d} | Score {score:.2f}"
            cv2.putText(img, header_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            out_path = obj_dir / f"{fid:06d}.jpg"
            cv2.imwrite(str(out_path), img)

    n_dirs = len(list(objects_dir.iterdir()))
    print(f"    Object galleries: {n_dirs} objects → objects/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_scene(scene_name: str):
    scene_dir = DATASET_ROOT / scene_name
    cg_dir = scene_dir / "conceptgraph"
    raw_dir = scene_dir / "raw"

    print(f"\n{'='*60}")
    print(f"  Scene: {scene_name}")
    print(f"{'='*60}")

    # Find SAM3 pcd file
    pcd_files = list(cg_dir.glob(f"pcd_saves/{PCD_GLOB}"))
    if not pcd_files:
        print(f"  ERROR: No SAM3 pcd file found in {cg_dir / 'pcd_saves'}")
        return
    pcd_file = pcd_files[0]

    # Load scene graph
    print(f"  Loading {pcd_file.name}...")
    data = pickle.load(gzip.open(pcd_file))
    objects = data["objects"]
    labels = [Counter(obj["class_name"]).most_common(1)[0][0] for obj in objects]
    print(f"  {len(objects)} objects loaded")

    # Load visibility index
    vis_index_path = cg_dir / "indices" / VIS_INDEX_NAME
    vis_index = {}
    if vis_index_path.exists():
        vis_index = pickle.load(open(vis_index_path, "rb"))
        n_obj = len(vis_index.get("object_to_views", {}))
        n_view = len(vis_index.get("view_to_objects", {}))
        print(f"  Visibility index: {n_obj} objects, {n_view} views")
    else:
        print(f"  WARNING: No visibility index at {vis_index_path}")

    # Detection dir
    det_dir = cg_dir / "gsa_detections_sam3_sn200_withbg"
    if not det_dir.exists():
        print(f"  ERROR: No detection dir at {det_dir}")
        return

    vis_dir = cg_dir / "sam3_vis"
    vis_dir.mkdir(exist_ok=True)

    # 1. PLY export
    print("  [1/5] Exporting PLY files...")
    export_ply(objects, labels, vis_dir)

    # 2. Object labels
    print("  [2/5] Exporting object labels...")
    export_labels(objects, labels, vis_dir)

    # 3. Interactive HTML
    print("  [3/5] Generating interactive mapping HTML...")
    if vis_index:
        export_mapping_html(objects, labels, vis_index, scene_name, vis_dir)

    # 4. Per-frame overlays
    print("  [4/5] Generating frame detection overlays...")
    export_frame_overlays(det_dir, raw_dir, vis_dir)

    # 5. Per-object galleries
    print("  [5/5] Generating per-object frame galleries...")
    if vis_index:
        export_object_galleries(objects, labels, vis_index, raw_dir, det_dir, vis_dir)

    print(f"\n  All outputs → {vis_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM 3 scene graph results")
    parser.add_argument("--all", action="store_true", help="Process all 5 scenes")
    parser.add_argument("--scene", type=str, help="Process specific scene")
    args = parser.parse_args()

    if args.all:
        for s in SCENES:
            process_scene(s)
    elif args.scene:
        process_scene(args.scene)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
