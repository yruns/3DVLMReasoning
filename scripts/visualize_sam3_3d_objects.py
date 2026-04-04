"""Visualize FINAL 3D objects projected back onto frames.

Unlike the raw 2D detection overlay, this shows objects that survived
the full SLAM pipeline (matching, merging, filtering). Each bbox is
derived by projecting the 3D object's point cloud onto the frame via
camera intrinsics + extrinsics.

Outputs per scene (in <scene>/conceptgraph/sam3_vis_3d/):
  1. scene_objects_instance.ply  — combined instance-colored PLY
  2. scene_objects_rgb.ply       — combined original-RGB PLY
  3. object_labels.json          — per-object metadata
  4. mapping.html                — interactive obj↔frame HTML
  5. frames/frame_XXXXXX.jpg     — sampled frames with projected 3D obj bbox
  6. objects/<id>_<label>/       — per-object folder with representative frames

Usage:
    python scripts/visualize_sam3_3d_objects.py --all
    python scripts/visualize_sam3_3d_objects.py --scene 002-scannet-scene0709_00
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
OBJ_FRAMES_DIR = "sam3_sn200_withbg_sam3_overlap_simsum1.2"
VIS_INDEX_NAME = "visibility_index.pkl"
N_SAMPLE_FRAMES = 8
N_OBJ_FRAMES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _write_ply(points, colors, path):
    n = len(points)
    header = (f"ply\nformat ascii 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")


def load_intrinsics(scene_dir: Path) -> np.ndarray:
    """Load 3x3 camera intrinsic matrix."""
    for name in ["intrinsic_color.txt", "intrinsic.txt"]:
        p = scene_dir / "raw" / name
        if p.exists():
            K4 = np.loadtxt(p)
            return K4[:3, :3]
        p2 = scene_dir / "conceptgraph" / name
        if p2.exists():
            K4 = np.loadtxt(p2)
            return K4[:3, :3]
    raise FileNotFoundError(f"No intrinsic file in {scene_dir}")


def project_points_to_2d(pts_3d: np.ndarray, K: np.ndarray,
                          cam_pose: np.ndarray, H: int, W: int):
    """Project 3D points to 2D pixel coords. Returns (u, v, valid_mask)."""
    # cam_pose is world-to-camera or camera-to-world?
    # In ConceptGraph, cam_pose is camera-to-world (T_wc).
    # We need world-to-camera: T_cw = inv(T_wc)
    T_cw = np.linalg.inv(cam_pose)
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]

    # Transform to camera frame
    pts_cam = (R @ pts_3d.T).T + t  # (N, 3)

    # Filter points behind camera
    valid = pts_cam[:, 2] > 0.1
    if not np.any(valid):
        return np.array([]), np.array([]), np.zeros(len(pts_3d), dtype=bool)

    # Project to pixel
    pts_proj = (K @ pts_cam.T).T  # (N, 3)
    u = pts_proj[:, 0] / (pts_proj[:, 2] + 1e-8)
    v = pts_proj[:, 1] / (pts_proj[:, 2] + 1e-8)

    # Check bounds
    valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    return u, v, valid


def get_projected_bbox(pts_3d: np.ndarray, K: np.ndarray,
                        cam_pose: np.ndarray, H: int, W: int):
    """Get 2D bounding box from projected 3D points. Returns (x1,y1,x2,y2) or None."""
    u, v, valid = project_points_to_2d(pts_3d, K, cam_pose, H, W)
    if not np.any(valid):
        return None
    u_valid = u[valid]
    v_valid = v[valid]
    x1 = max(0, int(np.min(u_valid)))
    y1 = max(0, int(np.min(v_valid)))
    x2 = min(W, int(np.max(u_valid)) + 1)
    y2 = min(H, int(np.max(v_valid)) + 1)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# 1. PLY export (same as before)
# ---------------------------------------------------------------------------

def export_ply(objects, labels, vis_dir):
    inst_colors = _generate_colors(len(objects))
    all_pts_inst, all_cols_inst = [], []
    all_pts_rgb, all_cols_rgb = [], []

    for i, obj in enumerate(objects):
        pts = np.array(obj.get("pcd_np", []))
        if len(pts) == 0:
            continue
        color_inst = (inst_colors[i] * 255).astype(np.uint8)
        all_pts_inst.append(pts)
        all_cols_inst.append(np.tile(color_inst, (len(pts), 1)))
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
# 2. Object labels
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
    print(f"    Labels: {len(info)} objects")
    return info


# ---------------------------------------------------------------------------
# 3. Mapping HTML (same as before, updated paths)
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
<html><head><meta charset="utf-8"><title>SAM 3 3D Objects: {scene_name}</title>
<style>
body {{ font-family: monospace; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }} h2 {{ color: #0f3460; background: #16213e; padding: 8px; border-radius: 4px; }}
.container {{ display: flex; gap: 20px; }} .panel {{ flex: 1; max-height: 80vh; overflow-y: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
td, th {{ border: 1px solid #333; padding: 4px 8px; text-align: left; }}
th {{ background: #16213e; position: sticky; top: 0; }}
tr:hover {{ background: #16213e; cursor: pointer; }} .selected {{ background: #0f3460 !important; }}
.badge {{ background: #e94560; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px; }}
.frame-list {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }}
.frame-tag {{ background: #16213e; padding: 2px 6px; border-radius: 3px; font-size: 11px; cursor: pointer; }}
.frame-tag:hover {{ background: #0f3460; }} #detail {{ background: #16213e; padding: 15px; border-radius: 8px; margin-top: 15px; min-height: 100px; }}
.thumb {{ max-width: 400px; max-height: 300px; margin: 5px; border: 2px solid #333; border-radius: 4px; }}
.note {{ color: #888; font-size: 12px; margin-top: 5px; }}
</style></head>
<body>
<h1>SAM 3 — Final 3D Objects: {scene_name}</h1>
<p>{len(objects)} 3D objects (post-pipeline), {len(view_to_objects)} views</p>
<p class="note">Bounding boxes are 3D object point clouds projected onto each frame — NOT raw 2D detections.</p>
<div class="container"><div class="panel"><h2>Objects → Views</h2>
<table id="obj-table"><tr><th>ID</th><th>Label</th><th>Pts</th><th>Dets</th><th>Views</th></tr>
"""
    for o in obj_info:
        html += (f'<tr onclick="showObj({o["id"]})" id="obj-{o["id"]}">'
                 f'<td>{o["id"]}</td><td>{o["label"]}</td><td>{o["n_points"]}</td>'
                 f'<td>{o["n_detections"]}</td><td><span class="badge">{o["n_views"]}</span></td></tr>\n')
    html += '</table></div><div class="panel"><h2>Views → Objects</h2><table id="view-table"><tr><th>Frame</th><th>Objs</th></tr>\n'
    for vid in sorted(view_to_objects.keys()):
        n = len(view_to_objects[vid])
        html += f'<tr onclick="showView({vid})" id="view-{vid}"><td>frame {vid*5:06d}</td><td><span class="badge">{n}</span></td></tr>\n'
    html += f"""</table></div></div><div id="detail">Click an object or frame to see details.</div>
<script>
const objData = {json.dumps(obj_info)};
const viewData = {json.dumps(view_info)};
function showObj(id) {{
    document.querySelectorAll('#obj-table tr').forEach(r => r.classList.remove('selected'));
    document.getElementById('obj-'+id)?.classList.add('selected');
    const o = objData[id];
    const safeName = `${{String(id).padStart(3,'0')}}_${{o.label.replace(/ /g,'_')}}`;
    let h = `<h3>Object ${{id}}: ${{o.label}}</h3><p>${{o.n_points}} 3D pts, ${{o.n_detections}} dets, ${{o.n_views}} views</p>`;
    h += `<div class="frame-list">`;
    o.top_views.forEach(([vid, score]) => {{ h += `<span class="frame-tag" onclick="showView(${{vid}})">frame ${{String(vid*5).padStart(6,'0')}} (${{score}})</span>`; }});
    h += `</div><div style="margin-top:10px">`;
    o.top_views.slice(0,3).forEach(([vid]) => {{ const fid = String(vid*5).padStart(6,'0'); h += `<img class="thumb" src="objects/${{safeName}}/${{fid}}.jpg" onerror="this.style.display='none'">`; }});
    h += `</div>`; document.getElementById('detail').innerHTML = h;
}}
function showView(vid) {{
    document.querySelectorAll('#view-table tr').forEach(r => r.classList.remove('selected'));
    document.getElementById('view-'+vid)?.classList.add('selected');
    const objs = viewData[vid] || []; const fid = String(vid*5).padStart(6,'0');
    let h = `<h3>Frame ${{fid}}</h3><p>${{objs.length}} 3D objects visible</p>`;
    h += `<img class="thumb" src="frames/frame_${{fid}}.jpg" style="max-width:700px;max-height:500px" onerror="this.style.display='none'">`;
    h += `<table><tr><th>ID</th><th>Label</th><th>Score</th></tr>`;
    objs.forEach(o => {{ h += `<tr onclick="showObj(${{o.id}})" style="cursor:pointer"><td>${{o.id}}</td><td>${{o.label}}</td><td>${{o.score}}</td></tr>`; }});
    h += `</table>`; document.getElementById('detail').innerHTML = h;
}}
</script></body></html>"""
    (vis_dir / "mapping.html").write_text(html)
    print(f"    Mapping HTML")


# ---------------------------------------------------------------------------
# 4. Per-frame: project 3D objects → 2D bbox overlay
# ---------------------------------------------------------------------------

def export_frame_projections(objects, labels, vis_index, K, raw_dir,
                              obj_frames_dir, vis_dir, n_frames=N_SAMPLE_FRAMES):
    frames_dir = vis_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    view_to_objects = vis_index.get("view_to_objects", {})
    inst_colors = _generate_colors(len(objects))

    # Pick evenly spaced view ids
    all_vids = sorted(view_to_objects.keys())
    if not all_vids:
        print("    No views!")
        return
    step = max(1, len(all_vids) // n_frames)
    sample_vids = all_vids[::step][:n_frames]

    # Load per-frame object snapshots for camera poses
    frame_files = sorted(obj_frames_dir.glob("*.pkl.gz"))
    frame_lookup = {}  # frame_index (1-based) → file
    for f in frame_files:
        try:
            idx = int(f.stem.replace(".pkl", ""))
            frame_lookup[idx] = f
        except ValueError:
            pass

    for vid in sample_vids:
        fid = vid * 5  # frame stride = 5
        frame_stem = f"{fid:06d}-rgb"
        img_path = raw_dir / f"{frame_stem}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # Get camera pose from objects_all_frames
        # vid is 0-indexed view, frame file is 1-indexed (frame 1 = first strided frame)
        frame_file_idx = vid + 1
        if frame_file_idx not in frame_lookup:
            continue
        frame_data = pickle.load(gzip.open(frame_lookup[frame_file_idx]))
        cam_pose = frame_data["camera_pose"]

        # Project each visible 3D object
        visible_objs = view_to_objects.get(vid, [])
        drawn = 0
        for obj_id, score in visible_objs:
            if obj_id >= len(objects):
                continue
            pts_3d = np.array(objects[obj_id].get("pcd_np", []))
            if len(pts_3d) == 0:
                continue

            bbox_2d = get_projected_bbox(pts_3d, K, cam_pose, H, W)
            if bbox_2d is None:
                continue

            x1, y1, x2, y2 = bbox_2d
            # Skip huge boxes (> 70% image)
            if (x2 - x1) * (y2 - y1) > 0.7 * H * W:
                continue

            label = labels[obj_id]
            color_f = inst_colors[obj_id % len(inst_colors)]
            color = (int(color_f[2] * 255), int(color_f[1] * 255), int(color_f[0] * 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"{obj_id}:{label}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            drawn += 1

        out = frames_dir / f"frame_{fid:06d}.jpg"
        cv2.imwrite(str(out), img)
        print(f"      Frame {fid:06d}: {drawn} 3D objects projected → {out.name}")

    print(f"    Frames: {len(sample_vids)} projected frames → frames/")


# ---------------------------------------------------------------------------
# 5. Per-object: representative frames with this object highlighted
# ---------------------------------------------------------------------------

def export_object_galleries(objects, labels, vis_index, K, raw_dir,
                             obj_frames_dir, vis_dir):
    obj_to_views = vis_index.get("object_to_views", {})
    view_to_objects = vis_index.get("view_to_objects", {})
    objects_dir = vis_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    inst_colors = _generate_colors(len(objects))

    # Camera pose lookup
    frame_files = sorted(obj_frames_dir.glob("*.pkl.gz"))
    frame_lookup = {}
    for f in frame_files:
        try:
            idx = int(f.stem.replace(".pkl", ""))
            frame_lookup[idx] = f
        except ValueError:
            pass

    for obj_id, obj in enumerate(objects):
        label = labels[obj_id]
        safe_name = f"{obj_id:03d}_{label.replace(' ', '_')}"
        obj_dir = objects_dir / safe_name
        obj_dir.mkdir(exist_ok=True)

        views = obj_to_views.get(obj_id, [])
        if not views:
            continue

        pts_3d = np.array(obj.get("pcd_np", []))
        if len(pts_3d) == 0:
            continue

        top_views = views[:N_OBJ_FRAMES]
        for vid, score in top_views:
            fid = vid * 5
            frame_stem = f"{fid:06d}-rgb"
            img_path = raw_dir / f"{frame_stem}.png"
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]

            frame_file_idx = vid + 1
            if frame_file_idx not in frame_lookup:
                continue
            frame_data = pickle.load(gzip.open(frame_lookup[frame_file_idx]))
            cam_pose = frame_data["camera_pose"]

            # Draw all other visible objects dimmed
            other_objs = view_to_objects.get(vid, [])
            for oid, _ in other_objs:
                if oid == obj_id or oid >= len(objects):
                    continue
                other_pts = np.array(objects[oid].get("pcd_np", []))
                if len(other_pts) == 0:
                    continue
                bbox = get_projected_bbox(other_pts, K, cam_pose, H, W)
                if bbox:
                    bx1, by1, bx2, by2 = bbox
                    if (bx2 - bx1) * (by2 - by1) < 0.7 * H * W:
                        cv2.rectangle(img, (bx1, by1), (bx2, by2), (80, 80, 80), 1)

            # Highlight THIS object
            bbox = get_projected_bbox(pts_3d, K, cam_pose, H, W)
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                text = f"{obj_id}:{label} ({score:.2f})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
                cv2.putText(img, text, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Header
            cv2.putText(img, f"Obj {obj_id}: {label} | Frame {fid:06d} | Score {score:.2f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imwrite(str(obj_dir / f"{fid:06d}.jpg"), img)

    n_dirs = sum(1 for _ in objects_dir.iterdir() if _.is_dir())
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

    # Load post-processed scene graph
    pcd_files = list(cg_dir.glob(f"pcd_saves/{PCD_GLOB}"))
    if not pcd_files:
        print(f"  ERROR: No SAM3 pcd in {cg_dir / 'pcd_saves'}")
        return
    pcd_file = pcd_files[0]

    print(f"  Loading {pcd_file.name}...")
    data = pickle.load(gzip.open(pcd_file))
    objects = data["objects"]
    labels = [Counter(obj["class_name"]).most_common(1)[0][0] for obj in objects]
    print(f"  {len(objects)} 3D objects")

    # Load visibility index
    vis_index_path = cg_dir / "indices" / VIS_INDEX_NAME
    if not vis_index_path.exists():
        print(f"  ERROR: No visibility index")
        return
    vis_index = pickle.load(open(vis_index_path, "rb"))

    # Camera intrinsics
    K = load_intrinsics(scene_dir)
    print(f"  Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")

    # objects_all_frames dir
    obj_frames_dir = cg_dir / "objects_all_frames" / OBJ_FRAMES_DIR
    if not obj_frames_dir.exists():
        print(f"  ERROR: No objects_all_frames dir at {obj_frames_dir}")
        return

    vis_dir = cg_dir / "sam3_vis_3d"
    vis_dir.mkdir(exist_ok=True)

    # 1. PLY
    print("  [1/5] PLY export...")
    export_ply(objects, labels, vis_dir)

    # 2. Labels
    print("  [2/5] Object labels...")
    export_labels(objects, labels, vis_dir)

    # 3. HTML
    print("  [3/5] Mapping HTML...")
    export_mapping_html(objects, labels, vis_index, scene_name, vis_dir)

    # 4. Frame projections
    print("  [4/5] Frame projections (3D→2D)...")
    export_frame_projections(objects, labels, vis_index, K, raw_dir,
                              obj_frames_dir, vis_dir)

    # 5. Object galleries
    print("  [5/5] Object galleries...")
    export_object_galleries(objects, labels, vis_index, K, raw_dir,
                             obj_frames_dir, vis_dir)

    print(f"\n  All outputs → {vis_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scene", type=str)
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
