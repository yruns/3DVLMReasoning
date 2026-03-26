"""YOLOE prompt-free detection pipeline for ConceptGraph.

Uses YOLOE-11L-seg-pf (prompt-free, 4585 LVIS+ categories) for detection
and segmentation in a single model — no separate SAM pass needed.

Output format is identical to ``generate_gsa.py`` so downstream 3-D mapping
(``conceptgraph.slam.pipeline``) works unchanged.

Usage::

    python conceptgraph/detection/generate_yoloe.py \\
        --dataset_root /path/to/OpenEQA/scannet \\
        --dataset_config conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml \\
        --scene_id 002-scannet-scene0709_00/conceptgraph \\
        --stride 5 --add_bg_classes
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import trange
from ultralytics import YOLO

from conceptgraph.dataset.loader import get_dataset

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BG_CLASSES = ["wall", "floor", "ceiling"]

# Scene-level / non-object categories to filter out
_SCENE_LEVEL_TAGS = {
    "house", "room", "apartment", "building", "space", "area",
    "kitchen", "bathroom", "bedroom", "office", "hallway",
    "living room", "dining room", "playroom", "darkness",
    "act", "action", "activity", "add", "adaptation",
    "3d cg rendering", "action film",
}


def _is_scene_level(label: str) -> bool:
    return label.lower().strip() in _SCENE_LEVEL_TAGS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLOE prompt-free detection")
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--dataset_config", type=str, required=True)
    p.add_argument("--scene_id", type=str, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--desired_height", type=int, default=480)
    p.add_argument("--desired_width", type=int, default=640)
    p.add_argument("--add_bg_classes", action="store_true")
    p.add_argument("--model", type=str, default="yoloe-11l-seg-pf.pt")
    p.add_argument("--conf_threshold", type=float, default=0.1,
                    help="Minimum confidence for detections")
    p.add_argument("--iou_threshold", type=float, default=0.5,
                    help="IoU threshold for NMS")
    p.add_argument("--max_bbox_area_ratio", type=float, default=0.8,
                    help="Filter detections covering more than this ratio of image")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--exp_suffix", type=str, default="")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_parser().parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = get_dataset(
        dataconfig=args.dataset_config,
        basedir=str(args.dataset_root),
        sequence=args.scene_id,
        start=args.start,
        end=args.end,
        stride=args.stride,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
    )
    scene_path = args.dataset_root / args.scene_id

    # ── YOLOE ────────────────────────────────────────────────────────
    print(f"Loading YOLOE ({args.model}) …")
    yolo_model = YOLO(args.model)

    # ── CLIP ─────────────────────────────────────────────────────────
    print("Loading CLIP …")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k",
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # ── Output dirs ──────────────────────────────────────────────────
    save_tag = "yoloe_pf"
    if args.add_bg_classes:
        save_tag += "_withbg"
    save_tag += "_allclasses"
    if args.exp_suffix:
        save_tag += f"_{args.exp_suffix}"
    det_dir = scene_path / f"gsa_detections_{save_tag}"
    det_dir.mkdir(parents=True, exist_ok=True)

    global_classes: set[str] = set()

    print(f"Processing {len(dataset)} frames (conf≥{args.conf_threshold}) …")

    # ── Per-frame processing ─────────────────────────────────────────
    for idx in trange(len(dataset)):
        color_path = Path(dataset.color_paths[idx])
        det_save = det_dir / color_path.with_suffix(".pkl.gz").name

        image_bgr = cv2.imread(str(color_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        H, W = image_bgr.shape[:2]
        img_area = H * W

        # ── Frame-level CLIP feature ─────────────────────────────────
        clip_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
        frame_clip_feat = clip_model.encode_image(clip_input)
        frame_clip_feat = F.normalize(frame_clip_feat, dim=-1)
        frame_clip_feat = frame_clip_feat.cpu().numpy().squeeze()

        # ── YOLOE detection ──────────────────────────────────────────
        results = yolo_model.predict(
            str(color_path),
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            verbose=False,
        )
        r = results[0]

        # Extract detections
        bboxes = r.boxes.xyxy.cpu().numpy()  # (N, 4)
        confidences = r.boxes.conf.cpu().numpy()  # (N,)
        class_ids_raw = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        labels = [r.names[cid] for cid in class_ids_raw]

        # Extract masks if available
        if r.masks is not None:
            masks_raw = r.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
            # Resize masks to original image size
            from scipy.ndimage import zoom
            masks_list = []
            for m in masks_raw:
                if m.shape[0] != H or m.shape[1] != W:
                    scale_h = H / m.shape[0]
                    scale_w = W / m.shape[1]
                    m_resized = zoom(m, (scale_h, scale_w), order=0) > 0.5
                else:
                    m_resized = m > 0.5
                masks_list.append(m_resized)
            has_masks = True
        else:
            has_masks = False

        # ── Filter ───────────────────────────────────────────────────
        keep = []
        for i in range(len(bboxes)):
            # Skip scene-level labels
            if _is_scene_level(labels[i]):
                continue
            # Skip detections covering too much of the image
            box_area = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
            if box_area / img_area > args.max_bbox_area_ratio:
                continue
            keep.append(i)

        bboxes = bboxes[keep]
        confidences = confidences[keep]
        labels = [labels[i] for i in keep]
        if has_masks:
            masks_list = [masks_list[i] for i in keep]

        # Add background classes
        if args.add_bg_classes:
            for bg_cls in BG_CLASSES:
                bboxes = np.vstack([bboxes, [[0, 0, W, H]]]) if len(bboxes) > 0 else np.array([[0, 0, W, H]], dtype=np.float32)
                confidences = np.append(confidences, 1.0)
                labels.append(bg_cls)
                if has_masks:
                    masks_list.append(np.ones((H, W), dtype=bool))

        if len(labels) == 0:
            _save_empty(det_save, frame_clip_feat)
            continue

        # Build detection arrays
        classes = sorted(set(labels))
        global_classes.update(classes)
        class_to_id = {c: i for i, c in enumerate(classes)}
        xyxy = np.array(bboxes, dtype=np.float32)
        class_id = np.array([class_to_id[l] for l in labels], dtype=np.int64)
        n_det = len(xyxy)

        # Build masks array
        if has_masks:
            masks_np = np.stack(masks_list, axis=0)
        else:
            # Fallback to box masks (shouldn't happen with -seg model)
            masks_np = np.zeros((n_det, H, W), dtype=bool)
            for i in range(n_det):
                x1, y1, x2, y2 = map(int, xyxy[i])
                masks_np[i, y1:y2, x1:x2] = True

        # ── CLIP features per crop ───────────────────────────────────
        image_feats = np.zeros((n_det, 1024), dtype=np.float32)
        text_feats = np.zeros((n_det, 1024), dtype=np.float32)
        for i in range(n_det):
            x1, y1, x2, y2 = map(int, xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_pil.crop((x1, y1, x2, y2))
            crop_input = clip_preprocess(crop).unsqueeze(0).to(device)
            feat = clip_model.encode_image(crop_input)
            feat = F.normalize(feat, dim=-1)
            image_feats[i] = feat.cpu().numpy().squeeze()

            tokens = clip_tokenizer([labels[i]]).to(device)
            tfeat = clip_model.encode_text(tokens)
            tfeat = F.normalize(tfeat, dim=-1)
            text_feats[i] = tfeat.cpu().numpy().squeeze()

        # ── Save pkl.gz ──────────────────────────────────────────────
        result = {
            "xyxy": xyxy,
            "confidence": np.array(confidences, dtype=np.float32),
            "class_id": class_id,
            "mask": masks_np,
            "classes": classes,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "frame_clip_feat": frame_clip_feat,
            "tagging_caption": "yoloe_pf",
            "tagging_text_prompt": ", ".join(labels),
        }
        with gzip.open(str(det_save), "wb") as f:
            pickle.dump(result, f)

    # Save class list
    cls_path = scene_path / f"gsa_classes_{save_tag}.json"
    cls_path.write_text(json.dumps(sorted(global_classes), indent=2))
    print(f"\nDone. {len(global_classes)} unique classes → {cls_path}")


def _save_empty(path: Path, frame_clip_feat: np.ndarray) -> None:
    result = {
        "xyxy": np.zeros((0, 4), dtype=np.float32),
        "confidence": np.zeros(0, dtype=np.float32),
        "class_id": np.zeros(0, dtype=np.int64),
        "mask": np.zeros((0, 1, 1), dtype=bool),
        "classes": [],
        "image_feats": np.zeros((0, 1024), dtype=np.float32),
        "text_feats": np.zeros((0, 1024), dtype=np.float32),
        "frame_clip_feat": frame_clip_feat,
        "tagging_caption": "",
        "tagging_text_prompt": "",
    }
    with gzip.open(str(path), "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
