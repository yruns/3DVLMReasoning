"""SAM 3 (Segment Anything with Concepts) detection pipeline for ConceptGraph.

Uses SAM 3's Promptable Concept Segmentation (PCS) — a single unified model
that replaces the entire RAM + GroundingDINO + SAM pipeline.

Given a text prompt (e.g., "chair"), SAM 3 directly returns:
  - bounding boxes (xyxy)
  - instance segmentation masks
  - confidence scores (presence × detection)

We iterate over the ScanNet200 vocabulary, prompting SAM 3 once per category
per frame. This eliminates the error cascade of tag→ground→segment.

Output format is identical to ``generate_gsa.py`` so downstream 3-D mapping
(``conceptgraph.slam.pipeline``) works unchanged.

Requirements:
  - conda env ``sam3``: Python 3.12+, PyTorch 2.7+, CUDA 12.6+
  - ``pip install -e .`` from https://github.com/facebookresearch/sam3
  - Weights from ModelScope: ``modelscope download --model facebook/sam3``

Usage::

    python conceptgraph/detection/generate_sam3.py \\
        --scene_dir /path/to/OpenEQA/scannet/002-scannet-scene0709_00 \\
        --raw_subdir raw \\
        --out_subdir conceptgraph \\
        --stride 5 --add_bg_classes
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BG_CLASSES = ["wall", "floor", "ceiling"]

_SCENE_LEVEL_TAGS = {
    "house", "room", "apartment", "building", "space", "area",
    "kitchen", "bathroom", "bedroom", "office", "hallway",
    "living room", "dining room", "playroom", "darkness",
}

_SCANNET200_VOCAB_PATH = (
    Path(__file__).resolve().parent.parent / "scannet200_classes.txt"
)


def _load_vocab(path: Path | None = None) -> list[str]:
    """Load ScanNet200 labels, filtering scene-level and bg classes."""
    if path is None:
        path = _SCANNET200_VOCAB_PATH
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    labels = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    labels = [
        l for l in labels
        if l.lower() not in _SCENE_LEVEL_TAGS
        and l.lower() not in {"wall", "floor", "ceiling"}
    ]
    return labels


def _discover_frames(raw_dir: Path, stride: int) -> list[Path]:
    """Find RGB frames in raw_dir, sorted by frame number, strided."""
    pattern = re.compile(r"^(\d+)-rgb\.png$")
    frames = []
    for p in sorted(raw_dir.iterdir()):
        m = pattern.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))
    frames.sort(key=lambda x: x[0])
    # Apply stride
    strided = [p for i, (_, p) in enumerate(frames) if i % stride == 0]
    return strided


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAM 3 PCS detection pipeline")
    p.add_argument("--scene_dir", type=Path, required=True,
                    help="Scene root, e.g. .../002-scannet-scene0709_00")
    p.add_argument("--raw_subdir", type=str, default="raw",
                    help="Subdirectory containing *-rgb.png frames")
    p.add_argument("--out_subdir", type=str, default="conceptgraph",
                    help="Subdirectory for output detections")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--add_bg_classes", action="store_true")
    p.add_argument(
        "--conf_threshold", type=float, default=0.3,
        help="Minimum confidence (presence × detection) for SAM 3",
    )
    p.add_argument(
        "--max_bbox_area_ratio", type=float, default=0.8,
        help="Filter detections covering more than this ratio of image",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--checkpoint", type=str,
                    default="/home/ysh/models/sam3/facebook/sam3/sam3.pt",
                    help="Path to SAM 3 checkpoint (.pt or .safetensors)")
    p.add_argument("--vocab", type=Path, default=None,
                    help="Custom vocab file (default: scannet200_classes.txt)")
    p.add_argument("--exp_suffix", type=str, default="")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_parser().parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Discover frames ──────────────────────────────────────────────
    raw_dir = args.scene_dir / args.raw_subdir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    color_paths = _discover_frames(raw_dir, args.stride)
    if not color_paths:
        raise FileNotFoundError(f"No *-rgb.png frames in {raw_dir}")
    print(f"Found {len(color_paths)} frames (stride={args.stride}) in {raw_dir}")

    scene_out = args.scene_dir / args.out_subdir

    # ── Vocabulary ───────────────────────────────────────────────────
    vocab = _load_vocab(args.vocab)
    print(f"Vocabulary: {len(vocab)} categories")

    # ── SAM 3 ────────────────────────────────────────────────────────
    print(f"Loading SAM 3 from {args.checkpoint} …")
    sam3_model = build_sam3_image_model(
        device=device,
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
    )
    processor = Sam3Processor(sam3_model)
    processor.set_confidence_threshold(args.conf_threshold)
    print(f"  SAM 3 ready on {device} (conf≥{args.conf_threshold})")

    # ── CLIP ─────────────────────────────────────────────────────────
    print("Loading CLIP …")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k",
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # ── Output dirs ──────────────────────────────────────────────────
    save_tag = "sam3_sn200"
    if args.add_bg_classes:
        save_tag += "_withbg"
    if args.exp_suffix:
        save_tag += f"_{args.exp_suffix}"
    det_dir = scene_out / f"gsa_detections_{save_tag}"
    det_dir.mkdir(parents=True, exist_ok=True)

    global_classes: set[str] = set()

    print(f"Processing {len(color_paths)} frames × {len(vocab)} categories …")

    # ── Per-frame processing ─────────────────────────────────────────
    for color_path in tqdm(color_paths, desc="Frames"):
        det_save = det_dir / color_path.with_suffix(".pkl.gz").name

        image_rgb = cv2.imread(str(color_path))
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        H, W = image_rgb.shape[:2]
        img_area = H * W

        # ── Frame-level CLIP feature ─────────────────────────────────
        clip_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
        frame_clip_feat = clip_model.encode_image(clip_input)
        frame_clip_feat = F.normalize(frame_clip_feat, dim=-1)
        frame_clip_feat = frame_clip_feat.cpu().numpy().squeeze()

        # ── SAM 3: set image once, prompt per category ───────────────
        state = processor.set_image(image_pil)

        all_boxes = []
        all_masks = []
        all_scores = []
        all_labels = []

        for label in vocab:
            processor.reset_all_prompts(state)
            result = processor.set_text_prompt(label, state)

            boxes = result.get("boxes")
            masks = result.get("masks")
            scores = result.get("scores")

            if boxes is None or len(boxes) == 0:
                continue

            boxes_np = boxes.cpu().numpy()      # (N, 4) xyxy
            masks_np = masks.cpu().numpy()      # (N, 1, H, W) bool
            scores_np = scores.cpu().numpy()    # (N,)

            for i in range(len(boxes_np)):
                x1, y1, x2, y2 = boxes_np[i]
                box_area = (x2 - x1) * (y2 - y1)
                if box_area / img_area > args.max_bbox_area_ratio:
                    continue
                all_boxes.append(boxes_np[i])
                all_masks.append(masks_np[i, 0])
                all_scores.append(float(scores_np[i]))
                all_labels.append(label)

        # ── Add background classes ───────────────────────────────────
        if args.add_bg_classes:
            for bg_cls in BG_CLASSES:
                all_boxes.append(np.array([0, 0, W, H], dtype=np.float32))
                all_masks.append(np.ones((H, W), dtype=bool))
                all_scores.append(1.0)
                all_labels.append(bg_cls)

        if len(all_labels) == 0:
            _save_empty(det_save, frame_clip_feat)
            continue

        # ── Build detection arrays ───────────────────────────────────
        xyxy = np.array(all_boxes, dtype=np.float32)
        confidences = np.array(all_scores, dtype=np.float32)
        masks_arr = np.stack(all_masks, axis=0)  # (N, H, W)

        classes = sorted(set(all_labels))
        global_classes.update(classes)
        class_to_id = {c: i for i, c in enumerate(classes)}
        class_id = np.array([class_to_id[l] for l in all_labels], dtype=np.int64)
        n_det = len(xyxy)

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

            tokens = clip_tokenizer([all_labels[i]]).to(device)
            tfeat = clip_model.encode_text(tokens)
            tfeat = F.normalize(tfeat, dim=-1)
            text_feats[i] = tfeat.cpu().numpy().squeeze()

        # ── Save pkl.gz ──────────────────────────────────────────────
        det_result = {
            "xyxy": xyxy,
            "confidence": confidences,
            "class_id": class_id,
            "mask": masks_arr,
            "classes": classes,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "frame_clip_feat": frame_clip_feat,
            "tagging_caption": "sam3_pcs",
            "tagging_text_prompt": ", ".join(all_labels),
        }
        with gzip.open(str(det_save), "wb") as f:
            pickle.dump(det_result, f)

    # Save class list
    cls_path = scene_out / f"gsa_classes_{save_tag}.json"
    cls_path.write_text(json.dumps(sorted(global_classes), indent=2))
    print(f"\nDone. {len(global_classes)} unique classes → {cls_path}")


def _save_empty(path: Path | str, frame_clip_feat: np.ndarray) -> None:
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
