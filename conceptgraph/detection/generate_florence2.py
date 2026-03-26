"""Florence-2 + SAM detection pipeline for ConceptGraph (v2 — hybrid).

Uses a two-pass strategy for high-recall open-vocabulary detection:

  1. ``<OD>`` — Florence-2 open-vocabulary detection (high precision, low recall)
  2. ``<CAPTION_TO_PHRASE_GROUNDING>`` — ground a broad indoor vocabulary against
     the image (high recall, may have duplicates)

Results are merged with IoU-based NMS to remove duplicates. SAM refines each
box into a pixel-accurate mask.

Output format is identical to ``generate_gsa.py`` so downstream 3-D mapping
(``conceptgraph.slam.pipeline``) works unchanged.

Usage::

    python conceptgraph/detection/generate_florence2.py \\
        --dataset_root /path/to/OpenEQA/scannet \\
        --dataset_config conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml \\
        --scene_id 002-scannet-scene0709_00/conceptgraph \\
        --stride 5 --add_bg_classes
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoProcessor

from conceptgraph.dataset.loader import get_dataset

# SAM (from Grounded-Segment-Anything) — MUST succeed, no fallback
from segment_anything import SamPredictor, sam_model_registry

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCENE_LEVEL_TAGS = {
    "house", "room", "apartment", "building", "space", "area",
    "kitchen", "bathroom", "bedroom", "office", "hallway",
    "living room", "dining room",
}

BG_CLASSES = ["wall", "floor", "ceiling"]

# ScanNet200 indoor vocabulary (200 categories), split into batches of ~10
# to avoid Florence-2's token-limit concatenation bug.
_SCANNET200_VOCAB_PATH = Path(__file__).parent.parent.parent / "conceptgraph" / "scannet200_classes.txt"

def _load_vocab_batches(path: Path | None = None, batch_size: int = 10) -> list[str]:
    """Load ScanNet200 labels and split into grounding batches."""
    if path is None:
        path = _SCANNET200_VOCAB_PATH
    if not path.exists():
        # Fallback: use hardcoded ScanNet200 list
        path = Path(__file__).resolve().parent.parent / "scannet200_classes.txt"
    labels = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    # Remove bg classes (handled separately) and scene-level tags
    labels = [l for l in labels if l.lower() not in _SCENE_LEVEL_TAGS
              and l.lower() not in {"wall", "floor", "ceiling"}]
    batches = []
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i + batch_size]
        batches.append(". ".join(batch) + ".")
    return batches

_INDOOR_VOCAB_BATCHES: list[str] = []  # populated at runtime

# IoU threshold for NMS deduplication across OD + grounding results
_NMS_IOU_THRESH = 0.7


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_scene_level(label: str) -> bool:
    return label.lower().strip() in _SCENE_LEVEL_TAGS


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_merge(
    bboxes: list[list[float]],
    labels: list[str],
    iou_thresh: float = _NMS_IOU_THRESH,
) -> tuple[list[list[float]], list[str]]:
    """Greedy NMS: keep first occurrence, suppress later duplicates.

    When two boxes overlap above threshold, keep the one with the shorter
    (more specific) label.
    """
    if not bboxes:
        return bboxes, labels

    n = len(bboxes)
    boxes_np = np.array(bboxes, dtype=np.float32)
    # Sort by box area ascending (prefer smaller, more specific detections)
    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
    order = np.argsort(areas)

    keep = []
    suppressed = set()
    for i_idx in range(n):
        i = order[i_idx]
        if i in suppressed:
            continue
        keep.append(i)
        for j_idx in range(i_idx + 1, n):
            j = order[j_idx]
            if j in suppressed:
                continue
            if _box_iou(boxes_np[i], boxes_np[j]) >= iou_thresh:
                suppressed.add(j)

    return [bboxes[i] for i in keep], [labels[i] for i in keep]


def _clean_grounding_label(raw: str) -> str:
    """Extract a clean noun from a grounding label.

    Florence-2 grounding sometimes returns multi-word phrases or the full
    sentence fragment. We take just the first noun phrase (before any period).
    """
    label = raw.strip().rstrip(".")
    # If it contains multiple items separated by ". ", it's a concatenation bug
    if ". " in label:
        label = label.split(". ")[0]
    return label.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Florence-2 + SAM detection (hybrid)")
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--dataset_config", type=str, required=True)
    p.add_argument("--scene_id", type=str, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--desired_height", type=int, default=480)
    p.add_argument("--desired_width", type=int, default=640)
    p.add_argument("--add_bg_classes", action="store_true")
    p.add_argument("--florence2_model", type=str,
                    default="microsoft/Florence-2-large")
    p.add_argument("--sam_checkpoint", type=str, default=None)
    p.add_argument("--sam_encoder", type=str, default="vit_h")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_vis", action="store_true")
    p.add_argument("--exp_suffix", type=str, default="")
    p.add_argument("--od_only", action="store_true",
                    help="Use only <OD> mode (no vocab grounding). For ablation.")
    p.add_argument("--clip_filter", type=float, default=0.0,
                    help="CLIP cosine similarity threshold for post-filtering. "
                         "0 = disabled. Recommended: 0.20-0.25.")
    p.add_argument("--vocab", type=str, default=None,
                    help="Path to vocab file (one label per line). "
                         "Default: conceptgraph/scannet200_classes.txt")
    return p


# ---------------------------------------------------------------------------
# Florence-2 inference helpers
# ---------------------------------------------------------------------------

def _florence2_infer(model, processor, image_pil, device, task, text_input=""):
    """Run a Florence-2 task and return the parsed result dict."""
    prompt = task + text_input
    inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(device)
    gen_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=task, image_size=(image_pil.width, image_pil.height),
    )
    return parsed.get(task, parsed)


def run_florence2_hybrid(
    model, processor, image_pil: Image.Image, device: str, od_only: bool = False,
) -> tuple[list[list[float]], list[str]]:
    """Two-pass Florence-2 detection: <OD> + vocabulary grounding.

    Returns merged, deduplicated (bboxes, labels).
    """
    # Pass 1: Open-vocabulary OD
    od = _florence2_infer(model, processor, image_pil, device, "<OD>")
    all_bboxes = list(od.get("bboxes", []))
    all_labels = list(od.get("labels", []))

    if od_only:
        return all_bboxes, all_labels

    # Pass 2: Vocabulary grounding (batched to avoid token-limit concatenation)
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    for vocab_batch in _INDOOR_VOCAB_BATCHES:
        result = _florence2_infer(
            model, processor, image_pil, device, task, vocab_batch,
        )
        for bbox, label in zip(result.get("bboxes", []), result.get("labels", [])):
            clean = _clean_grounding_label(label)
            if clean and not _is_scene_level(clean):
                all_bboxes.append(bbox)
                all_labels.append(clean)

    # Deduplicate
    all_bboxes, all_labels = _nms_merge(all_bboxes, all_labels)
    return all_bboxes, all_labels


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

    # ── Florence-2 ───────────────────────────────────────────────────
    print(f"Loading Florence-2 ({args.florence2_model}) …")
    f2_model = AutoModelForCausalLM.from_pretrained(
        args.florence2_model, trust_remote_code=True,
    ).to(device)
    f2_processor = AutoProcessor.from_pretrained(
        args.florence2_model, trust_remote_code=True,
    )

    # ── SAM (MUST load — no fallback to rectangles) ────────────────
    sam_ckpt = args.sam_checkpoint
    if sam_ckpt is None:
        sam_ckpt = os.path.join(
            os.environ.get("GSA_PATH", ""), "sam_vit_h_4b8939.pth",
        )
    if not os.path.isfile(sam_ckpt):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {sam_ckpt}\n"
            "Set --sam_checkpoint or GSA_PATH env var."
        )
    print(f"Loading SAM ({args.sam_encoder}) from {sam_ckpt} …")
    sam = sam_model_registry[args.sam_encoder](checkpoint=sam_ckpt)
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    # ── CLIP ─────────────────────────────────────────────────────────
    print("Loading CLIP …")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k",
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # ── Output dirs ──────────────────────────────────────────────────
    save_tag = "florence2"
    if args.add_bg_classes:
        save_tag += "_withbg"
    save_tag += "_allclasses"
    if args.exp_suffix:
        save_tag += f"_{args.exp_suffix}"
    det_dir = scene_path / f"gsa_detections_{save_tag}"
    det_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir = scene_path / f"gsa_vis_{save_tag}"
        vis_dir.mkdir(parents=True, exist_ok=True)

    global_classes: set[str] = set()

    # Load vocabulary batches
    global _INDOOR_VOCAB_BATCHES
    vocab_path = Path(args.vocab) if args.vocab else None
    _INDOOR_VOCAB_BATCHES = _load_vocab_batches(vocab_path)
    print(f"Loaded {sum(b.count('.') for b in _INDOOR_VOCAB_BATCHES)} vocab terms in {len(_INDOOR_VOCAB_BATCHES)} batches")

    mode = "OD-only" if args.od_only else "hybrid (OD + ScanNet200 vocab grounding)"
    print(f"Detection mode: {mode}")
    print(f"Processing {len(dataset)} frames …")

    # ── Per-frame processing ─────────────────────────────────────────
    for idx in trange(len(dataset)):
        color_path = Path(dataset.color_paths[idx])
        det_save = det_dir / color_path.with_suffix(".pkl.gz").name

        image_bgr = cv2.imread(str(color_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        H, W = image_bgr.shape[:2]

        # ── Frame-level CLIP feature ─────────────────────────────────
        clip_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
        frame_clip_feat = clip_model.encode_image(clip_input)
        frame_clip_feat = F.normalize(frame_clip_feat, dim=-1)
        frame_clip_feat = frame_clip_feat.cpu().numpy().squeeze()

        # ── Florence-2 detection (hybrid) ─────────────────────────────
        bboxes, labels = run_florence2_hybrid(
            f2_model, f2_processor, image_pil, device,
            od_only=args.od_only,
        )

        # Filter scene-level labels
        keep = [
            i for i, lbl in enumerate(labels)
            if not _is_scene_level(lbl)
        ]
        bboxes = [bboxes[i] for i in keep]
        labels = [labels[i] for i in keep]

        # Optionally add background classes
        if args.add_bg_classes:
            for bg_cls in BG_CLASSES:
                bboxes.append([0, 0, W, H])
                labels.append(bg_cls)

        if not bboxes:
            _save_empty(det_save, frame_clip_feat)
            continue

        # Build detection arrays
        classes = sorted(set(labels))
        global_classes.update(classes)
        class_to_id = {c: i for i, c in enumerate(classes)}
        xyxy = np.array(bboxes, dtype=np.float32)
        class_id = np.array([class_to_id[l] for l in labels], dtype=np.int64)
        n_det = len(xyxy)

        # ── SAM segmentation (always required) ───────────────────────
        sam_predictor.set_image(image_rgb)
        masks = []
        for box in xyxy:
            sam_masks, _, _ = sam_predictor.predict(
                box=box, multimask_output=True,
            )
            masks.append(sam_masks[0])
        masks_np = np.stack(masks, axis=0)  # (N, H, W)

        # ── Confidence (Florence-2 doesn't give scores; set 1.0) ─────
        confidence = np.ones(n_det, dtype=np.float32)

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

        # ── CLIP post-filtering (remove hallucinated detections) ────
        if args.clip_filter > 0:
            clip_sims = np.array([
                np.dot(image_feats[i], text_feats[i]) for i in range(n_det)
            ])
            clip_keep = clip_sims >= args.clip_filter
            # Always keep background classes
            for i in range(n_det):
                if labels[i] in BG_CLASSES:
                    clip_keep[i] = True

            if not clip_keep.all():
                keep_idx = np.where(clip_keep)[0]
                xyxy = xyxy[keep_idx]
                masks_np = masks_np[keep_idx]
                image_feats = image_feats[keep_idx]
                text_feats = text_feats[keep_idx]
                # Use CLIP similarity as confidence
                confidence = clip_sims[keep_idx].astype(np.float32)
                labels = [labels[i] for i in keep_idx]
                # Rebuild class arrays
                classes = sorted(set(labels))
                class_to_id = {c: i for i, c in enumerate(classes)}
                class_id = np.array([class_to_id[l] for l in labels], dtype=np.int64)
                n_det = len(xyxy)

        # ── Save pkl.gz ──────────────────────────────────────────────
        result = {
            "xyxy": xyxy,
            "confidence": confidence,
            "class_id": class_id,
            "mask": masks_np,
            "classes": classes,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "frame_clip_feat": frame_clip_feat,
            "tagging_caption": "florence2_hybrid",
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
