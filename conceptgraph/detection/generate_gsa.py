"""Extract Grounded SAM (RAM + GroundingDINO + SAM) detections on a
posed RGB-D dataset.  Results are saved as compressed pickles under
the scene directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import imageio
import matplotlib
import numpy as np
import open_clip
import supervision as sv
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TS
from PIL import Image
from tqdm import trange

from conceptgraph.dataset.loader import get_dataset
from conceptgraph.utils.clip import compute_clip_features
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption

matplotlib.use("Agg")

try:
    from groundingdino.util.inference import Model
    from segment_anything import (
        SamAutomaticMaskGenerator,
        SamPredictor,
        sam_model_registry,
    )
except ImportError as e:
    raise ImportError(
        "Please install Grounded Segment Anything following the "
        "instructions in README."
    ) from e

# ---------------------------------------------------------------------------
# Paths derived from GSA_PATH
# ---------------------------------------------------------------------------
GSA_PATH = os.environ.get("GSA_PATH")
if GSA_PATH is None:
    raise ValueError("Set the GSA_PATH environment variable to the GSA repo root.")

TAG2TEXT_PATH = os.path.join(GSA_PATH, "")
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
sys.path.append(GSA_PATH)
sys.path.append(TAG2TEXT_PATH)
sys.path.append(EFFICIENTSAM_PATH)

try:
    from ram import inference_ram, inference_tag2text
    from ram.models import ram, tag2text
except ImportError as e:
    raise ImportError("RAM sub-package not found. Please check your GSA_PATH.") from e

torch.set_grad_enabled(False)

GROUNDING_DINO_CONFIG_PATH = os.path.join(
    GSA_PATH,
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "groundingdino_swint_ogc.pth")

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "sam_vit_h_4b8939.pth")

TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "ram_swin_large_14m.pth")
RAM_PLUS_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "ram_plus_swin_large_14m.pth")

FOREGROUND_GENERIC_CLASSES = [
    "item",
    "furniture",
    "object",
    "electronics",
    "wall decoration",
    "door",
]

FOREGROUND_MINIMAL_CLASSES = ["item"]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to the dataset YAML config file.",
    )
    parser.add_argument("--scene_id", type=str, default="train_3")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--desired_height", type=int, default=480)
    parser.add_argument("--desired_width", type=int, default=640)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument(
        "--class_set",
        type=str,
        default="scene",
        choices=["scene", "generic", "minimal", "tag2text", "ram", "ram_plus", "none"],
        help=(
            "If 'none', tagging/detection is skipped and SAM runs in "
            "dense sampling mode."
        ),
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="dino",
        choices=["yolo", "dino"],
        help="Object detector backend.",
    )
    parser.add_argument(
        "--add_bg_classes",
        action="store_true",
        help="Add background classes (wall, floor, ceiling).",
    )
    parser.add_argument(
        "--accumu_classes",
        action="store_true",
        help="Accumulate the class set across frames.",
    )
    parser.add_argument(
        "--sam_variant",
        type=str,
        default="sam",
        choices=["fastsam", "mobilesam", "lighthqsam"],
    )
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--use_slow_vis",
        action="store_true",
        help="Use caption-annotated visualisation (ram/tag2text only).",
    )
    parser.add_argument(
        "--exp_suffix",
        type=str,
        default=None,
        help="Suffix appended to the output folder name.",
    )
    return parser


# ---------------------------------------------------------------------------
# SAM helpers
# ---------------------------------------------------------------------------


def get_sam_segmentation_from_xyxy(
    sam_predictor: SamPredictor,
    image: np.ndarray,
    xyxy: np.ndarray,
) -> np.ndarray:
    """Prompt SAM with bounding boxes and return best masks."""
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        result_masks.append(masks[np.argmax(scores)])
    return np.array(result_masks)


def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    """Instantiate a SAM predictor for the given variant."""
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        return SamPredictor(sam)

    if variant == "mobilesam":
        from MobileSAM.setup_mobile_sam import setup_model

        ckpt_path = os.path.join(GSA_PATH, "EfficientSAM/mobile_sam.pt")
        checkpoint = torch.load(ckpt_path)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        return SamPredictor(mobile_sam)

    if variant == "lighthqsam":
        from LightHQSAM.setup_light_hqsam import setup_model

        ckpt_path = os.path.join(GSA_PATH, "EfficientSAM/sam_hq_vit_tiny.pth")
        checkpoint = torch.load(ckpt_path)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)
        return SamPredictor(light_hqsam)

    raise NotImplementedError(f"SAM variant '{variant}' not supported")


def get_sam_segmentation_dense(
    variant: str,
    model: Any,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SAM automatic mask generation (no bbox prompting).

    Args:
        variant: SAM variant name.
        model: The mask generator (or YOLO model for fastsam).
        image: (H, W, 3) RGB image in [0, 255].

    Returns:
        Tuple of (masks, xyxy, confidence) arrays.
    """
    if variant == "sam":
        results = model.generate(image)
        masks, xyxy_list, confs = [], [], []
        for r in results:
            masks.append(r["segmentation"])
            box = r["bbox"].copy()
            box[2] += box[0]  # xywh -> xyxy
            box[3] += box[1]
            xyxy_list.append(box)
            confs.append(r["predicted_iou"])
        return (
            np.array(masks),
            np.array(xyxy_list),
            np.array(confs),
        )

    raise NotImplementedError(
        f"Dense segmentation for variant '{variant}' not implemented"
    )


def get_sam_mask_generator(
    variant: str, device: str | int
) -> SamAutomaticMaskGenerator:
    """Create a SAM automatic mask generator."""
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        return SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=12,
            points_per_batch=144,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=100,
        )

    raise NotImplementedError(f"Mask generator for variant '{variant}' not implemented")


# ---------------------------------------------------------------------------
# Tag / class processing
# ---------------------------------------------------------------------------


def process_tag_classes(
    text_prompt: str,
    add_classes: list[str] | None = None,
    remove_classes: list[str] | None = None,
) -> list[str]:
    """Convert a Tag2Text/RAM text prompt to a deduplicated class list."""
    add_classes = add_classes or []
    remove_classes = remove_classes or []

    classes = [c.strip() for c in text_prompt.split(",") if c.strip()]

    for c in add_classes:
        if c not in classes:
            classes.append(c)

    for c in remove_classes:
        classes = [cls for cls in classes if c not in cls.lower()]

    return classes


def process_ai2thor_classes(
    classes: list[str],
    add_classes: list[str] | None = None,
    remove_classes: list[str] | None = None,
) -> list[str]:
    """Pre-process AI2Thor objectType names into readable labels."""
    add_classes = add_classes or []
    remove_classes = remove_classes or []

    classes = list(set(classes))
    classes.extend(add_classes)

    for c in remove_classes:
        classes = [cls for cls in classes if c not in cls.lower()]

    # Split PascalCase into words (treat "TV" specially)
    classes = [cls.replace("TV", "Tv") for cls in classes]
    classes = [" ".join(re.findall("[A-Z][^A-Z]*", cls)) for cls in classes]
    return classes


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    # ---- GroundingDINO ----
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        device=args.device,
    )

    # ---- SAM ----
    if args.class_set == "none":
        mask_generator = get_sam_mask_generator(args.sam_variant, args.device)
    else:
        sam_predictor = get_sam_predictor(args.sam_variant, args.device)

    # ---- CLIP ----
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    # ---- Dataset ----
    dataset = get_dataset(
        dataconfig=args.dataset_config,
        start=args.start,
        end=args.end,
        stride=args.stride,
        basedir=args.dataset_root,
        sequence=args.scene_id,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
        device="cpu",
        dtype=torch.float,
    )

    global_classes: set[str] = set()

    # ---- YOLO-World (optional) ----
    if args.detector == "yolo":
        from ultralytics import YOLO

        yolo_model_w_classes = YOLO("yolov8l-world.pt")

    # ---- Determine initial class set ----
    caption = "NA"
    text_prompt = ""
    specified_tags = "None"
    tagging_model = None
    tagging_transform = None

    if args.class_set == "scene":
        obj_meta_path = args.dataset_root / args.scene_id / "obj_meta.json"
        with open(obj_meta_path) as f:
            obj_meta = json.load(f)
        classes: list[str] = process_ai2thor_classes(
            [obj["objectType"] for obj in obj_meta],
            remove_classes=["wall", "floor", "room", "ceiling"],
        )
    elif args.class_set == "generic":
        classes = list(FOREGROUND_GENERIC_CLASSES)
    elif args.class_set == "minimal":
        classes = list(FOREGROUND_MINIMAL_CLASSES)
    elif args.class_set in ("tag2text", "ram", "ram_plus"):
        if args.class_set == "tag2text":
            delete_tag_index = list(range(3012, 3429))
            specified_tags = "None"
            tagging_model = tag2text.tag2text_caption(
                pretrained=TAG2TEXT_CHECKPOINT_PATH,
                image_size=384,
                vit="swin_b",
                delete_tag_index=delete_tag_index,
            )
            tagging_model.threshold = 0.64
        elif args.class_set == "ram_plus":
            from ram.models import ram_plus
            tagging_model = ram_plus(
                pretrained=RAM_PLUS_CHECKPOINT_PATH,
                image_size=384,
                vit="swin_l",
            )
        else:
            tagging_model = ram(
                pretrained=RAM_CHECKPOINT_PATH,
                image_size=384,
                vit="swin_l",
            )

        tagging_model = tagging_model.eval().to(args.device)
        tagging_transform = TS.Compose(
            [
                TS.Resize((384, 384)),
                TS.ToTensor(),
                TS.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        classes = []  # populated per-frame below
    elif args.class_set == "none":
        classes = ["item"]
    else:
        raise ValueError(f"Unknown class_set: {args.class_set}")

    if args.class_set not in ("ram", "tag2text"):
        print(f"Total classes to detect: {len(classes)}")
    elif args.class_set == "none":
        print("Skipping tagging and detection models.")
    else:
        print(f"{args.class_set} will be used to detect classes.")

    # ---- Output naming ----
    save_name = args.class_set
    if args.sam_variant != "sam":
        save_name += f"_{args.sam_variant}"
    if args.exp_suffix:
        save_name += f"_{args.exp_suffix}"

    scene_dir = args.dataset_root / args.scene_id
    frames: list[np.ndarray] = []

    for idx in trange(len(dataset)):
        color_path = Path(dataset.color_paths[idx])

        vis_dir = scene_dir / f"gsa_vis_{save_name}"
        det_dir = scene_dir / f"gsa_detections_{save_name}"
        vis_save_path = vis_dir / color_path.name
        det_save_path = (det_dir / color_path.name).with_suffix(".pkl.gz")

        vis_dir.mkdir(parents=True, exist_ok=True)
        det_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(color_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Frame-level CLIP feature
        with torch.no_grad():
            clip_input = clip_preprocess(image_pil).unsqueeze(0).to(args.device)
            frame_clip_feat = clip_model.encode_image(clip_input)
            frame_clip_feat = F.normalize(frame_clip_feat, dim=-1)
            frame_clip_feat = frame_clip_feat.cpu().numpy().squeeze()

        # Per-frame tagging (RAM / Tag2Text)
        if args.class_set in ("ram", "ram_plus", "tag2text"):
            raw_image = image_pil.resize((384, 384))
            raw_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)

            if args.class_set in ("ram", "ram_plus"):
                res = inference_ram(raw_image, tagging_model)
                caption = "NA"
            else:
                res = inference_tag2text.inference(
                    raw_image, tagging_model, specified_tags
                )
                caption = res[2]

            text_prompt = res[0].replace(" |", ",")

            add_cls = ["other item"]
            remove_cls = [
                "room",
                "kitchen",
                "office",
                "house",
                "home",
                "building",
                "corner",
                "shadow",
                "carpet",
                "photo",
                "shade",
                "stall",
                "space",
                "aquarium",
                "apartment",
                "image",
                "city",
                "blue",
                "skylight",
                "hallway",
                "bureau",
                "modern",
                "salon",
                "doorway",
                "wall lamp",
                "wood floor",
            ]
            bg_cls = ["wall", "floor", "ceiling"]

            if args.add_bg_classes:
                add_cls += bg_cls
            else:
                remove_cls += bg_cls

            classes = process_tag_classes(
                text_prompt,
                add_classes=add_cls,
                remove_classes=remove_cls,
            )

        global_classes.update(classes)
        if args.accumu_classes:
            classes = list(global_classes)

        # ---- Detection & segmentation ----
        if args.class_set == "none":
            mask, xyxy, conf = get_sam_segmentation_dense(
                args.sam_variant, mask_generator, image_rgb
            )
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb,
                detections,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
                classes,
                args.device,
            )
            annotated_image, labels = vis_result_fast(
                image, detections, classes, instance_random_color=True
            )
            cv2.imwrite(str(vis_save_path), annotated_image)
        else:
            if args.detector == "dino":
                detections = grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=classes,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )

                if len(detections.class_id) > 0:
                    # NMS
                    nms_idx = (
                        torchvision.ops.nms(
                            torch.from_numpy(detections.xyxy),
                            torch.from_numpy(detections.confidence),
                            args.nms_threshold,
                        )
                        .numpy()
                        .tolist()
                    )
                    detections.xyxy = detections.xyxy[nms_idx]
                    detections.confidence = detections.confidence[nms_idx]
                    detections.class_id = detections.class_id[nms_idx]

                    # Remove invalid class IDs
                    valid = detections.class_id != -1
                    detections.xyxy = detections.xyxy[valid]
                    detections.confidence = detections.confidence[valid]
                    detections.class_id = detections.class_id[valid]

            elif args.detector == "yolo":
                yolo_model_w_classes.set_classes(classes)
                yolo_results = yolo_model_w_classes.predict(str(color_path))
                yolo_results[0].save(
                    str(vis_save_path).removesuffix(".png") + "_yolo_out.jpg"
                )
                boxes = yolo_results[0].boxes
                detections = sv.Detections(
                    xyxy=boxes.xyxy.cpu().numpy(),
                    confidence=boxes.conf.cpu().numpy(),
                    class_id=boxes.cls.cpu().numpy().astype(int),
                    mask=None,
                )

            if len(detections.class_id) > 0:
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy,
                )
                image_crops, image_feats, text_feats = compute_clip_features(
                    image_rgb,
                    detections,
                    clip_model,
                    clip_preprocess,
                    clip_tokenizer,
                    classes,
                    args.device,
                )
            else:
                image_crops, image_feats, text_feats = [], [], []

            annotated_image, labels = vis_result_fast(image, detections, classes)

            if args.class_set in ("ram", "tag2text") and args.use_slow_vis:
                annotated_caption = vis_result_slow_caption(
                    image_rgb,
                    detections.mask,
                    detections.xyxy,
                    labels,
                    caption,
                    text_prompt,
                )
                Image.fromarray(annotated_caption).save(str(vis_save_path))
            else:
                cv2.imwrite(str(vis_save_path), annotated_image)

        if args.save_video:
            frames.append(annotated_image)

        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "frame_clip_feat": frame_clip_feat,
        }

        if args.class_set in ("ram", "ram_plus", "tag2text"):
            results["tagging_caption"] = caption
            results["tagging_text_prompt"] = text_prompt

        with gzip.open(det_save_path, "wb") as f:
            pickle.dump(results, f)

    # Save accumulated class list
    cls_path = scene_dir / f"gsa_classes_{save_name}.json"
    with open(cls_path, "w") as f:
        json.dump(list(global_classes), f)

    if args.save_video:
        video_path = scene_dir / f"gsa_vis_{save_name}.mp4"
        imageio.mimsave(str(video_path), frames, fps=10)
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
