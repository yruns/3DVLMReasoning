from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

# ---------------------------------------------------------------------------
# SAM predictor loading
# ---------------------------------------------------------------------------


def get_sam_predictor(cfg) -> SamPredictor:
    """Build a SAM predictor from the given config."""
    if cfg.sam_variant == "sam":
        sam = sam_model_registry[cfg.sam_encoder_version](
            checkpoint=cfg.sam_checkpoint_path
        )
        sam.to(cfg.device)
        return SamPredictor(sam)

    if cfg.sam_variant == "mobilesam":
        from MobileSAM.setup_mobile_sam import setup_model

        checkpoint = torch.load(cfg.mobile_sam_path)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=cfg.device)
        return SamPredictor(mobile_sam)

    if cfg.sam_variant == "lighthqsam":
        from LightHQSAM.setup_light_hqsam import setup_model

        checkpoint_path = cfg.lighthqsam_checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=cfg.device)
        return SamPredictor(light_hqsam)

    raise NotImplementedError(f"SAM variant '{cfg.sam_variant}' is not supported")


# ---------------------------------------------------------------------------
# SAM segmentation from bounding boxes
# ---------------------------------------------------------------------------


def get_sam_segmentation_from_xyxy_batched(
    sam_predictor: SamPredictor,
    image: np.ndarray,
    xyxy_tensor: torch.Tensor,
) -> torch.Tensor:
    """Prompt SAM with detection boxes in a batch."""
    sam_predictor.set_image(image)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        xyxy_tensor, image.shape[:2]
    )
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.squeeze()


def get_sam_segmentation_from_xyxy(
    sam_predictor: SamPredictor,
    image: np.ndarray,
    xyxy: np.ndarray,
) -> np.ndarray:
    """Prompt SAM with detection boxes one-by-one."""
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        result_masks.append(masks[np.argmax(scores)])
    return np.array(result_masks)


# ---------------------------------------------------------------------------
# CLIP feature extraction
# ---------------------------------------------------------------------------

_CROP_PADDING = 20


def _padded_crop(
    image: Image.Image, xyxy: np.ndarray, padding: int = _CROP_PADDING
) -> Image.Image:
    """Crop a detection box with clamped padding."""
    x_min, y_min, x_max, y_max = xyxy
    w, h = image.size
    x_min = x_min - min(padding, x_min)
    y_min = y_min - min(padding, y_min)
    x_max = x_max + min(padding, w - x_max)
    y_max = y_max + min(padding, h - y_max)
    return image.crop((x_min, y_min, x_max, y_max))


def compute_clip_features(
    image: np.ndarray,
    detections,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    classes: list[str],
    device: str | torch.device,
) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    """Extract per-detection CLIP image and text features (unbatched)."""
    pil_image = Image.fromarray(image)

    image_crops: list[Image.Image] = []
    image_feats: list[np.ndarray] = []
    text_feats: list[np.ndarray] = []

    for idx in range(len(detections.xyxy)):
        crop = _padded_crop(pil_image, detections.xyxy[idx])

        prep = clip_preprocess(crop).unsqueeze(0).to(device)
        crop_ft = clip_model.encode_image(prep)
        crop_ft = crop_ft / crop_ft.norm(dim=-1, keepdim=True)

        class_id = detections.class_id[idx]
        tokens = clip_tokenizer([classes[class_id]]).to(device)
        text_ft = clip_model.encode_text(tokens)
        text_ft = text_ft / text_ft.norm(dim=-1, keepdim=True)

        image_crops.append(crop)
        image_feats.append(crop_ft.cpu().numpy())
        text_feats.append(text_ft.cpu().numpy())

    return (
        image_crops,
        np.concatenate(image_feats, axis=0),
        np.concatenate(text_feats, axis=0),
    )


def compute_clip_features_batched(
    image: np.ndarray,
    detections,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    classes: list[str],
    device: str | torch.device,
) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    """Extract per-detection CLIP image and text features (batched)."""
    pil_image = Image.fromarray(image)

    image_crops: list[Image.Image] = []
    preprocessed: list[torch.Tensor] = []
    text_tokens: list[str] = []

    for idx in range(len(detections.xyxy)):
        crop = _padded_crop(pil_image, detections.xyxy[idx])
        preprocessed.append(clip_preprocess(crop).unsqueeze(0))
        text_tokens.append(classes[detections.class_id[idx]])
        image_crops.append(crop)

    images_batch = torch.cat(preprocessed, dim=0).to(device)
    tokens_batch = clip_tokenizer(text_tokens).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(images_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = clip_model.encode_text(tokens_batch)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return (
        image_crops,
        image_features.cpu().numpy(),
        text_features.cpu().numpy(),
    )
