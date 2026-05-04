"""Batched CLIP scoring for VG visible-proposal crop augmentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass(frozen=True)
class ClipImageRequest:
    scene_id: str
    proposal_id: int
    frame_id: int
    raw_rgb_path: Path
    bbox_2d: tuple[int, int, int, int]
    visibility_weight: float = 1.0


@dataclass(frozen=True)
class ClipScore:
    proposal_id: int
    frame_id: int
    clip_score: float
    raw_rgb_path: Path
    bbox_2d: tuple[int, int, int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchedClipProvider:
    """Lazy open_clip provider with per-process text/image embedding caches."""

    def __init__(
        self,
        *,
        backbone: str = "ViT-H-14/laion2b_s32b_b79k",
        cache_dir: str | Path | None = None,
        batch_size: int = 16,
        dtype: str = "fp16",
        device: str | None = None,
    ) -> None:
        self.backbone = backbone
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model: Any | None = None
        self._preprocess: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None
        self._np: Any | None = None
        self._image_cache: dict[tuple[Any, ...], Any] = {}
        self._image_failure_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._text_cache: dict[str, Any] = {}
        self.last_failures: list[dict[str, Any]] = []

    def score(
        self,
        category: str,
        requests: list[ClipImageRequest] | tuple[ClipImageRequest, ...],
    ) -> list[ClipScore]:
        """Return cosine scores for category text against proposal crop requests."""
        if not requests:
            return []

        self.last_failures = []
        torch = self._torch_module()
        text_embedding = self._text_embedding(category).to(torch.float32)
        image_embeddings = self._image_embeddings(list(requests))

        scores: list[ClipScore] = []
        for request, image_embedding in zip(requests, image_embeddings, strict=True):
            if image_embedding is None:
                continue
            score = float((image_embedding.to(torch.float32) * text_embedding).sum())
            scores.append(
                ClipScore(
                    proposal_id=request.proposal_id,
                    frame_id=request.frame_id,
                    clip_score=score,
                    raw_rgb_path=request.raw_rgb_path,
                    bbox_2d=request.bbox_2d,
                )
            )
        return scores

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import numpy as np
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError(
                "CVRA clip_visible augmentation requires numpy, torch, and open_clip."
            ) from exc

        model_name, pretrained = _split_backbone(self.backbone)
        resolved_device = self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=resolved_device,
        )
        model.eval()
        if self.dtype == "fp16" and resolved_device.startswith("cuda"):
            model = model.half()

        self.device = resolved_device
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._torch = torch
        self._np = np

    def _torch_module(self) -> Any:
        self._load()
        return self._torch

    def _text_embedding(self, category: str) -> Any:
        self._load()
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        assert self._tokenizer is not None

        key = " ".join(category.strip().lower().split())
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached

        tokens = self._tokenizer([category]).to(self.device)
        with torch.no_grad():
            embedding = self._model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
        cached_embedding = embedding[0].detach().cpu().to(torch.float16)
        self._text_cache[key] = cached_embedding
        return cached_embedding

    def _image_embeddings(self, requests: list[ClipImageRequest]) -> list[Any]:
        self._load()
        torch = self._torch
        assert torch is not None
        assert self._model is not None

        missing: list[tuple[ClipImageRequest, tuple[Any, ...]]] = []
        for request in requests:
            key = self._image_cache_key(request)
            if key in self._image_cache:
                failure = self._image_failure_cache.get(key)
                if self._image_cache[key] is None and failure is not None:
                    self.last_failures.append(dict(failure))
                continue
            disk_embedding = self._load_disk_embedding(key)
            if disk_embedding is not None:
                self._image_cache[key] = disk_embedding
                continue
            missing.append((request, key))

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            tensors_with_keys: list[tuple[Any, tuple[Any, ...]]] = []
            for request, key in batch:
                try:
                    tensors_with_keys.append((self._crop_tensor(request), key))
                except (ValueError, OSError) as exc:
                    self._record_crop_failure(request, key, exc)
                    self._image_cache[key] = None
            if not tensors_with_keys:
                continue

            tensors = [tensor for tensor, _ in tensors_with_keys]
            images = torch.stack(tensors).to(self.device)
            if self.dtype == "fp16" and str(self.device).startswith("cuda"):
                images = images.half()
            with torch.no_grad():
                embeddings = self._model.encode_image(images)
                embeddings = embeddings / embeddings.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            for (_, key), embedding in zip(tensors_with_keys, embeddings, strict=True):
                cached_embedding = embedding.detach().cpu().to(torch.float16)
                self._image_cache[key] = cached_embedding
                self._save_disk_embedding(key, cached_embedding)

        return [
            self._image_cache[self._image_cache_key(request)] for request in requests
        ]

    def _crop_tensor(self, request: ClipImageRequest) -> Any:
        self._load()
        assert self._preprocess is not None

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "CVRA clip_visible augmentation requires Pillow."
            ) from exc

        with Image.open(request.raw_rgb_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            x1, y1, x2, y2 = request.bbox_2d
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Invalid bbox_2d for proposal {request.proposal_id}: "
                    f"{request.bbox_2d}"
                )
            pad_x = max(1, int(round((x2 - x1) * 0.10)))
            pad_y = max(1, int(round((y2 - y1) * 0.10)))
            crop_box = (
                max(0, x1 - pad_x),
                max(0, y1 - pad_y),
                min(width, x2 + pad_x),
                min(height, y2 + pad_y),
            )
            crop = rgb.crop(crop_box)
            return self._preprocess(crop)

    def _image_cache_key(self, request: ClipImageRequest) -> tuple[Any, ...]:
        try:
            stat = request.raw_rgb_path.stat()
            st_mtime_ns = stat.st_mtime_ns
            st_size = stat.st_size
        except OSError as exc:
            logger.warning(
                "[CVRA] raw_rgb_path stat failed for proposal {} frame {} at {}: {}",
                request.proposal_id,
                request.frame_id,
                request.raw_rgb_path,
                exc,
            )
            st_mtime_ns = 0
            st_size = 0
        return (
            request.scene_id,
            request.proposal_id,
            request.frame_id,
            str(request.raw_rgb_path),
            request.bbox_2d,
            st_mtime_ns,
            st_size,
        )

    def _record_crop_failure(
        self,
        request: ClipImageRequest,
        key: tuple[Any, ...],
        exc: Exception,
    ) -> None:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "[CVRA] dropping proposal {} frame {} crop from {}: {}",
            request.proposal_id,
            request.frame_id,
            request.raw_rgb_path,
            error,
        )
        failure = {
            "proposal_id": request.proposal_id,
            "frame_id": request.frame_id,
            "raw_rgb_path": str(request.raw_rgb_path),
            "source": "clip_visible",
            "metadata": {"crop_failed": True, "error": error},
        }
        self._image_failure_cache[key] = failure
        self.last_failures.append(dict(failure))

    def _disk_path(self, key: tuple[Any, ...]) -> Path | None:
        if self.cache_dir is None:
            return None
        serialized = json.dumps(key, sort_keys=True, default=str)
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _load_disk_embedding(self, key: tuple[Any, ...]) -> Any | None:
        path = self._disk_path(key)
        if path is None or not path.exists():
            return None
        self._load()
        assert self._np is not None
        assert self._torch is not None
        return self._torch.from_numpy(self._np.load(path))

    def _save_disk_embedding(self, key: tuple[Any, ...], embedding: Any) -> None:
        path = self._disk_path(key)
        if path is None:
            return
        self._load()
        assert self._np is not None
        self._np.save(path, embedding.numpy())


def _split_backbone(backbone: str) -> tuple[str, str]:
    if "/" in backbone:
        model_name, pretrained = backbone.split("/", 1)
        if model_name and pretrained:
            return model_name, pretrained
    if backbone == "ViT-B-32":
        return backbone, "openai"
    raise ValueError(
        "clip_visible_backbone must be '<open_clip_model>/<pretrained>' "
        "or the macOS smoke shorthand 'ViT-B-32'."
    )


__all__ = ["BatchedClipProvider", "ClipImageRequest", "ClipScore"]
