"""Unit tests for BatchedClipProvider internals."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.packs.vg_embodiedscan.clip_provider import (
    BatchedClipProvider,
    ClipImageRequest,
    _split_backbone,
)


def _request(path: Path, *, proposal_id: int = 1) -> ClipImageRequest:
    return ClipImageRequest(
        scene_id="scene_test",
        proposal_id=proposal_id,
        frame_id=10,
        raw_rgb_path=path,
        bbox_2d=(10, 20, 30, 50),
        visibility_weight=1.0,
    )


def test_split_backbone_accepts_canonical_and_macos_shorthand() -> None:
    assert _split_backbone("ViT-H-14/laion2b_s32b_b79k") == (
        "ViT-H-14",
        "laion2b_s32b_b79k",
    )
    assert _split_backbone("ViT-B-32") == ("ViT-B-32", "openai")
    with pytest.raises(ValueError, match="clip_visible_backbone"):
        _split_backbone("not-a-valid-backbone")


def test_image_cache_key_is_stable_for_same_missing_path(tmp_path: Path) -> None:
    provider = BatchedClipProvider()
    missing = tmp_path / "missing.png"

    key1 = provider._image_cache_key(_request(missing))
    key2 = provider._image_cache_key(_request(missing))

    assert key1 == key2
    assert str(missing) in key1


def test_text_embedding_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    provider = BatchedClipProvider()
    calls = {"encode_text": 0}

    class _Tokens:
        def to(self, device):  # noqa: ANN001
            return self

    class _Model:
        def encode_text(self, tokens):  # noqa: ANN001
            calls["encode_text"] += 1
            return torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    provider._torch = torch
    provider._model = _Model()
    provider._tokenizer = lambda texts: _Tokens()
    monkeypatch.setattr(provider, "_load", lambda: None)

    emb1 = provider._text_embedding(" Desk ")
    emb2 = provider._text_embedding("desk")

    assert calls["encode_text"] == 1
    assert torch.equal(emb1, emb2)


def test_crop_tensor_applies_padding_and_clamps(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    provider = BatchedClipProvider()
    provider._preprocess = lambda image: image.size
    monkeypatch.setattr(provider, "_load", lambda: None)

    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (100, 100), color="white").save(rgb)
    req = _request(rgb)

    assert provider._crop_tensor(req) == (24, 36)


def test_score_skips_bad_bbox_and_records_crop_failed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    provider = BatchedClipProvider(batch_size=4)
    good = tmp_path / "good.png"
    good.write_bytes(b"good")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"bad")

    good_req = _request(good, proposal_id=1)
    bad_req = ClipImageRequest(
        scene_id="scene_test",
        proposal_id=2,
        frame_id=10,
        raw_rgb_path=bad,
        bbox_2d=(10, 10, 10, 20),
    )

    class _Tokens:
        def to(self, device):  # noqa: ANN001
            return self

    class _Model:
        def encode_text(self, tokens):  # noqa: ANN001
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        def encode_image(self, images):  # noqa: ANN001
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    provider._torch = torch
    provider._model = _Model()
    provider._tokenizer = lambda texts: _Tokens()
    provider._preprocess = lambda crop: torch.tensor([1.0, 0.0], dtype=torch.float32)
    monkeypatch.setattr(provider, "_load", lambda: None)

    def fake_crop_tensor(request: ClipImageRequest):
        if request.proposal_id == 2:
            raise ValueError("bad bbox")
        return torch.tensor([1.0, 0.0], dtype=torch.float32)

    monkeypatch.setattr(provider, "_crop_tensor", fake_crop_tensor)

    scores = provider.score("desk", [good_req, bad_req])

    assert [score.proposal_id for score in scores] == [1]
    assert provider.last_failures == [
        {
            "proposal_id": 2,
            "frame_id": 10,
            "raw_rgb_path": str(bad),
            "source": "clip_visible",
            "metadata": {"crop_failed": True, "error": "ValueError: bad bbox"},
        }
    ]

    repeat_scores = provider.score("desk", [good_req, bad_req])
    assert [score.proposal_id for score in repeat_scores] == [1]
    assert provider.last_failures[0]["metadata"]["crop_failed"] is True


def test_disk_cache_round_trip(tmp_path: Path, monkeypatch) -> None:
    import torch

    provider = BatchedClipProvider(cache_dir=tmp_path / "cache")
    provider._torch = torch
    import numpy as np

    provider._np = np
    monkeypatch.setattr(provider, "_load", lambda: None)
    key = ("scene", 1, 2, "path", (1, 2, 3, 4), 0, 0)
    embedding = torch.tensor([0.25, 0.75], dtype=torch.float16)

    provider._save_disk_embedding(key, embedding)
    loaded = provider._load_disk_embedding(key)

    assert loaded is not None
    assert torch.equal(loaded, embedding)
