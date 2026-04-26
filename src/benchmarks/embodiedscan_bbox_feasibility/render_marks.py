"""Render set-of-marks annotated keyframes."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_COLORS = [
    (220, 50, 50), (50, 200, 80), (50, 100, 220), (220, 180, 30),
    (180, 50, 200), (50, 200, 200), (240, 130, 30),
]


def render_marked_keyframe(
    *,
    rgb_path: Path,
    out_path: Path,
    marks: list[dict],
    font_size: int = 18,
) -> None:
    """marks: list of {proposal_id, label, bbox_2d=(x1,y1,x2,y2)}."""
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    for i, m in enumerate(marks):
        color = _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = m["bbox_2d"]
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        tag = f"{m['proposal_id']}: {m['label']}"
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        draw.rectangle([(x1, y1 - th - 4), (x1 + tw + 6, y1)], fill=color)
        draw.text((x1 + 3, y1 - th - 3), tag, fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


__all__ = ["render_marked_keyframe"]
