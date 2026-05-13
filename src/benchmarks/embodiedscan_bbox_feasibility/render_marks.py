"""Render set-of-marks annotated keyframes."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_COLORS = [
    (230, 48, 48),
    (30, 180, 80),
    (40, 105, 235),
    (235, 185, 20),
    (190, 55, 220),
    (30, 200, 210),
    (245, 130, 25),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _fit_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if _text_size(draw, text, font)[0] <= max_width:
        return text
    suffix = "..."
    if _text_size(draw, suffix, font)[0] > max_width:
        return ""
    fitted = text
    while fitted and _text_size(draw, fitted + suffix, font)[0] > max_width:
        fitted = fitted[:-1].rstrip()
    return fitted + suffix if fitted else suffix


def render_marked_keyframe(
    *,
    rgb_path: Path,
    out_path: Path,
    marks: list[dict],
    font_size: int = 30,
    line_width: int = 6,
    include_legend: bool = False,
) -> None:
    """marks: list of {proposal_id, label, bbox_2d=(x1,y1,x2,y2)}."""
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    for i, m in enumerate(marks):
        color = _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = m["bbox_2d"]
        draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=line_width + 3)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        tag = f"#{m['proposal_id']} {m['label']}".strip()
        tw, th = _text_size(draw, tag, font)
        pad_x = max(8, font_size // 3)
        pad_y = max(5, font_size // 6)
        max_badge_w = max(48, img.width - 2)
        tag = _fit_text_to_width(draw, tag, font, max_badge_w - 2 * pad_x)
        tw, th = _text_size(draw, tag, font)
        badge_w = tw + 2 * pad_x
        badge_h = th + 2 * pad_y
        badge_x1 = max(0, min(x1, img.width - badge_w))
        badge_y1 = max(0, y1 - badge_h)
        badge_x2 = badge_x1 + badge_w
        badge_y2 = badge_y1 + badge_h
        draw.rectangle(
            [(badge_x1, badge_y1), (badge_x2, badge_y2)],
            fill=(0, 0, 0),
            outline=color,
            width=max(2, line_width // 2),
        )
        draw.text(
            (badge_x1 + pad_x, badge_y1 + pad_y - 1),
            tag,
            fill="white",
            font=font,
        )

    if include_legend and marks:
        legend_font = _load_font(max(22, int(font_size * 0.85)))
        row_h = max(34, int(font_size * 1.35))
        pad = 14
        min_col_w = 270
        cols = max(1, min(3, img.width // min_col_w))
        rows = (len(marks) + cols - 1) // cols
        legend_h = pad * 2 + rows * row_h
        canvas = Image.new("RGB", (img.width, img.height + legend_h), (18, 18, 18))
        canvas.paste(img, (0, 0))
        legend_draw = ImageDraw.Draw(canvas)
        col_w = img.width // cols
        for i, m in enumerate(marks):
            row = i // cols
            col = i % cols
            x = col * col_w + pad
            y = img.height + pad + row * row_h
            color = _COLORS[i % len(_COLORS)]
            legend_draw.rectangle(
                [(x, y + 5), (x + 28, y + row_h - 7)],
                fill=color,
                outline="white",
                width=2,
            )
            legend_draw.text(
                (x + 40, y + 1),
                f"#{m['proposal_id']} {m['label']}",
                fill="white",
                font=legend_font,
            )
        img = canvas

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


__all__ = ["render_marked_keyframe"]
