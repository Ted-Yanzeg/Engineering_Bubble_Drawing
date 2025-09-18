# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

def draw_bubbles(img: Image.Image, items: List[Dict[str, Any]], radius: int = 16, text_scale: float = 1.2,
                 font_path: Optional[str] = None, anchor: str = "tr",
                 offset: Tuple[int, int] = (8, -8), avoid_overlap: bool = True) -> Image.Image:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    if font_path:
        try:
            font = ImageFont.truetype(font_path, size=max(1, int(radius * text_scale)))
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    centers: List[Tuple[float, float]] = []

    def anchor_xy(box, anchor, offset):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        ax, ay = x2, y1  # tr
        if anchor == "tl": ax, ay = x1, y1
        elif anchor == "bl": ax, ay = x1, y2
        elif anchor == "br": ax, ay = x2, y2
        return ax + offset[0], ay + offset[1]

    def nudge(x, y):
        if not avoid_overlap:
            return x, y
        step = radius * 1.1
        tries = 0
        while any(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 < 1.7 * radius for (x0, y0) in centers):
            x += step; y -= step
            tries += 1
            if tries > 12:
                break
        return x, y

    for idx, it in enumerate(items, start=1):
        x, y = anchor_xy(it["box"], anchor, offset)
        x, y = nudge(x, y)
        centers.append((x, y))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#e33", outline="#e33", width=2)
        # —— 改动：优先使用 item 自带的 bubble_id；没有则退回 enumerate(idx)
        bubble_num = it.get("bubble_id", idx)
        num = str(bubble_num)
        l, t, r, b = draw.textbbox((0, 0), num, font=font)
        tw, th = r - l, b - t
        draw.text((x - tw / 2, y - th / 2), num, font=font, fill="white")

    return canvas
