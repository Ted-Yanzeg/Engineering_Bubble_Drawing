# drawing.py
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import math, numpy as np

RGBA = Tuple[int, int, int, int]

def _ensure_rgba(c: Tuple[int, int, int] | RGBA, alpha: int | float = 255) -> RGBA:
    if len(c) == 4:
        r,g,b,a = c  # type: ignore
        return (int(r), int(g), int(b), int(a))
    r,g,b = c  # type: ignore
    if isinstance(alpha, float):
        a = int(max(0, min(1, alpha)) * 255)
    else:
        a = int(max(0, min(255, alpha)))
    return (int(r), int(g), int(b), a)

def _to_pil(img_any) -> Image.Image:
    """接受 PIL.Image 或 np.ndarray，统一转为 PIL.Image(RGB)。"""
    if isinstance(img_any, Image.Image):
        return img_any.convert("RGB")
    if isinstance(img_any, np.ndarray):
        if img_any.dtype != np.uint8:
            img_any = img_any.astype(np.uint8)
        if img_any.ndim == 2:
            return Image.fromarray(img_any, mode="L").convert("RGB")
        return Image.fromarray(img_any).convert("RGB")
    raise TypeError(f"draw_bubbles: unsupported image type: {type(img_any)}")

def draw_bubbles(
    img,  # 可传 PIL 或 numpy
    items: List[Dict[str, Any]],
    radius: int = 16,
    text_scale: float = 1.2,
    font_path: Optional[str] = None,
    anchor: str = "tr",
    offset: Tuple[int, int] = (8, -8),
    avoid_overlap: bool = True,
    # ------- 样式参数 -------
    show_boxes: bool = False,
    bubble_fill: Tuple[int,int,int] = (255, 0, 0),
    bubble_alpha: float | int = 0.65,
    number_color: Tuple[int,int,int] = (255, 255, 255),
    dash_color: Tuple[int,int,int] = (255, 0, 0),
    box_color: Tuple[int,int,int] = (0, 255, 0),
    box_alpha: float | int = 0.35,
) -> Image.Image:
    """
    渲染：
      1) 可选：OCR 四点多边形框
      2) 半透明气泡 + 虚线连边
      3) 气泡中心编号（it['bubble_id'] 优先）
    """
    base_rgb = _to_pil(img)
    base = base_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    odraw = ImageDraw.Draw(overlay, "RGBA")
    w, h = base.size

    # 字体
    font_size = max(10, int(radius * text_scale))
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    def centroid(box: List[Tuple[float,float]]) -> Tuple[float,float]:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        return (sum(xs)/4.0, sum(ys)/4.0)

    def nearest_point_on_edges(pt: Tuple[float,float], poly: List[Tuple[float,float]]) -> Tuple[float,float]:
        x0, y0 = pt
        best = None
        for i in range(4):
            x1,y1 = poly[i]
            x2,y2 = poly[(i+1)%4]
            vx, vy = x2-x1, y2-y1
            seg_len2 = vx*vx + vy*vy
            if seg_len2 <= 1e-6:
                proj = (x1, y1)
            else:
                t = ((x0-x1)*vx + (y0-y1)*vy) / seg_len2
                t = max(0.0, min(1.0, t))
                proj = (x1 + t*vx, y1 + t*vy)
            d2 = (proj[0]-x0)**2 + (proj[1]-y0)**2
            if (best is None) or (d2 < best[0]):
                best = (d2, proj)
        return best[1] if best else (poly[0][0], poly[0][1])

    def draw_dashed_line(d: ImageDraw.ImageDraw, p1: Tuple[float,float], p2: Tuple[float,float], dash: int = 6, gap: int = 4):
        x1,y1 = p1; x2,y2 = p2
        dx, dy = x2-x1, y2-y1
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return
        ux, uy = dx/dist, dy/dist
        n = int(dist // (dash+gap)) + 1
        for i in range(n):
            a = i*(dash+gap)
            b = min(a + dash, dist)
            xa, ya = x1 + ux*a, y1 + uy*a
            xb, yb = x1 + ux*b, y1 + uy*b
            d.line((xa,ya,xb,yb), fill=_ensure_rgba(dash_color, 255), width=1)

    # 1) 可选显示 OCR 框
    if show_boxes:
        box_rgba = _ensure_rgba(box_color, box_alpha)
        for it in items:
            box = it.get("box")
            if not box or len(box) != 4:
                continue
            odraw.polygon(box, outline=box_rgba, fill=None)

    # 简单的防重叠
    used: List[Tuple[float,float]] = []
    def place_point(anchor_pt: Tuple[float,float]) -> Tuple[float,float]:
        x, y = anchor_pt
        for rtry in range(0, radius*3, max(2, radius//3)):
            for ang in (0,45,90,135,180,225,270,315):
                rx = x + (rtry*math.cos(math.radians(ang)))
                ry = y + (rtry*math.sin(math.radians(ang)))
                if all((rx-ux)**2 + (ry-uy)**2 > (radius*2.2)**2 for (ux,uy) in used):
                    used.append((rx,ry))
                    return (rx, ry)
        used.append((x,y))
        return (x,y)

    for idx, it in enumerate(items, start=1):
        box = it.get("box")
        if not box or len(box) != 4:
            cx, cy = it.get("center", (w*0.5, h*0.5))
            target = (cx, cy)
        else:
            target = centroid(box)

        bx, by = target
        sel_anchor = anchor
        if anchor == "auto":
            leftness = bx < w*0.33; rightness = bx > w*0.67
            topness = by < h*0.33; bottomness = by > h*0.67
            if leftness and topness:   sel_anchor="br"
            elif leftness and bottomness: sel_anchor="tr"
            elif rightness and topness: sel_anchor="bl"
            elif rightness and bottomness: sel_anchor="tl"
            else: sel_anchor="tr"

        ax = {"tl": -1, "tr": 1, "bl": -1, "br": 1}.get(sel_anchor, 1)
        ay = {"tl": -1, "tr": -1, "bl": 1, "br": 1}.get(sel_anchor, -1)

        cx = bx + ax*(abs(offset[0]) + radius*1.2)
        cy = by + ay*(abs(offset[1]) + radius*1.2)
        cx, cy = place_point((cx, cy))

        # 2) 画气泡
        fill_rgba = _ensure_rgba(bubble_fill, bubble_alpha)
        outline_rgba = _ensure_rgba(bubble_fill, 1.0 if isinstance(bubble_alpha, float) else 255)
        odraw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=fill_rgba, outline=outline_rgba, width=2)

        # 连线到最近边
        if box and len(box)==4:
            end_pt = nearest_point_on_edges((cx, cy), box)
            vx, vy = end_pt[0]-cx, end_pt[1]-cy
            vlen = math.hypot(vx, vy)
            if vlen > 1e-3:
                sx, sy = cx + vx/vlen*(radius+1), cy + vy/vlen*(radius+1)
                draw_dashed_line(odraw, (sx,sy), end_pt)

        # 3) 写编号
        bubble_num = it.get("bubble_id", idx)
        num = str(bubble_num)
        l, t, r, b = odraw.textbbox((0, 0), num, font=font)
        tw, th = r-l, b-t
        odraw.text((cx - tw/2, cy - th/2), num, font=font, fill=_ensure_rgba(number_color, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")
