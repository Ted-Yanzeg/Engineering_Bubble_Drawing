# -*- coding: utf-8 -*-
from typing import Tuple, List

def in_rect(pt: Tuple[float, float], rect: Tuple[float, float, float, float]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def default_exclusion_zones(w: int, h: int) -> List[Tuple[float, float, float, float]]:
    m = 0.025
    top_bar = (0, 0, w, 0.06 * h)
    left_bar = (0, 0, 0.06 * w, h)
    bottom_bar = (0, 0.94 * h, w, h)
    right_bar = (0.96 * w, 0, w, h)
    title_block = (0.62 * w, 0.62 * h, w, h)
    outer = [
        (0, 0, w * m, h), (0, 0, w, h * m), (w * (1 - m), 0, w, h), (0, h * (1 - m), w, h),
    ]
    return outer + [top_bar, left_bar, bottom_bar, right_bar, title_block]
