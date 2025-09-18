# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple
from .rules import classify_text
from .geometry import default_exclusion_zones, in_rect

# 清洗 OCR 结果，过滤低置信度、无类型、区域内的文本
def should_drop_by_zone(center: Tuple[float, float], zones: List[Tuple[float, float, float, float]]) -> bool:
    return any(in_rect(center, z) for z in zones)

def clean_items(items: List[Dict[str, Any]], img_w: int, img_h: int, min_conf: float,
                custom_excludes: Optional[List[Tuple[float, float, float, float]]] = None) -> List[Dict[str, Any]]:
    zones = default_exclusion_zones(img_w, img_h)
    if custom_excludes:
        zones += custom_excludes
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        txt = (it.get("text") or "").strip()
        conf = float(it.get("conf", 1.0))
        if txt == "0":
            continue
        center = it["center"]
        tp = classify_text(txt)
        if conf < min_conf:
            continue
        if tp is None:
            continue
        if should_drop_by_zone(center, zones):
            if tp not in ("R", "DIA", "ANG"):
                continue
        it["type"] = tp
        cleaned.append(it)
    return cleaned
