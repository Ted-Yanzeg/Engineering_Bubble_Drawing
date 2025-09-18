# -*- coding: utf-8 -*-
from typing import Any, Dict, List
import numpy as np

#  标签排序
def sort_reading_order(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return items
    ys = np.array([it["center"][1] for it in items], dtype=float)
    h = float(max(ys) - min(ys)) if len(ys) else 0.0
    lane = max(6.0, 0.02 * h)
    rows: List[List[Dict[str, Any]]] = []
    for it in sorted(items, key=lambda t: t["center"][1]):
        placed = False
        for r in rows:
            if abs(r[0]["center"][1] - it["center"][1]) <= lane:
                r.append(it); placed = True; break
        if not placed:
            rows.append([it])
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.extend(sorted(r, key=lambda t: t["center"][0]))
    return out
