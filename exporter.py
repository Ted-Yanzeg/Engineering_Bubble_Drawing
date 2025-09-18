# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import json

try:
    import pandas as pd
except Exception:
    pd = None

def export_tabular(items: List[Dict[str, Any]], out_base: str, image_name: str) -> Tuple[str, Optional[str]]:
    rows = []
    for idx, it in enumerate(items, start=1):
        box = it["box"]
        flat = [c for p in box for c in p]
        # —— 改动：bubble_id 优先用用户编辑后的
        bid = it.get("bubble_id", idx)
        rows.append({
            "bubble_id": bid,
            "text": it.get("text", ""),
            "type": it.get("type", ""),
            "conf": round(float(it.get("conf", 1.0)), 4),
            "cx": round(float(it["center"][0]), 2),
            "cy": round(float(it["center"][1]), 2),
            "box": json.dumps(box, ensure_ascii=False),
            "x1": flat[0], "y1": flat[1], "x2": flat[2], "y2": flat[3],
            "x3": flat[4], "y3": flat[5], "x4": flat[6], "y4": flat[7],
            "image": image_name,
        })

    # 可按 bubble_id 排序输出，便于对齐
    rows.sort(key=lambda r: (r["bubble_id"],))

    csv_path = f"{out_base}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = list(rows[0].keys()) if rows else ["bubble_id","text","type","conf"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            line = ",".join([str(r.get(c, "")) for c in cols])
            f.write(line + "\n")

    xlsx_path = None
    if pd is not None:
        try:
            df = pd.DataFrame(rows)
            xlsx_path = f"{out_base}.xlsx"
            df.to_excel(xlsx_path, index=False)
        except Exception:
            xlsx_path = None
    return csv_path, xlsx_path
