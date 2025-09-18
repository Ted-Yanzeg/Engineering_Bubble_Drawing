# -*- coding: utf-8 -*-
from __future__ import annotations
import inspect, tempfile
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

def _create_paddle_ocr(lang: str, det: bool, rec: bool):
    from paddleocr import PaddleOCR  # lazy import
    sig = inspect.signature(PaddleOCR)
    params = sig.parameters
    if "use_textline_orientation" in params:
        ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
        return ocr, "v3"
    elif "use_angle_cls" in params:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        return ocr, "v2"
    else:
        raise ValueError("Unsupported PaddleOCR version: neither `use_textline_orientation` nor `use_angle_cls` is available.")

def _parse_paddle_result(res: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        arr = res[0] if (isinstance(res, list) and len(res) > 0 and isinstance(res[0], list)) else res
        for item in arr:
            if not item:
                continue
            box = item[0]
            txt = item[1][0]
            conf = float(item[1][1]) if len(item[1]) > 1 else 1.0
            cx = sum(p[0] for p in box) / 4.0
            cy = sum(p[1] for p in box) / 4.0
            out.append({"text": txt, "conf": conf, "box": box, "center": (cx, cy)})
    except Exception:
        pass
    return out

def _parse_v3_predict(pred: Any) -> List[Dict[str, Any]]:
    pages = pred if isinstance(pred, list) else [pred]
    out: List[Dict[str, Any]] = []
    for r in pages:
        dt_polys = getattr(r, 'dt_polys', None) or getattr(r, 'boxes', None)
        if dt_polys is None and isinstance(r, dict):
            dt_polys = r.get('dt_polys') or r.get('boxes')
        rec_texts = getattr(r, 'rec_texts', None)
        if rec_texts is None and isinstance(r, dict):
            rec_texts = r.get('rec_texts') or r.get('texts')
        rec_scores = getattr(r, 'rec_scores', None)
        if rec_scores is None and isinstance(r, dict):
            rec_scores = r.get('rec_scores')
        if dt_polys is None:
            continue
        n = len(dt_polys)
        for i in range(n):
            poly = dt_polys[i]
            if poly is None or len(poly) < 4:
                continue
            box = [[float(poly[j][0]), float(poly[j][1])] for j in range(4)]
            cx = sum(p[0] for p in box) / 4.0
            cy = sum(p[1] for p in box) / 4.0
            txt = rec_texts[i] if (rec_texts is not None and i < len(rec_texts)) else ''
            conf = float(rec_scores[i]) if (rec_scores is not None and i < len(rec_scores)) else 1.0
            out.append({"text": txt, "conf": conf, "box": box, "center": (cx, cy)})
    return out

def run_ocr(img: Image.Image, lang: str = "en", det: bool = True, rec: bool = True) -> Tuple[List[Dict[str, Any]], str]:
    ocr, mode = _create_paddle_ocr(lang, det, rec)
    arr = np.array(img.convert("RGB"))
    if mode == "v3":
        parsed = []
        try:
            pred = ocr.predict(input=arr)
            parsed = _parse_v3_predict(pred)
        except Exception:
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
                    Image.fromarray(arr).save(tf.name, quality=95)
                    pred = ocr.predict(input=tf.name)
                    parsed = _parse_v3_predict(pred)
            except Exception:
                parsed = []
        if not parsed:
            try:
                res = ocr.ocr(arr)  # v3 内部处理 cls
                parsed = _parse_paddle_result(res)
                mode_out = "v3-legacy"
            except Exception:
                parsed = []
                mode_out = mode
        else:
            mode_out = mode
        return parsed, mode_out
    else:
        res = ocr.ocr(arr, cls=True)
        parsed = _parse_paddle_result(res)
        return parsed, mode

# ---- 小范围自动 OCR 预填（Gradio 用） ----
def ocr_prefill_at(img: Image.Image, x: float, y: float, lang: str, patch: int = 64) -> str:
    ocr, mode = _create_paddle_ocr(lang=lang, det=True, rec=True)
    W, H = img.size
    x1 = max(0, int(x - patch)); y1 = max(0, int(y - patch))
    x2 = min(W, int(x + patch)); y2 = min(H, int(y + patch))
    crop = img.crop((x1, y1, x2, y2)).convert("RGB")
    arr = np.array(crop)
    parsed: List[Dict[str, Any]] = []
    if mode == "v3":
        try:
            pred = ocr.predict(input=arr)
            parsed = _parse_v3_predict(pred)
        except Exception:
            parsed = []
        if not parsed:
            try:
                res = ocr.ocr(arr)
                parsed = _parse_paddle_result(res)
            except Exception:
                parsed = []
    else:
        res = ocr.ocr(arr, cls=True)
        parsed = _parse_paddle_result(res)
    if not parsed:
        return ""
    parsed.sort(key=lambda it: float(it.get("conf", 0.0)), reverse=True)
    for it in parsed:
        t = (it.get("text") or "").strip()
        if t and t != "0":
            return t
    return ""
