# -*- coding: utf-8 -*-
"""
ocr.py — PaddleOCR v2/v3 统一封装（修复 numpy 数组的布尔判断歧义）
- 关闭文档预处理（不改几何）
- v3: 预测前“最长边=960 + pad到32倍数”，预测后按比例反映射回原图
- v2: 直接 .ocr() + angle_cls
返回每项：
{
  "text": str, "conf": float,
  "box": [(x1,y1),(x2,y2),(x3,y3),(x4,y4)],  # 原图坐标
  "center": (cx, cy)
}
"""
from __future__ import annotations

import inspect
from math import ceil
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

Point = Tuple[float, float]
Quad  = List[Point]

# ---------------- utils ----------------
def _next_multiple(x: int, base: int = 32) -> int:
    return int(ceil(x / float(base)) * base)

def _quad_center(box: Quad) -> Tuple[float, float]:
    cx = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4.0
    cy = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4.0
    return (cx, cy)

def _is_empty(x) -> bool:
    """None 或 长度为 0 时返回 True；避免对 numpy.ndarray 做直接布尔判断"""
    if x is None:
        return True
    try:
        return len(x) == 0
    except Exception:
        return False

# -------- v3 resize + pad (before predict) --------
def _det_resize_v3(
    arr_rgb: np.ndarray,
    limit_side_len: int = 960,
    pad_stride: int = 32,
    allow_upscale: bool = False
) -> Tuple[np.ndarray, float, float]:
    if cv2 is None:
        raise RuntimeError("cv2 未安装（pip install opencv-python-headless）。")

    h, w = arr_rgb.shape[:2]
    side = max(h, w)

    if side > limit_side_len or (allow_upscale and side != limit_side_len):
        r = float(limit_side_len) / float(side)
        new_w = max(1, int(round(w * r)))
        new_h = max(1, int(round(h * r)))
        bgr = arr_rgb[:, :, ::-1]
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rw, rh = new_w / float(w), new_h / float(h)
    else:
        resized = arr_rgb[:, :, ::-1].copy()
        rw, rh = 1.0, 1.0
        new_h, new_w = h, w

    if pad_stride:
        pad_h = _next_multiple(resized.shape[0], pad_stride)
        pad_w = _next_multiple(resized.shape[1], pad_stride)
        if pad_h != resized.shape[0] or pad_w != resized.shape[1]:
            canvas = np.zeros((pad_h, pad_w, 3), dtype=resized.dtype)
            canvas[:resized.shape[0], :resized.shape[1]] = resized
            resized = canvas

    return resized, rw, rh

# -------- PaddleOCR init (v2/v3) --------
def _create_paddle_ocr(
    lang: str = "en",
    det: bool = True,
    rec: bool = True,
    det_model_dir: Optional[str] = None,
    rec_model_dir: Optional[str] = None
):
    from paddleocr import PaddleOCR
    sig = inspect.signature(PaddleOCR)
    params = sig.parameters

    kwargs: Dict[str, Any] = dict(lang=lang)
    if "use_doc_orientation_classify" in params:
        kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in params:
        kwargs["use_doc_unwarping"] = False
    if det_model_dir is not None and "det_model_dir" in params:
        kwargs["det_model_dir"] = det_model_dir
    if rec_model_dir is not None and "rec_model_dir" in params:
        kwargs["rec_model_dir"] = rec_model_dir

    if "use_textline_orientation" in params:
        kwargs["use_textline_orientation"] = True
        return PaddleOCR(**kwargs), "v3"
    if "use_angle_cls" in params:
        kwargs["use_angle_cls"] = True
        return PaddleOCR(**kwargs), "v2"
    raise ValueError("Unsupported PaddleOCR version")

# -------- parsers --------
def _parse_v2_ocr(res: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        pages = res if isinstance(res, list) else [res]
        for page in pages:
            if not isinstance(page, list):
                continue
            for item in page:
                if not item:
                    continue
                box = item[0]
                info = item[1] if len(item) > 1 else ["", 0.0]
                txt = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else ""
                conf = float(info[1]) if isinstance(info, (list, tuple)) and len(info) > 1 else 0.0
                if not (isinstance(box, (list, tuple)) and len(box) == 4):
                    continue
                quad: Quad = [(float(x), float(y)) for (x, y) in box]
                cx, cy = _quad_center(quad)
                out.append({"text": txt, "conf": conf, "box": quad, "center": (cx, cy)})
    except Exception:
        pass
    return out

def _as_attr_or_key(obj, name):
    """优先取属性，取不到再取 dict[key]；避免直接做布尔判断"""
    v = getattr(obj, name, None)
    if v is None and isinstance(obj, dict):
        v = obj.get(name)
    return v

def _parse_v3_predict(pred: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pages = pred if isinstance(pred, list) else [pred]
    for r in pages:
        dt_polys = _as_attr_or_key(r, "dt_polys") or _as_attr_or_key(r, "boxes")
        rec_texts = _as_attr_or_key(r, "rec_texts")
        rec_scores = _as_attr_or_key(r, "rec_scores")

        if _is_empty(dt_polys):
            continue

        # 兼容 numpy.ndarray
        n = int(len(dt_polys))
        for i in range(n):
            poly = dt_polys[i]
            if poly is None or len(poly) < 4:
                continue
            # poly 可能是 ndarray；统一转 float
            quad: Quad = [(float(poly[j][0]), float(poly[j][1])) for j in range(4)]
            txt = (rec_texts[i] if (rec_texts is not None and i < len(rec_texts)) else "") if not _is_empty(rec_texts) else ""
            conf = (float(rec_scores[i]) if (rec_scores is not None and i < len(rec_scores)) else 0.0) if not _is_empty(rec_scores) else 0.0
            cx, cy = _quad_center(quad)
            out.append({"text": txt, "conf": conf, "box": quad, "center": (cx, cy)})
    return out

# -------- main entry --------
def run_ocr(
    img: Image.Image,
    lang: str = "en",
    det: bool = True,
    rec: bool = True,
    limit_side_len: int = 960,
    pad_stride: int = 32,
    allow_upscale: bool = False,
    det_model_dir: Optional[str] = None,
    rec_model_dir: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    ocr, mode = _create_paddle_ocr(
        lang=lang, det=det, rec=rec,
        det_model_dir=det_model_dir, rec_model_dir=rec_model_dir
    )

    arr_rgb = np.array(img.convert("RGB"))

    if mode == "v3":
        det_in_bgr, rw, rh = _det_resize_v3(
            arr_rgb, limit_side_len=limit_side_len,
            pad_stride=pad_stride, allow_upscale=allow_upscale
        )
        pred = ocr.predict(input=det_in_bgr)
        parsed = _parse_v3_predict(pred)

        items: List[Dict[str, Any]] = []
        for it in parsed:
            box = it["box"]
            # 反映射回原图（pad 左上对齐，无偏移）
            mapped: Quad = [(x / rw, y / rh) for (x, y) in box]
            cx, cy = _quad_center(mapped)
            items.append({"text": it["text"], "conf": it["conf"], "box": mapped, "center": (cx, cy)})
        return items, "v3"

    # v2
    res = ocr.ocr(arr_rgb, cls=True)
    items = _parse_v2_ocr(res)
    return items, "v2"

# -------- small patch OCR (prefill) --------
def ocr_prefill_at(
    img: Image.Image,
    x: float, y: float,
    lang: str = "en",
    patch: int = 64,
    limit_side_len: int = 960,
    pad_stride: int = 32,
    det_model_dir: Optional[str] = None,
    rec_model_dir: Optional[str] = None
) -> str:
    ocr, mode = _create_paddle_ocr(
        lang=lang, det=True, rec=True,
        det_model_dir=det_model_dir, rec_model_dir=rec_model_dir
    )
    W, H = img.size
    x1 = max(0, int(x - patch)); y1 = max(0, int(y - patch))
    x2 = min(W, int(x + patch)); y2 = min(H, int(y + patch))
    crop = img.crop((x1, y1, x2, y2)).convert("RGB")
    arr_rgb = np.array(crop)

    candidates: List[Tuple[str, float]] = []

    if mode == "v3":
        det_in_bgr, _, _ = _det_resize_v3(arr_rgb, limit_side_len=limit_side_len, pad_stride=pad_stride)
        pred = ocr.predict(input=det_in_bgr)
        parsed = _parse_v3_predict(pred)
        for it in parsed:
            t = (it.get("text") or "").strip()
            s = float(it.get("conf") or 0.0)
            if t and t != "0":
                candidates.append((t, s))
    else:
        res = ocr.ocr(arr_rgb, cls=True)
        parsed = _parse_v2_ocr(res)
        for it in parsed:
            t = (it.get("text") or "").strip()
            s = float(it.get("conf") or 0.0)
            if t and t != "0":
                candidates.append((t, s))

    if not candidates:
        return ""
    candidates.sort(key=lambda z: z[1], reverse=True)
    return candidates[0][0]

__all__ = ["run_ocr", "ocr_prefill_at"]
