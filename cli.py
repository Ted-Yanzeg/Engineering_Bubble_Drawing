# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
from typing import Tuple, List

from PIL import Image

from .ocr import run_ocr
from .cleaning import clean_items
from .sorting import sort_reading_order
from .drawing import draw_bubbles
from .exporter import export_tabular
from .gradio_ui import cmd_gradio as _cmd_gradio

# 命令行入口，支持 run 和 gradio 两个子命令
# run 为命令行模式，gradio 为图形界面模式
def _parse_excludes(s: str) -> List[Tuple[float, float, float, float]]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for seg in s.split(";"):
        try:
            x1,y1,x2,y2 = [float(v) for v in seg.split(",")]
            out.append((x1,y1,x2,y2))
        except Exception:
            continue
    return out

# run + clean + sort + draw + export 流程
def cmd_run(args: argparse.Namespace) -> None:
    img = Image.open(args.input).convert("RGB")
    W, H = img.size
    print(f"[INFO] image loaded: {args.input} ({W}x{H})")

    ocr_items, mode = run_ocr(img, lang=args.lang, det=True, rec=True)
    print(f"[INFO] PaddleOCR mode: {mode}; raw items: {len(ocr_items)}")

    custom_excludes = _parse_excludes(args.exclude)
    items = clean_items(ocr_items, W, H, min_conf=args.min_conf, custom_excludes=custom_excludes)
    print(f"[INFO] after cleaning: {len(items)}")

    items_sorted = sort_reading_order(items)

    # 给每个气泡编号 1 到 n
    for i, it in enumerate(items_sorted, start=1):
        it["bubble_id"] = i

    dx, dy = args.offset
    anno = draw_bubbles(img, items_sorted, radius=args.bubble_radius, text_scale=args.label_scale, font_path=args.font,
                        anchor=args.anchor, offset=(dx, dy), avoid_overlap=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    out_img = str(Path(args.out_dir) / f"{stem}_bubbled.jpg")
    anno.save(out_img, quality=95)
    print(f"[OK] annotated image -> {out_img}")

    base = str(Path(args.out_dir) / f"{stem}_dims")
    csv_path, xlsx_path = export_tabular(items_sorted, base, image_name=Path(args.input).name)
    print(f"[OK] table -> {csv_path}" + (f" | {xlsx_path}" if xlsx_path else " (xlsx skipped)"))

    out_json = str(Path(args.out_dir) / f"{stem}_dims.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items_sorted, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON -> {out_json}")

# argparse 
def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Automatic dimension detection + bubble labeling + Excel/JSON export")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run", help="Run OCR->clean->bubble->export pipeline")
    ap_run.add_argument("--input", required=True, help="Input image file (JPG/PNG/TIF)")
    ap_run.add_argument("--out_dir", default="out", help="Output directory")
    ap_run.add_argument("--lang", default="en", help="OCR language")
    ap_run.add_argument("--min_conf", type=float, default=0.60)
    ap_run.add_argument("--bubble_radius", type=int, default=18)
    ap_run.add_argument("--label_scale", type=float, default=1.2, help="Scale for bubble number font size relative to radius (font_size = radius*label_scale)")
    ap_run.add_argument("--font", default=None, help="Optional TTF font path for bubble numbers")
    ap_run.add_argument("--anchor", default="tr", choices=["tl", "tr", "bl", "br"], help="Bubble anchor relative to text box")
    ap_run.add_argument("--offset", type=lambda s: tuple(map(int, s.split(","))), default=(10, -10), help="dx,dy for bubble from anchor")
    ap_run.add_argument("--exclude", default="", help="Exclude zones 'x1,y1,x2,y2;...' in pixels")
    ap_run.set_defaults(func=cmd_run)

    ap_ui = sub.add_parser("gradio", help="Launch Gradio UI")
    ap_ui.add_argument("--share", action="store_true", help="Enable public share link (慎用，涉及图纸隐私)")
    ap_ui.set_defaults(func=_cmd_gradio)

    return ap
