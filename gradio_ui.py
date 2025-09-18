# -*- coding: utf-8 -*-
import time, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image
import numpy as np
import gradio as gr

from .ocr import run_ocr, ocr_prefill_at
from .cleaning import clean_items
from .sorting import sort_reading_order
from .drawing import draw_bubbles
from .exporter import export_tabular
from .rules import classify_text

# generated table columns
_COLS = ["bubble_id","text","type","conf"]

try:
    import pandas as pd  
except Exception:
    pd = None

# rows 转到 DataFrame 或 list[list]
def _rows_to_grid(rows):
    if pd is not None:
        return pd.DataFrame(rows, columns=_COLS)
    return [[r.get(c, "") for c in _COLS] for r in rows]


def _parse_excludes(s: str):
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

def _to_table(items: List[Dict[str, Any]], image_name: str) -> List[Dict[str, Any]]:
    rows = []
    for idx, it in enumerate(items, start=1):
        bid = it.get("bubble_id", idx)
        rows.append({
            "bubble_id": bid,
            "text": (it.get("text", "") or ""),
            "type": (it.get("type", "") or ""),
            "conf": round(float(it.get("conf", 1.0)), 4),
        })
    return rows

def _coerce_table_value(table_val) -> List[Dict[str, Any]]:
    """把 DataFrame/list 转为 [{'bubble_id':..,'text':..,'type':..,'conf':..}, ...]"""
    rows_out = []
    if table_val is None:
        return rows_out
    if pd is not None and isinstance(table_val, pd.DataFrame):
        iter_rows = table_val[_COLS].fillna("").values.tolist()
    else:
        iter_rows = table_val
    for row in iter_rows:
        if not row:
            continue
        try:
            bid = int(float(row[0])) if str(row[0]).strip() != "" else None
        except Exception:
            bid = None
        txt = str(row[1]) if len(row) > 1 else ""
        tp  = str(row[2]) if len(row) > 2 else ""
        try:
            cf = float(row[3]) if len(row) > 3 and str(row[3]).strip() != "" else None
        except Exception:
            cf = None
        rows_out.append({"bubble_id": bid, "text": txt, "type": tp, "conf": cf})
    return rows_out

def build_gradio_app():
    import gradio as gr

    with gr.Blocks(title="工程图尺寸气泡标注（PaddleOCR，本地）") as demo:
        gr.Markdown("""
        # 工程图尺寸气泡标注
        - OCR 模型使用PaddleOCR v3/v2
        - 操作流程：上传图片 → 调整参数 → 运行OCR → 编辑气泡与text → 导出
        - app流程：上传图 → 识别 → 清洗 → 排序编号 → 画气泡 → 导出 CSV/XLSX/JSON
        """)

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="上传工程图", type="pil")
                lang = gr.Dropdown(["en","ch"], value="en", label="OCR语言参数")
                min_conf = gr.Slider(0, 1, value=0.60, step=0.01, label="最小置信度")
                bubble_radius = gr.Slider(6, 48, value=18, step=1, label="气泡半径")
                label_scale = gr.Slider(0.6, 2.0, value=1.2, step=0.05, label="数字字号缩放(半径×系数)")
                anchor = gr.Radio(["auto","tl","tr","bl","br"], value="auto", label="气泡锚点")
                offset = gr.Textbox(value="10,-10", label="气泡偏移 dx,dy")
                font_path = gr.Textbox(value="", label="可选：TTF字体路径(提高数字清晰度)")
                exclude = gr.Textbox(value="", label="可选：排除区域 'x1,y1,x2,y2;...'（像素）")
                run_btn = gr.Button("START")
            with gr.Column(scale=2):
                anno_out = gr.Image(label="可点击的预览（选择模式后点击图片）", interactive=True)
                table_out = gr.Dataframe(
                    headers=_COLS,
                    row_count=(1, "dynamic"),
                    interactive=True,                 # —— 允许编辑
                    datatype=["number","str","str","number"]
                )
                log_out = gr.Textbox(label="运行日志", interactive=False)
                with gr.Row():
                    edit_mode = gr.Radio(["添加","删除"], value="添加", label="点击模式")
                    manual_text = gr.Textbox(value="", label="添加时：手动文本（留空则自动OCR预填）")
                    prefill_patch = gr.Slider(16, 160, value=64, step=8, label="自动预填OCR范围(px)")
                export_btn = gr.Button("导出 CSV / (XLSX) / JSON 到 out/时间戳")
                csv_file = gr.File(label="CSV")
                xlsx_file = gr.File(label="XLSX")
                json_file = gr.File(label="JSON")

        items_state = gr.State([])        # list[dict]
        orig_img_state = gr.State(None)   # PIL Image

        def _run_and_store(img, lang, min_conf, bubble_radius, label_scale, anchor, offset, exclude, font_path):
            if img is None:
                raise gr.Error("请先上传一张图纸")
            img = img.convert("RGB")
            W, H = img.size
            ocr_items, mode = run_ocr(img, lang=lang, det=True, rec=True)
            zones = _parse_excludes(exclude)
            items = clean_items(ocr_items, W, H, min_conf=min_conf, custom_excludes=zones)
            items = sort_reading_order(items)

            # 为 items 赋初始 bubble_id（1..N）
            for i, it in enumerate(items, start=1):
                it["bubble_id"] = i

            try:
                dx, dy = tuple(map(int, (offset or "10,-10").split(",")))
            except Exception:
                dx, dy = (10, -10)
            anno = draw_bubbles(img, items, radius=bubble_radius, text_scale=label_scale,
                                font_path=(font_path or None), anchor=anchor, offset=(dx,dy), avoid_overlap=True)
            rows = _to_table(items, "uploaded_image")
            table = _rows_to_grid(rows)
            log = (
                f"[INFO] image: {W}x{H}\n"
                f"[INFO] PaddleOCR mode: {mode}; raw items: {len(ocr_items)}\n"
                f"[INFO] after cleaning: {len(items)}"
            )
            return anno, items, table, img, log

        def _on_click(evt: "gr.SelectData", mode: str, items: List[Dict[str,Any]],
              orig_img: Image.Image, label_scale: float, bubble_radius: int,
              anchor: str, offset: str, font_path: Optional[str], lang: str,
              prefill_patch: int, manual_text: str):
            if orig_img is None:
                raise gr.Error("请先上传并运行一次 OCR")

            # ------- 新增：把预览坐标 -> 原图坐标 -------
            x, y = float(evt.index[0]), float(evt.index[1])
            W, H = orig_img.size
            disp_w = None
            disp_h = None
            # gradio 里 evt.image 可能是 numpy 或 PIL，做两种兼容
            try:
                import numpy as _np  # 已在文件顶部 import 过就不会重复
                if isinstance(evt.image, _np.ndarray):
                    disp_h, disp_w = evt.image.shape[:2]
                elif hasattr(evt.image, "size"):
                    disp_w, disp_h = evt.image.size[0], evt.image.size[1]
            except Exception:
                if hasattr(evt.image, "size"):
                    disp_w, disp_h = evt.image.size[0], evt.image.size[1]
            if disp_w and disp_h and (disp_w != W or disp_h != H):
                sx = W / float(disp_w)
                sy = H / float(disp_h)
                x *= sx
                y *= sy
            # ---------------------------------------

            try:
                dx, dy = tuple(map(int, (offset or "10,-10").split(",")))
            except Exception:
                dx, dy = (10, -10)

            # 仍按阅读顺序排序，但不重置 bubble_id
            items = sort_reading_order(items)
            anno = draw_bubbles(orig_img, items, radius=bubble_radius, text_scale=label_scale,
                                font_path=(font_path or None), anchor=anchor, offset=(dx,dy), avoid_overlap=True)
            rows = _to_table(items, "uploaded_image")
            table = _rows_to_grid(rows)
            return anno, items, table

        def _on_table_edit(table_val, items: List[Dict[str,Any]], orig_img: Image.Image,
                           label_scale: float, bubble_radius: int, anchor: str, offset: str, font_path: Optional[str]):
            """用户编辑表格后：同步 items 的四个字段，并重绘"""
            if orig_img is None or items is None:
                raise gr.Error("请先上传并运行一次 OCR")
            coerced = _coerce_table_value(table_val)
            # 逐行覆盖（按行号对齐）
            n = min(len(items), len(coerced))
            for i in range(n):
                row = coerced[i]
                if row.get("bubble_id") is not None:
                    items[i]["bubble_id"] = int(row["bubble_id"])
                items[i]["text"] = row.get("text", items[i].get("text", ""))
                items[i]["type"] = row.get("type", items[i].get("type", ""))
                if row.get("conf") is not None:
                    try:
                        items[i]["conf"] = float(row["conf"])
                    except Exception:
                        pass
            # 不改变 items 顺序，仅用新的 bubble_id 绘制
            try:
                dx, dy = tuple(map(int, (offset or "10,-10").split(",")))
            except Exception:
                dx, dy = (10, -10)
            anno = draw_bubbles(orig_img, items, radius=bubble_radius, text_scale=label_scale,
                                font_path=(font_path or None), anchor=anchor, offset=(dx,dy), avoid_overlap=True)
            # 规范化表格显示（例如 conf 四舍五入）
            rows = _to_table(items, "uploaded_image")
            table_norm = _rows_to_grid(rows)
            return anno, items, table_norm

        def _on_export(items: List[Dict[str,Any]], orig_img: Image.Image):
            if not items:
                raise gr.Error("当前没有可导出的项。请先运行 OCR 或添加点位。")
            outdir = Path("out") / time.strftime("%Y%m%d_%H%M%S")
            outdir.mkdir(parents=True, exist_ok=True)

            # 导出图像（用当前 bubble_id 绘制）
            anno = draw_bubbles(orig_img, items, radius=18, text_scale=1.2, anchor="tr", offset=(10, -10))
            img_name = "uploaded_image"
            anno_path = outdir / f"{img_name}_bubbled.jpg"
            anno.save(anno_path, quality=95)

            # 表格（完整字段的 CSV/XLSX）
            base = str(outdir / f"{img_name}_dims")
            csv_path, xlsx_path = export_tabular(items, base, image_name=img_name)

            # JSON（完整 items）
            json_path = outdir / f"{img_name}_dims.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

            return str(csv_path), (str(xlsx_path) if xlsx_path else None), str(json_path)

        run_btn.click(_run_and_store,
                      inputs=[img_in, lang, min_conf, bubble_radius, label_scale, anchor, offset, exclude, font_path],
                      outputs=[anno_out, items_state, table_out, orig_img_state, log_out])

        anno_out.select(_on_click,
                        inputs=[edit_mode, items_state, orig_img_state, label_scale, bubble_radius, anchor, offset, font_path, lang, prefill_patch, manual_text],
                        outputs=[anno_out, items_state, table_out])

        # —— 新增：表格编辑事件
        table_out.change(_on_table_edit,
                         inputs=[table_out, items_state, orig_img_state, label_scale, bubble_radius, anchor, offset, font_path],
                         outputs=[anno_out, items_state, table_out])

        export_btn.click(_on_export, inputs=[items_state, orig_img_state], outputs=[csv_file, xlsx_file, json_file])

    return demo

def cmd_gradio(args):
    demo = build_gradio_app()
    demo.launch(share=args.share)
