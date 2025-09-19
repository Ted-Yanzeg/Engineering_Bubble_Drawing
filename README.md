# Engineering Drawing Bubble Annotation

## 目录

* [项目简介](#项目简介)
* [文件目录说明](#文件目录说明)
* [配置要求](#配置要求)
* [安装步骤](#安装步骤)
* [Quick Start](#quick-start)

---

## 项目简介

本项目面向2d工程图的“气泡标注 + 结构化导出”场景：

* 以 **PaddleOCR** 识别工程图中的尺寸/符号文本；
* 清洗并分类有效条目，并按“阅读顺序”排序；
* 在图中绘制带编号的气泡标注和引出线；
* 导出 **CSV / XLSX / JSON** 方便后续核对与统计；
* 提供 **命令行工具（CLI）** 与 **Gradio Web 界面** 两种使用方式。

> 适用输入：JPG/PNG 等图像文件（PDF 可在外部预先转图后再输入）。

---

## 文件目录说明

```
.
├─ Readme.md        # 本文件
├─ cli.py           # 命令行入口：参数解析、组装完整流水线并执行
├─ gradio_ui.py     # Web 前端：基于 Gradio 的交互式标注与导出
├─ ocr.py           # OCR 封装：创建/调用 PaddleOCR，统一结果结构
├─ cleaning.py      # 清洗与过滤：设置置信度阈值、文本规范化、去噪
├─ rules.py         # 规则与分类
├─ sorting.py       # 排序：按“上到下、左到右”的顺序对编号排序
├─ geometry.py      # 绘制几何工具，并做区域排除、IOU 等
├─ drawing.py       # 绘制：在图像上画气泡、编号、引出线
├─ exporter.py      # 导出：写出 CSV/XLSX/JSON，字段规范与表头
├─ requirements.txt # 依赖清单
```

**模块职责简述**

* **`ocr.py`**：创建并调用PaddleOCR，返回统一的 `文本 + 置信度 + 坐标` 列表。
* **`cleaning.py`**：按最小置信度、字符合法性、禁区（标题栏/边框等）清洗数据。
* **`rules.py`**：用正则/启发式将文本标成“尺寸/符号/其他”，并进行必要的格式清洗（如去掉孤立“0”等）。
* **`geometry.py`**：提供坐标变换、中心点计算、矩形并/交、默认排除区生成等。
* **`sorting.py`**：先按行聚类再行内从左到右排序；对高度/倾斜有一定鲁棒性。
* **`drawing.py`**：渲染半透明圆形气泡、白底编号、边框与引出线；输出标注图。
* **`exporter.py`**：把清洗 + 排序后的结果写出到 CSV / XLSX / JSON，字段包含 `bubble_id / text / type / conf `。
* **`cli.py`**：`python cli.py run --input ...` 一条命令完成“识别 → 清洗 → 排序 → 绘制 → 导出”。
* **`gradio_ui.py`**：浏览器中上传图片、一键识别、可视化核对、导出结果。

---

## 配置要求

* **Python**：3.10–3.12（建议 64 位）
* **操作系统**：macOS / Linux / Windows
* **必需依赖（见 requirements.txt）**

  * paddleocr ≥ 2.6.0
  * paddlepaddle ≥ 2.5.0（CPU 版即可；若需 GPU 请安装与你 CUDA 版本匹配的 GPU 版）
  * gradio ≥ 3.50.2
  * pillow, numpy, pandas, openpyxl

> 如在国内环境建议配置镜像源，以加速安装；Windows 如安装 GPU 版 PaddlePaddle，请参阅其官方安装指引选择匹配的 CUDA/CUDNN 版本。

---

## 安装步骤

> 以新建虚拟环境为例；如已有环境可跳过相应步骤。

### 1) 克隆项目并进入目录

```bash
git clone https://github.com/Ted-Yanzeg/Engineering_Bubble_Drawing.git
cd Path/to/Engineering_Bubble_Drawing
```

### 2) 创建并激活虚拟环境

**macOS / Linux**

```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) 安装依赖

```bash
pip install -r requirements.txt
```
> 如遇到安装慢或者readtimeout等问题，可以手动 pip install 或者使用清华镜像源 ：

```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple <package-name>
```

> 若安装 `paddlepaddle` 失败或速度慢：
>
> 1. 先仅安装 `paddleocr` 以自动拉取兼容版本；
> 2. 或根据你的 OS/硬件到 Paddle 官方指引选择对应wheel（CPU/GPU）。

---

## Quick Start

### 方式一：命令行（CLI）

最简示例：

```bash
cd Path/to/Engineering_Bubble_Drawing
cd ..
python -m Engineering_Bubble_Drawing run --input /Path/to/image.png --out_dir Path/to/outdir --min_conf 0.62 --bubble_radius 12
```

常用选项：

* `--input`：输入图像路径（JPG/PNG）。
* `--out_dir`：输出目录，保存标注图与 CSV/XLSX/JSON。
* `--min_conf`：最小置信度阈值，过滤低置信度文本。
* `--bubble_radius`：气泡半径像素值。
* `--exclude`：自定义排除区，格式如 `"x1,y1,x2,y2;..."`。

运行完毕后，`./out/` 输出文档：

* `annotated.png` 带编号气泡的图像；
* `results.csv|xlsx|json`：结构化导出，包含 `bubble_id / text / type / conf ` 等字段。

### 方式二：Gradio Web 界面

本地启动：

```bash
cd Path/to/Engineering_Bubble_Drawing
cd ..
python -m Engineering_Bubble_Drawing gradio
```

控制台会输出本地地址（如 `http://127.0.0.1:7860`），ctrl + 左键使用浏览器打开后：

1. 上传工程图图片；
2. 点击start，核对自动结果；
3. 需要时微调参数或手动修订；
4. 可添加，删除手动绘制气泡图图像；
5. 一键导出 CSV/XLSX/JSON。

---

> Tips
>
> * PDF 图可用外部工具先转成高DPI 的 PNG/JPG 再输入，识别效果更稳。
> * 若工程图右下角标题栏/边框干扰较多，可在 CLI 里传 `--exclude` 或在 UI 中设置“排除区域”。
> * 如需开启文本方向/倾斜识别，请在 `ocr.py` 中调整 PaddleOCR 的相关开关以契合你的版本。
