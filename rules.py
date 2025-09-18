# -*- coding: utf-8 -*-
import re
from typing import Optional

# -----------------------------
# Text classification heuristics
# -----------------------------
_DIM_PATTERNS = {
    "DIA": re.compile(r"^\s*[⌀Φφ]\s*\d+(\.\d+)?\s*(mm|cm|m)?\s*$"),
    "R":   re.compile(r"^\s*[Rr]\s*\d+(\.\d+)?\s*(mm|cm|m)?\s*$"),
    "ANG": re.compile(r"^\s*\d+(\.\d+)?\s*°\s*$"),
    "LEN": re.compile(r"^\s*\d+(\.\d+)?\s*(mm|cm|m)?\s*$"),
    "ROUGH": re.compile(r"^\s*(Ra|RA)\s*\d+(\.\d+)?\s*(μm|um|µm)?\s*$"),
    "THREAD": re.compile(r"^\s*M\d+(\.\d+)?([xX×]\d+(\.\d+)?)?.*$"),
}
_SECTION = re.compile(r"^[A-Z]-[A-Z]$")
_MISC_SKIP = re.compile(r"^[A-Za-z]+$")
_GRID_NUM = re.compile(r"^\d{1,2}$")
_GRID_LET = re.compile(r"^[A-Z]$")
_DATE_FMT = re.compile(r"^\d{1,2}/\d{1,2}$")

def classify_text(t: str) -> Optional[str]:
    s = (t or "").strip()
    if not s:
        return None
    if _SECTION.match(s) or _MISC_SKIP.match(s) or _GRID_LET.match(s) or _DATE_FMT.match(s):
        return None
    for k, pat in _DIM_PATTERNS.items():
        if pat.match(s):
            return k
    if _GRID_NUM.match(s):
        return "LEN"
    return None
