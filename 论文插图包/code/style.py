"""Thesis figure style: Chinese labels, print-ready, shared palette."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parent
FIG = PACK / "figures"
DATA = PACK / "data"
CODE = PACK / "code"
PAPER_IMG = ROOT / "图片"

WATER_FEATURES = ["WT", "pH", "DO", "COD_MN", "NH3_N", "TP", "TN", "EC", "TURB"]
WATER_CN = {
    "WT": "水温 WT",
    "pH": "pH",
    "DO": "溶解氧 DO",
    "COD_MN": "高锰酸盐指数 CODMn",
    "NH3_N": "氨氮 NH3-N",
    "TP": "总磷 TP",
    "TN": "总氮 TN",
    "EC": "电导率 EC",
    "TURB": "浊度 TURB",
}
WATER_UNIT = {
    "WT": "℃",
    "pH": "—",
    "DO": "mg/L",
    "COD_MN": "mg/L",
    "NH3_N": "mg/L",
    "TP": "mg/L",
    "TN": "mg/L",
    "EC": "μS/cm",
    "TURB": "NTU",
}

# Stable keys used in filenames / JSON
MODEL_ORDER = [
    "ha",
    "persist",
    "lstm",
    "tcn",
    "patchtst",
    "v3",
    "former",
    "p0_fut",
    "p0",
]
MODEL_CN = {
    "ha": "HA",
    "persist": "持续性基线",
    "lstm": "LSTM",
    "tcn": "TCN",
    "patchtst": "PatchTST",
    "v3": "MD-DySTGCN",
    "p0": "MD-DySTGCN-CA（本文）",
    "p0_fut": "未来气象扩展",
    "former": "MD-DySTFormer",
}
# Colors: this work in red; others muted
MODEL_COLOR = {
    "ha": "#7a7a7a",
    "persist": "#4b5563",
    "lstm": "#2563eb",
    "tcn": "#0d9488",
    "patchtst": "#7c3aed",
    "v3": "#1d4ed8",
    "p0": "#c00000",
    "p0_fut": "#ca8a04",
    "former": "#0891b2",
}

OBS_COLOR = "#111111"
PRED_COLOR = "#c00000"
IMPUTE_COLOR = "#c00000"
OUTLIER_COLOR = "#dc2626"
THRESH_COLOR = "#d97706"

DPI = 300


def _register_cjk():
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            font_manager.fontManager.addfont(p)
    return ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]


def apply_style():
    sans = _register_cjk()
    plt.rcParams.update(
        {
            "font.sans-serif": sans,
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "legend.framealpha": 0.92,
        }
    )


def ensure_dirs():
    for sub in ("ch1", "ch2", "ch3", "ch4", "ch5"):
        (FIG / sub).mkdir(parents=True, exist_ok=True)
    (DATA / "runs").mkdir(parents=True, exist_ok=True)
    PAPER_IMG.mkdir(parents=True, exist_ok=True)


def savefig(fig, rel: str, reset_style: bool = True):
    """Save under 论文插图包/figures and copy basename to 图片/."""
    if reset_style:
        apply_style()
    dest = FIG / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=DPI)
    paper_name = dest.name
    fig.savefig(PAPER_IMG / paper_name, dpi=DPI)
    plt.close(fig)
    print(f"[fig] {dest}")
    return dest
