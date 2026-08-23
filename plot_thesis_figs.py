#!/usr/bin/env python3
"""Generate Chapter-5 figures for the MD-DySTFormer thesis (horizon-1)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_tf import (
    DATA_DIR,
    NPZDataset,
    WATER_FEATURES,
    cfg_for_exp,
    MDDySTGCNtf,
    unpack_batch,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "图片"
CKPT = ROOT / "outputs" / "model_runs_tf" / "p0_patch_h1" / "best.pt"
LOSS = ROOT / "outputs" / "model_runs_tf" / "p0_patch_h1" / "loss_curves.png"
ATTN = ROOT / "outputs" / "model_runs_tf" / "p0_patch_h1" / "attn_zongguan.png"


def _font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def copy_static():
    OUT.mkdir(exist_ok=True)
    shutil.copy(LOSS, OUT / "图5-1_训练曲线.png")
    if ATTN.exists():
        shutil.copy(ATTN, OUT / "图5-4_注意力.png")
    src = ROOT / "outputs" / "figures"
    mapping = {
        "fig3_2_hannancun_9params.png": "图3-2_汉南村九参数.png",
        "fig3_3_zongguan_raw.png": "图3-3_宗关原始序列.png",
        "fig3_4_zongguan_imputed.png": "图3-4_宗关插补叠加.png",
        "fig3_5_nh3n_outliers.png": "图3-5_氨氮箱线异常.png",
        "meteo_zongguan_check.png": "图3-6_气象对齐抽检.png",
        "A_mask.png": "图3-7_河网物理掩码.png",
    }
    for a, b in mapping.items():
        p = src / a
        if p.exists():
            shutil.copy(p, OUT / b)
    print("[copy] static figures")


def plot_ablation():
    _font()
    names = ["Persistence", "MD-DySTGCN\n(v3)", "P0\nHCA decoder", "P0+future meteo", "MD-DySTFormer\n(this work)"]
    rmse = [0.3337, 0.3090, 0.2959, 0.3337, 0.2950]
    mae = [0.1766, 0.1728, 0.1635, 0.1771, 0.1636]
    mape = [6.41, 6.06, 5.77, 6.42, 5.85]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w, rmse, w, label="RMSE", color="#2563eb")
    ax.bar(x, mae, w, label="MAE", color="#0d9488")
    ax.bar(x + w, [m / 20 for m in mape], w, label="MAPE/20", color="#d97706")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("z-space error (MAPE scaled)")
    ax.set_title("Horizon-1 ablation at Zongguan")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "图5-2_消融对比.png", dpi=150)
    plt.close(fig)
    print("[plot] ablation")


@torch.no_grad()
def plot_forecast(n_steps=200):
    _font()
    cfg = cfg_for_exp("p0_patch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDDySTGCNtf(cfg).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    A = torch.from_numpy(np.load(DATA_DIR / "A_mask.npy").astype(np.float32)).to(device)
    loader = DataLoader(NPZDataset(DATA_DIR / "test.npz", False, cfg["T_out"]), batch_size=64, shuffle=False)
    scaler = json.loads((DATA_DIR / "scaler_stats.json").read_text())
    mu = np.asarray(scaler["water_mu"], dtype=np.float64)
    std = np.asarray(scaler["water_std"], dtype=np.float64)
    zi = cfg["zongguan_idx"]
    preds, trues = [], []
    for batch in loader:
        X, M, Y, Mf = unpack_batch(batch, device, False)
        Yhat = model(X, M, A, Mf)
        preds.append(Yhat[:, 0, zi, :].float().cpu().numpy())
        trues.append(Y[:, 0, zi, :].float().cpu().numpy())
    pred = np.concatenate(preds, 0) * std + mu
    true = np.concatenate(trues, 0) * std + mu
    sl = slice(0, min(n_steps, pred.shape[0]))
    labels = ["WT", "pH", "DO", "COD-MN", "NH3-N", "TP", "TN", "EC", "TURB"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    t = np.arange(sl.stop)
    for ax, i, lab in zip(axes.ravel(), range(9), labels):
        ax.plot(t, true[sl, i], color="black", lw=1.0, label="observed")
        ax.plot(t, pred[sl, i], color="#dc2626", lw=1.0, ls="--", label="MD-DySTFormer")
        ax.set_title(lab)
        ax.grid(True, ls=":", alpha=0.5)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Zongguan horizon-1 forecast (first 200 test steps, physical units)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "图5-3_宗关预测.png", dpi=150)
    plt.close(fig)
    print("[plot] forecast", pred.shape)


if __name__ == "__main__":
    copy_static()
    plot_ablation()
    plot_forecast()
    print("done", OUT)
