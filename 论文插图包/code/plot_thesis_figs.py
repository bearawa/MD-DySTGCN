#!/usr/bin/env python3
"""Chapter-5 thesis figures from real metrics and (optional) inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
ROOT = PACK.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from style import (
    DATA,
    MODEL_CN,
    MODEL_COLOR,
    MODEL_ORDER,
    OBS_COLOR,
    PRED_COLOR,
    WATER_CN,
    WATER_FEATURES,
    WATER_UNIT,
    apply_style,
    savefig,
)
from train_tf import NPZDataset, MDDySTGCNtf, cfg_for_exp, unpack_batch

ARRAY = ROOT / "outputs" / "arrays"
OUT_TF = ROOT / "outputs" / "model_runs_tf"


def load_h1(name: str):
    p = DATA / "runs" / name / "test_metrics.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    return raw.get("horizon_1", raw)


def load_history(name: str):
    p = DATA / "runs" / name / "history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_scaler():
    sc = json.loads((DATA / "scaler_stats.json").read_text())
    return np.asarray(sc["water_mu"], dtype=np.float64), np.asarray(sc["water_std"], dtype=np.float64)


def collect_comparison():
    rows = {}
    for name in MODEL_ORDER:
        h = load_h1(name)
        if h is None:
            continue
        rows[name] = {
            "MAE": float(h["MAE"]),
            "RMSE": float(h["RMSE"]),
            "MAPE": float(h["MAPE"]),
            "per_channel_rmse": h.get("per_channel_rmse") or {},
        }
    (DATA / "comparison_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rows


def fig_5_1_this():
    hist = load_history("p0")
    if hist is None:
        return
    apply_style()
    persist = 0.3337
    v3 = 0.3090
    best_ep = 23
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.8))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    ax[0].plot(ep, hist["train_loss"], color="#1d4ed8", lw=1.4, label="训练集")
    ax[0].plot(ep, hist["val_loss"], color="#c2410c", lw=1.4, label="验证集")
    ax[0].axvline(best_ep, color="#64748b", ls="--", lw=0.9, label=f"最佳 epoch {best_ep}")
    ax[0].set_xlabel("训练轮次")
    ax[0].set_ylabel("Huber 损失")
    ax[0].set_title("MD-DySTGCN-CA 训练 / 验证损失")
    ax[0].legend()
    ax[1].plot(ep, hist["val_rmse"], color="#15803d", lw=1.5, label="验证 RMSE")
    ax[1].axhline(persist, color="#6b7280", ls="--", lw=1.0, label=f"持续性 {persist:.3f}")
    ax[1].axhline(v3, color="#2563eb", ls="--", lw=1.0, label=f"MD-DySTGCN {v3:.3f}")
    ax[1].axvline(best_ep, color="#64748b", ls="--", lw=0.9)
    ax[1].set_xlabel("训练轮次")
    ax[1].set_ylabel("宗关 RMSE（标准化）")
    ax[1].set_title("验证集 horizon-1 RMSE")
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "ch5/图5-1_本文训练曲线.png")


def fig_5_1_multi():
    """Two panels: graph models vs classic TS (loss scales differ)."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharex=False)
    left = [("v3", "MD-DySTGCN"), ("p0_fut", "未来气象扩展"), ("p0", "MD-DySTGCN-CA（本文）")]
    right = [("lstm", "LSTM"), ("tcn", "TCN"), ("patchtst", "PatchTST")]
    for ax, group, title in (
        (axes[0], left, "图模型（全站加权 Huber）"),
        (axes[1], right, "纯时序基线（仅宗关 Huber）"),
    ):
        drawn = False
        for key, _lab in group:
            h = load_history(key)
            if not h or not h.get("train_loss"):
                continue
            ax.plot(np.arange(1, len(h["train_loss"]) + 1), h["train_loss"],
                    color=MODEL_COLOR[key], lw=1.5, label=MODEL_CN[key])
            drawn = True
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("训练损失")
        ax.set_title(title)
        if drawn:
            ax.legend()
    fig.suptitle("各模型训练损失下降曲线", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "ch5/图5-1_多模型训练损失.png")


def fig_5_1_senior_style():
    """Single-panel training-loss overlay in the senior-thesis Fig 5-1 style."""
    specs = [
        ("former_e50", "MD-DySTFormer (this work)", "#c00000", "-", "o", 1.6),
        ("v3_e50", "MD-DySTGCN", "#7f7f7f", "-", "o", 1.3),
        ("p0_e50", "P0", "#1f4e79", "--", "s", 1.3),
        ("p0_fut_e50", "P0+future meteo", "#c55a11", "-.", "^", 1.3),
        ("patchtst_e50", "PatchTST", "#7030a0", "-.", "v", 1.3),
        ("lstm_e50", "LSTM", "#548235", ":", "D", 1.3),
        ("tcn_e50", "TCN", "#2e75b6", ":", "*", 1.3),
    ]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif", "Nimbus Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for key, lab, color, ls, marker, lw in specs:
        h = load_history(key)
        if not h or not h.get("train_loss"):
            continue
        y = np.asarray(h["train_loss"], dtype=float)
        x = np.arange(1, len(y) + 1)
        ax.plot(
            x, y, color=color, ls=ls, lw=lw, marker=marker, markevery=5,
            markersize=5.5, markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=1.0, label=lab, zorder=3 if key.startswith("former") else 2,
        )
    ax.set_xlim(1, 50)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.set_title("Training Loss of Compared Models", fontsize=12, pad=8)
    ax.tick_params(axis="both", direction="in", which="both", top=True, right=True, labelsize=9)
    ax.minorticks_on()
    ax.grid(True, ls=":", color="#b0b0b0", alpha=0.85, which="major")
    ax.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False,
              edgecolor="#888888", framealpha=1.0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    fig.tight_layout()
    savefig(fig, "ch5/图5-1_多模型训练损失_同款.png", reset_style=False)


def fig_overall_bars(rows):
    apply_style()
    names = [k for k in MODEL_ORDER if k in rows and k != "ha"]
    x = np.arange(len(names))
    rmse = [rows[k]["RMSE"] for k in names]
    mae = [rows[k]["MAE"] for k in names]
    mape = [rows[k]["MAPE"] for k in names]
    labels = [MODEL_CN[k] for k in names]
    colors = [MODEL_COLOR[k] for k in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    w = 0.36
    axes[0].bar(x - w / 2, rmse, w, label="RMSE", color="#1d4ed8")
    axes[0].bar(x + w / 2, mae, w, label="MAE", color="#0d9488")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    axes[0].set_ylabel("标准化空间误差")
    axes[0].set_title("宗关 horizon-1 总体误差")
    axes[0].legend()
    axes[1].bar(x, mape, color=colors, edgecolor="white", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title("反标准化后的平均绝对百分比误差")
    fig.tight_layout()
    savefig(fig, "ch5/图5-2b_总体指标对比.png")


def fig_5_2_turb_do(rows):
    apply_style()
    mu, std = load_scaler()
    std_map = {f: s for f, s in zip(WATER_FEATURES, std)}
    names = [k for k in MODEL_ORDER if k in rows and rows[k].get("per_channel_rmse")]
    labels = [MODEL_CN[k] for k in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    for ax, feat, title, unit in (
        (axes[0], "TURB", "浊度 TURB", "NTU"),
        (axes[1], "DO", "溶解氧 DO", "mg/L"),
    ):
        vals = [rows[k]["per_channel_rmse"][feat] * std_map[feat] for k in names]
        cols = [MODEL_COLOR[k] for k in names]
        bars = ax.bar(x, vals, color=cols, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(f"RMSE ({unit})")
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("消融与基线：浊度、溶解氧物理量纲 RMSE", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "ch5/图5-2_浊度溶解氧RMSE.png")


def fig_per_channel(rows):
    apply_style()
    names = [k for k in MODEL_ORDER if k in rows and rows[k].get("per_channel_rmse")]
    # skip HA if it dwarfs the axis
    names_plot = [k for k in names if k != "ha"]
    x = np.arange(len(WATER_FEATURES))
    w = 0.09
    fig, ax = plt.subplots(figsize=(12.6, 4.8))
    n = len(names_plot)
    for i, k in enumerate(names_plot):
        vals = [rows[k]["per_channel_rmse"].get(f, np.nan) for f in WATER_FEATURES]
        ax.bar(x + (i - n / 2) * w + w / 2, vals, w, label=MODEL_CN[k], color=MODEL_COLOR[k])
    ax.set_xticks(x)
    ax.set_xticklabels([WATER_CN[f] for f in WATER_FEATURES], rotation=20, ha="right")
    ax.set_ylabel("标准化 RMSE")
    ax.set_title("宗关各水质通道 horizon-1 RMSE")
    ax.legend(ncol=4, fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "ch5/图5-2c_分通道RMSE.png")


def plot_forecast_array(true, pred, title, rel, n_steps=200):
    apply_style()
    sl = slice(0, min(n_steps, true.shape[0]))
    t = np.arange(sl.stop)
    fig, axes = plt.subplots(3, 3, figsize=(12.2, 8.2), sharex=True)
    for ax, i, feat in zip(axes.ravel(), range(9), WATER_FEATURES):
        ax.plot(t, true[sl, i], color=OBS_COLOR, lw=1.05, label="观测值")
        ax.plot(t, pred[sl, i], color=PRED_COLOR, lw=1.05, ls="--", label="预测值")
        ax.set_title(f"{WATER_CN[feat]}（{WATER_UNIT[feat]}）", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    axes[2, 1].set_xlabel("测试集时间步（4 h / 步）")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, rel)


def forecasts_from_npy():
    mu, std = load_scaler()
    true_p = None
    for name in MODEL_ORDER:
        p = DATA / "runs" / name / "pred_zongguan.npy"
        t = DATA / "runs" / name / "true_zongguan.npy"
        if not p.exists():
            continue
        pred = np.load(p) * std + mu
        true = np.load(t) * std + mu
        true_p = true
        plot_forecast_array(
            true, pred,
            f"宗关九通道 horizon-1 预测（{MODEL_CN[name]}）",
            f"ch5/pred_{name}.png",
        )
    return true_p


@torch.no_grad()
def infer_graph(name: str, exp: str, ckpt_path: Path, use_future=False):
    dest = DATA / "runs" / name / "pred_zongguan.npy"
    if dest.exists():
        return
    if not ckpt_path.exists():
        print(f"[infer] skip {name}, no ckpt")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg_for_exp(exp)
    cfg["T_out"] = 1
    model = MDDySTGCNtf(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    A = torch.from_numpy(np.load(ARRAY / "A_mask.npy").astype(np.float32)).to(device)
    loader = DataLoader(
        NPZDataset(ARRAY / "test.npz", use_future, 1),
        batch_size=64, shuffle=False, num_workers=2,
    )
    zi = 15
    preds, trues = [], []
    for batch in loader:
        X, M, Y, Mf = unpack_batch(batch, device, use_future)
        Yhat = model(X, M, A, Mf)
        preds.append(Yhat[:, 0, zi, :].float().cpu().numpy())
        trues.append(Y[:, 0, zi, :].float().cpu().numpy())
    pred = np.concatenate(preds, 0)
    true = np.concatenate(trues, 0)
    (DATA / "runs" / name).mkdir(parents=True, exist_ok=True)
    np.save(DATA / "runs" / name / "pred_zongguan.npy", pred.astype(np.float32))
    np.save(DATA / "runs" / name / "true_zongguan.npy", true.astype(np.float32))
    print(f"[infer] {name} {pred.shape}")


@torch.no_grad()
def fig_5_4_attn():
    ckpt_path = OUT_TF / "p0_h1" / "best.pt"
    if not ckpt_path.exists():
        return
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg_for_exp("p0")
    cfg["T_out"] = 1
    model = MDDySTGCNtf(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    A = torch.from_numpy(np.load(ARRAY / "A_mask.npy").astype(np.float32)).to(device)
    loader = DataLoader(NPZDataset(ARRAY / "test.npz", False, 1), batch_size=1, shuffle=False)
    batch = next(iter(loader))
    X, M, Y, Mf = unpack_batch(batch, device, False)
    _, attn_w = model(X, M, A, Mf, return_attn=True)
    w = attn_w.detach().cpu().numpy()
    heat = w[15] if w.shape[0] >= 16 else w.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10.6, 3.1))
    im = ax.imshow(heat, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("历史时间步（0 为窗口最早时刻）")
    ax.set_ylabel("预测步")
    ax.set_title("宗关断面 horizon 查询对 168 步历史的注意力")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="注意力权重")
    fig.tight_layout()
    savefig(fig, "ch5/图5-4_注意力.png")


def fig_5_3_main():
    mu, std = load_scaler()
    p = DATA / "runs" / "p0" / "pred_zongguan.npy"
    t = DATA / "runs" / "p0" / "true_zongguan.npy"
    if not p.exists():
        return
    plot_forecast_array(
        np.load(t) * std + mu,
        np.load(p) * std + mu,
        "宗关断面九项水质 horizon-1 预测（MD-DySTGCN-CA）",
        "ch5/图5-3_宗关预测.png",
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only-senior-loss", action="store_true",
                   help="redraw 图5-1 同款 from *_e50 only; do not touch other figs/metrics")
    args = p.parse_args()
    if args.only_senior_loss:
        fig_5_1_senior_style()
        print("[ch5] senior loss only")
        return
    apply_style()
    from collect_existing import main as copy_existing
    copy_existing()
    rows = collect_comparison()
    print("models with metrics:", list(rows))
    fig_5_1_this()
    fig_5_1_multi()
    fig_5_1_senior_style()
    fig_overall_bars(rows)
    fig_5_2_turb_do(rows)
    fig_per_channel(rows)

    infer_graph("former", "p0_patch", OUT_TF / "p0_patch_h1" / "best.pt", False)
    infer_graph("p0", "p0", OUT_TF / "p0_h1" / "best.pt", False)
    infer_graph("p0_fut", "p0_fut", OUT_TF / "p0_fut_h1" / "best.pt", True)
    forecasts_from_npy()
    fig_5_3_main()
    fig_5_4_attn()
    print("[ch5] done")


if __name__ == "__main__":
    main()
