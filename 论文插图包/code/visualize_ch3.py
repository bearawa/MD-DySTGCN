#!/usr/bin/env python3
"""Chapter-3 thesis figures from outputs/arrays (real series)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import (
    IMPUTE_COLOR,
    OUTLIER_COLOR,
    ROOT,
    THRESH_COLOR,
    WATER_CN,
    WATER_FEATURES,
    WATER_UNIT,
    apply_style,
    savefig,
)

sys.path.insert(0, str(ROOT))
from preprocess import config as cfg
from preprocess.dataset import build_amask_chain

ARRAY = ROOT / "outputs" / "arrays"
COLORS_9 = [
    "#1f4e79", "#c55a11", "#2e7d32", "#c00000", "#7030a0",
    "#7f6000", "#c2185b", "#595959", "#00838f",
]


def _load():
    water_raw = np.load(ARRAY / "water_raw.npy")
    water_clean = np.load(ARRAY / "water_clean.npy")
    impute_mask = np.load(ARRAY / "impute_mask.npy")
    outlier_mask = np.load(ARRAY / "outlier_mask.npy")
    meteo = np.load(ARRAY / "meteo.npy") if (ARRAY / "meteo.npy").exists() else None
    time_index = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ARRAY / "timestamps.csv")["timestamp"]))
    water_pre = water_clean.copy()
    water_pre[impute_mask] = np.nan
    return water_raw, water_clean, water_pre, impute_mask, outlier_mask, meteo, time_index


def fig_3_1_stations():
    apply_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    lons = [s["lon"] for s in cfg.STATIONS]
    lats = [s["lat"] for s in cfg.STATIONS]
    ax.plot(lons, lats, color="#9ca3af", lw=1.0, zorder=1)
    ax.scatter(lons[:-1], lats[:-1], c="#1d4ed8", s=36, zorder=2, label="干流断面")
    ax.scatter([lons[-1]], [lats[-1]], c="#c00000", s=64, zorder=3, label="评估站点（宗关）")
    for s in cfg.STATIONS:
        ax.annotate(s["name"], (s["lon"], s["lat"]), textcoords="offset points",
                    xytext=(4, 3), fontsize=7)
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.set_title("汉江干流 16 个水质监测断面空间分布")
    ax.legend(loc="lower left")
    ax.set_aspect("equal", adjustable="datalim")
    savefig(fig, "ch3/图3-1_监测断面分布.png")


def fig_3_2(water_clean, time_index):
    apply_style()
    si = cfg.STATION_INDEX["汉南村"]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.4), sharex=True)
    for ax, feat, color in zip(axes.ravel(), WATER_FEATURES, COLORS_9):
        ax.plot(time_index, water_clean[:, si, cfg.WATER_FEATURE_INDEX[feat]], color=color, lw=0.55)
        ax.set_title(f"{WATER_CN[feat]}（{WATER_UNIT[feat]}）", fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("汉南村断面九项水质参数时间序列", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "ch3/图3-2_汉南村九参数.png")


def fig_3_3(water_pre, time_index):
    apply_style()
    si = cfg.STATION_INDEX["宗关"]
    specs = [
        ("WT", "#7b2cbf", "温度 (°C)"),
        ("NH3_N", "#0d9488", "浓度 (mg/L)"),
        ("DO", "#1d4ed8", "浓度 (mg/L)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.6), sharex=True)
    for ax, (feat, color, ylab) in zip(axes, specs):
        ax.plot(time_index, water_pre[:, si, cfg.WATER_FEATURE_INDEX[feat]],
                color=color, lw=0.65, label=WATER_CN[feat])
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("监测时间")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("宗关断面水温、氨氮、溶解氧原始序列（插补前）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "ch3/图3-3_宗关原始序列.png")


def fig_3_4(water_pre, water_clean, impute_mask, time_index):
    apply_style()
    si = cfg.STATION_INDEX["宗关"]
    specs = [
        ("WT", "#7b2cbf", "温度 (°C)"),
        ("NH3_N", "#0d9488", "浓度 (mg/L)"),
        ("DO", "#1d4ed8", "浓度 (mg/L)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.6), sharex=True)
    for ax, (feat, color, ylab) in zip(axes, specs):
        ci = cfg.WATER_FEATURE_INDEX[feat]
        ax.plot(time_index, water_pre[:, si, ci], color=color, lw=0.65, label="原始观测")
        y_red = np.where(impute_mask[:, si, ci], water_clean[:, si, ci], np.nan)
        ax.plot(time_index, y_red, color=IMPUTE_COLOR, lw=1.05, label="算法填补")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("监测时间")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("宗关断面分层插补结果", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "ch3/图3-4_宗关插补叠加.png")


def fig_3_5(water_raw, outlier_mask, time_index, year=2021):
    apply_style()
    si = cfg.STATION_INDEX["宗关"]
    ci = cfg.WATER_FEATURE_INDEX["NH3_N"]
    series = water_raw[:, si, ci]
    valid = series[~np.isnan(series)]
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    bp = axes[0].boxplot(valid, vert=True, widths=0.4, patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("#cfe2f3")
    out_vals = series[outlier_mask[:, si, ci]]
    axes[0].scatter(np.ones_like(out_vals), out_vals, c=OUTLIER_COLOR, s=12, zorder=3, label="异常值")
    axes[0].axhline(upper, color=THRESH_COLOR, ls="--", lw=1.4, label=f"上限阈值 ({upper:.2f})")
    axes[0].set_ylabel("浓度 (mg/L)")
    axes[0].set_title("(a) 箱线异常判定")
    axes[0].set_xticks([])
    axes[0].legend(fontsize=8)

    mask_year = time_index.year == year
    t = time_index[mask_year]
    y = series[mask_year]
    out_m = outlier_mask[mask_year, si, ci]
    axes[1].plot(t, y, color="black", lw=0.7, label="原始监测数据")
    axes[1].axhline(upper, color=THRESH_COLOR, ls="--", lw=1.4, label=f"上限阈值 ({upper:.2f} mg/L)")
    axes[1].scatter(t[out_m], y[out_m], c=OUTLIER_COLOR, s=16, zorder=3, label="识别出的异常值")
    axes[1].set_xlabel("时间")
    axes[1].set_ylabel("浓度 (mg/L)")
    axes[1].set_title(f"(b) 宗关氨氮异常值检测（{year}）")
    axes[1].legend(fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("氨氮 NH3-N 箱线异常检测", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "ch3/图3-5_氨氮箱线异常.png")


def fig_3_6(water_clean, meteo, time_index):
    apply_style()
    si = cfg.STATION_INDEX["宗关"]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.4), sharex=True)
    axes[0].plot(time_index, meteo[:, si, 0], color="#1d4ed8", lw=0.55)
    axes[0].set_ylabel("降水 (mm / 4h)")
    axes[1].plot(time_index, meteo[:, si, 2], color="#c2410c", lw=0.55)
    axes[1].set_ylabel("气温 (°C)")
    axes[2].plot(time_index, water_clean[:, si, cfg.WATER_FEATURE_INDEX["TURB"]], color="#0f766e", lw=0.55)
    axes[2].set_ylabel("浊度 (NTU)")
    axes[2].set_xlabel("时间")
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("宗关断面气象通道与浊度对齐抽检", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "ch3/图3-6_气象对齐抽检.png")


def fig_3_7():
    apply_style()
    A = build_amask_chain()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(cfg.N_STATIONS))
    ax.set_yticks(range(cfg.N_STATIONS))
    ax.set_xticklabels(cfg.STATION_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(cfg.STATION_NAMES, fontsize=8)
    ax.set_xlabel("下游断面 j")
    ax.set_ylabel("上游断面 i")
    ax.set_title("河网链式有向物理掩码 $A_{\\mathrm{mask}}$")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    savefig(fig, "ch3/图3-7_河网物理掩码.png")


def main():
    apply_style()
    water_raw, water_clean, water_pre, impute_mask, outlier_mask, meteo, time_index = _load()
    fig_3_1_stations()
    fig_3_2(water_clean, time_index)
    fig_3_3(water_pre, time_index)
    fig_3_4(water_pre, water_clean, impute_mask, time_index)
    fig_3_5(water_raw, outlier_mask, time_index)
    if meteo is not None:
        fig_3_6(water_clean, meteo, time_index)
    fig_3_7()
    print("[ch3] done")


if __name__ == "__main__":
    main()
