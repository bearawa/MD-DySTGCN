"""Paper-style verification figures (Fig 3-2 ~ 3-5) plus meteo/A_mask checks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from . import config as cfg


def _setup_font():
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fig_hannancun_9grid(
    water: np.ndarray,
    time_index: pd.DatetimeIndex,
    out_path: Path,
    station: str = None,
):
    """Fig 3-2 style: 3x3 water-quality panels for Hannancun."""
    _setup_font()
    station = station or cfg.FIG_HANNANCUN
    si = cfg.STATION_INDEX[station]
    order = ["WT", "pH", "DO", "COD_MN", "NH3_N", "TP", "TN", "EC", "TURB"]
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#17becf",
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    for ax, feat, color in zip(axes, order, colors):
        ci = cfg.WATER_FEATURE_INDEX[feat]
        y = water[:, si, ci]
        ax.plot(time_index, y, color=color, lw=0.6)
        ax.set_title(cfg.WATER_DISPLAY[feat], fontsize=11)
        ax.grid(True, ls=":", alpha=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(f"{station}断面九项水质参数时间序列", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved {out_path}")


def fig_zongguan_raw_stack(
    water_pre: np.ndarray,
    time_index: pd.DatetimeIndex,
    out_path: Path,
    station: str = None,
):
    """Fig 3-3 style: WT / NH3-N / DO raw (with gaps) for Zongguan."""
    _setup_font()
    station = station or cfg.EVAL_STATION
    si = cfg.STATION_INDEX[station]
    specs = [
        ("WT", "水温 (WT)", "温度 (°C)", "#7b2cbf"),
        ("NH3_N", "氨氮 (NH3-N)", "浓度 (mg/L)", "#0d9488"),
        ("DO", "溶解氧 (DO)", "浓度 (mg/L)", "#2563eb"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, (feat, label, ylabel, color) in zip(axes, specs):
        ci = cfg.WATER_FEATURE_INDEX[feat]
        y = water_pre[:, si, ci]
        ax.plot(time_index, y, color=color, lw=0.7, label=label)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(True, ls=":", alpha=0.5)

    axes[-1].set_xlabel("监测时间")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(f"{station}断面水温、氨氮、溶解氧原始序列（插补前）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved {out_path}")


def _plot_impute_overlay(ax, time_index, original, imputed, impute_mask, color, label):
    """Plot original in color; imputed segments in red."""
    ax.plot(time_index, imputed, color=color, lw=0.7, label=f"原始观测 ({label})")
    # Red overlay only where imputed
    y_red = np.where(impute_mask, imputed, np.nan)
    ax.plot(time_index, y_red, color="red", lw=1.0, label="算法填补")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)


def fig_zongguan_impute_overlay(
    water_pre: np.ndarray,
    water_clean: np.ndarray,
    impute_mask: np.ndarray,
    time_index: pd.DatetimeIndex,
    out_path: Path,
    station: str = None,
):
    """Fig 3-4 style: red = imputed, colored = original observation."""
    _setup_font()
    station = station or cfg.EVAL_STATION
    si = cfg.STATION_INDEX[station]
    specs = [
        ("WT", "水温", "温度 (°C)", "#7b2cbf"),
        ("NH3_N", "氨氮", "浓度 (mg/L)", "#0d9488"),
        ("DO", "溶解氧", "浓度 (mg/L)", "#2563eb"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, (feat, label, ylabel, color) in zip(axes, specs):
        ci = cfg.WATER_FEATURE_INDEX[feat]
        # For display: show pre-impute (with NaN gaps) as "original", clean as filled
        original = water_pre[:, si, ci]
        imputed = water_clean[:, si, ci]
        mask = impute_mask[:, si, ci]
        ax.plot(time_index, original, color=color, lw=0.7, label=f"原始观测 ({label})")
        y_red = np.where(mask, imputed, np.nan)
        ax.plot(time_index, y_red, color="red", lw=1.0, label="算法填补")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)

    axes[-1].set_xlabel("监测时间")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(f"{station}断面分层插补结果", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved {out_path}")


def fig_nh3n_boxplot_outliers(
    water_raw_phys: np.ndarray,
    outlier_mask: np.ndarray,
    thresholds: Dict,
    time_index: pd.DatetimeIndex,
    out_path: Path,
    station: str = None,
    year: int = 2021,
):
    """Fig 3-5 style: (a) boxplot + threshold, (b) 2021 series with red outliers."""
    _setup_font()
    station = station or cfg.EVAL_STATION
    si = cfg.STATION_INDEX[station]
    ci = cfg.WATER_FEATURE_INDEX["NH3_N"]
    series = water_raw_phys[:, si, ci]
    thr = thresholds[station]["NH3_N"]
    upper = thr.get("upper", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) boxplot
    valid = series[~np.isnan(series)]
    bp = axes[0].boxplot(valid, vert=True, widths=0.4, patch_artist=True,
                         showfliers=False)
    bp["boxes"][0].set_facecolor("#cfe2f3")
    # plot outliers as red circles
    out_vals = series[outlier_mask[:, si, ci]]
    axes[0].scatter(np.ones_like(out_vals), out_vals, c="red", s=12, zorder=3, label="异常值")
    if not np.isnan(upper):
        axes[0].axhline(upper, color="orange", ls="--", lw=1.5,
                        label=f"判定阈值 ({upper:.2f})")
    axes[0].set_ylabel("浓度 (mg/L)")
    axes[0].set_title("(a) 异常判定统计原理")
    axes[0].set_xticks([])
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls=":", alpha=0.5)

    # (b) 2021 time series
    mask_year = (time_index.year == year)
    t = time_index[mask_year]
    y = series[mask_year]
    out_m = outlier_mask[mask_year, si, ci]
    axes[1].plot(t, y, color="black", lw=0.7, label="原始监测数据")
    if not np.isnan(upper):
        axes[1].axhline(upper, color="orange", ls="--", lw=1.5,
                        label=f"上限阈值 ({upper:.2f} mg/L)")
    axes[1].scatter(t[out_m], y[out_m], c="red", s=18, zorder=3, label="识别出的异常值")
    axes[1].set_xlabel("时间")
    axes[1].set_ylabel("浓度 (mg/L)")
    axes[1].set_title(f"(b) {station} NH3-N 异常值检测 ({year})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls=":", alpha=0.5)
    # Avoid locale-dependent Chinese strftime (can raise UnicodeEncodeError on Windows)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.suptitle("氨氮 NH3-N 箱线异常检测", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved {out_path}")


def fig_meteo_align_check(
    water_clean: np.ndarray,
    meteo: np.ndarray,
    time_index: pd.DatetimeIndex,
    out_path: Path,
    station: str = None,
):
    """Engineering check: precip / temp vs turbidity at Zongguan."""
    _setup_font()
    station = station or cfg.EVAL_STATION
    si = cfg.STATION_INDEX[station]
    turb = water_clean[:, si, cfg.WATER_FEATURE_INDEX["TURB"]]
    precip = meteo[:, si, 0]
    temp = meteo[:, si, 2]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_index, precip, color="#1d4ed8", lw=0.6)
    axes[0].set_ylabel("降水 (mm / 4h)")
    axes[0].set_title(f"{station} 气象-水质对齐抽检")
    axes[0].grid(True, ls=":", alpha=0.5)

    axes[1].plot(time_index, temp, color="#c2410c", lw=0.6)
    axes[1].set_ylabel("气温 (°C)")
    axes[1].grid(True, ls=":", alpha=0.5)

    axes[2].plot(time_index, turb, color="#0f766e", lw=0.6)
    axes[2].set_ylabel("浊度 (NTU)")
    axes[2].set_xlabel("时间")
    axes[2].grid(True, ls=":", alpha=0.5)
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved {out_path}")


def fig_amask_heatmap(A: np.ndarray, out_path: Path):
    _setup_font()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(cfg.N_STATIONS))
    ax.set_yticks(range(cfg.N_STATIONS))
    ax.set_xticklabels(cfg.STATION_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(cfg.STATION_NAMES, fontsize=8)
    ax.set_xlabel("下游 j")
    ax.set_ylabel("上游 i")
    ax.set_title("河网物理掩码 A_mask (链式有向)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] saved {out_path}")


def generate_all_figures(
    water_raw: np.ndarray,
    water_pre_impute: np.ndarray,
    water_clean: np.ndarray,
    meteo: Optional[np.ndarray],
    impute_mask: np.ndarray,
    outlier_mask: np.ndarray,
    thresholds: Dict,
    time_index: pd.DatetimeIndex,
    A: np.ndarray,
    figure_dir: Path,
    water_for_boxplot: np.ndarray,
):
    """Write all verification figures into figure_dir."""
    figure_dir = _ensure_dir(figure_dir)

    fig_hannancun_9grid(
        water_clean, time_index, figure_dir / "fig3_2_hannancun_9params.png"
    )
    fig_zongguan_raw_stack(
        water_pre_impute, time_index, figure_dir / "fig3_3_zongguan_raw.png"
    )
    fig_zongguan_impute_overlay(
        water_pre_impute,
        water_clean,
        impute_mask,
        time_index,
        figure_dir / "fig3_4_zongguan_imputed.png",
    )
    fig_nh3n_boxplot_outliers(
        water_for_boxplot,
        outlier_mask,
        thresholds,
        time_index,
        figure_dir / "fig3_5_nh3n_outliers.png",
    )
    fig_amask_heatmap(A, figure_dir / "A_mask.png")
    if meteo is not None:
        fig_meteo_align_check(
            water_clean, meteo, time_index, figure_dir / "meteo_zongguan_check.png"
        )
