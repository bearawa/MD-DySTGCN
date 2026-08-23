"""CLI entry: run MD-DySTGCN preprocessing and verification figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow `python -m preprocess.run_preprocess` from repo root
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "preprocess"

from . import config as cfg
from .clean import apply_physical_bounds, boxplot_outliers, stratified_impute
from .dataset import (
    apply_zscore,
    build_amask_chain,
    fit_zscore_train_only,
    save_dataset_bundle,
    split_time_indices,
)
from .io_meteo import load_meteo_aligned
from .io_water import crosscheck_sheet1_stations, load_raw_water, load_station_table
from .visualize import generate_all_figures


def _print_summary(water_raw, water_clean, impute_mask, outlier_mask, meteo, counts):
    print("\n========== Preprocess Summary ==========")
    print(f"Stations ({cfg.N_STATIONS}): {cfg.STATION_NAMES}")
    print(f"Water shape: {water_raw.shape}  |  Clean NaN: {int(np.isnan(water_clean).sum())}")
    print(f"Outlier cells: {int(outlier_mask.sum())}  |  Imputed cells: {int(impute_mask.sum())}")
    if meteo is not None:
        print(f"Meteo shape: {meteo.shape}  |  coverage: {100*(1-np.isnan(meteo).mean()):.1f}%")
    print(f"Window samples: {counts}")
    # Per-station missing rate before impute (on raw grid)
    print("\nMissing rate by station (raw grid, mean over 9 features):")
    for name in cfg.STATION_NAMES:
        si = cfg.STATION_INDEX[name]
        rate = float(np.isnan(water_raw[:, si, :]).mean())
        print(f"  {name:6s}: {100*rate:5.1f}%")
    print("========================================\n")


def run(skip_meteo: bool = False, figures_only: bool = False):
    cfg.ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if figures_only:
        print("[run] figures-only mode: loading arrays from outputs/arrays ...")
        water_raw = np.load(cfg.ARRAY_DIR / "water_raw.npy")
        water_clean = np.load(cfg.ARRAY_DIR / "water_clean.npy")
        impute_mask = np.load(cfg.ARRAY_DIR / "impute_mask.npy")
        outlier_mask = np.load(cfg.ARRAY_DIR / "outlier_mask.npy")
        A = np.load(cfg.ARRAY_DIR / "A_mask.npy")
        import pandas as pd
        import json

        time_index = pd.DatetimeIndex(
            pd.to_datetime(pd.read_csv(cfg.ARRAY_DIR / "timestamps.csv")["timestamp"])
        )
        meteo = None
        meteo_path = cfg.ARRAY_DIR / "meteo.npy"
        if meteo_path.exists() and not skip_meteo:
            meteo = np.load(meteo_path)
        # Rebuild pre-impute approx: clean where not imputed, else nan
        water_pre = water_clean.copy()
        water_pre[impute_mask] = np.nan
        # thresholds not saved — recompute quickly for fig3-5
        w_phys, _ = apply_physical_bounds(water_raw)
        _, _, thresholds = boxplot_outliers(w_phys)
        generate_all_figures(
            water_raw=water_raw,
            water_pre_impute=water_pre,
            water_clean=water_clean,
            meteo=meteo,
            impute_mask=impute_mask,
            outlier_mask=outlier_mask,
            thresholds=thresholds,
            time_index=time_index,
            A=A,
            figure_dir=cfg.FIGURE_DIR,
            water_for_boxplot=w_phys,
        )
        print("[run] figures regenerated.")
        return

    # ---- 1) stations ----
    stations = load_station_table()
    print(stations.to_string(index=False))
    try:
        crosscheck_sheet1_stations()
    except Exception as e:
        print(f"[run] Sheet1 crosscheck skipped: {e}")

    # ---- 2) water ----
    water_raw, time_index, obs_mask, water_meta = load_raw_water()

    # ---- 3) clean ----
    w_phys, phys_mask = apply_physical_bounds(water_raw)
    w_box, outlier_mask, thresholds = boxplot_outliers(w_phys)
    water_clean, impute_mask = stratified_impute(w_box)
    water_pre_impute = w_box  # after QC, before impute (NaNs remain)

    # ---- 4) meteo ----
    meteo = None
    if not skip_meteo:
        meteo, meteo_meta = load_meteo_aligned(time_index)
        # Fill any rare meteo NaNs with train-time channel means later; for now linear along time
        for si in range(meteo.shape[1]):
            for di in range(meteo.shape[2]):
                col = meteo[:, si, di]
                if np.isnan(col).any():
                    idx = np.arange(len(col))
                    good = ~np.isnan(col)
                    if good.sum() >= 2:
                        col[~good] = np.interp(idx[~good], idx[good], col[good])
                    elif good.sum() == 1:
                        col[:] = col[good][0]
                    else:
                        col[:] = 0.0
                    meteo[:, si, di] = col
    else:
        print("[run] --skip-meteo: using zeros for meteorology placeholders")
        meteo = np.zeros((len(time_index), cfg.N_STATIONS, cfg.D_METEO), dtype=np.float64)

    # ---- 5) split / zscore / windows ----
    splits = split_time_indices(len(time_index), cfg.SPLIT_RATIOS)
    print(f"[run] time split sizes (train/val/test): {splits['sizes']}")
    stats = fit_zscore_train_only(water_clean, meteo, splits["train"])
    water_z, meteo_z = apply_zscore(water_clean, meteo, stats)
    A = build_amask_chain()

    counts = save_dataset_bundle(
        out_dir=cfg.ARRAY_DIR,
        water_raw=water_raw,
        water_clean=water_clean,
        meteo=meteo,
        impute_mask=impute_mask,
        outlier_mask=outlier_mask,
        time_index=time_index,
        stats=stats,
        splits=splits,
        water_z=water_z,
        meteo_z=meteo_z,
    )
    np.save(cfg.ARRAY_DIR / "obs_mask.npy", obs_mask)
    np.save(cfg.ARRAY_DIR / "phys_mask.npy", phys_mask)

    # Save thresholds for reproducibility
    import json

    thr_serializable = {
        st: {
            feat: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}
            for feat, d in feats.items()
        }
        for st, feats in thresholds.items()
    }
    with open(cfg.ARRAY_DIR / "boxplot_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thr_serializable, f, ensure_ascii=False, indent=2)

    # ---- 6) figures ----
    generate_all_figures(
        water_raw=water_raw,
        water_pre_impute=water_pre_impute,
        water_clean=water_clean,
        meteo=None if skip_meteo else meteo,
        impute_mask=impute_mask,
        outlier_mask=outlier_mask,
        thresholds=thresholds,
        time_index=time_index,
        A=A,
        figure_dir=cfg.FIGURE_DIR,
        water_for_boxplot=w_phys,
    )

    _print_summary(water_raw, water_clean, impute_mask, outlier_mask, meteo, counts)
    print(f"[run] arrays -> {cfg.ARRAY_DIR}")
    print(f"[run] figures -> {cfg.FIGURE_DIR}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MD-DySTGCN preprocessing pipeline")
    parser.add_argument(
        "--skip-meteo",
        action="store_true",
        help="Skip ERA5 loading (faster water-only QC + figures)",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures from existing outputs/arrays",
    )
    args = parser.parse_args(argv)
    run(skip_meteo=args.skip_meteo, figures_only=args.figures_only)


if __name__ == "__main__":
    main()
