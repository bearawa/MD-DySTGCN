"""Physical bounds, boxplot outliers, and stratified imputation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import config as cfg


def apply_physical_bounds(water: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set out-of-physical-range values to NaN.

    Returns cleaned array and bool mask of violations (True = violated).
    """
    out = water.copy()
    viol = np.zeros_like(water, dtype=bool)
    for ci, feat in enumerate(cfg.WATER_FEATURES):
        lo, hi = cfg.PHYSICAL_BOUNDS[feat]
        col = out[:, :, ci]
        bad = (~np.isnan(col)) & ((col < lo) | (col > hi))
        viol[:, :, ci] = bad
        col[bad] = np.nan
        out[:, :, ci] = col
    n = int(viol.sum())
    print(f"[clean] Physical-bound violations set to NaN: {n}")
    return out, viol


def boxplot_outliers(
    water: np.ndarray,
    k: float = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Per-station, per-feature IQR boxplot outliers -> NaN.

    Returns cleaned array, outlier mask, and threshold dict.
    """
    k = cfg.BOXPLOT_K if k is None else k
    out = water.copy()
    outlier = np.zeros_like(water, dtype=bool)
    thresholds: Dict[str, Dict[str, Dict[str, float]]] = {}

    for si, name in enumerate(cfg.STATION_NAMES):
        thresholds[name] = {}
        for ci, feat in enumerate(cfg.WATER_FEATURES):
            series = out[:, si, ci]
            valid = series[~np.isnan(series)]
            if valid.size < 8:
                thresholds[name][feat] = {
                    "Q1": np.nan,
                    "Q3": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                }
                continue
            q1 = float(np.percentile(valid, 25))
            q3 = float(np.percentile(valid, 75))
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            thresholds[name][feat] = {
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "lower": lower,
                "upper": upper,
            }
            bad = (~np.isnan(series)) & ((series < lower) | (series > upper))
            outlier[:, si, ci] = bad
            series = series.copy()
            series[bad] = np.nan
            out[:, si, ci] = series

    n = int(outlier.sum())
    print(f"[clean] Boxplot outliers set to NaN: {n}")
    return out, outlier, thresholds


def _gap_runs(isnan: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive index runs where isnan is True."""
    runs = []
    n = len(isnan)
    i = 0
    while i < n:
        if not isnan[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and isnan[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _linear_interpolate_1d(y: np.ndarray) -> np.ndarray:
    """Linear interpolate NaNs; leave leading/trailing NaNs for later fill."""
    out = y.copy()
    idx = np.arange(len(out))
    good = ~np.isnan(out)
    if good.sum() < 2:
        return out
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def _mean_fill_run(y: np.ndarray, start: int, end: int, window: int = 42) -> None:
    """
    Fill a long gap with historical / sliding-window mean.

    Prefer mean of same clock-of-day slots in a local window; fallback to global mean.
    """
    n = len(y)
    # local window excluding the gap itself
    left = y[max(0, start - window) : start]
    right = y[end + 1 : min(n, end + 1 + window)]
    neigh = np.concatenate([left, right])
    neigh = neigh[~np.isnan(neigh)]
    if neigh.size > 0:
        fill = float(np.mean(neigh))
    else:
        valid = y[~np.isnan(y)]
        fill = float(np.mean(valid)) if valid.size else 0.0

    # Same hour-of-day bias: 6 samples/day, use modulo-6 peers
    period = 6
    slot = start % period
    peers = y[slot::period]
    peers = peers[~np.isnan(peers)]
    if peers.size >= 3:
        fill = float(np.mean(peers))

    y[start : end + 1] = fill


def _impute_series(y: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Impute one 1-D series.

    mode: 'linear' | 'spike' | 'hybrid'
    Returns (imputed, impute_mask) where impute_mask marks originally-NaN cells that were filled.
    """
    out = y.copy().astype(np.float64)
    was_nan = np.isnan(out)
    if not was_nan.any():
        return out, np.zeros_like(out, dtype=bool)

    if mode == "linear":
        out = _linear_interpolate_1d(out)
        # fill any remaining edge NaNs with nearest valid / mean
        if np.isnan(out).any():
            valid = out[~np.isnan(out)]
            fill = float(np.mean(valid)) if valid.size else 0.0
            out[np.isnan(out)] = fill
        return out, was_nan

    # spike / hybrid: short gaps linear, long gaps mean
    runs = _gap_runs(was_nan)
    # First pass: mark long gaps and mean-fill them
    long_mask = np.zeros_like(out, dtype=bool)
    for s, e in runs:
        length = e - s + 1
        if length > cfg.SHORT_GAP_MAX:
            long_mask[s : e + 1] = True
            _mean_fill_run(out, s, e)

    # Remaining NaNs (short gaps + edges): linear then edge fill
    out = _linear_interpolate_1d(out)
    if np.isnan(out).any():
        valid = out[~np.isnan(out)]
        fill = float(np.mean(valid)) if valid.size else 0.0
        out[np.isnan(out)] = fill

    return out, was_nan


def stratified_impute(water: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified imputation by feature category.

    Returns water_clean [T,N,C] and impute_mask (True = algorithm-filled).
    """
    out = water.copy()
    impute_mask = np.zeros_like(water, dtype=bool)

    for ci, feat in enumerate(cfg.WATER_FEATURES):
        if feat in cfg.STABLE_FEATURES:
            mode = "linear"
        elif feat in cfg.SPIKE_FEATURES:
            mode = "spike"
        else:
            mode = "hybrid"  # DO, COD_MN

        for si in range(water.shape[1]):
            series = out[:, si, ci]
            filled, mask = _impute_series(series, mode)
            out[:, si, ci] = filled
            impute_mask[:, si, ci] = mask

    n = int(impute_mask.sum())
    remain = int(np.isnan(out).sum())
    print(f"[clean] Imputed cells: {n}; remaining NaN: {remain}")
    if remain:
        # Final safety fill
        for ci in range(out.shape[2]):
            col = out[:, :, ci]
            if np.isnan(col).any():
                fill = float(np.nanmean(col))
                if np.isnan(fill):
                    fill = 0.0
                nan_idx = np.isnan(col)
                col[nan_idx] = fill
                impute_mask[:, :, ci] |= nan_idx
                out[:, :, ci] = col
        print(f"[clean] Safety fill applied; remaining NaN: {int(np.isnan(out).sum())}")

    return out, impute_mask


def clean_pipeline(water_raw: np.ndarray) -> Dict:
    """Full clean: physical bounds -> boxplot -> stratified impute."""
    w1, phys_mask = apply_physical_bounds(water_raw)
    w2, outlier_mask, thresholds = boxplot_outliers(w1)
    # Combined pre-impute NaN sources for visualization of "raw after QC"
    pre_impute = w2
    water_clean, impute_mask = stratified_impute(pre_impute)
    return {
        "water_pre_impute": pre_impute,
        "water_clean": water_clean,
        "phys_mask": phys_mask,
        "outlier_mask": outlier_mask,
        "impute_mask": impute_mask,
        "thresholds": thresholds,
    }
