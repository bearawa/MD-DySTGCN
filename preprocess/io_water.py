"""Load and grid Hanjiang water-quality Excel into [T, N, C] arrays."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as cfg


def load_station_table() -> pd.DataFrame:
    """Return the paper Table 3-1 station table (hard-coded coordinates)."""
    return pd.DataFrame(cfg.STATIONS)


def crosscheck_sheet1_stations(xlsx_path=None) -> pd.DataFrame:
    """Compare Sheet1 Hanjiang stations against Table 3-1; print diffs."""
    xlsx_path = xlsx_path or cfg.WATER_XLSX
    sheet1 = pd.read_excel(xlsx_path, sheet_name="Sheet1", engine="openpyxl")
    name_col = "断面" if "断面" in sheet1.columns else sheet1.columns[1]
    lon_col = "经度" if "经度" in sheet1.columns else None
    lat_col = "纬度" if "纬度" in sheet1.columns else None
    river_col = "河流" if "河流" in sheet1.columns else None

    df = sheet1.copy()
    if river_col:
        df = df[df[river_col].astype(str).str.contains("汉江", na=False)]

    paper = {s["name"]: (s["lon"], s["lat"]) for s in cfg.STATIONS}
    rows = []
    for _, r in df.iterrows():
        name = str(r[name_col]).strip()
        lon = float(r[lon_col]) if lon_col else np.nan
        lat = float(r[lat_col]) if lat_col else np.nan
        if name in paper:
            plon, plat = paper[name]
            rows.append(
                {
                    "name": name,
                    "sheet1_lon": lon,
                    "sheet1_lat": lat,
                    "paper_lon": plon,
                    "paper_lat": plat,
                    "d_lon": abs(lon - plon),
                    "d_lat": abs(lat - plat),
                }
            )
        else:
            print(f"[io_water] Sheet1 station not in Table 3-1 (skipped): {name}")

    missing = set(cfg.STATION_NAMES) - set(df[name_col].astype(str).str.strip())
    if missing:
        print(f"[io_water] Table 3-1 stations missing in Sheet1: {sorted(missing)}")

    return pd.DataFrame(rows)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_raw_water(
    xlsx_path=None,
) -> Tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, Dict]:
    """
    Load Sheet2, keep 16 Table-3-1 stations, reindex to a complete 4h grid.

    Returns
    -------
    water_raw : ndarray, shape [T, N, C]
    time_index : DatetimeIndex length T
    obs_mask : bool ndarray [T, N, C]  True where an original (pre-grid) observation existed
    meta : dict with missing rates etc.
    """
    xlsx_path = xlsx_path or cfg.WATER_XLSX
    print(f"[io_water] Reading {xlsx_path} ...")
    df = pd.read_excel(xlsx_path, sheet_name="Sheet2", engine="openpyxl")

    section_col = "断面名称"
    time_col = "监测时间"
    if section_col not in df.columns or time_col not in df.columns:
        raise KeyError(f"Expected columns {section_col}/{time_col}, got {list(df.columns)}")

    df[time_col] = pd.to_datetime(df[time_col])
    df[section_col] = df[section_col].astype(str).str.strip()

    all_sections = set(df[section_col].unique())
    extra = all_sections - set(cfg.STATION_NAMES)
    if extra:
        print(f"[io_water] Dropping extra sections: {sorted(extra)}")
    df = df[df[section_col].isin(cfg.STATION_NAMES)].copy()

    rename = {zh: en for zh, en in cfg.WATER_COL_MAP.items() if zh in df.columns}
    missing_cols = [zh for zh in cfg.WATER_COL_MAP if zh not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing water columns in Excel: {missing_cols}")

    keep = [section_col, time_col] + list(rename.keys())
    df = df[keep].rename(columns=rename)

    for feat in cfg.WATER_FEATURES:
        df[feat] = _coerce_numeric(df[feat])

    # Aggregate duplicate timestamps (mean of concurrent readings)
    df = (
        df.groupby([time_col, section_col], as_index=False)[cfg.WATER_FEATURES]
        .mean()
    )

    time_index = pd.date_range(cfg.TIME_START, cfg.TIME_END, freq=cfg.FREQ)
    T = len(time_index)
    N = cfg.N_STATIONS
    C = cfg.C_WATER

    water_raw = np.full((T, N, C), np.nan, dtype=np.float64)
    obs_mask = np.zeros((T, N, C), dtype=bool)

    time_to_i = {ts: i for i, ts in enumerate(time_index)}

    for _, row in df.iterrows():
        ts = row[time_col]
        if ts not in time_to_i:
            # Snap to nearest 4h if within 1h tolerance of a grid point
            nearest = time_index[np.argmin(np.abs(time_index - ts))]
            if abs((nearest - ts).total_seconds()) <= 3600:
                ti = time_to_i[nearest]
            else:
                continue
        else:
            ti = time_to_i[ts]

        si = cfg.STATION_INDEX[row[section_col]]
        for ci, feat in enumerate(cfg.WATER_FEATURES):
            val = row[feat]
            if pd.isna(val):
                continue
            water_raw[ti, si, ci] = float(val)
            obs_mask[ti, si, ci] = True

    # Per-station missing rates on the grid
    miss_rates = {}
    for name in cfg.STATION_NAMES:
        si = cfg.STATION_INDEX[name]
        miss = np.isnan(water_raw[:, si, :]).mean(axis=0)
        miss_rates[name] = {feat: float(miss[ci]) for ci, feat in enumerate(cfg.WATER_FEATURES)}

    n_obs = int(obs_mask.sum())
    n_total = T * N * C
    print(
        f"[io_water] Grid T={T}, N={N}, C={C}; "
        f"observed cells={n_obs}/{n_total} ({100 * n_obs / n_total:.1f}%)"
    )

    meta = {
        "T": T,
        "N": N,
        "C": C,
        "n_obs": n_obs,
        "miss_rates": miss_rates,
        "dropped_sections": sorted(extra),
    }
    return water_raw, time_index, obs_mask, meta
