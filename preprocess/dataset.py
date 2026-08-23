"""Graph mask, chronological split, Z-score, and sliding windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from . import config as cfg


def build_amask_chain(n: int = None) -> np.ndarray:
    """
    Directed chain adjacency: upstream i -> downstream i+1.

    A_mask[i, j] = 1 if i is the direct upstream neighbor of j
    (paper: A_mask_ij = 1 when i is upstream of j with direct hydraulic link).
    Convention used: row i influences column j, so A[i, i+1] = 1.
    """
    n = cfg.N_STATIONS if n is None else n
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    return A


def split_time_indices(T: int, ratios=(0.7, 0.1, 0.2)) -> Dict[str, slice]:
    """Chronological 7:1:2 split on the time axis."""
    r_train, r_val, r_test = ratios
    assert abs(r_train + r_val + r_test - 1.0) < 1e-6
    n_train = int(T * r_train)
    n_val = int(T * r_val)
    n_test = T - n_train - n_val
    return {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, T),
        "sizes": (n_train, n_val, n_test),
    }


def fit_zscore_train_only(
    water: np.ndarray,
    meteo: np.ndarray,
    train_slice: slice,
) -> Dict:
    """
    Fit per-channel mean/std on the training time range only (all stations pooled).

    water: [T,N,C], meteo: [T,N,D]
    """
    w_tr = water[train_slice]  # [Tt, N, C]
    m_tr = meteo[train_slice]

    w_mu = np.nanmean(w_tr.reshape(-1, w_tr.shape[-1]), axis=0)
    w_std = np.nanstd(w_tr.reshape(-1, w_tr.shape[-1]), axis=0)
    w_std = np.where(w_std < 1e-8, 1.0, w_std)

    m_mu = np.nanmean(m_tr.reshape(-1, m_tr.shape[-1]), axis=0)
    m_std = np.nanstd(m_tr.reshape(-1, m_tr.shape[-1]), axis=0)
    m_std = np.where(m_std < 1e-8, 1.0, m_std)

    stats = {
        "water_mu": w_mu.tolist(),
        "water_std": w_std.tolist(),
        "water_features": cfg.WATER_FEATURES,
        "meteo_mu": m_mu.tolist(),
        "meteo_std": m_std.tolist(),
        "meteo_features": cfg.METEO_FEATURES,
    }
    print("[dataset] Z-score fitted on train split only.")
    return stats


def apply_zscore(water: np.ndarray, meteo: np.ndarray, stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
    w_mu = np.asarray(stats["water_mu"], dtype=np.float64)
    w_std = np.asarray(stats["water_std"], dtype=np.float64)
    m_mu = np.asarray(stats["meteo_mu"], dtype=np.float64)
    m_std = np.asarray(stats["meteo_std"], dtype=np.float64)
    water_z = (water - w_mu) / w_std
    meteo_z = (meteo - m_mu) / m_std
    return water_z, meteo_z


def make_windows(
    water_z: np.ndarray,
    meteo_z: np.ndarray,
    time_slice: slice,
    t_in: int = None,
    t_out: int = None,
):
    """
    Build sliding windows inside a chronological segment.

    Returns X [B, Tin, N, C], M [B, Tin, N, D], Y [B, Tout, N, C],
    M_future [B, Tout, N, D] (ERA5 as 48h-forecast proxy; same indices as Y).
    Windows use absolute indices; only windows fully contained in the segment.
    """
    t_in = cfg.T_IN if t_in is None else t_in
    t_out = cfg.T_OUT if t_out is None else t_out

    start = time_slice.start or 0
    stop = time_slice.stop if time_slice.stop is not None else water_z.shape[0]

    xs, ms, ys, mfs = [], [], [], []
    # Need Tin history ending at t, and Tout future after t
    # Sample indexed by end-of-input time t (absolute)
    for t in range(start + t_in - 1, stop - t_out):
        x = water_z[t - t_in + 1 : t + 1]
        m = meteo_z[t - t_in + 1 : t + 1]
        y = water_z[t + 1 : t + 1 + t_out]
        mf = meteo_z[t + 1 : t + 1 + t_out]
        xs.append(x)
        ms.append(m)
        ys.append(y)
        mfs.append(mf)

    if not xs:
        n, c = water_z.shape[1], water_z.shape[2]
        d = meteo_z.shape[2]
        return (
            np.zeros((0, t_in, n, c)),
            np.zeros((0, t_in, n, d)),
            np.zeros((0, t_out, n, c)),
            np.zeros((0, t_out, n, d)),
        )

    return np.stack(xs), np.stack(ms), np.stack(ys), np.stack(mfs)


def save_dataset_bundle(
    out_dir: Path,
    water_raw: np.ndarray,
    water_clean: np.ndarray,
    meteo: np.ndarray,
    impute_mask: np.ndarray,
    outlier_mask: np.ndarray,
    time_index,
    stats: Dict,
    splits: Dict,
    water_z: np.ndarray,
    meteo_z: np.ndarray,
) -> Dict[str, int]:
    """Persist arrays, scaler, timestamps, A_mask, and windowed npz files."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A = build_amask_chain()
    np.save(out_dir / "water_raw.npy", water_raw)
    np.save(out_dir / "water_clean.npy", water_clean)
    np.save(out_dir / "meteo.npy", meteo)
    np.save(out_dir / "impute_mask.npy", impute_mask)
    np.save(out_dir / "outlier_mask.npy", outlier_mask)
    np.save(out_dir / "A_mask.npy", A)
    np.save(out_dir / "water_z.npy", water_z)
    np.save(out_dir / "meteo_z.npy", meteo_z)

    with open(out_dir / "scaler_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    pd.DataFrame({"timestamp": time_index}).to_csv(
        out_dir / "timestamps.csv", index=False
    )

    counts = {}
    for split_name in ("train", "val", "test"):
        sl = splits[split_name]
        X, M, Y, Mf = make_windows(water_z, meteo_z, sl)
        np.savez_compressed(out_dir / f"{split_name}.npz", X=X, M=M, Y=Y, M_future=Mf)
        counts[split_name] = int(X.shape[0])
        print(
            f"[dataset] {split_name}: samples={X.shape[0]}, "
            f"X={X.shape}, M={M.shape}, Y={Y.shape}, M_future={Mf.shape}"
        )

    meta = {
        "stations": cfg.STATION_NAMES,
        "water_features": cfg.WATER_FEATURES,
        "meteo_features": cfg.METEO_FEATURES,
        "T_in": cfg.T_IN,
        "T_out": cfg.T_OUT,
        "split_sizes_time": list(splits["sizes"]),
        "sample_counts": counts,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return counts


def rebuild_window_npz(out_dir: Path = None) -> Dict[str, int]:
    """Rebuild train/val/test.npz from existing water_z / meteo_z (no raw reload)."""
    out_dir = Path(out_dir) if out_dir is not None else cfg.ARRAY_DIR
    water_z = np.load(out_dir / "water_z.npy")
    meteo_z = np.load(out_dir / "meteo_z.npy")
    splits = split_time_indices(water_z.shape[0], cfg.SPLIT_RATIOS)
    print(f"[dataset] rebuild windows from {out_dir}  T={water_z.shape[0]} split={splits['sizes']}")

    counts = {}
    for split_name in ("train", "val", "test"):
        sl = splits[split_name]
        X, M, Y, Mf = make_windows(water_z, meteo_z, sl)
        np.savez_compressed(out_dir / f"{split_name}.npz", X=X, M=M, Y=Y, M_future=Mf)
        counts[split_name] = int(X.shape[0])
        print(
            f"[dataset] {split_name}: samples={X.shape[0]}, "
            f"X={X.shape}, M={M.shape}, Y={Y.shape}, M_future={Mf.shape}"
        )

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {
            "stations": cfg.STATION_NAMES,
            "water_features": cfg.WATER_FEATURES,
            "meteo_features": cfg.METEO_FEATURES,
            "T_in": cfg.T_IN,
            "T_out": cfg.T_OUT,
            "split_sizes_time": list(splits["sizes"]),
        }
    meta["sample_counts"] = counts
    meta["has_M_future"] = True
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return counts
