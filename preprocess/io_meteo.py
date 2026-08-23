"""Load ERA5-Land ZIP-NetCDF files and bilinear-interpolate onto stations.

Note: local files are already 4-hourly (00/04/08/12/16/20), so accumulated
fluxes (tp, sro) at each timestamp are used directly as the 4h window totals
(equivalent to summing hourly values over [t-3, t] in the paper).
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from . import config as cfg


def _list_era5_files(meteo_dir: Optional[Path] = None) -> List[Path]:
    meteo_dir = Path(meteo_dir or cfg.METEO_DIR)
    files = sorted(meteo_dir.glob("era5_land_*.nc"))
    if not files:
        raise FileNotFoundError(f"No era5_land_*.nc under {meteo_dir}")
    return files


def _open_zip_nc(path: Path) -> xr.Dataset:
    """Outer .nc is a ZIP containing data_0.nc (HDF5/NetCDF4)."""
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path, engine="h5netcdf")

    tmp = tempfile.mkdtemp(prefix="era5_")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"Empty ZIP: {path}")
        zf.extract(names[0], tmp)
    inner = Path(tmp) / names[0]
    return xr.open_dataset(inner, engine="h5netcdf")


def bilinear_extract(
    da: xr.DataArray,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolate DataArray (time, lat, lon) onto station points.

    Returns array shape [T, N].
    """
    # xarray interp expects coordinate names
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"

    # Ensure lat ascending for interp
    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.sortby(lat_name)

    points_lat = xr.DataArray(lats, dims="station")
    points_lon = xr.DataArray(lons, dims="station")
    out = da.interp({lat_name: points_lat, lon_name: points_lon}, method="linear")
    return np.asarray(out.values, dtype=np.float64)


def load_meteo_aligned(
    time_index: pd.DatetimeIndex,
    meteo_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Extract station meteorology aligned to ``time_index``.

    Returns
    -------
    meteo : ndarray [T, N, 4]
        Channels: precipitation(mm), surface_runoff(mm), air_temp_2m(C), wind_speed_10m(m/s)
    meta : coverage stats
    """
    files = _list_era5_files(meteo_dir)
    lons = np.array([s["lon"] for s in cfg.STATIONS], dtype=np.float64)
    lats = np.array([s["lat"] for s in cfg.STATIONS], dtype=np.float64)

    T = len(time_index)
    N = cfg.N_STATIONS
    meteo = np.full((T, N, cfg.D_METEO), np.nan, dtype=np.float64)

    time_to_i = {pd.Timestamp(ts): i for i, ts in enumerate(time_index)}
    n_filled = 0

    print(f"[io_meteo] Loading {len(files)} monthly ERA5 files ...")
    for fi, path in enumerate(files):
        ds = _open_zip_nc(path)
        try:
            tcoord = "valid_time" if "valid_time" in ds.coords else "time"
            times = pd.to_datetime(ds[tcoord].values)

            tp = bilinear_extract(ds["tp"], lons, lats)      # [Tm, N] meters
            sro = bilinear_extract(ds["sro"], lons, lats)
            t2m = bilinear_extract(ds["t2m"], lons, lats)    # Kelvin
            u10 = bilinear_extract(ds["u10"], lons, lats)
            v10 = bilinear_extract(ds["v10"], lons, lats)
            wind = np.sqrt(u10 ** 2 + v10 ** 2)

            precip_mm = tp * 1000.0
            runoff_mm = sro * 1000.0
            temp_c = t2m - 273.15

            for j, ts in enumerate(times):
                key = pd.Timestamp(ts)
                if key not in time_to_i:
                    continue
                ti = time_to_i[key]
                meteo[ti, :, 0] = precip_mm[j]
                meteo[ti, :, 1] = runoff_mm[j]
                meteo[ti, :, 2] = temp_c[j]
                meteo[ti, :, 3] = wind[j]
                n_filled += 1
        finally:
            ds.close()

        if (fi + 1) % 12 == 0 or fi == len(files) - 1:
            print(f"[io_meteo]   processed {fi + 1}/{len(files)} months")

    cover = 1.0 - float(np.isnan(meteo).mean())
    print(
        f"[io_meteo] Aligned timesteps with any meteo write: {n_filled}; "
        f"cell coverage={100 * cover:.1f}%"
    )
    meta = {
        "n_files": len(files),
        "n_timestep_writes": n_filled,
        "coverage": cover,
        "features": cfg.METEO_FEATURES,
    }
    return meteo, meta
