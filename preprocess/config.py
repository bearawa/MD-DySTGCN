"""Paths, station table, column maps, and hyperparameters for MD-DySTGCN preprocessing."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "原始数据"
WATER_XLSX = RAW_DIR / "汉江.xlsx"
METEO_DIR = RAW_DIR / "hanjiang(原始气象数据)"
OUTPUT_DIR = ROOT / "outputs"
ARRAY_DIR = OUTPUT_DIR / "arrays"
FIGURE_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Stations (Table 3-1 order: upstream -> downstream)
# ---------------------------------------------------------------------------
STATIONS = [
    {"name": "烈金坝", "lon": 106.2589, "lat": 33.0438, "city": "汉中市"},
    {"name": "梁西渡", "lon": 106.9289, "lat": 33.1074, "city": "汉中市"},
    {"name": "小钢桥", "lon": 108.2210, "lat": 33.0361, "city": "安康市"},
    {"name": "老君关", "lon": 109.0742, "lat": 32.7156, "city": "安康市"},
    {"name": "羊尾", "lon": 110.1478, "lat": 32.8153, "city": "十堰市"},
    {"name": "陈家坡", "lon": 110.9144, "lat": 32.8139, "city": "十堰市"},
    {"name": "沈湾", "lon": 111.6000, "lat": 32.4600, "city": "襄阳市"},
    {"name": "白家湾", "lon": 112.0409, "lat": 32.0584, "city": "襄阳市"},
    {"name": "余家湖", "lon": 112.1764, "lat": 31.9142, "city": "襄阳市"},
    {"name": "转斗", "lon": 112.4403, "lat": 31.4661, "city": "荆门市"},
    {"name": "皇庄", "lon": 112.5683, "lat": 31.1964, "city": "荆门市"},
    {"name": "罗汉闸", "lon": 112.6081, "lat": 30.6823, "city": "荆门市"},
    {"name": "岳口", "lon": 113.0694, "lat": 30.5017, "city": "天门市"},
    {"name": "汉南村", "lon": 113.2414, "lat": 30.2351, "city": "仙桃市"},
    {"name": "小河", "lon": 113.9455, "lat": 30.6806, "city": "孝感市"},
    {"name": "宗关", "lon": 114.2177, "lat": 30.5773, "city": "武汉市"},
]

STATION_NAMES = [s["name"] for s in STATIONS]
N_STATIONS = len(STATION_NAMES)
STATION_INDEX = {name: i for i, name in enumerate(STATION_NAMES)}

# Extra sections present in Excel but not used
EXTRA_SECTIONS = {"南柳渡", "黄金峡", "小河口"}

# ---------------------------------------------------------------------------
# Water-quality columns: Excel Chinese name -> model symbol
# Order matches C=9 feature channels
# ---------------------------------------------------------------------------
WATER_COL_MAP = {
    "水温(℃)": "WT",
    "pH": "pH",
    "溶解氧(mg/L)": "DO",
    "高锰酸盐指数(mg/L)": "COD_MN",
    "氨氮(mg/L)": "NH3_N",
    "总磷(mg/L)": "TP",
    "总氮(mg/L)": "TN",
    "电导率(μS/cm)": "EC",
    "浊度(NTU)": "TURB",
}

WATER_FEATURES = ["WT", "pH", "DO", "COD_MN", "NH3_N", "TP", "TN", "EC", "TURB"]
C_WATER = len(WATER_FEATURES)
WATER_FEATURE_INDEX = {name: i for i, name in enumerate(WATER_FEATURES)}

# Display labels for figures (paper style)
WATER_DISPLAY = {
    "WT": "WT",
    "pH": "PH",
    "DO": "DO",
    "COD_MN": "COD-MN",
    "NH3_N": "NH3-N",
    "TP": "TP",
    "TN": "TN",
    "EC": "EC",
    "TURB": "TURB",
}

# Stratified imputation categories (TN grouped with spike type; paper unspecified)
STABLE_FEATURES = {"WT", "pH", "EC"}
SPIKE_FEATURES = {"NH3_N", "TP", "TURB", "TN"}
HYBRID_FEATURES = {"DO", "COD_MN"}

# Short gap: <= 6 points (= 24 h at 4 h sampling)
SHORT_GAP_MAX = 6

# ---------------------------------------------------------------------------
# Physical bounds (out-of-range -> NaN)
# ---------------------------------------------------------------------------
PHYSICAL_BOUNDS = {
    "WT": (-5.0, 45.0),
    "pH": (0.0, 14.0),
    "DO": (0.0, 20.0),
    "COD_MN": (0.0, 50.0),
    "NH3_N": (0.0, 20.0),
    "TP": (0.0, 5.0),
    "TN": (0.0, 30.0),
    "EC": (1.0, 5000.0),  # EC==0 treated as invalid
    "TURB": (0.0, 2000.0),
}

BOXPLOT_K = 1.5

# ---------------------------------------------------------------------------
# Meteorology
# ---------------------------------------------------------------------------
METEO_FEATURES = ["precipitation", "surface_runoff", "air_temp_2m", "wind_speed_10m"]
D_METEO = len(METEO_FEATURES)
# ERA5 variable mapping after extraction
# tp (m, accum) -> precipitation (mm)
# sro (m, accum) -> surface_runoff (mm)
# t2m (K, instant) -> air_temp_2m (C)
# sqrt(u10^2+v10^2) -> wind_speed_10m (m/s)

# ---------------------------------------------------------------------------
# Time / window / split
# ---------------------------------------------------------------------------
TIME_START = "2021-01-01 00:00:00"
TIME_END = "2025-05-31 20:00:00"
FREQ = "4h"  # pandas>=2.2 uses lowercase 'h' (was '4H')

T_IN = 168   # 28 days
T_OUT = 12   # 48 hours
SPLIT_RATIOS = (0.7, 0.1, 0.2)  # train / val / test, chronological

EVAL_STATION = "宗关"
FIG_HANNANCUN = "汉南村"
