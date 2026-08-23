# MD-DySTGCN 数据预处理交付说明

本文档说明汉江水质–气象双流数据集的预处理代码、运行方式、落盘产物与可视化检验结果，对应论文《基于气象动态图卷积网络的汉江水质参数预测研究与应用》第 3 章流程。

---

## 1. 交付范围

| 类别 | 内容 |
|------|------|
| 代码包 | [`preprocess/`](preprocess/) |
| 依赖清单 | [`requirements-preprocess.txt`](requirements-preprocess.txt) |
| 数组产物 | [`outputs/arrays/`](outputs/arrays/) |
| 检验图 | [`outputs/figures/`](outputs/figures/) |
| **不包含** | MD-DySTGCN 模型训练与推理代码 |

原始数据目录 [`原始数据/`](原始数据/) 只读使用，脚本不会改写其中文件。

---

## 2. 目录结构

```
cursor/
├── 原始数据/
│   ├── 汉江.xlsx                      # 水质（Sheet2 监测 + Sheet1 站点）
│   └── hanjiang(原始气象数据)/         # era5_land_YYYY_MM.nc（ZIP 封装）
├── preprocess/
│   ├── config.py                      # 路径、16 站、列映射、超参
│   ├── io_water.py                    # 水质读取与 4h 网格
│   ├── io_meteo.py                    # ERA5 解压、双线性插值、单位换算
│   ├── clean.py                       # 物理极值 → 箱线异常 → 分层插补
│   ├── dataset.py                     # A_mask、划分、Z-Score、滑窗
│   ├── visualize.py                   # 图 3-2～3-5 及抽检图
│   └── run_preprocess.py              # CLI 入口
├── outputs/
│   ├── arrays/                        # npy / npz / json / csv
│   └── figures/                       # 检验 PNG
└── requirements-preprocess.txt
```

---

## 3. 环境与运行

### 3.1 依赖安装

```bash
pip install -r requirements-preprocess.txt
```

主要依赖：`pandas`、`openpyxl`、`numpy`、`xarray`、`h5netcdf`、`scipy`、`matplotlib`。

### 3.2 一键运行

在项目根目录（`cursor/`）执行：

```bash
# 完整流水线（水质 + 气象 + 落盘 + 画图，约数分钟）
python -m preprocess.run_preprocess

# 仅水质清洗与图（跳过 ERA5，便于快速验图）
python -m preprocess.run_preprocess --skip-meteo

# 已有 outputs/arrays 时只重画图
python -m preprocess.run_preprocess --figures-only
```

---

## 4. 处理流程摘要

```
汉江.xlsx ──► 筛 16 站 + 4h 网格
                    │
                    ▼
            物理极值置 NaN
                    │
                    ▼
            箱线图异常置 NaN
                    │
                    ▼
            分层差异化插补
                    │
ERA5 ZIP-NC ──► 双线性插值到站点 ──► 与水质时间戳对齐
                    │
                    ▼
        时序 7:1:2 划分 → 训练集 Z-Score → 滑窗 Tin=168 / Tout=12
                    │
                    ▼
              outputs/arrays + outputs/figures
```

### 4.1 水质

- **站点（表 3-1 顺序）**：烈金坝 → … → 宗关（共 16 站）
- **9 参数**：WT, pH, DO, COD_MN, NH3_N, TP, TN, EC, TURB
- **时间**：2021-01-01 00:00 ~ 2025-05-31 20:00，频率 4h
- **丢弃**：南柳渡、黄金峡；叶绿素 / 藻密度列
- **插补**：
  - 平稳（WT, pH, EC）：线性插值
  - 突变（NH3_N, TP, TURB, **TN**）：连续缺失 ≤6 点线性，>6 点均值填补
  - 复合（DO, COD_MN）：同突变型长短阈值混合

### 4.2 气象（相对论文字面的适配）

本地 ERA5-Land 已是 **4 小时** 步长，故：

- 瞬时量（气温、风速）：直接时刻对齐
- 累积量（降水 `tp`、径流 `sro`）：该 4h 时刻值视为窗口累积通量（等价于原文对 `[t-3,t]` 小时求和）
- 单位：`m → mm`，`K → ℃`，风速 `√(u10²+v10²)`
- 文件外表为 `.nc`，实为 ZIP，内含 `data_0.nc`，用 `h5netcdf` 打开

### 4.3 图结构与样本

- **A_mask**：干流链式有向边（上游 `i` → 下游 `i+1`）
- **划分**：严格时序 **7 : 1 : 2**
- **Z-Score**：仅用训练集估计 μ/σ，再变换全部分割
- **窗口**：输入 168 步（28 天）、预测 12 步（48 小时）

---

## 5. 落盘产物说明

### 5.1 `outputs/arrays/`

| 文件 | 形状 / 说明 |
|------|-------------|
| `water_raw.npy` | `[T,16,9]` 网格化、未插补 |
| `water_clean.npy` | `[T,16,9]` 清洗插补后（物理量纲） |
| `meteo.npy` | `[T,16,4]` precip_mm, runoff_mm, temp_c, wind |
| `water_z.npy` / `meteo_z.npy` | Z-Score 后 |
| `impute_mask.npy` | bool，`True` = 算法填补 |
| `outlier_mask.npy` | bool，箱线异常 |
| `obs_mask.npy` / `phys_mask.npy` | 原始观测 / 物理越界 |
| `A_mask.npy` | `[16,16]` 河网物理掩码 |
| `scaler_stats.json` | 训练集 μ/σ |
| `boxplot_thresholds.json` | 各站各参数箱线阈值 |
| `timestamps.csv` | 时间轴 |
| `meta.json` | 站序、特征名、样本数等 |
| `train.npz` / `val.npz` / `test.npz` | 键 `X,M,Y` |

**样本张量约定**

- `X`：`[B, 168, 16, 9]` 历史水质（标准化）
- `M`：`[B, 168, 16, 4]` 历史气象（标准化）
- `Y`：`[B, 12, 16, 9]` 未来水质（标准化）

评估论文主指标时，对站点维切片 **宗关**（索引 15）即可。

### 5.2 本次运行规模（已生成）

| 项 | 数值 |
|----|------|
| T（时间步） | 9672 |
| 时序划分长度 | 6770 / 967 / 1935 |
| 滑窗样本数 | train **6591** / val **788** / test **1756** |
| 气象格点覆盖 | 100% |

---

## 6. 可视化检验

对照论文图 3-2～3-5，输出在 [`outputs/figures/`](outputs/figures/)：

| 文件 | 对照 | 内容 |
|------|------|------|
| `fig3_2_hannancun_9params.png` | 图 3-2 | 汉南村 9 参数 3×3 |
| `fig3_3_zongguan_raw.png` | 图 3-3 | 宗关 WT / NH3-N / DO（插补前） |
| `fig3_4_zongguan_imputed.png` | 图 3-4 | 红线 = 算法填补，彩线 = 原始 |
| `fig3_5_nh3n_outliers.png` | 图 3-5 | 箱线阈值 + 2021 异常点 |
| `meteo_zongguan_check.png` | 工程抽检 | 宗关降水 / 气温 / 浊度对齐 |
| `A_mask.png` | 拓扑 | 链式有向掩码热力图 |

---

## 7. 加载示例（下游训练）

```python
import json
import numpy as np

train = np.load("outputs/arrays/train.npz")
X, M, Y = train["X"], train["M"], train["Y"]  # 已 Z-Score

A = np.load("outputs/arrays/A_mask.npy")
with open("outputs/arrays/scaler_stats.json", encoding="utf-8") as f:
    stats = json.load(f)

# 宗关切片（论文主评估断面）
zongguan = 15
Y_zg = Y[:, :, zongguan, :]  # [B, 12, 9]

# 反标准化示例（单通道）
mu = np.array(stats["water_mu"])
std = np.array(stats["water_std"])
Y_phys = Y * std + mu
```

---

## 8. 已知约定与局限

1. **TN 插补类别**：原文未明确归类，本实现并入突变型（与 NH3_N / TP 同策）。
2. **A_mask**：采用干流相邻有向边；未建模支流汇入。
3. **箱线异常**：按站、按参数独立 IQR×1.5；极端水文事件尖峰可能被判为异常并插补，复现时如需保留真实暴雨脉冲，可放宽阈值或对 TURB 等做豁免。
4. **Sheet1「小河口」**：卫河站点，已忽略；坐标以论文表 3-1 硬编码为准（皇庄在 Sheet1 中缺失，不影响）。
5. 本交付 **不含** 模型训练；`train.npz` 可直接对接后续 MD-DySTGCN 实现。

---

## 9. 验收清单

- [x] 16 站 × 9 水质 × 4 气象，时间严格对齐
- [x] 物理极值 + 箱线异常 + 分层插补，并保留掩码
- [x] 训练集-only Z-Score，时序 7:1:2，Tin=168 / Tout=12
- [x] `A_mask` 与 `train/val/test.npz` 落盘
- [x] 图 3-2～3-5 风格检验图 + 气象 / 掩码抽检图

---

## 10. 联系与版本

| 项 | 说明 |
|----|------|
| 对应论文章节 | 第 3 章数据集构建与预处理；第 5.1 节实验设置 |
| 代码版本 | `preprocess` 0.1.0 |
| 生成命令 | `python -m preprocess.run_preprocess` |
