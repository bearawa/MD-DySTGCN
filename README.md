# 汉江水质预测：MD-DySTGCN 与 MD-DySTFormer

基于汉江 16 个干流断面、9 项水质指标与 ERA5-Land 气象场的时空预测实验仓库。

- 复现骨架：**MD-DySTGCN**（气象驱动动态图 + 双塔 TCN + 门控融合），见 `train_improved.py` 与 `MD_DySTGCN_Local_v3*.ipynb`。
- 本文最终方案：**MD-DySTFormer**（通道独立 PatchTST 水质塔 + horizon 交叉注意力解码），见 `train_tf.py`。
- 学位论文草稿：[`基于气象动态图与分片Transformer的汉江水质参数预测研究.md`](基于气象动态图与分片Transformer的汉江水质参数预测研究.md)（不含原文第 6 章预警系统）。

主评估协议为 **horizon-1**（下一 4 小时时刻）、站点 **宗关**。最终权重测试集：MAE 0.1636 / RMSE 0.2950 / MAPE 5.85%。

## 目录

| 路径 | 说明 |
|------|------|
| `preprocess/` | 水质清洗、ERA5 对齐、划分、滑窗、第 3 章插图 |
| `原始数据/` | `汉江.xlsx` + ERA5-Land 月文件 |
| `outputs/arrays/` | 标准化序列、掩码、scaler；**滑窗 npz 不入库** |
| `outputs/figures/` | 预处理复现图 |
| `outputs/model_runs_v3_nb/` | 复现版 MD-DySTGCN（v3）权重与指标 |
| `outputs/model_runs_tf/p0_patch_h1/` | MD-DySTFormer 最终权重与训练曲线 |
| `train_improved.py` | v3 训练 |
| `train_tf.py` | Transformer 混合实验（`p0` / `p0_fut` / `p0_patch`） |
| `plot_thesis_figs.py` | 论文第 5 章插图 |
| `图片/` | 原文扫描件 + 本文 PNG |
| `reports/` | 组会报告 |

## 环境

```bash
pip install -r requirements-preprocess.txt
pip install -r requirements-train.txt   # PyTorch 请按本机 CUDA 自行安装
```

## 数据与预处理

滑窗 `train.npz` / `val.npz` / `test.npz` 超过 GitHub 单文件 100MB 限制，**不提交**。有原始数据时在仓库根目录执行：

```bash
python -m preprocess.run_preprocess
```

产物：`outputs/arrays/`（含 npz）与 `outputs/figures/`。样本数约 train 6591 / val 788 / test 1756。训练脚本读 npz 后将 `Y` 切为 `T_out=1`。

## 训练与评估

```bash
# 复现 v3（12 步解码，评估可看第一步）
python train_improved.py

# 本文最终方案 horizon-1
python train_tf.py --exp p0_patch
```

权重与 `test_metrics.json` 写在 `outputs/model_runs_tf/<exp>_h1/`。

## 论文插图

```bash
python plot_thesis_figs.py
```

生成 `图片/图5-1_训练曲线.png` 等；第 3 章图从 `outputs/figures/` 拷贝。不要把原文扫描件 `图5-1.jpg`–`图5-3.jpg`、`图6-*.jpg` 当作本文实验结果。
