#!/usr/bin/env python3
"""Thesis schematic figures: 1-1, 2-1, 4-1~4-4."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style import apply_style, savefig


def _box(ax, xy, w, h, text, fc="#e8eef6", ec="#1f4e79", fs=8.5, bold=False):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=1.1,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
        fontweight="bold" if bold else "normal", wrap=True,
    )
    return p


def _arrow(ax, x1, y1, x2, y2, color="#334155"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.15),
    )


def fig_1_1():
    apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("研究技术路线（数据—模型—实验）", fontsize=13, pad=8)

    _box(ax, (0.3, 5.1), 2.4, 1.3, "原始水质 Excel\n+ ERA5-Land 月文件", fc="#dbeafe")
    _box(ax, (3.2, 5.1), 2.5, 1.3, "清洗 / 插补 / 对齐\n河网掩码 / 滑窗", fc="#dbeafe")
    _box(ax, (6.2, 5.1), 2.6, 1.3, "MD-DySTFormer\n训练（AdamW + Huber）", fc="#fee2e2")
    _box(ax, (9.3, 5.1), 2.4, 1.3, "宗关 horizon-1\n评估与消融", fc="#dcfce7")
    _arrow(ax, 2.7, 5.75, 3.2, 5.75)
    _arrow(ax, 5.7, 5.75, 6.2, 5.75)
    _arrow(ax, 8.8, 5.75, 9.3, 5.75)

    _box(ax, (0.3, 2.7), 3.5, 1.7, "数据治理\n16 站 × 9 通道 × 4 h\n物理极值 + 箱线 + 分层插补", fc="#eff6ff")
    _box(ax, (4.2, 2.7), 3.6, 1.7, "时空建模\nPatchTST 水质塔 + TCN 气象塔\n动态图 + 门控 + 交叉注意力", fc="#fff7ed")
    _box(ax, (8.2, 2.7), 3.5, 1.7, "实验验证\n持续性 / v3 / P0 / 经典基线\n本文方法对比", fc="#f0fdf4")
    _arrow(ax, 1.8, 5.1, 1.8, 4.4)
    _arrow(ax, 7.5, 5.1, 6.0, 4.4)
    _arrow(ax, 10.5, 5.1, 10.0, 4.4)

    _box(ax, (0.3, 0.35), 11.4, 1.6,
         "本文范围止于数据—模型—实验，不含预警系统落地。\n"
         "训练任务：输入 168 步双流历史，预测下一 4 小时时刻 9 维水质（残差解码）。",
         fc="#f8fafc", ec="#64748b", fs=9)
    savefig(fig, "ch1/图1-1_技术路线.png")


def fig_2_1():
    apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("气象驱动下的水质多变量耦合演变机理", fontsize=13, pad=8)
    _box(ax, (0.4, 4.6), 2.6, 1.2, "降水 / 径流", fc="#dbeafe")
    _box(ax, (3.7, 4.6), 2.6, 1.2, "气温 / 风速", fc="#dbeafe")
    _box(ax, (7.0, 4.6), 2.6, 1.2, "人类活动（点源/面源）", fc="#fde68a")
    _arrow(ax, 1.7, 4.6, 1.7, 3.7)
    _arrow(ax, 5.0, 4.6, 5.0, 3.7)
    _arrow(ax, 8.3, 4.6, 8.3, 3.7)
    _box(ax, (0.4, 2.5), 4.2, 1.15, "冲刷抬升浊度、营养盐\n持续降雨可能稀释溶解性物质", fc="#e0f2fe")
    _box(ax, (5.4, 2.5), 4.2, 1.15, "热力与复氧过程\n影响 DO / 反应速率", fc="#ffedd5")
    _arrow(ax, 5.0, 2.5, 5.0, 1.7)
    _box(ax, (2.2, 0.35), 5.6, 1.3, "九项水质共变\n（但不在网络中做跨通道注意力混合）", fc="#fee2e2", bold=True)
    savefig(fig, "ch2/图2-1_气象驱动机理.png")


def fig_4_1():
    apply_style()
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("MD-DySTFormer 总体结构", fontsize=13, pad=8)

    _box(ax, (0.3, 6.2), 2.6, 1.2, "水质历史 X\n168×16×9", fc="#dbeafe")
    _box(ax, (0.3, 4.4), 2.6, 1.2, "气象历史 M\n168×16×4", fc="#dbeafe")
    _box(ax, (3.4, 6.2), 3.2, 1.2, "通道独立 PatchTST\n水质塔  patch_len=8", fc="#ede9fe")
    _box(ax, (3.4, 4.4), 3.2, 1.2, "空洞 TCN 气象塔\n6 层 膨胀 1…32", fc="#ccfbf1")
    _box(ax, (3.4, 2.6), 3.2, 1.2, "气象注意力动态邻接\n+ 河网掩码", fc="#fef3c7")
    _arrow(ax, 2.9, 6.8, 3.4, 6.8)
    _arrow(ax, 2.9, 5.0, 3.4, 5.0)
    _arrow(ax, 1.6, 4.4, 1.6, 3.2)
    _arrow(ax, 2.9, 3.2, 3.4, 3.2)

    _box(ax, (7.2, 6.2), 2.8, 1.2, "DynGCN 水质路\n4 层", fc="#e0e7ff")
    _box(ax, (7.2, 4.4), 2.8, 1.2, "DynGCN 气象路\n4 层", fc="#e0e7ff")
    _arrow(ax, 6.6, 6.8, 7.2, 6.8)
    _arrow(ax, 6.6, 5.0, 7.2, 5.0)
    _arrow(ax, 5.0, 3.8, 7.2, 6.4)
    _arrow(ax, 5.0, 3.2, 7.2, 4.8)

    _box(ax, (10.5, 5.1), 3.1, 1.5, "门控融合\nH = g⊙Zm+(1-g)⊙Zw", fc="#ffe4e6")
    _arrow(ax, 10.0, 6.8, 10.5, 6.2)
    _arrow(ax, 10.0, 5.0, 10.5, 5.6)

    _box(ax, (7.2, 1.5), 3.4, 1.5, "Horizon 交叉注意力解码\n1 个查询 × 168 步记忆", fc="#fce7f3")
    _box(ax, (10.9, 1.5), 2.7, 1.5, "残差输出\nŶ = Xt + Δ", fc="#fee2e2", bold=True)
    _arrow(ax, 12.0, 5.1, 8.9, 3.0)
    _arrow(ax, 10.6, 2.25, 10.9, 2.25)
    _box(ax, (0.3, 0.25), 6.4, 1.0, "评估：宗关断面 · horizon-1 · 9 通道", fc="#f8fafc", ec="#64748b", fs=9)
    savefig(fig, "ch4/图4-1_MD-DySTFormer结构.png")


def fig_4_2():
    apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.set_title("气象驱动的动态邻接矩阵生成", fontsize=13, pad=8)
    labels = [
        ("站点气象 Mt\nN×4", "#dbeafe"),
        ("嵌入 + 点积\n相似度 S", "#ede9fe"),
        ("MLP 得亲和\n+ 河网掩码", "#fef3c7"),
        ("行 softmax\n得到 At", "#fee2e2"),
    ]
    for i, (txt, fc) in enumerate(labels):
        x = 0.35 + i * 2.7
        _box(ax, (x, 1.15), 2.3, 1.5, txt, fc=fc)
        if i < 3:
            _arrow(ax, x + 2.3, 1.9, x + 2.7, 1.9)
    savefig(fig, "ch4/图4-2_动态邻接.png")


def fig_4_3():
    apply_style()
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("残差空洞 TCN 模块（本文仅用于气象塔）", fontsize=13, pad=8)
    _box(ax, (0.4, 2.1), 1.8, 1.1, "输入 xt", fc="#dbeafe")
    _box(ax, (2.7, 3.3), 2.4, 1.1, "空洞因果卷积\n膨胀 d", fc="#ccfbf1")
    _box(ax, (5.5, 3.3), 2.0, 1.1, "ReLU + Dropout", fc="#ccfbf1")
    _box(ax, (2.7, 0.8), 2.4, 1.1, "1×1 投影", fc="#e2e8f0")
    _box(ax, (8.0, 2.1), 1.7, 1.1, "残差相加", fc="#fee2e2", bold=True)
    _arrow(ax, 2.2, 2.65, 2.7, 3.85)
    _arrow(ax, 5.1, 3.85, 5.5, 3.85)
    _arrow(ax, 7.5, 3.85, 8.4, 3.2)
    _arrow(ax, 2.2, 2.4, 2.7, 1.35)
    _arrow(ax, 5.1, 1.35, 8.0, 2.4)
    savefig(fig, "ch4/图4-3_残差TCN.png")


def fig_4_4():
    apply_style()
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    ax.set_title("双路动态图卷积与物理感知门控融合", fontsize=13, pad=8)
    _box(ax, (0.4, 3.3), 2.6, 1.1, "水质塔特征 Zw", fc="#ede9fe")
    _box(ax, (0.4, 0.7), 2.6, 1.1, "气象塔特征 Zm", fc="#ccfbf1")
    _box(ax, (3.6, 3.3), 2.6, 1.1, "DynGCN × 4", fc="#e0e7ff")
    _box(ax, (3.6, 0.7), 2.6, 1.1, "DynGCN × 4", fc="#e0e7ff")
    _box(ax, (6.8, 1.8), 3.7, 1.6, "门控 g=σ([Zw;Zm]W)\nH = g⊙Zm + (1-g)⊙Zw", fc="#ffe4e6", bold=True)
    _arrow(ax, 3.0, 3.85, 3.6, 3.85)
    _arrow(ax, 3.0, 1.25, 3.6, 1.25)
    _arrow(ax, 6.2, 3.85, 7.4, 3.2)
    _arrow(ax, 6.2, 1.25, 7.4, 2.0)
    savefig(fig, "ch4/图4-4_门控融合.png")


def main():
    fig_1_1()
    fig_2_1()
    fig_4_1()
    fig_4_2()
    fig_4_3()
    fig_4_4()
    print("[schematic] done")


if __name__ == "__main__":
    main()
