#!/usr/bin/env python3
"""MD-DySTGCN + Transformer decoder (horizon-1 only).

Experiments:
  eval_v3   v3 checkpoint, report horizon-1
  p0        HorizonCrossAttnDecoder
  p0_ch     P0 + TURB/NH3-N channel Huber weights
  p0_fut    P0 + future-meteo cross-attention (ERA5 4h proxy)
  p0_patch  P0 + channel-independent PatchTST water tower
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "arrays"
V3_CKPT = ROOT / "outputs" / "model_runs_v3_nb" / "md_dystgcn_v3_best.pt"
WATER_FEATURES = ["WT", "pH", "DO", "COD_MN", "NH3_N", "TP", "TN", "EC", "TURB"]
# Frozen v3_nb baseline (same test.npz, Zongguan, z-space)
V3_H1 = {"MAE": 0.172787, "RMSE": 0.308992, "MAPE": 6.06497, "TURB": 0.5537, "NH3_N": 0.4091}

BASE_CFG = dict(
    T_in=168,
    T_out=1,  # horizon-1 only; 12-step protocol dropped
    N=16,
    C=9,
    D=4,
    F=128,
    d_E=64,
    mlp_hidden=128,
    L_tcn=6,
    K_tcn=3,
    L_g=4,
    dropout=0.1,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    huber_delta=1.0,
    epochs=60,
    patience=15,
    zongguan_idx=15,
    num_workers=4,
    use_amp=True,
    use_cosine=True,
    seed=42,
    zongguan_loss_weight=4.0,
    residual=True,
    # transformer / ablation flags (overridden per exp)
    use_cross_attn=False,
    use_future_meteo=False,
    use_patchtst_water=False,
    n_heads=4,
    n_dec_layers=2,
    patch_len=8,
    patch_layers=2,
    patch_dropout=0.15,
    channel_loss_weight=None,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def cfg_for_exp(exp: str) -> dict:
    cfg = dict(BASE_CFG)
    cfg["exp"] = exp
    if exp == "eval_v3":
        cfg["T_out"] = 12  # original v3 decoder width; metrics still use step 0
        cfg["eval_h1_only"] = True
    if exp in ("p0", "p0_ch", "p0_fut", "p0_patch"):
        cfg["use_cross_attn"] = True
    if exp == "p0_ch":
        w = np.ones(cfg["C"], dtype=np.float32)
        w[WATER_FEATURES.index("NH3_N")] = 2.0
        w[WATER_FEATURES.index("TURB")] = 2.5
        cfg["channel_loss_weight"] = w.tolist()
    if exp == "p0_fut":
        cfg["use_future_meteo"] = True
    if exp == "p0_patch":
        cfg["use_patchtst_water"] = True
        cfg["dropout"] = 0.15
        cfg["weight_decay"] = 2e-4
    return cfg


class NPZDataset(Dataset):
    def __init__(self, path: Path, load_future: bool = False, t_out: int = 1):
        d = np.load(path)
        self.X = torch.from_numpy(d["X"].astype(np.float32))
        self.M = torch.from_numpy(d["M"].astype(np.float32))
        self.Y = torch.from_numpy(d["Y"][:, :t_out].astype(np.float32))
        self.Mf = None
        if load_future:
            if "M_future" not in d.files:
                raise KeyError(f"{path} missing M_future; rebuild windows first")
            self.Mf = torch.from_numpy(d["M_future"][:, :t_out].astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.Mf is None:
            return self.X[i], self.M[i], self.Y[i]
        return self.X[i], self.M[i], self.Y[i], self.Mf[i]


class ResidualTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.cut = pad

    def _causal(self, y, T):
        if self.cut > 0:
            y = y[:, :, : -self.cut]
        return y[:, :, :T]

    def forward(self, x):
        T = x.size(2)
        y = self.drop(F.relu(self._causal(self.conv1(x), T)))
        y = self.drop(F.relu(self._causal(self.conv2(y), T)))
        return F.relu(y + self.proj(x))


class TemporalTower(nn.Module):
    def __init__(self, in_dim, hidden, n_layers=6, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        ch_in = in_dim
        for i in range(n_layers):
            layers.append(
                ResidualTCNBlock(ch_in, hidden, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            )
            ch_in = hidden
        self.blocks = nn.ModuleList(layers)

    def forward(self, x):
        B, T, N, C = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        for blk in self.blocks:
            h = blk(h)
        return h.reshape(B, N, h.shape[1], T).permute(0, 3, 1, 2)


class ChannelIndepTemporalTower(nn.Module):
    def __init__(self, n_channels, hidden, n_layers=6, kernel_size=3, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        layers = []
        ch_in = 1
        for i in range(n_layers):
            layers.append(
                ResidualTCNBlock(ch_in, hidden, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            )
            ch_in = hidden
        self.blocks = nn.ModuleList(layers)
        self.fuse = nn.Sequential(
            nn.Linear(n_channels * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        B, T, N, C = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N * C, 1, T)
        for blk in self.blocks:
            h = blk(h)
        Fh = h.shape[1]
        h = h.reshape(B, N, C, Fh, T).permute(0, 4, 1, 2, 3).contiguous()
        h = h.reshape(B, T, N, C * Fh)
        return self.fuse(h)


class ChannelIndepPatchTST(nn.Module):
    """PatchTST-style channel-independent encoder; upsample patches back to Tin."""

    def __init__(self, n_channels, hidden, t_in=168, patch_len=8, n_layers=2, n_heads=4, dropout=0.15):
        super().__init__()
        if t_in % patch_len != 0:
            raise ValueError(f"T_in={t_in} must be divisible by patch_len={patch_len}")
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.n_patches = t_in // patch_len
        self.hidden = hidden
        self.patch_embed = nn.Linear(patch_len, hidden)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, hidden) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.fuse = nn.Sequential(
            nn.Linear(n_channels * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        B, T, N, C = x.shape
        P, Fh = self.n_patches, self.hidden
        h = x.permute(0, 2, 3, 1).reshape(B * N * C, P, self.patch_len)
        h = self.patch_embed(h) + self.pos
        h = self.encoder(h)
        h = h.reshape(B, N, C, P, Fh).permute(0, 3, 1, 2, 4).contiguous()
        h = self.fuse(h.reshape(B, P, N, C * Fh))  # [B,P,N,F]
        h = h.permute(0, 2, 3, 1).reshape(B * N, Fh, P)
        h = F.interpolate(h, size=T, mode="linear", align_corners=False)
        return h.reshape(B, N, Fh, T).permute(0, 3, 1, 2).contiguous()


class DynamicAdjacency(nn.Module):
    def __init__(self, d_meteo, d_E=64, mlp_hidden=128):
        super().__init__()
        self.embed = nn.Linear(d_meteo, d_E)
        self.mlp = nn.Sequential(nn.Linear(1, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))
        self.d_E = d_E

    def forward(self, M, A_mask):
        B, T, N, D = M.shape
        E = F.relu(self.embed(M.reshape(B * T, N, D)))
        S = torch.matmul(E, E.transpose(1, 2)) / math.sqrt(self.d_E)
        A_prime = F.relu(self.mlp(S.unsqueeze(-1)).squeeze(-1))
        mask = A_mask.to(A_prime.device).clone()
        eye = torch.eye(N, device=mask.device, dtype=mask.dtype)
        mask = torch.clamp(mask + eye, max=1.0)
        logits = A_prime.masked_fill(mask.unsqueeze(0) < 0.5, -1e4)
        A = torch.softmax(logits, dim=-1)
        return A.reshape(B, T, N, N)


def normalize_adj(A):
    deg = A.sum(dim=-1).clamp(min=1e-6)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)


class DynGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.theta = nn.Linear(dim, dim, bias=True)
        self.res = nn.Identity()

    def forward(self, Z, A):
        A_prop = A.transpose(-1, -2)
        A_norm = normalize_adj(A_prop)
        return F.relu(self.theta(torch.matmul(A_norm, Z))) + self.res(Z)


class DynGCN(nn.Module):
    def __init__(self, hidden, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([DynGCNLayer(hidden) for _ in range(n_layers)])

    def forward(self, Z, A):
        for layer in self.layers:
            Z = layer(Z, A)
        return Z


class GatedFusion(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden * 2, hidden)

    def forward(self, Zw, Zm):
        gate = torch.sigmoid(self.fc(torch.cat([Zw, Zm], dim=-1)))
        return gate * Zm + (1.0 - gate) * Zw


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv, need_weights=False):
        attn_out, w = self.attn(q, kv, kv, need_weights=need_weights, average_attn_weights=True)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q, w


class HorizonCrossAttnDecoder(nn.Module):
    """Horizon queries cross-attend historical fused states (and optional future meteo)."""

    def __init__(
        self,
        hidden,
        t_in,
        t_out,
        n_channels,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        use_future_meteo=False,
        d_meteo=4,
    ):
        super().__init__()
        self.t_out = t_out
        self.use_future_meteo = use_future_meteo
        self.query_from_last = nn.Linear(hidden, hidden)
        self.queries = nn.Parameter(torch.zeros(1, t_out, hidden))
        self.horizon_pos = nn.Parameter(torch.zeros(1, t_out, hidden))
        nn.init.normal_(self.horizon_pos, std=0.02)
        self.hist_pos = nn.Parameter(torch.zeros(1, t_in, hidden))
        nn.init.normal_(self.hist_pos, std=0.02)
        self.hist_blocks = nn.ModuleList([CrossAttnBlock(hidden, n_heads, dropout) for _ in range(n_layers)])
        if use_future_meteo:
            self.fut_proj = nn.Sequential(nn.Linear(d_meteo, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.fut_pos = nn.Parameter(torch.zeros(1, t_out, hidden))
            nn.init.normal_(self.fut_pos, std=0.02)
            self.fut_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden, n_heads, dropout) for _ in range(n_layers)]
            )
        else:
            self.fut_proj = None
            self.fut_blocks = None
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, n_channels))

    def forward(self, H, M_future=None):
        B, T, N, Fh = H.shape
        orig_dtype = H.dtype
        with torch.amp.autocast("cuda", enabled=False):
            H = H.float()
            mem = H.permute(0, 2, 1, 3).reshape(B * N, T, Fh) + self.hist_pos[:, :T]
            last = mem[:, -1]
            q = self.query_from_last(last).unsqueeze(1) + self.queries + self.horizon_pos
            want_w = not self.training
            attn_w = None
            for i, blk in enumerate(self.hist_blocks):
                q, w = blk(q, mem, need_weights=want_w and i == len(self.hist_blocks) - 1)
                if w is not None:
                    attn_w = w
            if self.fut_proj is not None:
                if M_future is None:
                    raise ValueError("M_future required when use_future_meteo=True")
                fut = self.fut_proj(M_future.float()).permute(0, 2, 1, 3).reshape(B * N, self.t_out, Fh)
                fut = fut + self.fut_pos
                for blk in self.fut_blocks:
                    q, _ = blk(q, fut, need_weights=False)
            delta = self.out(q)
        delta = delta.to(orig_dtype)
        delta = delta.reshape(B, N, self.t_out, -1).permute(0, 2, 1, 3).contiguous()
        return delta, attn_w


class PoolDecoder(nn.Module):
    """Original v3 last+mean decoder."""

    def __init__(self, hidden, t_out, n_channels, dropout=0.1):
        super().__init__()
        self.t_out = t_out
        self.n_channels = n_channels
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, t_out * n_channels),
        )

    def forward(self, H, M_future=None):
        h = torch.cat([H[:, -1], H.mean(dim=1)], dim=-1)
        out = self.mlp(h)
        B, N, _ = out.shape
        delta = out.view(B, N, self.t_out, self.n_channels).permute(0, 2, 1, 3).contiguous()
        return delta, None


class MDDySTGCNtf(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        Fh = cfg["F"]
        drop = cfg["dropout"]
        self.cfg = cfg
        if cfg.get("use_patchtst_water"):
            self.water_tower = ChannelIndepPatchTST(
                cfg["C"],
                Fh,
                t_in=cfg["T_in"],
                patch_len=cfg["patch_len"],
                n_layers=cfg["patch_layers"],
                n_heads=cfg["n_heads"],
                dropout=cfg.get("patch_dropout", drop),
            )
        else:
            self.water_tower = ChannelIndepTemporalTower(
                cfg["C"], Fh, n_layers=cfg["L_tcn"], kernel_size=cfg["K_tcn"], dropout=drop
            )
        self.meteo_tower = TemporalTower(
            cfg["D"], Fh, n_layers=cfg["L_tcn"], kernel_size=cfg["K_tcn"], dropout=drop
        )
        self.dyn_adj = DynamicAdjacency(cfg["D"], d_E=cfg["d_E"], mlp_hidden=cfg["mlp_hidden"])
        self.gcn_w = DynGCN(Fh, n_layers=cfg["L_g"])
        self.gcn_m = DynGCN(Fh, n_layers=cfg["L_g"])
        self.gate = GatedFusion(Fh)
        if cfg.get("use_cross_attn"):
            self.decoder = HorizonCrossAttnDecoder(
                hidden=Fh,
                t_in=cfg["T_in"],
                t_out=cfg["T_out"],
                n_channels=cfg["C"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_dec_layers"],
                dropout=drop,
                use_future_meteo=cfg.get("use_future_meteo", False),
                d_meteo=cfg["D"],
            )
        else:
            self.decoder = PoolDecoder(Fh, cfg["T_out"], cfg["C"], dropout=drop)

    def forward(self, X, M, A_mask, M_future=None, return_attn=False):
        Hw = self.water_tower(X)
        Hm = self.meteo_tower(M)
        A = self.dyn_adj(M, A_mask)
        H = self.gate(self.gcn_w(Hw, A), self.gcn_m(Hm, A))
        delta, attn_w = self.decoder(H, M_future)
        if self.cfg.get("residual", True):
            yhat = delta + X[:, -1, :, :].unsqueeze(1)
        else:
            yhat = delta
        if return_attn:
            return yhat, attn_w
        return yhat


def unpack_batch(batch, device, load_future, meteo_clip=None):
    if load_future:
        X, M, Y, Mf = batch
        Mf = Mf.to(device, non_blocking=True)
    else:
        X, M, Y = batch
        Mf = None
    X = X.to(device, non_blocking=True)
    M = M.to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)
    if meteo_clip is not None:
        M = M.clamp(-meteo_clip, meteo_clip)
        if Mf is not None:
            Mf = Mf.clamp(-meteo_clip, meteo_clip)
    return X, M, Y, Mf


def station_channel_huber(pred, target, cfg):
    per = F.huber_loss(pred, target, delta=cfg["huber_delta"], reduction="none")
    w_s = torch.ones(cfg["N"], device=pred.device, dtype=per.dtype)
    w_s[cfg["zongguan_idx"]] = cfg["zongguan_loss_weight"]
    per = per * w_s.view(1, 1, -1, 1)
    ch_w = cfg.get("channel_loss_weight")
    if ch_w is not None:
        w_c = torch.tensor(ch_w, device=pred.device, dtype=per.dtype)
        per = per * w_c.view(1, 1, 1, -1)
    return per.mean()


@torch.no_grad()
def eval_loader(model, loader, A_mask, cfg, water_mu, water_std, compute_mape=False):
    model.eval()
    device = next(model.parameters()).device
    zi = cfg["zongguan_idx"]
    use_amp = cfg["use_amp"] and device.type == "cuda"
    load_future = cfg.get("use_future_meteo", False)
    total_loss = n = 0.0
    se = ae = count = 0.0
    mape_num = mape_den = 0.0
    per_se = np.zeros(cfg["C"], dtype=np.float64)
    per_n = 0

    for batch in loader:
        X, M, Y, Mf = unpack_batch(batch, device, load_future, cfg.get("meteo_clip"))
        with torch.amp.autocast("cuda", enabled=use_amp):
            Yhat = model(X, M, A_mask, Mf)
            if cfg.get("eval_h1_only"):
                Yhat = Yhat[:, :1]
                Y = Y[:, :1]
            loss = station_channel_huber(Yhat, Y, cfg)
        bs = X.size(0)
        total_loss += loss.item() * bs
        n += bs
        pred_z = Yhat[:, :, zi, :].float()
        true_z = Y[:, :, zi, :].float()
        se += ((pred_z - true_z) ** 2).sum().item()
        ae += (pred_z - true_z).abs().sum().item()
        count += pred_z.numel()
        diff = (pred_z - true_z).detach().cpu().numpy()
        per_se += (diff**2).sum(axis=(0, 1))
        per_n += diff.shape[0] * diff.shape[1]
        if compute_mape:
            pred_p = pred_z.cpu().numpy() * water_std + water_mu
            true_p = true_z.cpu().numpy() * water_std + water_mu
            denom = np.maximum(np.abs(true_p), 1e-3)
            mape_num += np.abs((pred_p - true_p) / denom).sum()
            mape_den += true_p.size

    out = {
        "loss": total_loss / max(n, 1),
        "rmse": math.sqrt(se / max(count, 1)),
        "mae": ae / max(count, 1),
        "mse": se / max(count, 1),
        "per_channel_rmse": (np.sqrt(per_se / max(per_n, 1))).tolist(),
    }
    if compute_mape:
        out["mape"] = 100.0 * mape_num / max(mape_den, 1)
    return out


def persistence_metrics(test_path: Path, cfg, water_mu, water_std):
    te = np.load(test_path)
    t_out = cfg["T_out"]
    if cfg.get("eval_h1_only"):
        t_out = 1
    Yte = te["Y"][:, :t_out, cfg["zongguan_idx"], :]
    Xte = te["X"][:, :, cfg["zongguan_idx"], :]
    persist = np.repeat(Xte[:, -1:, :], Yte.shape[1], axis=1)
    p_rmse = float(np.sqrt(((persist - Yte) ** 2).mean()))
    p_mae = float(np.abs(persist - Yte).mean())
    pred_p = persist * water_std + water_mu
    true_p = Yte * water_std + water_mu
    denom = np.maximum(np.abs(true_p), 1e-3)
    p_mape = float(100.0 * np.abs((pred_p - true_p) / denom).mean())
    return {"rmse": p_rmse, "mae": p_mae, "mape": p_mape}


def metrics_payload(test_m, persist, cfg, n_params, ckpt, feats):
    per = {f: round(r, 4) for f, r in zip(feats, test_m["per_channel_rmse"])}
    rmse = test_m["rmse"]
    vs_v3 = 100.0 * (rmse - V3_H1["RMSE"]) / V3_H1["RMSE"]
    return {
        "exp": cfg.get("exp"),
        "protocol": "horizon-1",
        "horizon_1": {
            "MAE": test_m["mae"],
            "MSE": test_m["mse"],
            "RMSE": rmse,
            "MAPE": test_m.get("mape"),
            "persistence": persist,
            "per_channel_rmse": per,
            "v3_baseline": V3_H1,
            "rmse_vs_v3_pct": round(vs_v3, 2),
        },
        "best_epoch": ckpt.get("epoch") if ckpt else None,
        "best_val_rmse": ckpt.get("val_rmse") if ckpt else None,
        "n_params": n_params,
        "cfg": {k: v for k, v in cfg.items() if k != "channel_loss_weight" or v is None or isinstance(v, list)},
    }


def plot_curves(history, persist_rmse, out_path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Huber Loss (zg-weighted)")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[0].grid(True, ls=":")
    ax[1].plot(history["val_rmse"], color="C2", label="val horizon-1")
    ax[1].axhline(persist_rmse, color="gray", ls="--", label=f"persist {persist_rmse:.3f}")
    ax[1].axhline(V3_H1["RMSE"], color="C0", ls="--", label=f"v3 {V3_H1['RMSE']:.3f}")
    ax[1].set_title("Val RMSE horizon-1 (Zongguan, z-space)")
    ax[1].set_xlabel("epoch")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, ls=":")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def save_attn_heatmap(model, loader, A_mask, cfg, out_path: Path):
    if not cfg.get("use_cross_attn"):
        return
    model.eval()
    device = next(model.parameters()).device
    batch = next(iter(loader))
    X, M, Y, Mf = unpack_batch(
        batch, device, cfg.get("use_future_meteo", False), cfg.get("meteo_clip")
    )
    _, attn_w = model(X[:1], M[:1], A_mask, None if Mf is None else Mf[:1], return_attn=True)
    if attn_w is None:
        return
    # attn_w: [N, Tout, Tin] after B=1 → decoder used B*N
    w = attn_w.detach().cpu().numpy()
    zi = cfg["zongguan_idx"]
    heat = w[zi] if w.shape[0] >= cfg["N"] else w.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(heat, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("history timestep (0=oldest)")
    ax.set_ylabel("horizon")
    ax.set_title("Zongguan horizon-query attention over Tin")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_loaders(cfg):
    load_future = cfg.get("use_future_meteo", False)
    kw = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    t_out = cfg["T_out"]
    train_loader = DataLoader(
        NPZDataset(DATA_DIR / "train.npz", load_future, t_out), shuffle=True, drop_last=True, **kw
    )
    val_loader = DataLoader(NPZDataset(DATA_DIR / "val.npz", load_future, t_out), shuffle=False, **kw)
    test_loader = DataLoader(NPZDataset(DATA_DIR / "test.npz", load_future, t_out), shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def ensure_windows():
    need = [("train.npz", 6591), ("val.npz", 788), ("test.npz", 1756)]
    rebuild = False
    for name, n in need:
        p = DATA_DIR / name
        if not p.exists():
            rebuild = True
            break
        with np.load(p) as d:
            if d["X"].shape[0] != n or "M_future" not in d.files:
                rebuild = True
                break
    if not rebuild:
        print("[data] window npz ok")
        return
    print("[data] rebuilding windows with M_future ...")
    from preprocess.dataset import rebuild_window_npz

    counts = rebuild_window_npz(DATA_DIR)
    print("[data] rebuilt", counts)


def run_eval_existing(cfg, out_dir: Path):
    """Load v3_nb weights into a pool-decoder model and report dual-horizon metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = dict(cfg)
    eval_cfg["use_cross_attn"] = False
    eval_cfg["use_future_meteo"] = False
    eval_cfg["use_patchtst_water"] = False
    eval_cfg["exp"] = "eval_v3"
    scaler = json.loads((DATA_DIR / "scaler_stats.json").read_text())
    water_mu = np.asarray(scaler["water_mu"], dtype=np.float64)
    water_std = np.asarray(scaler["water_std"], dtype=np.float64)
    feats = scaler["water_features"]
    A_mask = torch.from_numpy(np.load(DATA_DIR / "A_mask.npy").astype(np.float32)).to(device)
    _, _, test_loader = make_loaders(eval_cfg)
    persist = persistence_metrics(DATA_DIR / "test.npz", eval_cfg, water_mu, water_std)
    model = MDDySTGCNtf(eval_cfg).to(device)
    ckpt = torch.load(V3_CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    # Map v3 key names: decoder.0/2 -> decoder.mlp.0/2 if needed
    if missing:
        sd = ckpt["model"]
        remap = {}
        for k, v in sd.items():
            if k.startswith("decoder.") and not k.startswith("decoder.mlp."):
                remap["decoder.mlp." + k[len("decoder.") :]] = v
            else:
                remap[k] = v
        missing, unexpected = model.load_state_dict(remap, strict=False)
        print("remapped v3 keys; missing", missing, "unexpected", unexpected)
    else:
        print("loaded v3_nb checkpoint strictly")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_m = eval_loader(model, test_loader, A_mask, eval_cfg, water_mu, water_std, compute_mape=True)
    payload = metrics_payload(test_m, persist, eval_cfg, n_params, ckpt, feats)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: payload[k] for k in ("horizon_1", "n_params")}, indent=2, ensure_ascii=False))
    return payload


def train_one(cfg, out_dir: Path):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"DEVICE={device} EXP={cfg['exp']} OUT={out_dir}")

    scaler = json.loads((DATA_DIR / "scaler_stats.json").read_text())
    water_mu = np.asarray(scaler["water_mu"], dtype=np.float64)
    water_std = np.asarray(scaler["water_std"], dtype=np.float64)
    feats = scaler["water_features"]
    A_mask = torch.from_numpy(np.load(DATA_DIR / "A_mask.npy").astype(np.float32)).to(device)

    train_loader, val_loader, test_loader = make_loaders(cfg)
    persist = persistence_metrics(DATA_DIR / "test.npz", cfg, water_mu, water_std)
    print(
        f"persist_h1 RMSE={persist['rmse']:.4f}  v3_h1 RMSE={V3_H1['RMSE']:.4f}  "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    model = MDDySTGCNtf(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-5)
        if cfg["use_cosine"]
        else None
    )
    use_amp = cfg["use_amp"] and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    patience_left = cfg["patience"]
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "lr": []}
    load_future = cfg.get("use_future_meteo", False)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        run_loss = n = 0.0
        for batch in train_loader:
            X, M, Y, Mf = unpack_batch(batch, device, load_future, cfg.get("meteo_clip"))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                Yhat = model(X, M, A_mask, Mf)
                loss = station_channel_huber(Yhat, Y, cfg)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print("  skip non-finite loss")
                continue
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            grads_ok = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_ok = False
                    break
            if not grads_ok:
                opt.zero_grad(set_to_none=True)
                print("  skip non-finite grad")
                scaler_amp.update()
                continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            run_loss += loss.item() * X.size(0)
            n += X.size(0)
        if scheduler is not None:
            scheduler.step()
        train_loss = (run_loss / n) if n > 0 else (
            history["train_loss"][-1] if history["train_loss"] else float("nan")
        )
        val_m = eval_loader(model, val_loader, A_mask, cfg, water_mu, water_std)
        cur_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_rmse"].append(val_m["rmse"])
        history["lr"].append(cur_lr)
        print(
            f"Epoch {epoch:03d}/{cfg['epochs']}  train={train_loss:.4f}  "
            f"val_RMSE_h1={val_m['rmse']:.4f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)"
        )
        if val_m["rmse"] < best_val - 1e-5:
            best_val = val_m["rmse"]
            patience_left = cfg["patience"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "val_rmse": best_val,
                    "val_metrics": val_m,
                },
                best_path,
            )
            print(f"  -> saved best val_RMSE_h1={best_val:.4f}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_m = eval_loader(model, test_loader, A_mask, cfg, water_mu, water_std, compute_mape=True)
    payload = metrics_payload(test_m, persist, cfg, n_params, ckpt, feats)
    print(json.dumps({k: payload[k] for k in ("horizon_1", "n_params", "best_epoch")}, indent=2, ensure_ascii=False))
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_curves(history, persist["rmse"], out_dir / "loss_curves.png")
    try:
        save_attn_heatmap(model, val_loader, A_mask, cfg, out_dir / "attn_zongguan.png")
    except Exception as e:
        print("attn heatmap skipped:", e)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="p0",
        choices=["eval_v3", "v3", "p0", "p0_ch", "p0_fut", "p0_patch"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--meteo-clip", type=float, default=None,
                        help="clamp M / M_future to +/- value (stabilizes p0_fut)")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args(argv)
    ensure_windows()
    cfg = cfg_for_exp(args.exp)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.patience is not None:
        cfg["patience"] = args.patience
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.no_amp:
        cfg["use_amp"] = False
    if args.meteo_clip is not None:
        cfg["meteo_clip"] = args.meteo_clip
    tag = args.exp if args.exp == "eval_v3" else f"{args.exp}_h1"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "model_runs_tf" / tag
    if args.exp == "eval_v3":
        run_eval_existing(cfg, out_dir)
    else:
        train_one(cfg, out_dir)


if __name__ == "__main__":
    main()
