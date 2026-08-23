#!/usr/bin/env python3
"""MD-DySTGCN improved training — close the gap vs paper RMSE 0.2649.

Root causes addressed (vs v2 smoke / prior ~0.52 RMSE):
1. Model was ≈ persistence (0.47): add residual decoding from last observation.
2. Multi-channel shared TCN mixes heterogeneous WQ vars: channel-independent water tower.
3. Decoder used only H[:,-1]: fuse last + mean temporal pooling.
4. Early-stop on Huber loss selected weak checkpoints: stop on val RMSE (Zongguan).
5. Softmax mask used -inf fill that can zero out rows: keep self-loop stable path.
"""
from __future__ import annotations

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
OUT_DIR = ROOT / "outputs" / "model_runs_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = dict(
    T_in=168,
    T_out=12,
    N=16,
    C=9,
    D=4,
    F=128,
    d_E=64,
    mlp_hidden=128,
    L_tcn=6,  # dilations 1..32 → RF covers most of Tin
    K_tcn=3,
    L_g=4,  # 4 hops enough for 16-node chain; less over-smooth than 8
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
    zongguan_loss_weight=4.0,  # emphasize eval station in train loss
    residual=True,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class NPZDataset(Dataset):
    def __init__(self, path: Path):
        d = np.load(path)
        self.X = torch.from_numpy(d["X"].astype(np.float32))
        self.M = torch.from_numpy(d["M"].astype(np.float32))
        self.Y = torch.from_numpy(d["Y"].astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.M[i], self.Y[i]


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
    """Shared multi-channel TCN. [B,T,N,C] -> [B,T,N,F]."""

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
    """Channel-independent TCN (PatchTST-style), then fuse channels -> F."""

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

        # Softmax only over valid neighbors; self-loop always present
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
        A_prop = A.transpose(-1, -2)  # downstream aggregates upstream
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


class MDDySTGCNv3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        Fh = cfg["F"]
        drop = cfg["dropout"]
        self.cfg = cfg
        # Water: channel-indep; meteo: shared multi-channel (D=4, homogeneous-ish)
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
        # last + mean temporal pool → 2F
        self.decoder = nn.Sequential(
            nn.Linear(Fh * 2, Fh * 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(Fh * 2, cfg["T_out"] * cfg["C"]),
        )

    def forward(self, X, M, A_mask):
        Hw = self.water_tower(X)
        Hm = self.meteo_tower(M)
        A = self.dyn_adj(M, A_mask)
        Zw = self.gcn_w(Hw, A)
        Zm = self.gcn_m(Hm, A)
        H = self.gate(Zw, Zm)  # [B,T,N,F]
        h_last = H[:, -1]
        h_mean = H.mean(dim=1)
        h = torch.cat([h_last, h_mean], dim=-1)
        out = self.decoder(h)
        B, N, _ = out.shape
        Tout, C = self.cfg["T_out"], self.cfg["C"]
        delta = out.view(B, N, Tout, C).permute(0, 2, 1, 3).contiguous()
        if self.cfg.get("residual", True):
            # residual from last water observation
            base = X[:, -1, :, :].unsqueeze(1)  # [B,1,N,C]
            return delta + base
        return delta


def huber_loss(pred, target, delta=1.0):
    return F.huber_loss(pred, target, delta=delta, reduction="none")


def station_weighted_huber(pred, target, cfg):
    per = huber_loss(pred, target, cfg["huber_delta"])  # [B,Tout,N,C]
    w = torch.ones(cfg["N"], device=pred.device, dtype=per.dtype)
    w[cfg["zongguan_idx"]] = cfg["zongguan_loss_weight"]
    # broadcast over B,Tout,C
    return (per * w.view(1, 1, -1, 1)).mean()


@torch.no_grad()
def eval_loader(model, loader, A_mask, cfg, water_mu, water_std, compute_mape=False):
    model.eval()
    device = next(model.parameters()).device
    total_loss = n = 0.0
    se = ae = count = 0.0
    mape_num = mape_den = 0.0
    zi = cfg["zongguan_idx"]
    use_amp = cfg["use_amp"] and device.type == "cuda"
    per_se = np.zeros(cfg["C"], dtype=np.float64)
    per_n = 0

    for X, M, Y in loader:
        X = X.to(device, non_blocking=True)
        M = M.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            Yhat = model(X, M, A_mask)
            loss = station_weighted_huber(Yhat, Y, cfg)
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


def main():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE={device} OUT_DIR={OUT_DIR}")

    scaler = json.loads((DATA_DIR / "scaler_stats.json").read_text())
    water_mu = np.asarray(scaler["water_mu"], dtype=np.float64)
    water_std = np.asarray(scaler["water_std"], dtype=np.float64)
    feats = scaler["water_features"]

    A_mask = torch.from_numpy(np.load(DATA_DIR / "A_mask.npy").astype(np.float32)).to(device)
    # ensure chain has no forced self in file; DynamicAdjacency adds eye
    print("A_mask sum", float(A_mask.sum()), "shape", tuple(A_mask.shape))

    kw = dict(
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(NPZDataset(DATA_DIR / "train.npz"), shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(NPZDataset(DATA_DIR / "val.npz"), shuffle=False, **kw)
    test_loader = DataLoader(NPZDataset(DATA_DIR / "test.npz"), shuffle=False, **kw)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    # Persistence baseline on test
    te = np.load(DATA_DIR / "test.npz")
    Yte = te["Y"][:, :, CFG["zongguan_idx"], :]
    Xte = te["X"][:, :, CFG["zongguan_idx"], :]
    persist = np.repeat(Xte[:, -1:, :], Yte.shape[1], axis=1)
    p_rmse = float(np.sqrt(((persist - Yte) ** 2).mean()))
    print(f"Persistence baseline test RMSE(zg)={p_rmse:.4f}  paper={0.2649}")

    model = MDDySTGCNv3(CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"], eta_min=1e-5)
        if CFG["use_cosine"]
        else None
    )
    use_amp = CFG["use_amp"] and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_path = OUT_DIR / "md_dystgcn_v3_best.pt"
    patience_left = CFG["patience"]
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "lr": []}

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        t0 = time.time()
        run_loss = n = 0.0
        for X, M, Y in train_loader:
            X = X.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                Yhat = model(X, M, A_mask)
                loss = station_weighted_huber(Yhat, Y, CFG)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            run_loss += loss.item() * X.size(0)
            n += X.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = run_loss / max(n, 1)
        val_m = eval_loader(model, val_loader, A_mask, CFG, water_mu, water_std)
        cur_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_rmse"].append(val_m["rmse"])
        history["lr"].append(cur_lr)
        print(
            f"Epoch {epoch:03d}/{CFG['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_m['loss']:.4f}  "
            f"val_RMSE(zg)={val_m['rmse']:.4f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)"
        )

        # Early stop on val RMSE (metric that matches paper table)
        if val_m["rmse"] < best_val - 1e-5:
            best_val = val_m["rmse"]
            patience_left = CFG["patience"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": CFG,
                    "epoch": epoch,
                    "val_rmse": best_val,
                    "val_metrics": val_m,
                },
                best_path,
            )
            print(f"  -> saved best val_RMSE={best_val:.4f}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best epoch={ckpt['epoch']} val_RMSE={ckpt['val_rmse']:.4f}")

    test_m = eval_loader(model, test_loader, A_mask, CFG, water_mu, water_std, compute_mape=True)
    per = {f: round(r, 4) for f, r in zip(feats, test_m["per_channel_rmse"])}
    metrics = {
        "test_loss": test_m["loss"],
        "test_rmse_zongguan": test_m["rmse"],
        "test_mae_zongguan": test_m["mae"],
        "test_mse_zongguan": test_m["mse"],
        "test_mape_zongguan_pct": test_m["mape"],
        "per_channel_rmse_zongguan": per,
        "persistence_rmse": p_rmse,
        "paper_ref_rmse": 0.2649,
        "paper_ref_mape_pct": 7.67,
        "best_epoch": ckpt["epoch"],
        "best_val_rmse": ckpt["val_rmse"],
        "n_params": n_params,
        "device": str(device),
        "cfg": CFG,
        "improvements": [
            "residual_decoding",
            "channel_indep_water_tcn",
            "temporal_last+mean_pool",
            "L_tcn=6_RF_cover_Tin",
            "L_g=4_less_oversmooth",
            "early_stop_on_val_rmse",
            "zongguan_loss_weight=4",
            "cosine_eta_min=1e-5",
        ],
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "cfg"}, indent=2, ensure_ascii=False))
    with open(OUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Huber Loss (zg-weighted)")
    ax[0].legend()
    ax[0].grid(True, ls=":")
    ax[1].plot(history["val_rmse"], color="C1", label="val")
    ax[1].axhline(p_rmse, color="gray", ls="--", label=f"persist {p_rmse:.3f}")
    ax[1].axhline(0.2649, color="green", ls="--", label="paper 0.265")
    ax[1].set_title("Val RMSE (Zongguan, z-space)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, ls=":")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curves.png", dpi=140)
    print("wrote", OUT_DIR / "test_metrics.json")


if __name__ == "__main__":
    main()
