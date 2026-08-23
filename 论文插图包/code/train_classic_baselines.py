#!/usr/bin/env python3
"""Horizon-1 Zongguan classic baselines: HA, persistence, LSTM, TCN, PatchTST."""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
ROOT = PACK.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from style import DATA, WATER_FEATURES
from train_tf import ChannelIndepPatchTST, TemporalTower

ARRAY = ROOT / "outputs" / "arrays"
ZI = 15
C = 9
T_IN = 168
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ZongguanH1(Dataset):
    """X: [T,C] at Zongguan, y: next-step [C]."""

    def __init__(self, npz_path: Path):
        d = np.load(npz_path, mmap_mode="r")
        self.X = np.asarray(d["X"][:, :, ZI, :], dtype=np.float32)
        self.Y = np.asarray(d["Y"][:, 0, ZI, :], dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])


def load_scaler():
    sc = json.loads((ARRAY / "scaler_stats.json").read_text())
    mu = np.asarray(sc["water_mu"], dtype=np.float64)
    std = np.asarray(sc["water_std"], dtype=np.float64)
    return mu, std


def metrics_from_arrays(pred, true, mu, std):
    """pred/true: [N,C] z-space."""
    se = (pred - true) ** 2
    ae = np.abs(pred - true)
    rmse = float(np.sqrt(se.mean()))
    mae = float(ae.mean())
    mse = float(se.mean())
    per = np.sqrt(se.mean(axis=0))
    pred_p = pred * std + mu
    true_p = true * std + mu
    mape = float(100.0 * np.abs((pred_p - true_p) / np.maximum(np.abs(true_p), 1e-3)).mean())
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "per_channel_rmse": {f: round(float(v), 4) for f, v in zip(WATER_FEATURES, per)},
    }


def dump_run(name: str, payload: dict, pred=None, true=None, history=None):
    out = DATA / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if history is not None:
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if pred is not None:
        np.save(out / "pred_zongguan.npy", pred.astype(np.float32))
    if true is not None:
        np.save(out / "true_zongguan.npy", true.astype(np.float32))
    print(f"[{name}] RMSE={payload['horizon_1']['RMSE']:.4f} MAE={payload['horizon_1']['MAE']:.4f} "
          f"MAPE={payload['horizon_1']['MAPE']:.2f}%")


def eval_stateless(name, pred, true, mu, std, extra=None):
    h1 = metrics_from_arrays(pred, true, mu, std)
    payload = {
        "exp": name,
        "protocol": "horizon-1",
        "station": "宗关",
        "horizon_1": h1,
        "n_params": 0,
    }
    if extra:
        payload.update(extra)
    dump_run(name, payload, pred=pred, true=true)
    return payload


def run_ha_persist(mu, std):
    tr = ZongguanH1(ARRAY / "train.npz")
    te = ZongguanH1(ARRAY / "test.npz")
    true = te.Y
    last = te.X[:, -1, :]
    ha = np.broadcast_to(tr.Y.mean(axis=0, keepdims=True), true.shape).copy()
    eval_stateless("ha", ha, true, mu, std, extra={"note": "训练集宗关各通道均值"})
    eval_stateless("persist", last, true, mu, std, extra={"note": "复制输入窗最后一拍"})
    np.save(DATA / "runs" / "true_zongguan.npy", true.astype(np.float32))


class LSTMBaseline(nn.Module):
    def __init__(self, c=9, hidden=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(c, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, c))

    def forward(self, x):
        h, _ = self.lstm(x)
        return x[:, -1] + self.head(h[:, -1])


class TCNBaseline(nn.Module):
    def __init__(self, c=9, hidden=128, n_layers=6, dropout=0.1):
        super().__init__()
        self.tower = TemporalTower(c, hidden, n_layers=n_layers, kernel_size=3, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, c))

    def forward(self, x):
        h = self.tower(x.unsqueeze(2))  # [B,T,1,F]
        return x[:, -1] + self.head(h[:, -1, 0])


class PatchTSTBaseline(nn.Module):
    def __init__(self, c=9, hidden=128, dropout=0.15):
        super().__init__()
        self.tower = ChannelIndepPatchTST(
            c, hidden, t_in=T_IN, patch_len=8, n_layers=2, n_heads=4, dropout=dropout
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, c))

    def forward(self, x):
        h = self.tower(x.unsqueeze(2))
        return x[:, -1] + self.head(h[:, -1, 0])


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        yhat = model(x)
        preds.append(yhat.float().cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds, 0), np.concatenate(trues, 0)


def train_nn(name: str, model: nn.Module, mu, std, epochs=60, patience=15, lr=1e-3, wd=1e-4, batch=64):
    out = DATA / "runs" / name
    if (out / "test_metrics.json").exists() and (out / "best.pt").exists():
        print(f"[{name}] skip, already have {out}")
        return json.loads((out / "test_metrics.json").read_text())
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_ds = ZongguanH1(ARRAY / "train.npz")
    val_ds = ZongguanH1(ARRAY / "val.npz")
    test_ds = ZongguanH1(ARRAY / "test.npz")
    kw = dict(batch_size=batch, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] device={device} params={n_params:,} train={len(train_ds)}")

    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    wait = patience
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "lr": []}
    out = DATA / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = n = 0.0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                yhat = model(x)
                loss = F.huber_loss(yhat, y, delta=1.0)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        pred_v, true_v = predict_all(model, val_loader, device)
        val_rmse = float(np.sqrt(((pred_v - true_v) ** 2).mean()))
        val_loss = float(np.mean(np.where(np.abs(pred_v - true_v) < 1.0,
                                          0.5 * (pred_v - true_v) ** 2,
                                          np.abs(pred_v - true_v) - 0.5)))
        history["train_loss"].append(run_loss / max(n, 1))
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(opt.param_groups[0]["lr"])
        print(f"  epoch {epoch:02d}  train={history['train_loss'][-1]:.4f}  "
              f"val_rmse={val_rmse:.4f}  {time.time()-t0:.1f}s")
        if val_rmse < best_rmse - 1e-6:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = patience
        else:
            wait -= 1
            if wait <= 0:
                print(f"  early stop at {epoch}, best epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    torch.save({"model": best_state, "epoch": best_epoch, "val_rmse": best_rmse}, out / "best.pt")
    pred, true = predict_all(model, test_loader, device)
    h1 = metrics_from_arrays(pred, true, mu, std)
    payload = {
        "exp": name,
        "protocol": "horizon-1",
        "station": "宗关",
        "horizon_1": h1,
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "n_params": n_params,
    }
    dump_run(name, payload, pred=pred, true=true, history=history)
    return payload


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--e50", action="store_true", help="train 50 epochs, no early stop, write *_e50")
    args = p.parse_args()
    (DATA / "runs").mkdir(parents=True, exist_ok=True)
    mu, std = load_scaler()
    if args.e50:
        kw = dict(epochs=50, patience=999)
        print("[classic] LSTM e50")
        train_nn("lstm_e50", LSTMBaseline(), mu, std, **kw)
        print("[classic] TCN e50")
        train_nn("tcn_e50", TCNBaseline(), mu, std, **kw)
        print("[classic] PatchTST e50")
        train_nn("patchtst_e50", PatchTSTBaseline(), mu, std, wd=2e-4, **kw)
        print("[classic] e50 done")
        return
    print("[classic] HA / persistence")
    run_ha_persist(mu, std)
    print("[classic] LSTM")
    train_nn("lstm", LSTMBaseline(), mu, std)
    print("[classic] TCN")
    train_nn("tcn", TCNBaseline(), mu, std)
    print("[classic] PatchTST")
    train_nn("patchtst", PatchTSTBaseline(), mu, std, wd=2e-4)
    print("[classic] done")


if __name__ == "__main__":
    main()
