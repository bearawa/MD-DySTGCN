#!/usr/bin/env python3
"""Copy existing graph-model metrics/history into 论文插图包/data/runs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
ROOT = PACK.parent
DATA = PACK / "data"
OUT = ROOT / "outputs"

MAP = {
    "v3": {
        "metrics": OUT / "model_runs_tf" / "eval_v3" / "test_metrics.json",
        "history": OUT / "model_runs_v3_nb" / "history.json",
    },
    "p0": {
        "metrics": OUT / "model_runs_tf" / "p0_h1" / "test_metrics.json",
        "history": OUT / "model_runs_tf" / "p0_h1" / "history.json",
    },
    "p0_fut": {
        "metrics": OUT / "model_runs_tf" / "p0_fut_h1" / "test_metrics.json",
        "history": OUT / "model_runs_tf" / "p0_fut_h1" / "history.json",
    },
    "former": {
        "metrics": OUT / "model_runs_tf" / "p0_patch_h1" / "test_metrics.json",
        "history": OUT / "model_runs_tf" / "p0_patch_h1" / "history.json",
    },
}


def flatten(name: str, raw: dict) -> dict:
    h = raw.get("horizon_1", raw)
    per = h.get("per_channel_rmse")
    return {
        "exp": name,
        "protocol": "horizon-1",
        "station": "宗关",
        "horizon_1": {
            "MAE": h["MAE"],
            "MSE": h.get("MSE"),
            "RMSE": h["RMSE"],
            "MAPE": h["MAPE"],
            "per_channel_rmse": per,
        },
        "best_epoch": raw.get("best_epoch"),
        "best_val_rmse": raw.get("best_val_rmse"),
        "n_params": raw.get("n_params"),
        "source": str(MAP[name]["metrics"].relative_to(ROOT)),
    }


def main():
    (DATA / "runs").mkdir(parents=True, exist_ok=True)
    src_scaler = ROOT / "outputs" / "arrays" / "scaler_stats.json"
    if src_scaler.exists():
        shutil.copy2(src_scaler, DATA / "scaler_stats.json")
    for name, paths in MAP.items():
        dest = DATA / "runs" / name
        dest.mkdir(parents=True, exist_ok=True)
        raw = json.loads(paths["metrics"].read_text())
        (dest / "test_metrics.json").write_text(
            json.dumps(flatten(name, raw), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if paths["history"].exists():
            shutil.copy2(paths["history"], dest / "history.json")
        print("[copy]", name)
    print("ok")


if __name__ == "__main__":
    main()
