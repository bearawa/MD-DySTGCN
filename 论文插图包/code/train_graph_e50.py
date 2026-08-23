#!/usr/bin/env python3
"""Train graph models 50 epochs (no early-stop cutoff) into *_e50 dirs.

Does not overwrite published p0_h1 / p0_fut_h1 / p0_patch_h1.
Copies history.json into 论文插图包/data/runs/<name>_e50/.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "论文插图包"
RUNS = PACK / "data" / "runs"
TRAIN = ROOT / "train_tf.py"

# exp -> dest, batch, use_amp, meteo_clip
EXPS = [
    ("v3", "v3_e50", 32, True, None),
    ("p0", "p0_e50", 32, True, None),
    ("p0_fut", "p0_fut_e50", 16, True, 8.0),
    ("p0_patch", "former_e50", 16, True, None),
]


def copy_history(exp: str, dest_name: str) -> None:
    src = ROOT / "outputs" / "model_runs_tf" / f"{exp}_e50"
    dest = RUNS / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    hist = src / "history.json"
    if not hist.exists():
        raise FileNotFoundError(hist)
    shutil.copy2(hist, dest / "history.json")
    metrics = src / "test_metrics.json"
    if metrics.exists():
        shutil.copy2(metrics, dest / "test_metrics.json")
    print(f"[copy] {hist} -> {dest / 'history.json'}")


def main() -> None:
    python = sys.executable
    for exp, dest_name, batch, use_amp, meteo_clip in EXPS:
        dest = RUNS / dest_name / "history.json"
        if dest.exists():
            hist = __import__("json").loads(dest.read_text())
            y = hist.get("train_loss", [])
            ok = len(y) >= 50 and all(isinstance(v, (int, float)) and v > 1e-8 for v in y[-5:])
            if ok:
                print(f"[{dest_name}] skip, already have {len(y)} epochs")
                continue
            print(f"[{dest_name}] history incomplete/collapsed, retrain")
        out_dir = ROOT / "outputs" / "model_runs_tf" / f"{exp}_e50"
        cmd = [
            python,
            str(TRAIN),
            "--exp",
            exp,
            "--epochs",
            "50",
            "--patience",
            "999",
            "--batch-size",
            str(batch),
            "--out-dir",
            str(out_dir),
        ]
        if not use_amp:
            cmd.append("--no-amp")
        if meteo_clip is not None:
            cmd.extend(["--meteo-clip", str(meteo_clip)])
        print("[run]", " ".join(cmd), flush=True)
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        subprocess.check_call(cmd, cwd=str(ROOT), env=env)
        copy_history(exp, dest_name)
    print("[graph e50] done")


if __name__ == "__main__":
    main()
