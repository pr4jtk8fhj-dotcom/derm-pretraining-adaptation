#!/usr/bin/env python3
"""Estimate GPU-hours of the LR sweep and of the matrix from MEASURED 1-epoch benchmarks.

Benchmarks: train.py --train-frac 1.0 --epochs 1 --no-test --no-save --tag bench, for {panderm, dinov2} x {lora, full}.

    python scripts/estimate_matrix.py --bench /workspace/runs/bench --manifest /workspace/data/partition/manifest.csv \
        [--gpus 1] [--margin 0.3]

Per run: setup + epochs * (train_s_per_img * n_train + val_s_per_img * n_val) + final val/test prediction.
The first epoch includes warm-up (cudnn autotune, worker start), so per-image times are conservative.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PRICE_PER_GPU_HOUR, PROJECT_DIR  # noqa: E402
from data import read_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=os.path.join(PROJECT_DIR, "configs", "hparams.json"))
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--budget-h", type=float, default=7.0, help="wall-clock hours available for the matrix")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    ep_by = cfg["common"]["epochs_by_frac"]

    n = {}
    for f in (0.1, 0.3, 1.0):
        df = read_manifest(a.manifest, f)
        n[f] = {s: int((df.split == s).sum()) for s in ("train", "val", "test")}
    print(f"sizes: {n}")

    bench = {}
    for path in glob.glob(os.path.join(a.bench, "*", "results.json")):
        r = json.load(open(path))
        run, ep = r["run"], r["run"]["epochs"][0]
        key = (r["backbone"], "lora" if r["method"].startswith("lora") else "full")
        bench[key] = {"train_s_img": ep["train_seconds"] / r["n_train"], "val_s_img": ep["val_seconds"] / n[1.0]["val"],
                      "setup_s": run["setup_seconds"], "img_per_s": ep["img_per_s"], "img_per_s_steady": ep["img_per_s_steady"],
                      "data_wait_pct": ep["data_wait_pct"], "gpu_idle_pct": run["gpu_train"]["gpu_idle_pct"],
                      "peak_mem_gb": run["peak_mem_allocated_gb"]}
    need = {(b, m) for b in ("panderm", "dinov2") for m in ("lora", "full")}
    if need - set(bench):
        sys.exit(f"missing benchmarks: {need - set(bench)}")
    for k, v in sorted(bench.items()):
        print(f"measured {k}: {v}")

    def hours(key, frac):
        b, s = bench[key], n[frac]
        return (b["setup_s"] + ep_by[str(frac)] * (b["train_s_img"] * s["train"] + b["val_s_img"] * s["val"])
                + b["val_s_img"] * (s["val"] + s["test"])) / 3600

    sweep = sum(3 * hours(k, 0.3) for k in need)
    seed_all = sum(hours(k, f) for k in need for f in (0.1, 0.3, 1.0))
    seed_100 = sum(hours(k, 1.0) for k in need)
    m = 1 + a.margin
    print(f"\nGPU-hours (no margin): sweep {sweep:.2f}; one seed all fractions {seed_all:.2f}; one seed 100% only {seed_100:.2f}")
    for k in sorted(need):
        print(f"  {k}: " + ", ".join(f"frac {f}: {hours(k, f) * 60:.1f} min" for f in (0.1, 0.3, 1.0)))
    plans = {
        "A: sweep + seed1 all fractions": sweep + seed_all,
        "B: A + seeds 2,3 at 100%": sweep + seed_all + 2 * seed_100,
        "C: sweep + seeds 1,2,3 all fractions": sweep + 3 * seed_all,
    }
    print(f"\nWith {int(a.margin * 100)}% margin, {a.gpus} GPU(s) running independent runs:")
    for name, h in plans.items():
        wall = h * m / a.gpus
        cost = wall * a.gpus * PRICE_PER_GPU_HOUR
        print(f"  {name}: {h * m:.2f} GPU-h -> {wall:.2f} h wall, ~{cost:.0f} $ "
              f"{'OK' if wall <= a.budget_h else 'OVER the ' + str(a.budget_h) + ' h window'}")
    print("(Wall time with >1 GPU assumes balanced shards; cost covers only these runs, not the rest of the pod time.)")


if __name__ == "__main__":
    main()
