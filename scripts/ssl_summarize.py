#!/usr/bin/env python3
"""Summary of the continued-pretraining smoke test (engineering check, NOT a model result).

    python scripts/ssl_summarize.py --run $WS/runs/ssl_smoke --batch-per-gpu 48 --gpus 1 [--gpu-log logs/ssl_gpu.csv]

Reads DINOv2's training_metrics.json (one JSON per logged iteration). Throughput is reported in ORIGINAL images/s
(each image yields 2 global + 8 local crops; crops are not counted as images), as the proposal's sizing formula needs:
GPU-hours = g x M / (3600 x Qg). Skips the first --skip iterations (compilation, worker start).
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gpu_info, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch-per-gpu", type=int, required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--skip", type=int, default=20)
    ap.add_argument("--gpu-log", help="nvidia-smi csv (timestamp, utilization.gpu, memory.used)")
    a = ap.parse_args()
    path = os.path.join(a.run, "training_metrics.json")
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        sys.exit(f"no metrics in {path}")
    print(f"logged entries: {len(rows)}; keys: {sorted(rows[-1].keys())}")
    steady = [r for r in rows if r.get("iteration", 0) >= a.skip] or rows
    it_time = [r["iter_time"] for r in steady if "iter_time" in r]
    data_time = [r["data_time"] for r in steady if "data_time" in r]
    loss = [r.get("total_loss") for r in rows if r.get("total_loss") is not None]
    out = {"entries": len(rows), "last_iteration": rows[-1].get("iteration"),
           "batch_per_gpu": a.batch_per_gpu, "gpus": a.gpus, "gpu": gpu_info()}
    if it_time:
        t = statistics.median(it_time)
        out["median_iter_time_s"] = round(t, 4)
        out["original_images_per_s_aggregate"] = round(a.batch_per_gpu * a.gpus / t, 1)
        out["original_images_per_s_per_gpu"] = round(a.batch_per_gpu / t, 1)
    if data_time and it_time:
        out["data_time_fraction_median"] = round(statistics.median(data_time) / statistics.median(it_time), 3)
    if loss:
        k = max(1, len(loss) // 10)
        out["loss_first_10pct_mean"] = round(sum(loss[:k]) / k, 4)
        out["loss_last_10pct_mean"] = round(sum(loss[-k:]) / k, 4)
        out["loss_non_finite"] = sum(1 for x in loss if not math.isfinite(x))
    mm = [r["max_mem"] for r in rows if "max_mem" in r]
    if mm:
        out["max_mem_logged_MB"] = max(mm)
    if a.gpu_log and os.path.exists(a.gpu_log):
        mem = []
        for line in open(a.gpu_log):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[2].split()[0].replace(".", "").isdigit():
                mem.append(float(parts[2].split()[0]))
        if mem:
            out["nvidia_smi_peak_memory_GiB"] = round(max(mem) / 1024, 2)
    out["note"] = "smoke test of continued pretraining: engineering evidence only, not a domain-adapted model"
    write_json(os.path.join(a.run, "ssl_smoke_summary.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
