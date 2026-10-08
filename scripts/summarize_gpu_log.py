#!/usr/bin/env python3
"""Summarize an nvidia-smi CSV log (timestamp, util.gpu, util.mem, mem.used MiB, power W).

    python scripts/summarize_gpu_log.py logs/original_profile/nvidia_smi.csv [--skip-s 60]
--skip-s drops the first seconds (model loading) so idle% refers to the training loop.
"""
import argparse
import csv
import datetime
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip-s", type=float, default=0.0)
    a = ap.parse_args()
    rows = [r for r in csv.reader(open(a.csv)) if len(r) >= 4]
    ts = [datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f") for r in rows]
    t0 = ts[0]
    keep = [r for r, t in zip(rows, ts) if (t - t0).total_seconds() >= a.skip_s]
    util = [float(r[1]) for r in keep]
    mem = [float(r[3]) for r in keep]
    out = {"samples": len(keep), "duration_s": round((ts[-1] - t0).total_seconds(), 1),
           "gpu_util_mean": round(sum(util) / len(util), 1), "gpu_idle_pct": round(100 - sum(util) / len(util), 1),
           "pct_samples_util_0": round(100 * sum(u == 0 for u in util) / len(util), 1),
           "mem_used_max_gb": round(max(mem) / 1024, 2)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
