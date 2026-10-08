#!/usr/bin/env python3
"""Inputs for the proposal's resource table (section 7): storage, files, host resources, checkpoint sizes.
Measured on the rented H100 pod: label them as such, not as Leonardo measurements.

    python scripts/resource_report.py --ws /workspace --out logs/resource_report.json
"""
import argparse
import glob
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gpu_info, write_json  # noqa: E402


def du(path):
    n, size = 0, 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                size += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return {"files": n, "GiB": round(size / 2 ** 30, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default=os.environ.get("WS", "/workspace"))
    ap.add_argument("--out", default="logs/resource_report.json")
    a = ap.parse_args()
    ws = a.ws
    rep = {"host": platform.node(), "cpus": os.cpu_count(), "gpu": gpu_info()}
    try:
        with open("/proc/meminfo") as f:
            rep["host_ram_GiB"] = round(int(f.readline().split()[1]) / 2 ** 20, 1)
    except OSError:
        pass
    rep["storage"] = {name: du(os.path.join(ws, p)) for name, p in [
        ("labelled_images_original", "data/images"), ("labelled_cache_256", "data/images_256"),
        ("labelled_cache_512", "data/images_512"), ("ssl_images_original", "data/ssl_images"),
        ("ssl_cache_256", "data/ssl_images_256"), ("frozen_features", "runs/frozen"), ("matrix_runs", "runs/matrix"),
        ("ssl_runs", "runs/ssl_smoke"), ("checkpoints_pretrained", "checkpoints")] if os.path.exists(os.path.join(ws, p))}
    ck = [os.path.getsize(p) for p in glob.glob(os.path.join(ws, "runs/matrix/*/ckpt_best.pt"))]
    if ck:
        rep["downstream_checkpoint_GiB"] = {"min": round(min(ck) / 2 ** 30, 3), "max": round(max(ck) / 2 ** 30, 3),
                                            "count": len(ck)}
    bl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "bench_loader.json")
    if os.path.exists(bl):
        rep["input_pipeline_throughput"] = json.load(open(bl)).get("cases")
    write_json(a.out, rep)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
