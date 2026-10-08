#!/usr/bin/env python3
"""Input-pipeline throughput alone (no model): how many training images/s each loader can deliver to the GPU.

    python scripts/bench_loader.py --manifest /workspace/data/partition/manifest.csv \
        --orig /workspace/data/images --cache256 /workspace/data/images_256 --cache512 /workspace/data/images_512

Cases (each at the resolution it serves): torchvision+PIL on original JPEGs, torchvision on the resized cache,
DALI GPU decoding on original JPEGs. If a loader delivers fewer images/s than the model consumes (train.py
img_per_s_steady), training is input-bound. Writes logs/bench_loader.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROJECT_DIR, run_record, write_json  # noqa: E402
from data import ManifestDataset, build_transform, read_manifest  # noqa: E402


def time_iter(gen, n_batches, warm=5):
    """images/s after warm-up, plus per-channel mean/std of the produced (normalized) tensors: a distributional
    check of augmentation semantics between pipelines (stochastic pipelines cannot be compared bit by bit)."""
    t0, seen, s1, s2, npx = None, 0, 0.0, 0.0, 0
    for k, (x, y) in enumerate(gen):
        if k == warm:
            torch.cuda.synchronize(); t0, seen = time.time(), 0
        x = x.cuda(non_blocking=True)
        seen += x.shape[0]
        xf = x.float()
        s1 = s1 + xf.sum((0, 2, 3)); s2 = s2 + (xf ** 2).sum((0, 2, 3)); npx += xf.numel() // xf.shape[1]
        if k + 1 >= warm + n_batches:
            break
    torch.cuda.synchronize()
    mean = (s1 / npx); std = (s2 / npx - mean ** 2).clamp_min(0).sqrt()
    return {"img_per_s": round(seen / (time.time() - t0), 1), "shape": list(x.shape[1:]),
            "channel_mean": [round(v, 4) for v in mean.tolist()], "channel_std": [round(v, 4) for v in std.tolist()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--cache256")
    ap.add_argument("--cache512")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    df = read_manifest(a.manifest, 1.0)
    tr = df[df.split == "train"].reset_index(drop=True)
    res = {"cpu_count": os.cpu_count(), "workers": a.workers, "batch": a.batch, "cases": {}}

    def torch_case(root, img):
        ds = ManifestDataset(tr, root, build_transform("panderm", train=True, img=img))
        dl = DataLoader(ds, batch_size=a.batch, sampler=RandomSampler(ds, generator=torch.Generator().manual_seed(0)),
                        num_workers=a.workers, pin_memory=True, drop_last=True, prefetch_factor=4)
        return time_iter(((x, y) for x, y, _ in dl), a.batches)

    for img in (224, 448):
        res["cases"][f"torchvision_original_{img}"] = torch_case(a.orig, img)
        cache = a.cache256 if img == 224 else a.cache512
        if cache:
            res["cases"][f"torchvision_cache_{img}"] = torch_case(cache, img)
        try:
            from dali_loader import DaliTrainLoader
            dl = DaliTrainLoader(tr, a.orig, "panderm", img, a.batch, 0, False,  # uniform sampling, like torchvision case
                                 n_samples=(a.batches + 10) * a.batch, threads=min(16, os.cpu_count()))
            res["cases"][f"dali_gpu_decode_original_{img}"] = time_iter(dl.epoch(), a.batches)
        except Exception as e:
            res["cases"][f"dali_gpu_decode_original_{img}"] = f"failed: {type(e).__name__}: {e}"
    for img in (224, 448):
        tv, da = res["cases"].get(f"torchvision_original_{img}"), res["cases"].get(f"dali_gpu_decode_original_{img}")
        if isinstance(tv, dict) and isinstance(da, dict):
            res[f"distribution_check_{img}"] = {
                "abs_diff_channel_mean": [round(abs(p - q), 4) for p, q in zip(tv["channel_mean"], da["channel_mean"])],
                "abs_diff_channel_std": [round(abs(p - q), 4) for p, q in zip(tv["channel_std"], da["channel_std"])],
                "note": "same images, same augmentation family; differences reflect resampling and crop implementation"}
    for k, v in res.items():
        if k.startswith(("cases", "distribution")):
            print(f"{k}: {json.dumps(v)}")
    write_json(os.path.join(PROJECT_DIR, "logs", "bench_loader.json"), {"run": run_record(seed=0), **res})


if __name__ == "__main__":
    main()
