#!/usr/bin/env python3
"""Frozen backbone + logistic regression, for each training fraction. Features are extracted ONCE per backbone
for every image of the manifest and cached; the classifier is refit per fraction on its train rows.

    python scripts/frozen_baseline.py --backbone panderm --images /workspace/data/images_256 \
        --manifest /workspace/data/partition/manifest.csv --out /workspace/runs/frozen

Classifier fixed in advance (no tuning): StandardScaler + LogisticRegression(C=1.0, class_weight='balanced').
Threshold locked on val at 95% sensitivity, then applied to test. Per-institution results: aggregate.py.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import BACKBONES, build_backbone  # noqa: E402
from common import PANDERM_REPO, WS, GpuSampler, Tee, run_record, sha256_file, write_json  # noqa: E402
from data import ManifestDataset, build_transform, read_manifest  # noqa: E402
from evaluation import evaluate_locked  # noqa: E402

FRACS = (0.1, 0.3, 1.0)


@torch.no_grad()
def extract(model, df, image_dir, backbone, bs, workers):
    ds = ManifestDataset(df, image_dir, build_transform(backbone, train=False, img=224))
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=True)
    feats = []
    torch.cuda.synchronize()
    t0 = time.time()
    for x, _, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(model(x.cuda(non_blocking=True)).float().cpu())
    torch.cuda.synchronize()
    return torch.cat(feats).numpy(), len(ds) / (time.time() - t0), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--cpt-ckpt", help="teacher checkpoint of the continued pretraining (backbone dinov2_cpt)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.path.join(WS, "checkpoints", "panderm_ll_data6_checkpoint-499.pth"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--target-sens", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    out = os.path.join(a.out, a.backbone)
    os.makedirs(out, exist_ok=True)
    Tee(os.path.join(out, "log.txt"))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rec = run_record(seed=a.seed, extra={"args": vars(a), "manifest_sha256": sha256_file(a.manifest)})
    t_start = time.time()

    full = read_manifest(a.manifest, 1.0).sort_values("image_id").reset_index(drop=True)
    feat_path = os.path.join(out, "features.npz")
    gs = GpuSampler()
    gs.start()
    if os.path.exists(feat_path):
        z = np.load(feat_path, allow_pickle=False)
        assert list(z["image_id"]) == list(full.image_id), "cached features do not match the manifest"
        F, ips, ext_s = z["feats"], float(z["ips"]), float(z["seconds"])
        print(f"loaded cached features {F.shape}")
    else:
        model = build_backbone(a.backbone, panderm_repo=PANDERM_REPO, panderm_ckpt=a.ckpt,
                               cpt_ckpt=a.cpt_ckpt).cuda().eval()
        if a.cpt_ckpt:
            rec["cpt_ckpt_sha256"] = sha256_file(a.cpt_ckpt)
        rec["load_report"] = model.load_report
        torch.cuda.reset_peak_memory_stats()
        F, ips, ext_s = extract(model, full, a.images, a.backbone, a.batch, a.workers)
        rec["extract_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        np.savez(feat_path, image_id=full.image_id.values.astype(str), feats=F, ips=ips, seconds=ext_s)
        del model
    gs.stop()
    rec.update({"n_images": len(full), "extract_img_per_s": round(ips, 1), "extract_seconds": round(ext_s, 1),
                "gpu_extract": gs.summary(skip_first_s=5)})
    print(f"extraction: {len(full)} images, {ips:.0f} img/s, {ext_s:.0f} s")

    pos = {k: i for i, k in enumerate(full.image_id)}
    results = {}
    for frac in FRACS:
        df = read_manifest(a.manifest, frac)
        part = {s: df[df.split == s] for s in ("train", "val", "test")}
        X = {s: F[[pos[k] for k in p.image_id]] for s, p in part.items()}
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                                                                 random_state=a.seed))
        t0 = time.time()
        clf.fit(X["train"], part["train"].y.values)
        sv, st = clf.predict_proba(X["val"])[:, 1], clf.predict_proba(X["test"])[:, 1]
        r = evaluate_locked(part["val"].y.values, sv, part["test"].y.values, st, part["test"].group_id.values,
                            a.target_sens, a.n_boot, a.seed)
        r["fit_seconds"], r["n_train"] = round(time.time() - t0, 1), len(part["train"])
        key = f"frac{frac}"
        results[key] = r
        va, te = part["val"], part["test"]
        pd.DataFrame({"image_id": list(va.image_id) + list(te.image_id), "split": ["val"] * len(va) + ["test"] * len(te),
                      "group_id": list(va.group_id) + list(te.group_id), "source": list(va.source) + list(te.source),
                      "y": list(va.y) + list(te.y), "score": list(sv) + list(st)}
                     ).to_csv(os.path.join(out, f"predictions_{key}.csv"), index=False)
        t = r["test"]
        print(f"[{key}] n_train {len(part['train'])} thr={r['threshold']:.4f} | test AUROC {t['auroc']:.4f} "
              f"sens {t['sensitivity']:.3f} spec {t['specificity']:.3f} FN {t['false_negatives']} flagged {t['flag_rate']:.3f}")

    rec["wall_seconds"] = round(time.time() - t_start, 1)
    rec["gpu_hours"] = round(rec["wall_seconds"] / 3600, 4)
    write_json(os.path.join(out, "results.json"), {"run": rec, "method": "frozen", "backbone": a.backbone,
                                                   "results": results})


if __name__ == "__main__":
    main()
