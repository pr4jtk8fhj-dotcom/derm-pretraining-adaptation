#!/usr/bin/env python3
"""LoRA or full fine-tuning of PanDerm / DINOv2 on the fixed partition, for one training fraction and resolution.

    python scripts/train.py --backbone panderm --method lora --rank 16 --train-frac 0.3 \
        --manifest /workspace/data/partition/manifest.csv --images /workspace/data/images_256 \
        --config configs/hparams_selected.json --out /workspace/runs/matrix --seed 1

Benchmark (test never touched):  ... --epochs 1 --no-test --no-save --tag bench
LR sweep (val only):             ... --lr 3e-4 --no-test --no-save --tag sweep
448 px:                          ... --img 448 --images /workspace/data/images_512 [--grad-ckpt] [--accum 2]
GPU JPEG decoding (train only):  ... --loader dali   (validation/test always use the same torchvision pipeline)
Profiler trace:                  ... --torch-profile  (steps 20-30 of epoch 0 -> <run>/trace.json)

Epochs per fraction from config "epochs_by_frac" (same for every method). Effective batch = config batch;
with --accum k the micro-batch is batch/k (same optimization, less activation memory).
Model selection: epoch with best val AUROC. Threshold: locked on val (95% sensitivity), then applied to test.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import (BACKBONES, Classifier, apply_grad_checkpointing, apply_lora, build_backbone,  # noqa: E402
                       param_groups, patch_panderm_sdpa)
from common import PANDERM_REPO, PROJECT_DIR, WS, GpuSampler, Tee, run_record, sha256_file, write_json  # noqa: E402
from data import ManifestDataset, build_transform, read_manifest  # noqa: E402
from evaluation import evaluate_locked  # noqa: E402

LICENSE_NOTICE = ("PanDerm-derived weights in this directory (fine-tuned checkpoints, LoRA adapters) are under "
                  "CC-BY-NC-ND 4.0 terms: keep them local, do not publish or share.\n")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, amp_dtype):
    model.eval()
    scores = []
    for x, _, _ in loader:
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(x.cuda(non_blocking=True))
        scores.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
    return torch.cat(scores).numpy()


def make_train_iter(a, cfg, tr, micro):
    """Returns (epoch_iterator_factory, steps_per_epoch). Each epoch yields micro-batches (x_gpu_or_cpu, y)."""
    steps = len(tr) // cfg["batch"]
    if a.loader == "dali":
        from dali_loader import DaliTrainLoader
        dl = DaliTrainLoader(tr, a.images, a.backbone, a.img, micro, a.seed, cfg["weighted_sampler"],
                             n_samples=steps * cfg["batch"], threads=a.workers)
        return dl.epoch, steps
    ds = ManifestDataset(tr, a.images, build_transform(a.backbone, train=True, img=a.img))
    g = torch.Generator().manual_seed(a.seed)
    w = (1.0 / np.bincount(tr.y.values)[tr.y.values]) if cfg["weighted_sampler"] else np.ones(len(tr))
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), steps * cfg["batch"], replacement=True,
                                    generator=g)
    dl = DataLoader(ds, batch_size=micro, sampler=sampler, drop_last=True, num_workers=a.workers, pin_memory=True,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None,
                    worker_init_fn=lambda i: seed_all(a.seed * 1000 + i))
    return (lambda: ((x, y) for x, y, _ in dl)), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--cpt-ckpt", help="teacher checkpoint of the continued pretraining (backbone dinov2_cpt)")
    ap.add_argument("--method", required=True, choices=["lora", "full"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train-frac", type=float, default=1.0, choices=[0.1, 0.3, 1.0])
    ap.add_argument("--img", type=int, default=224, choices=[224, 448])
    ap.add_argument("--lr", type=float, help="override config lr (LR sweep only)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default=os.path.join(PROJECT_DIR, "configs", "hparams.json"))
    ap.add_argument("--ckpt", default=os.path.join(WS, "checkpoints", "panderm_ll_data6_checkpoint-499.pth"))
    ap.add_argument("--epochs", type=int, help="override (benchmark only)")
    ap.add_argument("--batch", type=int, help="override config batch (smoke test on tiny data only)")
    ap.add_argument("--amp", choices=["bf16", "fp16"], help="override config")
    ap.add_argument("--sdpa", action="store_true", help="PanDerm fused attention (only after test_backbones --sdpa)")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing on every block")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (micro-batch = batch/accum)")
    ap.add_argument("--loader", choices=["torch", "dali"], default="torch")
    ap.add_argument("--torch-profile", action="store_true")
    ap.add_argument("--no-test", action="store_true", help="never compute test predictions (benchmarks, sweep)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t_start = time.time()

    cfg_all = json.load(open(a.config))
    cfg = {**cfg_all["common"], **cfg_all[a.method]}
    sel = cfg_all.get("selected_lr", {}).get(f"{a.backbone}_{a.method}")
    if sel is not None:
        cfg["lr"] = sel
    elif "selected_lr" in cfg_all and a.lr is None:
        sys.exit(f"no swept lr for {a.backbone}_{a.method} in {a.config}: run lr_sweep.sh for it first "
                 "(equal tuning budget for every encoder and method)")
    if a.lr is not None:
        cfg["lr"] = a.lr
    cfg["epochs"] = a.epochs or cfg["epochs_by_frac"][str(a.train_frac)]
    if a.amp:
        cfg["amp"] = a.amp
    if a.batch:
        cfg["batch"] = a.batch
    assert cfg["batch"] % a.accum == 0
    micro = cfg["batch"] // a.accum
    meth = f"lora{a.rank}" if a.method == "lora" else "full"
    name = (f"{a.backbone}_{meth}_frac{a.train_frac}_s{a.seed}" + (f"_img{a.img}" if a.img != 224 else "")
            + (f"_lr{cfg['lr']:g}" if a.lr is not None else "") + (f"_{a.tag}" if a.tag else ""))
    out = os.path.join(a.out, name)
    if os.path.exists(os.path.join(out, "results.json")) and not a.force:
        print(f"{name}: results.json exists, skipping (use --force to rerun)")
        return
    os.makedirs(out, exist_ok=True)
    Tee(os.path.join(out, "log.txt"))
    if a.backbone == "panderm":
        with open(os.path.join(a.out, "NOTICE_LICENSE.txt"), "w") as f:
            f.write(LICENSE_NOTICE)
    seed_all(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rec = run_record(seed=a.seed, extra={"args": vars(a), "config": cfg, "run_name": name, "micro_batch": micro,
                                         "config_sha256": sha256_file(a.config),
                                         "manifest_sha256": sha256_file(a.manifest)})
    print(json.dumps({"run": name, "config": cfg, "micro_batch": micro}, indent=1))

    # ---- data
    df = read_manifest(a.manifest, a.train_frac)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"train {len(tr)} (malignant {tr.y.sum()}), val {len(va)} ({va.y.sum()}), test {len(te)} ({te.y.sum()})")
    train_epoch, steps_per_epoch = make_train_iter(a, cfg, tr, micro)
    if steps_per_epoch < 1:
        sys.exit(f"training set ({len(tr)} images) smaller than one batch ({cfg['batch']})")
    eval_kw = dict(batch_size=256 if a.img == 224 else 64, num_workers=a.workers, pin_memory=True,
                   persistent_workers=a.workers > 0)
    mk_eval = lambda d: DataLoader(ManifestDataset(d, a.images, build_transform(a.backbone, train=False, img=a.img)),
                                   **eval_kw)
    dl_va = mk_eval(va)

    # ---- model
    bb = build_backbone(a.backbone, drop_path=cfg["drop_path"], panderm_repo=PANDERM_REPO, panderm_ckpt=a.ckpt,
                        img_size=a.img, cpt_ckpt=a.cpt_ckpt)
    if a.cpt_ckpt:
        rec["cpt_ckpt_sha256"] = sha256_file(a.cpt_ckpt)
    rec["load_report"] = bb.load_report
    if a.method == "lora":
        apply_lora(bb, a.rank, cfg["alpha_over_rank"] * a.rank, tuple(cfg["targets"]))
    if a.sdpa:
        assert a.backbone == "panderm", "--sdpa is for PanDerm only"
        patch_panderm_sdpa(bb)
    if a.grad_ckpt:
        apply_grad_checkpointing(bb)
    model = Classifier(bb).cuda()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    rec["params"] = {"trainable": n_train, "total": n_total}
    print(f"trainable params {n_train:,} / {n_total:,}")

    groups = param_groups(model, cfg["lr"], cfg["weight_decay"], cfg.get("layer_decay"))
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(cfg["betas"]), fused=True)
    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] == "fp16")
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return cfg["lr"] * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return cfg["min_lr"] + (cfg["lr"] - cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * p))

    prof = None
    if a.torch_profile:
        from torch.profiler import ProfilerActivity, profile, schedule
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(wait=19, warmup=1, active=10, repeat=1), record_shapes=False)
        prof.start()

    # ---- train
    gs = GpuSampler()
    gs.start()
    torch.cuda.reset_peak_memory_stats()
    t_train0 = time.time()
    rec["setup_seconds"] = round(t_train0 - t_start, 1)
    best, best_state, best_ep, epochs_log, step, t_crit = -1.0, None, -1, [], 0, None
    for ep in range(cfg["epochs"]):
        model.train()
        torch.cuda.synchronize()
        t_ep, wait, seen, loss_sum = time.time(), 0.0, 0, torch.zeros((), device="cuda")
        t_mark, seen_mark = None, 0
        it = train_epoch()
        for k in range(steps_per_epoch):
            if k == 10:
                torch.cuda.synchronize()
                t_mark, seen_mark = time.time(), seen
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step) * gr["lr_scale"]
            opt.zero_grad(set_to_none=True)
            for _ in range(a.accum):
                t0 = time.time()
                x, y = next(it)
                wait += time.time() - t0
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = crit(model(x), y) / a.accum
                scaler.scale(loss).backward()
                loss_sum += loss.detach()
                seen += len(y)
            scaler.step(opt)
            scaler.update()
            step += 1
            if prof is not None:
                prof.step()
                if step == 31:
                    prof.stop()
                    prof.export_chrome_trace(os.path.join(out, "trace.json"))
                    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
                    prof = None
        torch.cuda.synchronize()
        t_tr = time.time() - t_ep
        ips_steady = (seen - seen_mark) / (time.time() - t_mark) if t_mark else None
        t_v0 = time.time()
        sv = predict(model, dl_va, amp_dtype)
        auc = float(roc_auc_score(va.y.values, sv))
        t_val = time.time() - t_v0
        elapsed = time.time() - t_train0
        if t_crit is None and auc >= cfg["val_criterion_auroc"]:
            t_crit = elapsed
        if auc > best:
            best, best_ep = auc, ep
            keep = (lambda k: "lora_" in k or k.startswith("head.")) if a.method == "lora" else (lambda k: True)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if keep(k)}
        e = {"epoch": ep, "loss": round(float(loss_sum) / steps_per_epoch, 5), "val_auroc": round(auc, 5),
             "train_seconds": round(t_tr, 1), "val_seconds": round(t_val, 1), "img_per_s": round(seen / t_tr, 1),
             "img_per_s_steady": round(ips_steady, 1) if ips_steady else None,
             "data_wait_pct": round(100 * wait / t_tr, 1), "elapsed_s": round(elapsed, 1)}
        epochs_log.append(e)
        print(json.dumps(e))
    gs.stop()
    rec["peak_mem_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    rec["peak_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    rec["gpu_train"] = gs.summary(skip_first_s=10)
    rec["train_seconds_total"] = round(time.time() - t_train0, 1)
    rec["time_to_val_criterion_s"] = round(t_crit, 1) if t_crit is not None else None
    rec["best_epoch"], rec["best_val_auroc"] = best_ep, best
    rec["epochs"] = epochs_log

    # ---- final evaluation with the best checkpoint
    model.load_state_dict(best_state, strict=a.method == "full")
    sv = predict(model, dl_va, amp_dtype)
    result = {"run": rec, "backbone": a.backbone, "method": meth, "train_frac": a.train_frac, "img": a.img,
              "seed": a.seed, "lr": cfg["lr"], "n_train": len(tr), "loader": a.loader}
    if not a.no_test:
        st = predict(model, mk_eval(te), amp_dtype)
        result["metrics"] = evaluate_locked(va.y.values, sv, te.y.values, st, te.group_id.values,
                                            cfg["target_sensitivity"], a.n_boot, a.seed)
        pd.DataFrame({"image_id": list(va.image_id) + list(te.image_id), "split": ["val"] * len(va) + ["test"] * len(te),
                      "group_id": list(va.group_id) + list(te.group_id), "source": list(va.source) + list(te.source),
                      "y": list(va.y) + list(te.y), "score": list(sv) + list(st)}
                     ).to_csv(os.path.join(out, "predictions.csv"), index=False)
        t = result["metrics"]["test"]
        print(f"TEST frac{a.train_frac}: AUROC {t['auroc']:.4f} sens {t['sensitivity']:.3f} spec {t['specificity']:.3f} "
              f"FN {t['false_negatives']} flagged {t['flag_rate']:.3f} (thr {result['metrics']['threshold']:.4f})")
    if not a.no_save:
        t_ck = time.time()
        torch.save({"state": best_state, "method": meth, "backbone": a.backbone, "img": a.img, "config": cfg},
                   os.path.join(out, "ckpt_best.pt"))
        rec["checkpoint_save_seconds"] = round(time.time() - t_ck, 2)
        rec["checkpoint_bytes"] = os.path.getsize(os.path.join(out, "ckpt_best.pt"))
    rec["wall_seconds"] = round(time.time() - t_start, 1)
    rec["gpu_hours"] = round(rec["wall_seconds"] / 3600, 4)
    write_json(os.path.join(out, "results.json"), result)
    print(f"done: {name} in {rec['wall_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
