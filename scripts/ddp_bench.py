#!/usr/bin/env python3
"""Optional (pod with >= 2 GPUs): data-parallel scaling of full fine-tuning, throughput only, validation/test untouched.

    torchrun --nproc_per_node 1 scripts/ddp_bench.py --backbone panderm --manifest .../manifest.csv --images .../images_256
    torchrun --nproc_per_node 2 scripts/ddp_bench.py --backbone panderm --manifest .../manifest.csv --images .../images_256

Per-GPU batch fixed (default 128): global batch grows with GPUs (weak scaling). Reports img/s after warm-up and
scaling efficiency = ips_N / (N * ips_1) when --ref-ips (the 1-GPU result) is given.
"""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import Classifier, build_backbone  # noqa: E402
from common import PANDERM_REPO, PROJECT_DIR, WS, run_record, write_json  # noqa: E402
from data import ManifestDataset, build_transform, read_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["panderm", "dinov2"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", default=os.path.join(WS, "checkpoints", "panderm_ll_data6_checkpoint-499.pth"))
    ap.add_argument("--img", type=int, default=224, choices=[224, 448])
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--batch", type=int, default=128, help="per GPU")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ref-ips", type=float)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(1)
    df = read_manifest(a.manifest, 1.0)
    tr = df[df.split == "train"].reset_index(drop=True)
    ds = ManifestDataset(tr, a.images, build_transform(a.backbone, train=True, img=a.img))
    sampler = DistributedSampler(ds, shuffle=True, seed=1, drop_last=True)
    dl = DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    bb = build_backbone(a.backbone, drop_path=0.1, panderm_repo=PANDERM_REPO, panderm_ckpt=a.ckpt, img_size=a.img)
    if a.grad_ckpt:
        from backbones import apply_grad_checkpointing
        apply_grad_checkpointing(bb)
    model = Classifier(bb).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    crit = nn.CrossEntropyLoss()
    step, t0, epoch = 0, None, 0
    while step < a.steps:
        sampler.set_epoch(epoch)
        for x, y, _ in dl:
            if step == a.warmup:
                torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x.cuda(non_blocking=True)), y.cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= a.steps:
                break
        epoch += 1
    torch.cuda.synchronize(); dist.barrier()
    ips = (a.steps - a.warmup) * a.batch * world / (time.time() - t0)
    if rank == 0:
        res = {"run": run_record(seed=1, extra={"args": vars(a)}), "world_size": world, "img_per_s": round(ips, 1),
               "peak_mem_gb_rank0": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "scaling_efficiency": round(ips / (world * a.ref_ips), 3) if a.ref_ips else None}
        print(res)
        write_json(os.path.join(PROJECT_DIR, "logs", f"ddp_bench_{a.backbone}_img{a.img}_{world}gpu.json"), res)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
