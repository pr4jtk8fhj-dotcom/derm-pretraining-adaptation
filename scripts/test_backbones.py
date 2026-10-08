#!/usr/bin/env python3
"""Mandatory checks before any training run.

1. LoRALinear unit test (math, gradients only on A/B).
2. For each backbone: checkpoint loads (missing/unexpected keys printed); LoRA with B=0 gives outputs
   IDENTICAL to the original (CPU fp32, torch.equal); with B != 0 the output changes (LoRA is not bypassed);
   gradients reach every lora_A/lora_B and no frozen parameter.
3. --sdpa: PanDerm fused attention vs original, max |diff| of outputs and gradients (fp32 CPU and bf16 GPU).

    python scripts/test_backbones.py --ckpt /workspace/checkpoints/panderm_ll_data6_checkpoint-499.pth [--sdpa]
Writes logs/test_backbones.json.
"""
import argparse
import copy
import inspect
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import LoRALinear, apply_lora, build_backbone, patch_panderm_sdpa  # noqa: E402
from common import PANDERM_REPO, PROJECT_DIR, run_record, write_json  # noqa: E402


def unit_lora():
    torch.manual_seed(0)
    lin = nn.Linear(16, 48, bias=False)
    x = torch.randn(4, 16)
    y0 = lin(x)
    lo = LoRALinear(copy.deepcopy(lin), r=4, alpha=8)
    assert torch.equal(lo(x), y0) and torch.equal(nn.functional.linear(x, lo.weight), y0)
    nn.init.normal_(lo.lora_B)
    ref = x @ (lin.weight + 2.0 * lo.lora_B @ lo.lora_A).T
    assert torch.allclose(lo(x), ref, atol=1e-5)
    lo(x).sum().backward()
    assert lo.lora_A.grad is not None and lo.lora_B.grad is not None and lo.base_weight.grad is None
    return "ok"


def check_backbone(name, ckpt, res):
    torch.manual_seed(0)
    m = build_backbone(name, panderm_repo=PANDERM_REPO, panderm_ckpt=ckpt).eval()
    r = {"load_report": m.load_report, "feat_dim": m.feat_dim}
    if name == "panderm":
        src = inspect.getsource(type(m.blocks[0].attn).forward)
        r["attention_reads_qkv_weight"] = "qkv.weight" in src
        print(f"[{name}] Attention.forward reads qkv.weight directly: {r['attention_reads_qkv_weight']}")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y0 = m(x)
        apply_lora(m, r=8, alpha=16)
        y1 = m(x)
    r["out_shape"] = list(y0.shape)
    r["lora_zero_identical"] = bool(torch.equal(y0, y1))
    r["lora_zero_max_abs_diff"] = float((y0 - y1).abs().max())
    print(f"[{name}] LoRA B=0 identical to original: {r['lora_zero_identical']} (max diff {r['lora_zero_max_abs_diff']})")

    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_B, std=0.02)
    m.train()
    y2 = m(x)
    r["lora_nonzero_changes_output"] = bool((y2.detach() - y0).abs().max() > 1e-4)
    y2.float().pow(2).mean().backward()
    lora_p = [(n, p) for n, p in m.named_parameters() if "lora_" in n]
    r["n_lora_tensors"] = len(lora_p)
    r["n_lora_params"] = sum(p.numel() for _, p in lora_p)
    r["lora_all_have_grad"] = all(p.grad is not None and p.grad.abs().sum() > 0 for _, p in lora_p)
    r["frozen_with_grad"] = [n for n, p in m.named_parameters() if "lora_" not in n and p.grad is not None]
    print(f"[{name}] B!=0 changes output: {r['lora_nonzero_changes_output']}; lora tensors {r['n_lora_tensors']} "
          f"({r['n_lora_params']} params) all with grad: {r['lora_all_have_grad']}; frozen with grad: {r['frozen_with_grad']}")
    ok = (r["lora_zero_identical"] and r["lora_nonzero_changes_output"] and r["lora_all_have_grad"]
          and not r["frozen_with_grad"] and r["n_lora_tensors"] == 4 * len(m.blocks))
    r["PASS"] = bool(ok)
    res[name] = r
    del m


def check_sdpa(ckpt, res):
    out = {}
    for dev, dtype in [("cpu", torch.float32)] + ([("cuda", torch.bfloat16)] if torch.cuda.is_available() else []):
        torch.manual_seed(0)
        ref = build_backbone("panderm", panderm_repo=PANDERM_REPO, panderm_ckpt=ckpt).to(dev).train()
        new = copy.deepcopy(ref)
        patch_panderm_sdpa(new)
        for mm in (ref, new):  # deterministic comparison: no dropout / drop path
            for mod in mm.modules():
                if isinstance(mod, nn.Dropout) or type(mod).__name__ == "DropPath":
                    mod.p = 0.0
                    if hasattr(mod, "drop_prob"):
                        mod.drop_prob = 0.0
        x = torch.randn(2, 3, 224, 224, device=dev)
        ys = []
        for mm in (ref, new):
            with torch.autocast(dev, dtype=dtype, enabled=dtype != torch.float32):
                y = mm(x)
            y.float().pow(2).mean().backward()
            ys.append(y.float().detach())
        gdiff = max(float((p1.grad - p2.grad).abs().max()) for p1, p2 in zip(ref.parameters(), new.parameters())
                    if p1.grad is not None)
        gref = max(float(p.grad.abs().max()) for p in ref.parameters() if p.grad is not None)
        out[f"{dev}_{dtype}"] = {"out_max_abs_diff": float((ys[0] - ys[1]).abs().max()),
                                 "out_max_abs": float(ys[0].abs().max()),
                                 "grad_max_abs_diff": gdiff, "grad_max_abs": gref}
        print(f"[sdpa {dev} {dtype}] {out[f'{dev}_{dtype}']}")
        del ref, new
    res["sdpa"] = out


def check_448(ckpt, res):
    """448 px: PanDerm loads with interpolated position tables; both backbones run a forward pass + LoRA backward;
    peak memory of one full-fine-tuning step at micro-batch 16 with and without activation checkpointing."""
    from backbones import apply_grad_checkpointing
    out = {}
    for name in ["panderm", "dinov2"]:
        r = {}
        m = build_backbone(name, panderm_repo=PANDERM_REPO, panderm_ckpt=ckpt, img_size=448).cuda()
        r["interpolated"] = m.load_report.get("interpolated")
        x = torch.randn(16, 3, 448, 448, device="cuda")
        for gc in (False, True):
            if gc:
                apply_grad_checkpointing(m)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(x)
            y.float().pow(2).mean().backward()
            m.zero_grad(set_to_none=True)
            r[f"peak_mem_gb_full_ft_mb16_gradckpt_{gc}"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        r["out_shape"] = list(y.shape)
        print(f"[448 {name}] {r}")
        out[name] = r
        del m
        torch.cuda.empty_cache()
    res["img448"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sdpa", action="store_true")
    ap.add_argument("--img448", action="store_true", help="also check 448 px loading and memory (GPU)")
    ap.add_argument("--only", choices=["panderm", "dinov2"])
    a = ap.parse_args()
    res = {"unit_lora": unit_lora()}
    print("LoRALinear unit test: ok")
    for name in ["panderm", "dinov2"]:
        if a.only in (None, name):
            check_backbone(name, a.ckpt, res)
    if a.sdpa:
        check_sdpa(a.ckpt, res)
    if a.img448:
        check_448(a.ckpt, res)
    write_json(os.path.join(PROJECT_DIR, "logs", "test_backbones.json"), {"run": run_record(seed=0), "results": res})
    fails = [k for k, v in res.items() if isinstance(v, dict) and v.get("PASS") is False]
    print("FAILED: " + ", ".join(fails) if fails else "ALL LoRA CHECKS PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
