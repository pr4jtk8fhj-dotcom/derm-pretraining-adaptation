#!/usr/bin/env python3
"""Check that continued pretraining really started from the released DINOv2 weights: the DINOv2 trainer loads
student.pretrained_weights with strict=False, so a silent mismatch would mean training from random init.
Compares the earliest saved teacher backbone with the converted init weights (relative L2 difference per tensor).

    python scripts/ssl_check_init.py --run /workspace/runs/ssl_smoke
"""
import argparse
import glob
import os
import re
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tol", type=float, default=1e-2, help="max relative diff allowed (mask_token excluded)")
    a = ap.parse_args()
    init = torch.load(os.path.join(a.run, "init_vitl14_224.pth"), map_location="cpu", weights_only=True)["model"]
    ck = sorted(glob.glob(os.path.join(a.run, "eval", "*", "teacher_checkpoint.pth")),
                key=lambda p: int(re.findall(r"\d+", os.path.basename(os.path.dirname(p)))[-1]))
    if not ck:
        sys.exit("no teacher checkpoint found")
    t = torch.load(ck[0], map_location="cpu", weights_only=False)["teacher"]  # our own training output
    tb = {k[len("backbone."):]: v for k, v in t.items() if k.startswith("backbone.")}
    missing = sorted(set(init) - set(tb))
    rel = {k: float((tb[k].float() - init[k].float()).norm() / (init[k].float().norm() + 1e-8))
           for k in init if k in tb and k != "mask_token"}  # mask_token is all zeros in the released weights
    worst = max(rel, key=rel.get)
    res = {"checkpoint": ck[0], "n_init": len(init), "n_teacher_backbone": len(tb), "missing_in_teacher": missing,
           "median_rel_diff": statistics.median(rel.values()), "max_rel_diff": rel[worst], "max_rel_diff_key": worst,
           "PASS": not missing and rel[worst] < a.tol}
    print(res)
    write_json(os.path.join(a.run, "init_check.json"), res)
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
