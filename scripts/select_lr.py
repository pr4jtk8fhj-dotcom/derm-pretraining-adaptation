#!/usr/bin/env python3
"""Pick, for each (backbone, method), the sweep LR with the best validation AUROC; write configs/hparams_selected.json.

    python scripts/select_lr.py --sweep /workspace/runs/sweep
Stops if any (backbone, method) does not have all its grid values completed.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROJECT_DIR, sha256_file, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--config", default=os.path.join(PROJECT_DIR, "configs", "hparams.json"))
    ap.add_argument("--backbones", nargs="+", default=["panderm", "dinov2"],
                    help="Phase B: --backbones dinov2_cpt adds its choice without changing the existing ones")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    path = os.path.join(PROJECT_DIR, "configs", "hparams_selected.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    table = {}
    for f in glob.glob(os.path.join(a.sweep, "*", "results.json")):
        r = json.load(open(f))
        key = f"{r['backbone']}_{'lora' if r['method'].startswith('lora') else 'full'}"
        table.setdefault(key, {})[r["lr"]] = r["run"]["best_val_auroc"]
    sel = dict(prev.get("selected_lr", {}))
    for bb in a.backbones:
        for m in ["lora", "full"]:
            key, grid = f"{bb}_{m}", cfg[m]["lr_grid"]
            got = table.get(key, {})
            missing = [lr for lr in grid if not any(abs(lr - g) < 1e-12 for g in got)]
            if missing:
                sys.exit(f"{key}: missing sweep runs for lr {missing}; found {got}")
            best = max(got.items(), key=lambda kv: (kv[1], -kv[0]))
            if key in sel and sel[key] != best[0]:
                sys.exit(f"{key} already fixed at {sel[key]} in {path}: not overwriting a frozen choice")
            sel[key] = best[0]
            edge = " (EDGE OF GRID: report it)" if best[0] in (min(grid), max(grid)) else ""
            print(f"{key}: " + ", ".join(f"lr {lr:g} -> val AUROC {v:.4f}" for lr, v in sorted(got.items()))
                  + f" => {best[0]:g}{edge}")
    out = dict(cfg)
    out["selected_lr"] = sel
    out["sweep_val_auroc"] = {**prev.get("sweep_val_auroc", {}),
                              **{k: {f"{lr:g}": v for lr, v in d.items()} for k, d in table.items()}}
    out["_source_config_sha256"] = sha256_file(a.config)
    write_json(path, out)
    print(f"written {path}: from now on fixed (CLAUDE.md rule c)")


if __name__ == "__main__":
    main()
