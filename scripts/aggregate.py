#!/usr/bin/env python3
"""Final tables from saved predictions (no number typed by hand).

    python scripts/aggregate.py --frozen /workspace/runs/frozen --matrix /workspace/runs/matrix --out results

Per run: threshold locked on validation (95% sensitivity, deterministic rule in evaluation.py), external metrics
pooled and per test institution, 95% CI by bootstrap over patient/lesion groups (1000 resamples, threshold fixed).
Between-seed variation is reported separately (curves.csv: mean and SD across seeds).
Paired bootstrap on identical external images (same resolution, fraction, seed):
  - PRIMARY contrast of the proposal: dinov2_cpt minus dinov2, full fine-tuning, 224 px, 100% annotations, AUROC
    (written to primary.md when both arms exist; otherwise reported as missing);
  - secondary: dinov2_cpt vs dinov2 for every other method/fraction/resolution; PanDerm vs DINOv2 (external
    reference, not attributable to domain pretraining alone); full vs LoRA, LoRA vs frozen, full vs frozen.
Outputs: runs_long.csv, curves.csv, paired.csv, primary.md, tables.md.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PRICE_PER_GPU_HOUR, run_record, write_json  # noqa: E402
from evaluation import bootstrap, metrics_at, paired_bootstrap, threshold_at_sensitivity  # noqa: E402

KEYS = ["auroc", "sensitivity", "specificity", "flag_rate", "false_negatives"]
PRIMARY = {"treated": "dinov2_cpt", "reference": "dinov2", "method": "full", "img": 224, "frac": 1.0, "metric": "auroc"}


def load_runs(frozen_dir, matrix_dir):
    runs = []
    for f in glob.glob(os.path.join(frozen_dir, "*", "results.json")):
        r = json.load(open(f))
        d = os.path.dirname(f)
        for key in r["results"]:
            p = pd.read_csv(os.path.join(d, f"predictions_{key}.csv"), dtype={"image_id": str, "group_id": str})
            runs.append({"backbone": r["backbone"], "method": "frozen", "frac": float(key[4:]), "img": 224, "seed": 0,
                         "pred": p, "gpu_hours": r["run"]["gpu_hours"] / len(r["results"]),
                         "img_per_s": r["run"]["extract_img_per_s"], "peak_mem_gb": r["run"].get("extract_peak_mem_gb"),
                         "gpu_idle_pct": r["run"]["gpu_extract"]["gpu_idle_pct"], "time_to_crit_s": None})
    for f in glob.glob(os.path.join(matrix_dir, "*", "results.json")):
        r = json.load(open(f))
        if "metrics" not in r:
            continue
        run = r["run"]
        p = pd.read_csv(os.path.join(os.path.dirname(f), "predictions.csv"), dtype={"image_id": str, "group_id": str})
        steady = [e["img_per_s_steady"] for e in run["epochs"][1:] if e["img_per_s_steady"]]
        runs.append({"backbone": r["backbone"], "method": r["method"], "frac": r["train_frac"], "img": r.get("img", 224),
                     "seed": r["seed"], "pred": p, "gpu_hours": run["gpu_hours"],
                     "img_per_s": round(float(np.median(steady)), 1) if steady else None,
                     "peak_mem_gb": run["peak_mem_allocated_gb"], "gpu_idle_pct": run["gpu_train"]["gpu_idle_pct"],
                     "time_to_crit_s": run["time_to_val_criterion_s"]})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-sens", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load_runs(a.frozen, a.matrix)
    print(f"{len(runs)} runs found")
    long = []
    for r in runs:
        p = r["pred"]
        va, te = p[p.split == "val"], p[p.split == "test"].sort_values("image_id").reset_index(drop=True)
        t = threshold_at_sensitivity(va.y.values, va.score.values, a.target_sens)
        r["thr"], r["te"] = t, te
        for inst in ["ALL"] + sorted(te.source.unique()):
            s = te if inst == "ALL" else te[te.source == inst]
            m = metrics_at(s.y.values, s.score.values, t)
            ci = bootstrap(s.y.values, s.score.values, t, s.group_id.values, a.n_boot, a.seed)
            row = {k: r[k] for k in ["backbone", "method", "frac", "img", "seed", "gpu_hours", "img_per_s",
                                     "peak_mem_gb", "gpu_idle_pct", "time_to_crit_s"]}
            row.update({"test_source": inst, "threshold": t, **m, "n_groups": ci["n_groups"]})
            for k in KEYS:
                row[f"{k}_lo"], row[f"{k}_hi"] = ci[k]
            long.append(row)
    L = pd.DataFrame(long).sort_values(["test_source", "img", "backbone", "method", "frac", "seed"])
    L.to_csv(os.path.join(a.out, "runs_long.csv"), index=False)

    pooled = L[L.test_source == "ALL"]
    agg = {k: ["mean", "std"] for k in KEYS}
    agg.update({"seed": "count", "gpu_hours": "mean"})
    C = pooled.groupby(["img", "backbone", "method", "frac"]).agg(agg)
    C.columns = ["_".join(c) if c[1] else c[0] for c in C.columns]
    C = C.rename(columns={"seed_count": "n_seeds"}).reset_index()
    C.to_csv(os.path.join(a.out, "curves.csv"), index=False)

    idx = {(r["backbone"], r["method"], r["frac"], r["img"], r["seed"]): r for r in runs}
    frozen = {(r["backbone"], r["frac"], r["img"]): r for r in runs if r["method"] == "frozen"}
    pairs = []
    for (bb, meth, frac, img, seed), r in idx.items():
        comps = []
        if meth.startswith("lora"):
            if (bb, "full", frac, img, seed) in idx:
                comps.append((f"full_minus_{meth}", r, idx[(bb, "full", frac, img, seed)]))
            if (bb, frac, img) in frozen:
                comps.append((f"{meth}_minus_frozen", frozen[(bb, frac, img)], r))
        if meth == "full" and (bb, frac, img) in frozen:
            comps.append(("full_minus_frozen", frozen[(bb, frac, img)], r))
        if bb == "dinov2":
            if ("dinov2_cpt", meth, frac, img, seed) in idx:
                comps.append((f"cpt_minus_dinov2_{meth}", r, idx[("dinov2_cpt", meth, frac, img, seed)]))
            if ("panderm", meth, frac, img, seed) in idx:
                comps.append((f"panderm_minus_dinov2_{meth}", r, idx[("panderm", meth, frac, img, seed)]))
        for name, r1, r2 in comps:
            t1, t2 = r1["te"], r2["te"]
            assert (t1.image_id.values == t2.image_id.values).all(), "different test images"
            d = paired_bootstrap(t1.y.values, t1.score.values, r1["thr"], t2.score.values, r2["thr"],
                                 t1.group_id.values, a.n_boot, a.seed)
            for k, v in d.items():
                pairs.append({"comparison": name, "img": img, "frac": frac, "seed": seed, "metric": k,
                              "diff": v["diff"], "ci_lo": v["ci95"][0], "ci_hi": v["ci95"][1]})
    P = pd.DataFrame(pairs, columns=["comparison", "img", "frac", "seed", "metric", "diff", "ci_lo", "ci_hi"])
    P.to_csv(os.path.join(a.out, "paired.csv"), index=False)

    f3 = lambda x: "" if x is None or pd.isna(x) else f"{x:.3f}"
    pr = P[(P.comparison == f"cpt_minus_dinov2_{PRIMARY['method']}") & (P.img == PRIMARY["img"])
           & (P.frac == PRIMARY["frac"]) & (P.metric == PRIMARY["metric"])]
    pm = [f"# Primary contrast: {PRIMARY['treated']} minus {PRIMARY['reference']}, {PRIMARY['method']} fine-tuning, "
          f"{PRIMARY['img']} px, {int(PRIMARY['frac'] * 100)}% annotations, paired external {PRIMARY['metric'].upper()}", ""]
    if len(pr):
        pm += ["| Seed | Difference | 95% CI (patient-clustered paired bootstrap) |", "|---|---|---|"]
        pm += [f"| {r.seed} | {f3(r['diff'])} | {f3(r.ci_lo)} to {f3(r.ci_hi)} |" for _, r in pr.iterrows()]
        pm += ["", f"Seeds completed: {len(pr)}. Mean difference across seeds: {f3(pr['diff'].mean())}; "
               f"SD across seeds: {f3(pr['diff'].std()) if len(pr) > 1 else 'n/a (1 seed)'}.",
               "A confidence interval including 0 is not evidence of equivalence."]
    else:
        pm += ["NOT AVAILABLE: one or both arms have no completed run. Do not report a primary result."]
    with open(os.path.join(a.out, "primary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(pm) + "\n")

    md = [f"# Results: diagnostic proxy (benign vs malignant), threshold locked on validation "
          f"(target sensitivity {a.target_sens:.0%})", "",
          "95% CI: patient-clustered bootstrap (lesion where patient is missing), 1,000 resamples, fixed threshold.", "",
          "## Curves (pooled external test; mean and SD across seeds)", "",
          "| px | Backbone | Method | Annotations | Seeds | AUROC | Sensitivity | Specificity | Flagged | GPU-h/run |",
          "|" + "---|" * 10]
    for _, r in C.iterrows():
        md.append(f"| {r.img} | {r.backbone} | {r.method} | {r.frac} | {r.n_seeds} | {f3(r.auroc_mean)} ({f3(r.auroc_std)}) | "
                  f"{f3(r.sensitivity_mean)} ({f3(r.sensitivity_std)}) | {f3(r.specificity_mean)} ({f3(r.specificity_std)}) | "
                  f"{f3(r.flag_rate_mean)} | {f3(r.gpu_hours_mean)} |")
    md += ["", "## Per run and per external institution", "",
           "| Test | px | Backbone | Method | Annotations | Seed | AUROC [CI] | Sens [CI] | Spec [CI] | FN | Flagged | img/s | Mem GB |",
           "|" + "---|" * 13]
    for _, r in L.iterrows():
        md.append(f"| {r.test_source} | {r.img} | {r.backbone} | {r.method} | {r.frac} | {r.seed} | "
                  f"{f3(r.auroc)} [{f3(r.auroc_lo)}-{f3(r.auroc_hi)}] | {f3(r.sensitivity)} [{f3(r.sensitivity_lo)}-{f3(r.sensitivity_hi)}] | "
                  f"{f3(r.specificity)} [{f3(r.specificity_lo)}-{f3(r.specificity_hi)}] | {r.false_negatives} | {f3(r.flag_rate)} | "
                  f"{r.img_per_s} | {r.peak_mem_gb} |")
    tot_h = pooled.gpu_hours.sum()
    md += ["", f"GPU-hours of the runs in this table: {tot_h:.2f} (hardware in each run's results.json). "
           f"List-price equivalent on the rented pod: ~{tot_h * PRICE_PER_GPU_HOUR:.0f} USD (not part of the proposal)."]
    with open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    write_json(os.path.join(a.out, "aggregate_run.json"), run_record(seed=a.seed, extra={"n_runs": len(runs)}))
    print("\n".join(pm))
    print("\n".join(md[:8 + len(C)]))


if __name__ == "__main__":
    main()
