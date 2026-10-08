#!/usr/bin/env python3
"""One fixed partition: held-out test institutions + development (train/val), with nested training fractions.

    python scripts/make_partitions.py --cohort /workspace/data/cohort/cohort.csv \
        --dedup /workspace/data/dedup/dedup_decisions.csv --test-sources SRC_A SRC_B [SRC_C] \
        --out /workspace/data/partition --seed 0

Steps (policy fixed in CLAUDE.md):
 1. drop duplicate clusters with conflicting labels;
 2. union-find over patient_id, lesion_id (ISIC ids are global) and duplicate clusters -> group_id;
 3. any group touching a test institution: its non-test images are dropped (test stays intact);
 4. exact copies (same SHA-256) inside one split: keep the lowest image_id;
 5. development groups -> val (val_frac, stratified on (source, label)) and train;
 6. train groups -> 10 stratified folds; train_frac 0.1 = fold 0, 0.3 = folds 0-2, 1.0 = all (nested).
Outputs: manifest.csv, panderm.csv, partition_summary.json (counts, group basis, SHA-256 of files).
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_record, sha256_file, write_json  # noqa: E402
from build_cohort import load_source_map  # noqa: E402
from dedup_audit import UF  # noqa: E402


def describe(g):
    """Counts reported separately, as in the proposal: images, patients, lesions, linked groups, institutions."""
    return {"images": len(g), "patients": g.loc[g.patient_id != "", "patient_id"].nunique(),
            "lesions": g.loc[g.lesion_id != "", "lesion_id"].nunique(), "linked_groups": g.group_id.nunique(),
            "institutions": g.source.nunique(), "malignant": int((g.label == "malignant").sum()),
            "benign": int((g.label == "benign").sum()),
            "patient_id_coverage_pct": round(100 * (g.patient_id != "").mean(), 1) if len(g) else None}

MANIFEST_COLS = ["image_id", "file", "source", "patient_id", "lesion_id", "group_id", "label", "label_source", "license",
                 "attribution", "diagnosis_detail", "age_band", "sex", "body_site", "split", "train_frac"]
FRACS = (0.1, 0.3, 1.0)


def one_fold(df, n_splits, seed, fold=0):
    strat = df.source + "_" + df.label
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [idx for _, idx in sg.split(df, strat, groups=df.group_id)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--dedup", required=True)
    ap.add_argument("--test-sources", nargs="+", required=True)
    ap.add_argument("--source-map", required=True, help="configs/source_map.csv: test sources must be verified there")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-require-patient-id", nargs="*", default=[],
                    help="test sources whose images WITHOUT patient_id are excluded from every split (user decision "
                         "8 Oct 2026: MSKCC; the >=90%% patient-coverage rule is then met by the retained subset)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = Counter()

    c = pd.read_csv(a.cohort, dtype=str, keep_default_na=False)
    dd = pd.read_csv(a.dedup, dtype=str, keep_default_na=False)
    unknown = set(a.test_sources) - set(c.source)
    if unknown:
        sys.exit(f"unknown test sources: {unknown}")
    if "anonymous" in a.test_sources:
        sys.exit("anonymous cannot be a test source")
    _, verified = load_source_map(a.source_map)
    unverified = [s for s in a.test_sources if s not in verified]
    if unverified:
        sys.exit(f"test sources not verified in {a.source_map}: {unverified} (unresolved sources are development only)")
    bad = set(a.test_require_patient_id) - set(a.test_sources)
    if bad:
        sys.exit(f"--test-require-patient-id must list test sources only: {bad}")
    m = c.source.isin(a.test_require_patient_id) & (c.patient_id == "")
    log["excluded_test_source_without_patient_id"] = int(m.sum())
    c = c[~m].reset_index(drop=True)
    d = c.merge(dd[["image_id", "file", "sha256", "dup_cluster", "action"]], on="image_id", how="inner")
    log["not_downloaded"] = len(c) - len(d)
    m = d.action == "exclude_label_conflict"
    log["dropped_label_conflict_cluster"] = int(m.sum())
    d = d[~m].reset_index(drop=True)

    uf = UF(len(d))
    for col in ["patient_id", "lesion_id", "dup_cluster"]:
        first = {}
        for i, k in enumerate(d[col]):
            if k == "":
                continue
            if k in first:
                uf.union(i, first[k])
            else:
                first[k] = i
    d["group_id"] = [f"g{uf.find(i)}" for i in range(len(d))]
    d["is_test"] = d.source.isin(a.test_sources)
    touch = set(d.group_id[d.is_test])
    m = (~d.is_test) & d.group_id.isin(touch)
    log["dropped_dev_images_linked_to_test"] = int(m.sum())
    d = d[~m].copy()

    d["split"] = np.where(d.is_test, "test", "")
    dev = d[~d.is_test]
    folds = one_fold(dev, round(1 / a.val_frac), a.seed)
    val_idx = dev.index[folds[0]]
    d.loc[dev.index, "split"] = "train"
    d.loc[val_idx, "split"] = "val"

    # exact copies inside one split
    before = len(d)
    d = d.sort_values("image_id").drop_duplicates(["split", "sha256"], keep="first")
    log["dropped_exact_copies_same_split"] = before - len(d)
    # exact copies across dev splits (train/val) would be leakage: they share dup_cluster -> same group -> same split
    assert d.groupby("group_id").split.nunique().max() == 1, "group in more than one split"
    assert not (d[d.split != "test"].sha256.isin(d[d.split == "test"].sha256)).any(), "test image copy in dev"

    tr = d[d.split == "train"]
    tf = one_fold(tr, 10, a.seed)
    frac = pd.Series(1.0, index=tr.index)
    frac[tr.index[np.concatenate(tf[1:3])]] = 0.3
    frac[tr.index[tf[0]]] = 0.1
    d["train_frac"] = ""
    d.loc[tr.index, "train_frac"] = frac.astype(str)

    basis = {s: dict(Counter(np.where(g.patient_id != "", "patient", np.where(g.lesion_id != "", "lesion", "image"))))
             for s, g in d.groupby("source")}
    man = os.path.join(a.out, "manifest.csv")
    d[MANIFEST_COLS].sort_values(["split", "source", "image_id"]).to_csv(man, index=False)
    pdc = os.path.join(a.out, "panderm.csv")
    pd.DataFrame({"image": d.file, "label": (d.label == "malignant").astype(int),
                  "binary_label": (d.label == "malignant").astype(int), "split": d.split}).to_csv(pdc, index=False)

    tfv = pd.to_numeric(d.train_frac, errors="coerce")
    subsets = {f"Labelled training {int(f * 100)}%": describe(d[(d.split == "train") & (tfv <= f + 1e-9)]) for f in FRACS}
    subsets["Validation"] = describe(d[d.split == "val"])
    subsets["External test"] = describe(d[d.split == "test"])
    by_split_source = {"/".join(k): describe(g) for k, g in d.groupby(["split", "source"])}
    md = ["| Subset | Images | Patients | Lesions | Linked groups | Institutions | Malignant | Benign |", "|" + "---|" * 8]
    for k, v in subsets.items():
        md.append(f"| {k} | {v['images']} | {v['patients']} | {v['lesions']} | {v['linked_groups']} | "
                  f"{v['institutions']} | {v['malignant']} | {v['benign']} |")
    with open(os.path.join(a.out, "subsets_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n\n(Continued-pretraining pool: see data/ssl/ssl_set_summary.json)\n")
    counts = {**subsets, **{f"by_split_source/{k}": v for k, v in by_split_source.items()}}
    summary = {"seed": a.seed, "val_frac": a.val_frac, "test_sources": a.test_sources,
               "test_require_patient_id": a.test_require_patient_id, "log": dict(log),
               "group_basis_images": basis, "counts": counts,
               "patient_id_coverage_by_institution": {s: round(100 * (g.patient_id != "").mean(), 1)
                                                      for s, g in d.groupby("source")},
               "license_by_split": {s: dict(Counter(g.license)) for s, g in d.groupby("split")},
               "file_sha256": {"manifest.csv": sha256_file(man), "panderm.csv": sha256_file(pdc)}}
    write_json(os.path.join(a.out, "partition_summary.json"), summary)
    write_json(os.path.join(a.out, "partition_run.json"), run_record(seed=a.seed, extra={
        "cohort_sha256": sha256_file(a.cohort), "dedup_sha256": sha256_file(a.dedup)}))
    print(f"log: {dict(log)}")
    print("\n".join(md))
    print("\nNext: python controllo_dataset.py manifest.csv (label/label_source warnings are expected)")


if __name__ == "__main__":
    main()
