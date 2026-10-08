#!/usr/bin/env python3
"""Unlabelled pretraining set for continued DINOv2 pretraining: ISIC dermoscopic images with a permissive licence.

    # 1. select ids (needs metadata of ALL dermoscopic images: scripts/01_metadata.sh downloads metadata_dermoscopic.csv)
    python scripts/ssl_build_set.py select --meta $WS/data/metadata/metadata_dermoscopic.csv \
        --source-map configs/source_map.csv --manifest $WS/data/partition/manifest.csv --out $WS/data/ssl \
        [--licenses CC-0 CC-BY] [--smoke 12000]
    # 2. download (same script as the labelled cohort), resize, then write the file list from downloads.csv
    python scripts/download_images.py --cohort $WS/data/ssl/ssl_smoke.csv --out $WS/data/ssl_images
    python scripts/resize_cache.py --src $WS/data/ssl_images --dst $WS/data/ssl_images_256 --side 256
    python scripts/ssl_build_set.py list --ids $WS/data/ssl/ssl_smoke.csv --images $WS/data/ssl_images_256 \
        --out $WS/data/ssl/ssl_smoke.txt

Exclusions (leakage control, as in the proposal): every image of the external-test institutions; every image in
the labelled validation and test splits; any image sharing a lesion or patient id with a validation or test image;
after download, exact and perceptual duplicates of validation or test images (ssl_dedup_vs_test.py).
Labelled TRAINING images may stay (their labels are never used). Licences: CC-0 and CC-BY only.
"""
import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cohort import load_source_map, source_of  # noqa: E402
from common import run_record, sha256_file, write_json  # noqa: E402


def select(a):
    m = pd.read_csv(a.meta, dtype=str, keep_default_na=False)
    for c in ["isic_id", "copyright_license", "attribution", "image_type"]:
        if c not in m.columns:
            sys.exit(f"column {c} missing; columns: {list(m.columns)}")
    m = m[m.image_type.str.lower().str.startswith("dermoscopic")].copy()
    n0 = len(m)
    smap, _ = load_source_map(a.source_map)
    m["source"] = [source_of(x, smap) for x in m.attribution]  # same definition as build_cohort.py
    man = pd.read_csv(a.manifest, dtype=str, keep_default_na=False)
    test_sources = set(man.source[man.split == "test"])
    held = man[man.split.isin(["val", "test"])]
    log = Counter({"dermoscopic": n0})

    drop = m.source.isin(test_sources)
    log["excluded_test_institution"] = int(drop.sum()); m = m[~drop]
    drop = m.isic_id.isin(set(held.image_id))
    log["excluded_labelled_val_test_image"] = int(drop.sum()); m = m[~drop]
    for col in ["lesion_id", "patient_id"]:
        if col in m.columns:
            ids = set(held[col]) - {""}
            drop = m[col].isin(ids) & (m[col] != "")
            log[f"excluded_shares_{col}_with_val_or_test"] = int(drop.sum()); m = m[~drop]
    drop = ~m.copyright_license.isin(a.licenses)
    log["excluded_licence"] = int(drop.sum())
    log_lic = dict(Counter(m.copyright_license))
    m = m[~drop]
    log["selected"] = len(m)
    out = m.rename(columns={"isic_id": "image_id", "copyright_license": "license"})[
        ["image_id", "license", "attribution", "source"]].sort_values("image_id")
    os.makedirs(a.out, exist_ok=True)
    full = os.path.join(a.out, "ssl_full.csv")
    out.to_csv(full, index=False)
    files = {"ssl_full.csv": sha256_file(full)}
    if a.smoke:
        sm = out.sample(n=min(a.smoke, len(out)), random_state=0).sort_values("image_id")
        p = os.path.join(a.out, "ssl_smoke.csv")
        sm.to_csv(p, index=False)
        files["ssl_smoke.csv"] = sha256_file(p)
    pat = int(m.loc[m.patient_id != "", "patient_id"].nunique()) if "patient_id" in m.columns else None
    summary = {"log": dict(log), "licences_before_filter": log_lic, "licences_kept": a.licenses,
               "pool": {"images": len(out), "institutions": out.source.nunique(), "patients_with_id": pat},
               "by_source": dict(Counter(out.source)), "test_sources": sorted(test_sources), "files": files,
               "note": "duplicates of validation/test images are removed later by ssl_dedup_vs_test.py"}
    write_json(os.path.join(a.out, "ssl_set_summary.json"), summary)
    write_json(os.path.join(a.out, "ssl_set_run.json"), run_record(extra={"meta_sha256": sha256_file(a.meta)}))
    for k, v in summary.items():
        print(f"{k}: {v}")


def make_list(a):
    ids = pd.read_csv(a.ids, dtype=str, keep_default_na=False)
    dl = pd.read_csv(os.path.join(a.images, "downloads.csv"), dtype=str).drop_duplicates("image_id", keep="last")
    d = ids.merge(dl[["image_id", "file"]], on="image_id")
    d = d[[os.path.exists(os.path.join(a.images, f)) for f in d.file]]
    if a.exclude:
        ex = set(pd.read_csv(a.exclude, dtype=str).image_id)
        d = d[~d.image_id.isin(ex)]
    with open(a.out, "w") as f:
        f.write("\n".join(d.file) + "\n")
    att = os.path.splitext(a.out)[0] + "_attribution.csv"
    d[["image_id", "license", "attribution"]].to_csv(att, index=False)
    print(f"{len(d)} files listed in {a.out} (of {len(ids)} selected); attribution list: {att}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--meta", required=True)
    s.add_argument("--source-map")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--licenses", nargs="+", default=["CC-0", "CC-BY"])
    s.add_argument("--smoke", type=int, default=0)
    l = sub.add_parser("list")
    l.add_argument("--ids", required=True)
    l.add_argument("--images", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--exclude", help="CSV with image_id to drop (e.g. duplicates of test images)")
    a = ap.parse_args()
    select(a) if a.cmd == "select" else make_list(a)


if __name__ == "__main__":
    main()
