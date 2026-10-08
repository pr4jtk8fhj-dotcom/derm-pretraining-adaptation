#!/usr/bin/env python3
"""Eligible cohort from ISIC metadata (no images needed). Source = institution, from `attribution`.

Filters: dermoscopic image_type; diagnosis_confirm_type = histopathology; diagnosis_1 in {Benign, Malignant}.
Indeterminate / unknown diagnoses are EXCLUDED (never mapped to benign).

    python scripts/build_cohort.py --meta /workspace/data/metadata/metadata_all.csv --out /workspace/data/cohort \
        [--source-map configs/source_map.csv]

First run without --source-map: it writes attributions.csv (every distinct attribution with counts). Fill
configs/source_map.csv (columns: attribution, source, country) to merge spelling variants and record the country,
then rerun. Unmapped attributions become their own source; "Anonymous"/empty become source "anonymous" (dev only).
Outputs: cohort.csv, institutions.md/.json (with test-candidate flags per the CLAUDE.md rule), exclusions.json.
"""
import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_record, sha256_file, write_json  # noqa: E402

FIELDS = {
    "isic_id": ["isic_id"],
    "license": ["copyright_license", "license"],
    "attribution": ["attribution"],
    "image_type": ["image_type"],
    "diagnosis_1": ["diagnosis_1"],
    "diagnosis_2": ["diagnosis_2"],
    "diagnosis_3": ["diagnosis_3"],
    "confirm": ["diagnosis_confirm_type"],
    "lesion_id": ["lesion_id"],
    "patient_id": ["patient_id"],
    "age": ["age_approx"],
    "sex": ["sex"],
    "body_site": ["anatom_site_general", "anatom_site_special", "anatom_site"],
}
REQUIRED = ["isic_id", "license", "attribution", "image_type", "diagnosis_1", "confirm"]
LABELS = {"benign": "benign", "malignant": "malignant"}
# test-candidate rule (CLAUDE.md): fixed before any result
MIN_PATIENT_COV, MIN_PER_CLASS, EXCLUDED_COUNTRIES = 90.0, 150, {"australia"}


def resolve(cols, name):
    for c in FIELDS[name]:
        if c in cols:
            return c
    return None


def age_band(v):
    try:
        a = int(float(v))
    except (TypeError, ValueError):
        return ""
    return f"{(a // 10) * 10}-{(a // 10) * 10 + 9}"


def norm_attr(s):
    return re.sub(r"\s+", " ", s).strip().casefold()


def load_source_map(path):
    """configs/source_map.csv -> ({normalized attribution: source}, {source: country})."""
    smap, country = {}, {}
    if path:
        sm = pd.read_csv(path, dtype=str, keep_default_na=False)
        for r in sm.itertuples():
            smap[norm_attr(r.attribution)] = r.source
            country[r.source] = r.country
    return smap, country


def source_of(attribution, smap):
    """Single definition of 'source' shared by build_cohort.py and ssl_build_set.py."""
    k = norm_attr(attribution)
    if k in ("", "anonymous"):
        return "anonymous"
    return smap.get(k, re.sub(r"[^A-Za-z0-9]+", "_", attribution).strip("_")[:60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="append", required=True, help="metadata CSV (repeatable; rows merged by isic_id)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-map", help="CSV: attribution, source, country")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames, colmap_all = [], {}
    for path in a.meta:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        cols = list(df.columns)
        print(f"== {path}: {len(df)} rows, {len(cols)} columns\n   columns: {cols}")
        cmap = {k: resolve(cols, k) for k in FIELDS}
        colmap_all[path] = cmap
        missing = [k for k in REQUIRED if cmap[k] is None]
        if missing:
            sys.exit(f"ERROR: required columns not found: {missing}. Fix FIELDS after reading the header.")
        frames.append(pd.DataFrame({k: (df[c].fillna("").astype(str).str.strip() if c else "") for k, c in cmap.items()}))
    d = pd.concat(frames, ignore_index=True)
    n_rows = len(d)
    d = d.drop_duplicates()
    conflicting = d.isic_id[d.isic_id.duplicated(keep=False)].unique()
    ex = Counter({"identical_duplicate_rows": n_rows - len(d)})
    if len(conflicting):
        print(f"!! {len(conflicting)} isic_id with different metadata in different files: excluded")
        d = d[~d.isic_id.isin(conflicting)]
    ex["isic_id_conflicting_metadata"] = len(conflicting)
    for k in ["image_type", "confirm", "diagnosis_1", "license"]:
        print(f"   {k} values: {dict(Counter(d[k]))}")

    m = d.image_type.str.lower().str.startswith("dermoscopic"); ex["not_dermoscopic"] = int((~m).sum()); d = d[m]
    m = d.confirm.str.lower() == "histopathology"; ex["not_histopathology"] = int((~m).sum()); d = d[m]
    m = d.diagnosis_1.str.lower().isin(LABELS)
    ex["diagnosis_not_benign_malignant"] = int((~m).sum())
    ex_dx = dict(Counter(d.loc[~m, "diagnosis_1"]))
    d = d[m].copy()
    m = d.isic_id != ""; ex["empty_isic_id"] = int((~m).sum()); d = d[m].copy()
    print(f"exclusions: {dict(ex)}; excluded diagnosis_1 values: {ex_dx}")

    # ---- institutions
    d["attr_key"] = d.attribution.map(norm_attr)
    attr = (d.groupby("attr_key").agg(attribution=("attribution", lambda s: s.mode().iloc[0]), images=("isic_id", "size"))
            .reset_index().sort_values("images", ascending=False))
    attr.to_csv(os.path.join(a.out, "attributions.csv"), index=False)
    smap, country = load_source_map(a.source_map)
    if a.source_map:
        unmapped = attr[~attr.attr_key.isin(smap)]
        print(f"source map: {len(smap)} entries; {len(unmapped)} attributions unmapped ({int(unmapped.images.sum())} images)")
    d["source"] = [source_of(x, smap) for x in d.attribution]

    d["label"] = d.diagnosis_1.str.lower().map(LABELS)
    d["label_source"] = "istologia"
    d["diagnosis_detail"] = (d.diagnosis_2 + " > " + d.diagnosis_3).str.strip(" >")
    d["age_band"] = d.age.map(age_band)
    d["image_id"] = d.isic_id
    out = d[["image_id", "source", "patient_id", "lesion_id", "label", "label_source", "license", "attribution",
             "diagnosis_detail", "age_band", "sex", "body_site"]].sort_values(["source", "image_id"])
    path = os.path.join(a.out, "cohort.csv")
    out.to_csv(path, index=False)

    rows = []
    for src, g in out.groupby("source"):
        les = g.loc[g.lesion_id != "", ["lesion_id", "label"]].drop_duplicates()
        r = {"source": src, "country": country.get(src, "?"), "images": len(g),
             "benign": int((g.label == "benign").sum()), "malignant": int((g.label == "malignant").sum()),
             "pct_malignant": round(100 * (g.label == "malignant").mean(), 1),
             "patients": g.loc[g.patient_id != "", "patient_id"].nunique(),
             "lesions": g.loc[g.lesion_id != "", "lesion_id"].nunique(),
             "patient_cov_pct": round(100 * (g.patient_id != "").mean(), 1),
             "lesion_cov_pct": round(100 * (g.lesion_id != "").mean(), 1),
             "lesion_label_conflicts": int((les.groupby("lesion_id").label.nunique() > 1).sum()),
             "licenses": dict(Counter(g.license))}
        r["verified"] = src in country  # listed in configs/source_map.csv after checking the attribution
        why = []
        if src == "anonymous":
            why.append("anonymous")
        elif not r["verified"]:
            why.append("institution not verified in source_map")
        if r["patient_cov_pct"] < MIN_PATIENT_COV:
            why.append(f"patient_cov<{MIN_PATIENT_COV}")
        if min(r["benign"], r["malignant"]) < MIN_PER_CLASS:
            why.append(f"<{MIN_PER_CLASS} per class (before dedup)")
        if r["country"].casefold() in EXCLUDED_COUNTRIES:
            why.append("Australia")
        if r["country"] == "?":
            why.append("country unknown")
        r["test_candidate"] = "yes" if not why else "no: " + ", ".join(why)
        rows.append(r)
    rows.sort(key=lambda r: -r["images"])
    tot = {"images": len(out), "benign": int((out.label == "benign").sum()), "malignant": int((out.label == "malignant").sum()),
           "sources": len(rows), "licenses": dict(Counter(out.license))}
    write_json(os.path.join(a.out, "institutions.json"), {"per_source": rows, "total": tot})
    write_json(os.path.join(a.out, "exclusions.json"), {**ex, "excluded_diagnosis_values": ex_dx, "eligible": len(out)})

    hdr = ["Istituzione", "Verificata", "Paese", "Immagini", "Benigne", "Maligne", "% maligne", "Pazienti", "Lesioni",
           "Copertura patient_id", "Copertura lesion_id", "Lesioni in conflitto", "Licenze", "Candidata test"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        md.append(f"| {r['source']} | {'si' if r['verified'] else 'no'} | {r['country']} | {r['images']} | {r['benign']} | {r['malignant']} | {r['pct_malignant']} | "
                  f"{r['patients']} | {r['lesions']} | {r['patient_cov_pct']}% | {r['lesion_cov_pct']}% | "
                  f"{r['lesion_label_conflicts']} | {r['licenses']} | {r['test_candidate']} |")
    md.append(f"| TOTALE | | | {tot['images']} | {tot['benign']} | {tot['malignant']} | | | | | | | {tot['licenses']} | |")
    with open(os.path.join(a.out, "institutions.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nNext: confirm the test institutions with the user (CLAUDE.md rule a) before freezing partitions.")
    write_json(os.path.join(a.out, "build_cohort_run.json"), run_record(extra={
        "inputs": {p: sha256_file(p) for p in a.meta}, "source_map": a.source_map,
        "source_map_sha256": sha256_file(a.source_map) if a.source_map else None,
        "column_mapping": colmap_all, "cohort_sha256": sha256_file(path)}))


if __name__ == "__main__":
    main()
