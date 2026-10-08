#!/usr/bin/env python3
"""Synthetic ISIC-like data to test the whole pipeline end to end in minutes. NOT real data: never report results.

Creates <out>/metadata_all.csv and <out>/metadata_dermoscopic.csv in the ISIC column format, <out>/images/*.jpg
(random textures, 640x480) with downloads.csv, including one exact and one near duplicate.
4 institutions (SynthA..SynthD), 2 images per patient; plus unlabelled dermoscopic images for the SSL set.
"""
import argparse
import csv
import hashlib
import os

import numpy as np
from PIL import Image, ImageFilter

COLS = ["isic_id", "copyright_license", "attribution", "image_type", "diagnosis_1", "diagnosis_2", "diagnosis_3",
        "diagnosis_confirm_type", "lesion_id", "patient_id", "age_approx", "sex", "anatom_site_general"]


def img(rng, label):
    base = rng.integers(60, 200, size=(48, 64, 3), dtype=np.uint8)
    if label == "Malignant":
        base[16:32, 20:44] = rng.integers(0, 60, size=(16, 24, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((640, 480), Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-inst", type=int, default=120)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    imdir = os.path.join(a.out, "images")
    os.makedirs(imdir, exist_ok=True)
    rows, dl, k = [], [], 0
    lic = ["CC-0", "CC-BY", "CC-BY-NC"]
    for inst in ["SynthA", "SynthB", "SynthC", "SynthD"]:
        for p in range(a.per_inst // 2):
            label = "Malignant" if p % 2 else "Benign"
            for j in range(2):
                k += 1
                iid = f"ISIC_{9000000 + k:07d}"
                im = img(rng, label)
                path = os.path.join(imdir, iid + ".jpg")
                im.save(path, quality=90)
                rows.append([iid, lic[k % 3], f"{inst} Hospital", "dermoscopic", label, "", "", "histopathology",
                             f"IL_{inst}_{p}", f"IP_{inst}_{p}", str(30 + p % 50), "female" if k % 2 else "male", "trunk"])
        # unlabelled / non-histology dermoscopic images (pretraining pool only)
        for u in range(a.per_inst // 2):
            k += 1
            iid = f"ISIC_{9000000 + k:07d}"
            img(rng, "Benign").save(os.path.join(imdir, iid + ".jpg"), quality=90)
            rows.append([iid, lic[k % 2], f"{inst} Hospital", "dermoscopic", "Benign", "", "",
                         "single image expert consensus", "", "", "", "", ""])
    # one exact duplicate (same bytes) and one near duplicate (re-encoded) inside SynthA
    src = os.path.join(imdir, rows[0][0] + ".jpg")
    for kind in ("exact", "near"):
        k += 1
        iid = f"ISIC_{9000000 + k:07d}"
        dst = os.path.join(imdir, iid + ".jpg")
        if kind == "exact":
            open(dst, "wb").write(open(src, "rb").read())
        else:
            Image.open(src).save(dst, quality=70)
        r = list(rows[0]); r[0] = iid; r[8] = f"IL_dup_{kind}"; r[9] = f"IP_dup_{kind}"
        rows.append(r)
    with open(os.path.join(a.out, "metadata_dermoscopic.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
    with open(os.path.join(a.out, "metadata_all.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows([r for r in rows if r[7] == "histopathology"])
    with open(os.path.join(imdir, "downloads.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "file", "bytes", "sha256", "api_license"])
        for r in rows:
            p = os.path.join(imdir, r[0] + ".jpg")
            w.writerow([r[0], r[0] + ".jpg", os.path.getsize(p), hashlib.sha256(open(p, "rb").read()).hexdigest(), r[1]])
    with open(os.path.join(a.out, "source_map.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["attribution", "source", "country"])
        for inst, c in [("SynthA", "Italy"), ("SynthB", "Spain"), ("SynthC", "USA"), ("SynthD", "Argentina")]:
            w.writerow([f"{inst} Hospital", inst, c])
    print(f"synthetic data: {len(rows)} images in {imdir}")


if __name__ == "__main__":
    main()
