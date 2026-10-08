#!/usr/bin/env python3
"""Download the cohort images by ISIC id (ISIC API v2), recording size and SHA-256 of every file.

    python scripts/download_images.py --cohort /workspace/data/cohort/cohort.csv \
        --out /workspace/data/images [--allow-nc]

Without --allow-nc, images whose license contains "NC" are skipped (CLAUDE.md: user OK required first).
Resumable: files already listed in downloads.csv with a matching size are skipped.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_record, sha256_file, write_json  # noqa: E402

API = "https://api.isic-archive.com/api/v2/images/{}/"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download_one(isic_id, out_dir, retries=4):
    err = None
    for k in range(retries):
        try:
            meta = fetch_json(API.format(isic_id))
            url = meta["files"]["full"]["url"]
            ext = os.path.splitext(url.split("?")[0])[1].lower() or ".jpg"
            if ext not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"unexpected extension {ext}")
            path = os.path.join(out_dir, isic_id + ext)
            tmp = path + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(tmp, path)
            return isic_id, os.path.basename(path), os.path.getsize(path), sha256_file(path), meta.get("copyright_license")
        except Exception as e:  # network errors, 5xx, rate limiting
            err = e
            time.sleep(2 ** k)
    raise RuntimeError(f"{isic_id}: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-nc", action="store_true", help="only after the user accepted CC-BY-NC (log it in PROGRESS.md)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    c = pd.read_csv(a.cohort, dtype=str, keep_default_na=False)
    nc = c.license.str.upper().str.contains("NC")
    if not a.allow_nc:
        print(f"Skipping {int(nc.sum())} NonCommercial-licensed images (use --allow-nc after the user's OK)")
        c = c[~nc]
    lic = dict(zip(c.image_id, c.license))

    log_path = os.path.join(a.out, "downloads.csv")
    done = {}
    if os.path.exists(log_path):
        for r in csv.DictReader(open(log_path, encoding="utf-8")):
            p = os.path.join(a.out, r["file"])
            if os.path.exists(p) and os.path.getsize(p) == int(r["bytes"]):
                done[r["image_id"]] = r
    todo = [i for i in c.image_id if i not in done]
    print(f"{len(c)} images selected, {len(done)} already present, {len(todo)} to download")

    failed, license_mismatch = [], 0
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.writer(f)
        if new_file:
            w.writerow(["image_id", "file", "bytes", "sha256", "api_license"])
        futs = {ex.submit(download_one, i, a.out): i for i in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                row = fu.result()
                w.writerow(row)
                if row[4] is not None and row[4] != lic[row[0]]:
                    license_mismatch += 1
            except Exception as e:
                failed.append(str(e))
            if n % 200 == 0:
                f.flush()
                print(f"  {n}/{len(todo)} (failed {len(failed)})")

    print(f"Done. Failed: {len(failed)}. License differs between metadata and API: {license_mismatch}")
    for e in failed[:20]:
        print("  FAILED", e)
    write_json(os.path.join(a.out, "download_run.json"), run_record(extra={
        "cohort": a.cohort, "allow_nc": a.allow_nc, "n_selected": len(c), "n_failed": len(failed),
        "failed": failed, "license_mismatch": license_mismatch}))


if __name__ == "__main__":
    main()
