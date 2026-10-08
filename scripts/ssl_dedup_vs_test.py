#!/usr/bin/env python3
"""Remove from the pretraining pool every exact or near duplicate of a labelled VALIDATION or TEST image.

    python scripts/ssl_dedup_vs_test.py --ssl-ids $WS/data/ssl/ssl_smoke.csv --ssl-images $WS/data/ssl_images_256 \
        --manifest $WS/data/partition/manifest.csv --held-images $WS/data/images_256 --out $WS/data/ssl/exclude_smoke.csv
Exact = same SHA-256 of the original file; near = 256-bit dHash distance <= --max-hamming (same rule as dedup_audit.py).
Writes the image_ids to exclude (pass to `ssl_build_set.py list --exclude`) and a summary JSON.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_record, write_json  # noqa: E402
from dedup_audit import MAX_HAMMING, dhash, hamming  # noqa: E402


def hashes(paths, workers=None):
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        return np.stack(list(ex.map(dhash, paths, chunksize=64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl-ids", required=True)
    ap.add_argument("--ssl-images", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--held-images", required=True, help="image dir (with downloads.csv) of the labelled images")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hamming", type=int, default=MAX_HAMMING)
    ap.add_argument("--workers", type=int, default=None, help="dHash processes (default: all CPUs)")
    a = ap.parse_args()
    ids = pd.read_csv(a.ssl_ids, dtype=str)
    dl = pd.read_csv(os.path.join(a.ssl_images, "downloads.csv"), dtype=str).drop_duplicates("image_id", keep="last")
    s = ids.merge(dl[["image_id", "file", "sha256"]], on="image_id").reset_index(drop=True)
    man = pd.read_csv(a.manifest, dtype=str, keep_default_na=False)
    h = man[man.split.isin(a.splits)].reset_index(drop=True)
    hdl = pd.read_csv(os.path.join(a.held_images, "downloads.csv"), dtype=str).drop_duplicates("image_id", keep="last")
    h = h.merge(hdl[["image_id", "sha256"]], on="image_id", how="left")

    exact = set(s.image_id[s.sha256.isin(set(h.sha256.dropna()))])
    hs = hashes([os.path.join(a.ssl_images, f) for f in s.file], a.workers)
    hh = hashes([os.path.join(a.held_images, f) for f in h.file], a.workers)
    near = set()
    for i in range(0, len(hs), 128):
        d = hamming(hs[i:i + 128], hh)
        rows = np.nonzero((d <= a.max_hamming).any(1))[0]
        near.update(s.image_id[i + rows])
    ex = sorted(exact | near)
    pd.DataFrame({"image_id": ex}).to_csv(a.out, index=False)
    summary = {"pool_images_checked": len(s), "held_images": len(h), "held_splits": a.splits,
               "exact_copies": len(exact), "near_duplicates_only": len(near - exact), "excluded_total": len(ex),
               "max_hamming": a.max_hamming}
    write_json(os.path.splitext(a.out)[0] + "_summary.json", {"run": run_record(), **summary})
    print(summary)


if __name__ == "__main__":
    main()
