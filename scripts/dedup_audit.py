#!/usr/bin/env python3
"""Duplicate audit: exact (SHA-256 of file) + perceptual (256-bit dHash, 16x16), within AND across sources.

Threshold: Hamming distance <= 10 of 256 bits (user decision 8 Oct 2026, rule c). The pre-specified 64-bit dHash
<= 6 was replaced after the audit on real data: 847,995 pairs, 99.7% with different lesion_id, visually different
lesions (dermoscopic images share vignetting and a central lesion), 36% of the cohort excluded by chaining.
With 256 bits, visual check of random pairs: distance 6-10 = 24/24 same image; 11-15 mixed; 16-20 different
lesions. The threshold is conservative: some cropped copies (11-15) may be missed (declare it).

    python scripts/dedup_audit.py --cohort /workspace/data/cohort/cohort.csv \
        --images /workspace/data/images --out /workspace/data/dedup [--max-hamming 6]

This script only FINDS duplicates and marks clusters; it does not know the test set. Policy applied in
make_partitions.py (CLAUDE.md):
  * cluster with conflicting labels             -> excluded (action = exclude_label_conflict here);
  * cluster spanning test and development        -> development copies dropped, test kept intact;
  * exact copies (same SHA-256) in the same split -> one kept (lowest image_id);
  * otherwise near duplicates                    -> kept, forced into the same group (same split).
Outputs: dup_pairs.csv, dedup_decisions.csv (image_id, dup_cluster, action), dedup_summary.json,
pairs_sheet.jpg (side-by-side thumbnails of up to 48 near pairs, for visual checking of the threshold).
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run_record, write_json  # noqa: E402


HASH_SIZE, MAX_HAMMING = 16, 10
_POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path, size=HASH_SIZE):
    """Difference hash, size x size bits, packed into size*size/8 uint8."""
    with Image.open(path) as im:
        im.draft("L", (4 * size, 4 * size))  # JPEG DCT-domain downscale: much faster on large images
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(im, dtype=np.int16)
    return np.packbits((px[:, 1:] > px[:, :-1]).flatten())


def hamming(a, b):
    """a: (k, B) uint8, b: (n, B) uint8 packed hashes -> (k, n) Hamming distances."""
    return _POPCNT[a[:, None, :] ^ b[None, :, :]].sum(-1, dtype=np.int32)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--images", required=True, help="dir with downloads.csv and the image files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hamming", type=int, default=MAX_HAMMING)
    ap.add_argument("--hash-size", type=int, default=HASH_SIZE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    c = pd.read_csv(a.cohort, dtype=str, keep_default_na=False)
    dl = pd.read_csv(os.path.join(a.images, "downloads.csv"), dtype=str).drop_duplicates("image_id", keep="last")
    d = c.merge(dl[["image_id", "file", "sha256"]], on="image_id", how="inner").reset_index(drop=True)
    print(f"{len(c)} in cohort, {len(d)} downloaded")

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        from functools import partial
        hl = list(ex.map(partial(dhash, size=a.hash_size), [os.path.join(a.images, f) for f in d.file], chunksize=64))
    hashes = np.stack(hl)
    print(f"dHash ({a.hash_size}x{a.hash_size} bits) computed for {len(hashes)} images")
    d["dhash"] = [h.tobytes().hex() for h in hashes]

    pairs, min_dist = [], np.full(len(d), a.hash_size ** 2)
    B = 128  # 128 x N x 32 bytes (~160 MB at N = 40k)
    for s in range(0, len(d), B):
        dist = hamming(hashes[s:s + B], hashes).astype(np.int64)
        for r in range(dist.shape[0]):
            i = s + r
            dist[r, i] = 9999
            min_dist[i] = dist[r].min()  # nearest other image, any index
            dist[r, : i + 1] = 9999  # each pair once (j > i)
            row = dist[r]
            for j in np.nonzero(row <= a.max_hamming)[0]:
                pairs.append((i, int(j), int(row[j])))
    sha_groups = d.groupby("sha256").indices
    exact = {(min(v[k], v[m]), max(v[k], v[m])) for v in sha_groups.values() if len(v) > 1
             for k in range(len(v)) for m in range(k + 1, len(v))}
    pairset = {(i, j): dist for i, j, dist in pairs}
    for i, j in exact:
        pairset.setdefault((i, j), 0)

    uf = UF(len(d))
    rows = []
    for (i, j), dist in sorted(pairset.items()):
        uf.union(i, j)
        rows.append({"image_a": d.image_id[i], "image_b": d.image_id[j], "source_a": d.source[i], "source_b": d.source[j],
                     "hamming": dist, "exact_sha256": (i, j) in exact, "same_label": d.label[i] == d.label[j],
                     "same_lesion": d.lesion_id[i] != "" and d.lesion_id[i] == d.lesion_id[j],
                     "same_patient": d.patient_id[i] != "" and d.patient_id[i] == d.patient_id[j]})
    pr = pd.DataFrame(rows, columns=["image_a", "image_b", "source_a", "source_b", "hamming", "exact_sha256",
                                     "same_label", "same_lesion", "same_patient"])
    pr.to_csv(os.path.join(a.out, "dup_pairs.csv"), index=False)

    clusters = defaultdict(list)
    for i in range(len(d)):
        clusters[uf.find(i)].append(i)
    action = ["keep"] * len(d)
    cluster_id = [""] * len(d)
    stats = Counter()
    for root, members in clusters.items():
        if len(members) == 1:
            continue
        cid = f"dup{root}"
        for i in members:
            cluster_id[i] = cid
        srcs = {d.source[i] for i in members}
        labs = {d.label[i] for i in members}
        stats["clusters_cross_source" if len(srcs) > 1 else "clusters_within_source"] += 1
        if len(labs) > 1:
            stats["clusters_label_conflict"] += 1
            for i in members:
                action[i] = "exclude_label_conflict"
            continue
        for i in members:
            action[i] = "cluster"
    d["dup_cluster"], d["action"] = cluster_id, action
    d[["image_id", "source", "file", "sha256", "dhash", "dup_cluster", "action"]].to_csv(
        os.path.join(a.out, "dedup_decisions.csv"), index=False)

    summary = {
        "n_images": len(d), "max_hamming": a.max_hamming, "hash_bits": a.hash_size ** 2,
        "pairs_total": len(pr), "pairs_exact_sha256": int(pr.exact_sha256.sum()) if len(pr) else 0,
        "pairs_near_only": int((~pr.exact_sha256).sum()) if len(pr) else 0,
        "pairs_cross_source": int((pr.source_a != pr.source_b).sum()) if len(pr) else 0,
        "pairs_within_source": int((pr.source_a == pr.source_b).sum()) if len(pr) else 0,
        "pairs_label_conflict": int((~pr.same_label).sum()) if len(pr) else 0,
        "pairs_different_lesion_id": int((~pr.same_lesion).sum()) if len(pr) else 0,
        **stats, "actions": dict(Counter(action)),
        "actions_by_source": {s: dict(Counter(g.action)) for s, g in d.groupby("source")},
        "min_hamming_histogram_0_20": {int(k): int(v) for k, v in zip(*np.unique(np.minimum(min_dist, 21), return_counts=True))},
    }
    write_json(os.path.join(a.out, "dedup_summary.json"), summary)
    write_json(os.path.join(a.out, "dedup_run.json"), run_record(extra={"summary": summary}))
    for k, v in summary.items():
        print(f"{k}: {v}")

    # contact sheet for visual check (near pairs first, largest distance first: the doubtful ones)
    near = pr[~pr.exact_sha256].sort_values("hamming", ascending=False).head(48)
    if len(near):
        T = 160
        sheet = Image.new("RGB", (4 * 2 * T, ((len(near) + 3) // 4) * (T + 14)), "white")
        fmap = dict(zip(d.image_id, d.file))
        for k, r in enumerate(near.itertuples()):
            x0, y0 = (k % 4) * 2 * T, (k // 4) * (T + 14)
            for m, iid in enumerate([r.image_a, r.image_b]):
                im = Image.open(os.path.join(a.images, fmap[iid])).convert("RGB")
                im.thumbnail((T, T))
                sheet.paste(im, (x0 + m * T, y0 + 14))
            from PIL import ImageDraw
            ImageDraw.Draw(sheet).text((x0 + 2, y0), f"d={r.hamming} {r.source_a}/{r.source_b}", fill="black")
        sheet.save(os.path.join(a.out, "pairs_sheet.jpg"), quality=90)
        print("Visual check: pairs_sheet.jpg")


if __name__ == "__main__":
    main()
