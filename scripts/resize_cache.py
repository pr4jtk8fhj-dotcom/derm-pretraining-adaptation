#!/usr/bin/env python3
"""Pre-resize images (shorter side = --side, the first transform of the pipeline: 256 for 224 px, 512 for 448 px)
to remove the full-resolution JPEG decode from the training loop. Same file names; point --images to the cache.
Differences vs. resizing on the fly (to declare): JPEG DCT-domain draft decoding to >= 2x target size, then
bilinear resize, then JPEG re-encoding at quality 95.

    python scripts/resize_cache.py --src /workspace/data/images --dst /workspace/data/images_256 --side 256
    python scripts/resize_cache.py --src /workspace/data/images --dst /workspace/data/images_512 --side 512
"""
import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

from PIL import Image


def one(args):
    src, dst, side = args
    with Image.open(src) as im:
        w, h = im.size
        s = side / min(w, h)
        if s < 0.5:
            im.draft("RGB", (round(w * s * 2), round(h * s * 2)))
        im = im.convert("RGB")
        w, h = im.size
        s = side / min(w, h)
        if s < 1:
            im = im.resize((max(side, int(w * s)), max(side, int(h * s))), Image.BILINEAR)
        fmt = "PNG" if dst.lower().endswith(".png") else "JPEG"
        im.save(dst + ".part", fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    os.replace(dst + ".part", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--side", type=int, default=256)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    files = [f for f in os.listdir(a.src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    jobs = [(os.path.join(a.src, f), os.path.join(a.dst, f), a.side) for f in files
            if not os.path.exists(os.path.join(a.dst, f))]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs, chunksize=64))
    shutil.copy(os.path.join(a.src, "downloads.csv"), os.path.join(a.dst, "downloads.csv"))
    print(f"{len(jobs)} resized to shorter side {a.side}, {len(files) - len(jobs)} already present")


if __name__ == "__main__":
    main()
