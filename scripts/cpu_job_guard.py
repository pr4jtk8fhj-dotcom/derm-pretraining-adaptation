#!/usr/bin/env python3
"""Pause (SIGSTOP) a low-priority CPU job if GPU training slows down or the volume fills up (user rules, 8/10):
- training img/s (median of the last 3 epochs finished after the job started/resumed) < 90% of the 224 px
  benchmark img/s of the same (backbone, method) -> pause 10 min, then resume and re-check;
- free space on /workspace < 40 GB (+5 GB margin) -> pause, delete SSL ORIGINALS that already have SHA-256 in
  downloads.csv and a 256 cache file, then resume.
    python scripts/cpu_job_guard.py --pgid <process group> --log logs/cpu_job_guard.log
"""
import argparse, csv, glob, json, os, shutil, signal, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--pgid", type=int, required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--orig", default="/workspace/data/ssl_images")
ap.add_argument("--cache", default="/workspace/data/ssl_images_256")
a = ap.parse_args()
BENCH = {}  # 224 px benchmark (1 epoch, 100%): img_per_s of its epoch line
for f in glob.glob("/workspace/runs/bench/*/log.txt"):
    name = os.path.basename(os.path.dirname(f))
    ep = [json.loads(l) for l in open(f) if l.startswith("{") and '"img_per_s"' in l]
    if ep:
        BENCH[(name.split("_")[0], "lora" if "_lora" in name else "full")] = ep[-1]["img_per_s"]


def log(msg):
    with open(a.log, "a") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg + "\n")


def alive():
    try:
        os.killpg(a.pgid, 0); return True
    except ProcessLookupError:
        return False


def bench_ips(name):
    bb = "panderm" if name.startswith("panderm") else "dinov2" if name.startswith("dinov2") else None
    meth = "lora" if "_lora" in name else "full"
    return BENCH.get((bb, meth))


def current_run():
    logs = [p for p in glob.glob("/workspace/runs/*/*/log.txt") if "bench" not in p and "profile" not in p
            and "dali" not in p and time.time() - os.path.getmtime(p) < 180]
    return max(logs, key=os.path.getmtime) if logs else None


def epochs_after(path, t0):
    out = []
    for ln in open(path):
        if ln.startswith("{") and '"img_per_s"' in ln:
            out.append(json.loads(ln))
    # epochs written after t0: approximate with elapsed_s relative to file start (mtime-based)
    start = os.path.getmtime(path) - (out[-1]["elapsed_s"] if out else 0)
    return [e for e in out if start + e["elapsed_s"] >= t0]


def free_gb():
    return shutil.disk_usage("/workspace").free / 1e9


def prune_originals():
    done = {}
    dl = os.path.join(a.orig, "downloads.csv")
    if os.path.exists(dl):
        for r in csv.DictReader(open(dl)):
            if r.get("sha256"): done[r["file"]] = True
    n = 0
    for f in list(done):
        p, c = os.path.join(a.orig, f), os.path.join(a.cache, f)
        if os.path.exists(p) and os.path.exists(c):
            os.remove(p); n += 1
    return n


BENCH = {k: v for k, v in BENCH.items() if v}
log(f"guard start pgid={a.pgid} bench img/s={BENCH}")
t0, paused = time.time(), False
while alive():
    fg = free_gb()
    if fg < 45:
        os.killpg(a.pgid, signal.SIGSTOP); log(f"free {fg:.1f} GB < 45: paused; pruning SSL originals with hash+cache")
        n = prune_originals(); fg = free_gb(); log(f"deleted {n} originals, free now {fg:.1f} GB")
        if fg >= 45:
            os.killpg(a.pgid, signal.SIGCONT); log("resumed")
        else:
            log("still < 45 GB: job stays paused"); time.sleep(300); continue
    run = current_run()
    if run and "img448" not in run:
        name = os.path.basename(os.path.dirname(run)); ref = bench_ips(name)
        ep = epochs_after(run, t0)
        if ref and len(ep) >= 3:
            med = statistics.median(e["img_per_s"] for e in ep[-3:])
            if med < 0.9 * ref:
                os.killpg(a.pgid, signal.SIGSTOP)
                log(f"{name}: median img/s {med:.1f} < 90% of bench {ref:.1f}: PAUSED 10 min")
                time.sleep(600); os.killpg(a.pgid, signal.SIGCONT); t0 = time.time(); log("resumed, re-checking")
                continue
            log(f"ok {name}: median img/s {med:.1f} vs bench {ref:.1f}; free {fg:.1f} GB")
    time.sleep(60)
log("job finished, guard exits")
