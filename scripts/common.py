"""Shared helpers: file hashing, run records (command, commits, versions, GPU), GPU sampling."""
import datetime
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time

WS = os.environ.get("WS", "/workspace")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANDERM_REPO = os.environ.get("PANDERM_REPO", os.path.join(WS, "PanDerm"))
# RunPod H100 SXM list price per GPU, 27 Sep 2026 (CLAUDE.md). Pod price = per-GPU price x N_GPUS.
PRICE_PER_GPU_HOUR = float(os.environ.get("GPU_PRICE", "3.49"))
N_GPUS = int(os.environ.get("N_GPUS", "1"))
PRICE_PER_HOUR = PRICE_PER_GPU_HOUR * N_GPUS


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_commit(path):
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "n/a"
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "n/a"


def lib_versions():
    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ["torch", "torchvision", "timm", "numpy", "sklearn", "PIL", "pandas", "xformers"]:
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "?")
        except Exception:
            vers[name] = None
    try:
        import torch
        vers["cuda"] = torch.version.cuda
        vers["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return vers


def gpu_info():
    info = {}
    try:
        import torch
        if torch.cuda.is_available():
            info["name"] = torch.cuda.get_device_name(0)
            info["count"] = torch.cuda.device_count()
            info["total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = out.stdout.strip()
    except Exception:
        pass
    return info


def run_record(seed=None, extra=None):
    rec = {
        "command": " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv),
        "cwd": os.getcwd(),
        "started_utc": now_iso(),
        "host": platform.node(),
        "seed": seed,
        "project_commit": git_commit(PROJECT_DIR),
        "panderm_commit": git_commit(PANDERM_REPO),
        "libs": lib_versions(),
        "gpu": gpu_info(),
    }
    if extra:
        rec.update(extra)
    return rec


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


class Tee:
    """Duplicate stdout/stderr into a log file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.out, self.err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self, self

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding... from the real stream
        return getattr(self.out, name)


class GpuSampler(threading.Thread):
    """Samples GPU utilization (%) and memory used via NVML every `period` seconds.

    utilization.gpu is the fraction of the sample window in which at least one kernel was running,
    so idle% = 100 - mean(utilization).
    """

    def __init__(self, period=0.5, index=0):
        super().__init__(daemon=True)
        self.period, self.index = period, index
        self.util, self.mem = [], []
        self._stop = threading.Event()
        self.ok = True
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as e:
            print(f"[GpuSampler] NVML unavailable ({e}); GPU utilization will not be reported")
            self.ok = False

    def run(self):
        while self.ok and not self._stop.is_set():
            u = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            m = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.util.append(u.gpu)
            self.mem.append(m.used / 1e9)
            time.sleep(self.period)

    def stop(self):
        self._stop.set()

    def summary(self, skip_first_s=0.0):
        if not self.util:
            return {"gpu_util_mean": None, "gpu_idle_pct": None, "n_samples": 0}
        k = int(skip_first_s / self.period)
        u = self.util[k:] or self.util
        mean = sum(u) / len(u)
        return {"gpu_util_mean": round(mean, 1), "gpu_idle_pct": round(100 - mean, 1),
                "gpu_mem_used_max_gb": round(max(self.mem), 2), "n_samples": len(u)}
