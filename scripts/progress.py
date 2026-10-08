#!/usr/bin/env python3
"""Keep PROGRESS.md up to date: elapsed pod hours and estimated cost.

    python scripts/progress.py start                      # once, when the pod is up
    python scripts/progress.py log "Fase" "fatto" "manca"   # at each phase
    python scripts/progress.py status                     # print elapsed hours and cost
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PRICE_PER_HOUR, PROJECT_DIR  # noqa: E402

PROGRESS = os.path.join(PROJECT_DIR, "PROGRESS.md")
START = os.path.join(PROJECT_DIR, ".pod_start_utc")
BUDGET_CAP, BUDGET_TARGET = 100.0, 90.0
# Cost assumes the pod price did not change since start (N_GPUS / GPU_PRICE env vars); if the pod is
# replaced (e.g. 1 -> 2 GPUs), log it in PROGRESS.md and compute the real cost from RunPod billing.


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def elapsed():
    with open(START) as f:
        t0 = datetime.datetime.fromisoformat(f.read().strip())
    h = (now() - t0).total_seconds() / 3600
    return t0, h, h * PRICE_PER_HOUR


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        if os.path.exists(START):
            sys.exit(f"Start time already recorded in {START}; not overwriting.")
        t0 = now().isoformat(timespec="seconds")
        with open(START, "w") as f:
            f.write(t0)
        with open(PROGRESS, "a", encoding="utf-8") as f:
            f.write(f"# PROGRESS\n\nAvvio pod (UTC): {t0}\nPrezzo: {PRICE_PER_HOUR} $/h (stima, fatturazione al secondo). "
                    f"Tetto {BUDGET_CAP} $, obiettivo {BUDGET_TARGET} $.\n\n")
        print(f"Start recorded: {t0}")
        return
    t0, h, cost = elapsed()
    line = f"{h:.2f} h trascorse, costo stimato {cost:.2f} $ (tetto {BUDGET_CAP} $, obiettivo {BUDGET_TARGET} $)"
    if cmd == "status":
        print(line)
        return
    if cmd == "log":
        phase, done, todo = (sys.argv[2:5] + ["", "", ""])[:3]
        with open(PROGRESS, "a", encoding="utf-8") as f:
            f.write(f"## {now().strftime('%Y-%m-%d %H:%M')} UTC - {phase}\n- {line}\n- Fatto: {done}\n- Manca: {todo}\n\n")
        print(line)
        if cost > BUDGET_TARGET:
            print("ATTENZIONE: superato l'obiettivo di spesa.")
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
