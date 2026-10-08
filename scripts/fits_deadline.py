#!/usr/bin/env python3
"""Exit 0 if a new GPU run is expected to end before the deadline, else 1 (scheduling guard, no effect on results).

    python scripts/fits_deadline.py --deadline 2026-10-09T05:30:00+00:00 --ref <run dir> --minutes 62.7 [--factor 1.15]
The expected duration is the measured wall time of --ref (same config, earlier seed) when it exists,
otherwise --minutes (estimate_matrix.py estimate); both multiplied by --factor.
"""
import argparse
import datetime
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--deadline", required=True)
ap.add_argument("--ref", default="")
ap.add_argument("--minutes", type=float, required=True)
ap.add_argument("--factor", type=float, default=1.15)
ap.add_argument("--label", default="")
a = ap.parse_args()
mins, src = a.minutes, "estimate"
res = os.path.join(a.ref, "results.json")
if a.ref and os.path.exists(res):
    mins, src = json.load(open(res))["run"]["wall_seconds"] / 60, f"measured {a.ref}"
now = datetime.datetime.now(datetime.timezone.utc)
end = now + datetime.timedelta(minutes=mins * a.factor)
ok = end <= datetime.datetime.fromisoformat(a.deadline)
print(f"[fits_deadline] {a.label}: {mins:.1f} min ({src}) x {a.factor} -> end {end:%Y-%m-%d %H:%M} UTC "
      f"vs deadline {a.deadline}: {'RUN' if ok else 'SKIP'}", flush=True)
sys.exit(0 if ok else 1)
