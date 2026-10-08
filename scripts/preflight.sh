#!/usr/bin/env bash
# Preflight (1-2 min, no data needed): syntax of every script, imports, GPU, disk, config files.
# The scripts were written without being executed; this catches typos before any GPU time is spent.
#   bash scripts/preflight.sh 2>&1 | tee logs/preflight.log
set -uo pipefail
WS=${WS:-/workspace}
cd "$(dirname "$0")/.."
FAIL=0
ok() { echo "  OK   $1"; }
ko() { echo "  FAIL $1"; FAIL=1; }

echo "== Python syntax"
for f in scripts/*.py scripts/smoke/*.py controllo_dataset.py; do
  python -m py_compile "$f" 2>/tmp/pyc.err && ok "$f" || { ko "$f"; cat /tmp/pyc.err; }
done
echo "== Bash syntax"
for f in scripts/*.sh scripts/smoke/*.sh; do bash -n "$f" && ok "$f" || ko "$f"; done
echo "== JSON / YAML / CSV configs"
python -c "import json; json.load(open('configs/hparams.json'))" && ok hparams.json || ko hparams.json
python -c "import yaml; yaml.safe_load(open('configs/ssl_dinov2_vitl14.yaml'))" && ok ssl yaml || ko "ssl yaml"
python -c "import pandas as pd; c=pd.read_csv('configs/source_map.csv'); assert list(c.columns)==['attribution','source','country']" \
  && ok source_map.csv || ko "source_map.csv columns"
echo "== Imports of our modules"
for m in common evaluation data backbones build_cohort dedup_audit make_partitions frozen_baseline train \
         estimate_matrix aggregate select_lr ssl_build_set ssl_dedup_vs_test ssl_summarize resource_report \
         resize_cache bench_loader; do
  (cd scripts && python -c "import $m") 2>/tmp/imp.err && ok "$m" || { ko "$m"; tail -3 /tmp/imp.err; }
done
(cd scripts && python -c "import dali_loader") 2>/dev/null && ok "dali_loader (DALI available)" \
  || echo "  WARN dali_loader: DALI not importable -> --loader dali unavailable"
(cd scripts && PYTHONPATH=$WS/dinov2 python -c "import ssl_dinov2, dinov2.train.train as T; T.get_args_parser") 2>/tmp/imp.err \
  && ok "ssl_dinov2 + dinov2.train" || { echo "  WARN ssl_dinov2:"; tail -3 /tmp/imp.err; }
echo "== Environment"
python -c "import torch; assert torch.cuda.is_available(); print('  GPU', torch.cuda.get_device_name(0), torch.cuda.device_count())" || ko "CUDA"
[ -f "$WS/checkpoints/panderm_ll_data6_checkpoint-499.pth" ] && ok "PanDerm checkpoint" || ko "PanDerm checkpoint missing"
FREE=$(df -BG --output=avail "$WS" | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 120 ] && ok "free disk on $WS: ${FREE} GB" || echo "  WARN free disk on $WS: ${FREE} GB (< 120 GB: original ISIC images are large)"
echo "  CPUs: $(nproc)  RAM: $(free -g | awk '/Mem:/{print $2}') GB"
[ $FAIL -eq 0 ] && echo "== PREFLIGHT PASSED. Next: bash scripts/smoke/run_smoke.sh" || { echo "== PREFLIGHT FAILED: fix the FAIL lines first"; exit 1; }
