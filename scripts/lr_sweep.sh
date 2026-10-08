#!/usr/bin/env bash
# LR sweep: same grid size (3 values) for every (backbone, method); train fraction 0.3, seed 1, VALIDATION ONLY.
# Then: python scripts/select_lr.py --sweep $WS/runs/sweep
# Two GPUs: run "bash scripts/lr_sweep.sh 0 2" and "bash scripts/lr_sweep.sh 1 2" in two tmux windows.
set -uo pipefail
WS=${WS:-/workspace}
SHARD=${1:-0}; NSHARDS=${2:-1}
IMAGES=${IMAGES:-$WS/data/images_256}
MANIFEST=${MANIFEST:-$WS/data/partition/manifest.csv}
EXTRA=${EXTRA:-}
BBS=${BBS:-"panderm dinov2"}   # Phase B: BBS=dinov2_cpt EXTRA="--cpt-ckpt <teacher_checkpoint.pth>"
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$SHARD}

i=0
for BB in $BBS; do
  for M in lora full; do
    for LR in $(python -c "import json;print(' '.join(map(str,json.load(open('configs/hparams.json'))['$M']['lr_grid'])))"); do
      if [ $((i % NSHARDS)) -eq "$SHARD" ]; then
        echo "== $(date -u +%FT%TZ) sweep $BB $M lr=$LR (gpu $CUDA_VISIBLE_DEVICES)"
        python scripts/train.py --backbone $BB --method $M --train-frac 0.3 --lr $LR --seed 1 \
          --manifest "$MANIFEST" --images "$IMAGES" --out "$WS/runs/sweep" --no-test --no-save $EXTRA \
          || echo "!! FAILED sweep $BB $M lr=$LR"
      fi
      i=$((i + 1))
    done
  done
done
echo "== $(date -u +%FT%TZ) sweep shard $SHARD/$NSHARDS done"
