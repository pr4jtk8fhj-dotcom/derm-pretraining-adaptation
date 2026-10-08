#!/usr/bin/env bash
# Matrix: {LoRA r16, full} x {panderm, dinov2} x fractions, for the given seeds (frozen: frozen_baseline.py).
# Usage: bash scripts/run_matrix.sh "<seeds>" "<fractions>" [shard] [nshards]
#   seed 1, all fractions, 1 GPU:   bash scripts/run_matrix.sh "1" "1.0 0.3 0.1"
#   seeds 2-3, 100% only, 2 GPUs:   bash scripts/run_matrix.sh "2 3" "1.0" 0 2   (and ... 1 2 in another window)
# Completed runs (results.json present) are skipped, so it can be relaunched after an interruption.
# In tmux:  tmux new -s m0 'bash scripts/run_matrix.sh "1" "1.0 0.3 0.1" 2>&1 | tee -a logs/matrix_s1.log'
set -uo pipefail
WS=${WS:-/workspace}
SEEDS=${1:-1}; FRACS=${2:-"1.0 0.3 0.1"}; SHARD=${3:-0}; NSHARDS=${4:-1}
IMAGES=${IMAGES:-$WS/data/images_256}
MANIFEST=${MANIFEST:-$WS/data/partition/manifest.csv}
OUT=${OUT:-$WS/runs/matrix}
RANK=${RANK:-16}
EXTRA=${EXTRA:-}
BBS=${BBS:-"panderm dinov2"}   # Phase B: BBS=dinov2_cpt EXTRA="--cpt-ckpt <teacher_checkpoint.pth>"
METHODS=${METHODS:-"lora full"}
# Primary-contrast reference first (3 seeds):  BBS=dinov2 METHODS=full bash scripts/run_matrix.sh "2 3" "1.0"
cd "$(dirname "$0")/.."
[ -f configs/hparams_selected.json ] || { echo "run lr_sweep.sh + select_lr.py first"; exit 1; }
# Hold: while logs/MATRIX_HOLD exists (e.g. user reviewing the selected LRs) wait before starting any run.
while [ -f logs/MATRIX_HOLD ]; do echo "== $(date -u +%FT%TZ) MATRIX_HOLD present: waiting"; sleep 60; done
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$SHARD}

declare -A FAILS
i=0
for SEED in $SEEDS; do
  for F in $FRACS; do
    for BB in $BBS; do
      for M in $METHODS; do
        if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i + 1)); continue; fi
        i=$((i + 1))
        KEY="$BB-$M-$F"
        echo "== $(date -u +%FT%TZ) START $KEY seed $SEED (gpu $CUDA_VISIBLE_DEVICES)"
        if ! python scripts/train.py --backbone $BB --method $M --rank $RANK --train-frac $F --seed $SEED \
             --manifest "$MANIFEST" --images "$IMAGES" --config configs/hparams_selected.json --out "$OUT" $EXTRA; then
          FAILS[$KEY]=$(( ${FAILS[$KEY]:-0} + 1 ))
          echo "!! FAILED $KEY seed $SEED (failures for this config: ${FAILS[$KEY]})"
          if [ "${FAILS[$KEY]}" -ge 2 ]; then echo "!! same config failed twice: stopping (CLAUDE.md rule d)"; exit 1; fi
        fi
        python scripts/progress.py status || true
      done
    done
  done
done
echo "== $(date -u +%FT%TZ) MATRIX SHARD $SHARD/$NSHARDS DONE (seeds: $SEEDS, fracs: $FRACS). Stop the pod when all shards are done."
