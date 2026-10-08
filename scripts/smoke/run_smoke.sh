#!/usr/bin/env bash
# End-to-end test on SYNTHETIC data (~20 min on one H100): every script of the pipeline runs once, tiny settings,
# including the Phase B path (continued pretraining -> dinov2_cpt evaluation -> primary contrast).
# Purpose: find bugs before spending hours on real data. Outputs go to $WS/smoke and are NOT results: never report them.
#   bash scripts/smoke/run_smoke.sh 2>&1 | tee logs/smoke.log
# Each step prints PASS/FAIL; the script continues after a failure so all problems show up in one run.
set -uo pipefail
WS=${WS:-/workspace}
SM=$WS/smoke
cd "$(dirname "$0")/../.."
rm -rf "$SM"; mkdir -p "$SM"
declare -a RES
step() { local name=$1; shift; echo; echo "########## $name"; if "$@"; then RES+=("PASS $name"); else RES+=("FAIL $name"); fi; }
P=python
NB="--n-boot 50"
SMAP=$SM/raw/source_map.csv

# ---- data
step synthetic   $P scripts/smoke/make_synthetic.py --out $SM/raw
step cohort      $P scripts/build_cohort.py --meta $SM/raw/metadata_all.csv --out $SM/cohort --source-map $SMAP
step resize256   $P scripts/resize_cache.py --src $SM/raw/images --dst $SM/img256 --side 256
step resize512   $P scripts/resize_cache.py --src $SM/raw/images --dst $SM/img512 --side 512
step dedup       $P scripts/dedup_audit.py --cohort $SM/cohort/cohort.csv --images $SM/img256 --out $SM/dedup
step dedup_found_planted_copies $P -c "
import json; s = json.load(open('$SM/dedup/dedup_summary.json'))
assert s['pairs_exact_sha256'] >= 1 and s['pairs_total'] >= 2, s
print('planted exact and near duplicates found:', s['pairs_exact_sha256'], s['pairs_total'])"
step partition_rejects_unverified_test bash -c "head -3 $SMAP > $SM/map_without_D.csv; \
  ! $P scripts/make_partitions.py --cohort $SM/cohort/cohort.csv --dedup $SM/dedup/dedup_decisions.csv \
    --test-sources SynthD --source-map $SM/map_without_D.csv --out $SM/part_bad"
step partition   $P scripts/make_partitions.py --cohort $SM/cohort/cohort.csv --dedup $SM/dedup/dedup_decisions.csv \
                    --test-sources SynthD --source-map $SMAP --out $SM/part
step controllo   $P controllo_dataset.py $SM/part/manifest.csv
M=$SM/part/manifest.csv

# ---- correctness checks and Phase A runs
step test_backbones $P scripts/test_backbones.py --ckpt $WS/checkpoints/panderm_ll_data6_checkpoint-499.pth --sdpa --img448
step frozen_panderm $P scripts/frozen_baseline.py --backbone panderm --images $SM/img256 --manifest $M --out $SM/frozen $NB --workers 4
step frozen_dinov2  $P scripts/frozen_baseline.py --backbone dinov2 --images $SM/img256 --manifest $M --out $SM/frozen $NB --workers 4
T="--manifest $M --config configs/hparams.json --workers 4 --batch 16 $NB"
step train_panderm_lora $P scripts/train.py --backbone panderm --method lora --train-frac 1.0 --epochs 2 --images $SM/img256 --out $SM/matrix $T
step train_dinov2_full  $P scripts/train.py --backbone dinov2 --method full --train-frac 1.0 --epochs 1 --images $SM/img256 --out $SM/matrix $T
step train_frac01       $P scripts/train.py --backbone dinov2 --method lora --train-frac 0.1 --epochs 1 --images $SM/img256 --out $SM/matrix $T
step train_sdpa_ckpt_accum_profile $P scripts/train.py --backbone panderm --method full --train-frac 1.0 --epochs 2 --sdpa --grad-ckpt \
                    --accum 2 --torch-profile --no-test --no-save --tag feat --images $SM/img256 --out $SM/bench $T
step train_448      $P scripts/train.py --backbone panderm --method lora --train-frac 1.0 --epochs 1 --img 448 --grad-ckpt \
                    --no-test --no-save --tag bench --images $SM/img512 --out $SM/bench $T
step train_dali     $P scripts/train.py --backbone dinov2 --method lora --train-frac 1.0 --epochs 2 --loader dali \
                    --no-test --no-save --tag dali --images $SM/raw/images --out $SM/bench $T
step bench_loader   $P scripts/bench_loader.py --manifest $M --orig $SM/raw/images --cache256 $SM/img256 --cache512 $SM/img512 \
                    --batches 3 --batch 16 --workers 4
step sweep bash -c "for lr in 0.0001 0.0002 0.0005; do $P scripts/train.py --backbone panderm --method full --train-frac 0.3 \
                    --epochs 1 --lr \$lr --no-test --no-save --images $SM/img256 --out $SM/sweep $T || exit 1; done"

# ---- continued-pretraining pool and smoke run
step ssl_select     $P scripts/ssl_build_set.py select --meta $SM/raw/metadata_dermoscopic.csv --source-map $SMAP \
                    --manifest $M --out $SM/ssl --smoke 100000
step ssl_dedup      $P scripts/ssl_dedup_vs_test.py --ssl-ids $SM/ssl/ssl_smoke.csv --ssl-images $SM/img256 --manifest $M \
                    --held-images $SM/img256 --out $SM/ssl/exclude.csv
step ssl_list       $P scripts/ssl_build_set.py list --ids $SM/ssl/ssl_smoke.csv --images $SM/img256 --exclude $SM/ssl/exclude.csv \
                    --out $SM/ssl/ssl_smoke.txt
step ssl_no_val_test_leak $P -c "
import pandas as pd
m = pd.read_csv('$M', dtype=str, keep_default_na=False)
held = m[m.split.isin(['val', 'test'])]
s = set(open('$SM/ssl/ssl_smoke.txt').read().split())
assert not (set(held.file) & s), 'validation/test images in the pretraining list'
src = dict(zip(m.file, m.source))
assert not any(src.get(f) == 'SynthD' for f in s), 'test-institution image in the pretraining list'
assert len(s) > 0
print('no validation/test image and no test-institution image in the pretraining list;', len(s), 'images')"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > $SM/ssl_gpu.csv &
SMI=$!
step ssl_dinov2     timeout 15m torchrun --nproc_per_node 1 scripts/ssl_dinov2.py --dinov2-repo $WS/dinov2 \
                    --hub-weights $HOME/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth \
                    --config-file configs/ssl_dinov2_vitl14.yaml --output-dir $SM/ssl_run \
                    train.dataset_path=FileList:root=$SM/img256:list=$SM/ssl/ssl_smoke.txt \
                    train.batch_size_per_gpu=16 train.OFFICIAL_EPOCH_LENGTH=10 optim.epochs=2 optim.warmup_epochs=1 \
                    train.num_workers=4 evaluation.eval_period_iterations=10
kill $SMI 2>/dev/null
step ssl_summarize  $P scripts/ssl_summarize.py --run $SM/ssl_run --batch-per-gpu 16 --gpus 1 --skip 2 --gpu-log $SM/ssl_gpu.csv

# ---- Phase B evaluation path: continued-pretrained encoder -> same protocol -> primary contrast
CPT=$(ls -1 $SM/ssl_run/eval/*/teacher_checkpoint.pth 2>/dev/null | tail -1)
echo "teacher checkpoint for dinov2_cpt: ${CPT:-NONE}"
step frozen_cpt     $P scripts/frozen_baseline.py --backbone dinov2_cpt --cpt-ckpt "$CPT" --images $SM/img256 --manifest $M \
                    --out $SM/frozen $NB --workers 4
step train_cpt_full $P scripts/train.py --backbone dinov2_cpt --cpt-ckpt "$CPT" --method full --train-frac 1.0 --epochs 1 \
                    --images $SM/img256 --out $SM/matrix $T
step aggregate      $P scripts/aggregate.py --frozen $SM/frozen --matrix $SM/matrix --out $SM/results --n-boot 50
step primary_contrast_present $P -c "
t = open('$SM/results/primary.md').read(); print(t); assert 'NOT AVAILABLE' not in t"
step resource_report $P scripts/resource_report.py --ws $SM --out $SM/resource_report.json

echo; echo "################ SMOKE SUMMARY (synthetic data: these are NOT results)"
printf '%s\n' "${RES[@]}"
printf '%s\n' "${RES[@]}" | grep -q '^FAIL' && { echo "Fix the FAIL steps (see their output above) before real runs."; exit 1; } \
  || echo "ALL SMOKE STEPS PASSED"
