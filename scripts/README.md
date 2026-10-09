# Command order (phase A, single GPU pod)

Project folder: `/workspace/project` (always run from there). `WS=/workspace` (default). With 2 GPUs: `export N_GPUS=2`.
Every script saves command, commit, library versions, seed, GPU and file hashes as JSON. Run long jobs inside tmux.
Variables used below: `M=$WS/data/partition/manifest.csv`, `CK=$WS/checkpoints/panderm_ll_data6_checkpoint-499.pth`.

## 0. Setup and checks (before any real data)
| # | Command | Notes |
|---|---|---|
| 0 | `python scripts/progress.py start` | records the pod start time for elapsed-time tracking |
| 1 | `bash scripts/00_setup_pod.sh 2>&1 \| tee -a logs/setup.log` | dependencies with torch pinned, PanDerm, DINOv2, weights, DALI, isic-cli |
| 2 | `bash scripts/preflight.sh 2>&1 \| tee logs/preflight.log` | syntax, imports, GPU, disk: must end with PREFLIGHT PASSED |
| 3 | `bash scripts/smoke/run_smoke.sh 2>&1 \| tee logs/smoke.log` | whole pipeline on SYNTHETIC data (~15 min); every step must PASS before real data. Outputs are not results |

## 1. Data
| # | Command | Notes |
|---|---|---|
| 4 | `bash scripts/01_metadata.sh` | metadata: histopathology-labelled set and all dermoscopic images |
| 5 | `python scripts/build_cohort.py --meta $WS/data/metadata/metadata_all.csv --out $WS/data/cohort` | then fill `configs/source_map.csv` from `attributions.csv` and rerun with `--source-map configs/source_map.csv` |
| — | Choose the test institutions from `institutions.md` with the rule fixed before any training (see the top-level README) | phase A: HIBA and the patient-identified MSKCC images |
| 6 | `python scripts/download_images.py --cohort $WS/data/cohort/cohort.csv --out $WS/data/images --allow-nc` | CC-BY-NC images accepted for the labelled set (non-commercial research) |
| 7 | `python scripts/resize_cache.py --src $WS/data/images --dst $WS/data/images_256 --side 256` (and `--side 512 --dst $WS/data/images_512`) | all CPUs |
| 8 | `python scripts/dedup_audit.py --cohort $WS/data/cohort/cohort.csv --images $WS/data/images_256 --out $WS/data/dedup` | inspect `pairs_sheet.jpg` |
| 9 | `python scripts/make_partitions.py --cohort $WS/data/cohort/cohort.csv --dedup $WS/data/dedup/dedup_decisions.csv --test-sources A B [C] --source-map configs/source_map.csv --out $WS/data/partition` | verified institutions only; table `subsets_table.md`; then `python controllo_dataset.py $M` |

## 2. Baselines, profile, benchmark
| # | Command | Notes |
|---|---|---|
| 10 | `python scripts/frozen_baseline.py --backbone panderm --images $WS/data/images_256 --manifest $M --out $WS/runs/frozen` (then `dinov2`) | three annotation fractions |
| 11 | `bash scripts/profile_original.sh help`, write `logs/original_cmd.sh`, then `bash scripts/profile_original.sh run` | ORIGINAL PanDerm script, original images, 1 epoch; `summarize_gpu_log.py` |
| 12 | `python scripts/train.py --backbone B --method X --train-frac 1.0 --epochs 1 --no-test --no-save --tag bench --manifest $M --images $WS/data/images_256 --out $WS/runs/bench` for B in {panderm, dinov2}, X in {lora, full} | then `python scripts/estimate_matrix.py --bench $WS/runs/bench --manifest $M [--gpus 2]` |

## 3. Engineering measurements (before/after)
| # | Command |
|---|---|
| 13 | `python scripts/bench_loader.py --manifest $M --orig $WS/data/images --cache256 $WS/data/images_256 --cache512 $WS/data/images_512` |
| 14 | same benchmark as step 12 (PanDerm full) on `--images $WS/data/images` with `--loader torch` and with `--loader dali`, `--tag dali_cmp` |
| 15 | benchmark with `--sdpa` (equivalence checked by `test_backbones.py --sdpa`) and one with `--torch-profile` (or `profile_nsys.sh` where Nsight Systems is installed) |
| 16 | 448 px: `train.py ... --img 448 --images $WS/data/images_512 --train-frac 0.1 --epochs 1 --no-test --no-save --tag bench --out $WS/runs/bench448` for the 4 cases (+ `--grad-ckpt` / `--accum 2` if memory is short); then `estimate_matrix.py --bench $WS/runs/bench448 --manifest $M` |

## 4. Sweep, matrix, continued-pretraining set
| # | Command | Notes |
|---|---|---|
| 17 | `bash scripts/lr_sweep.sh` (2 GPUs: `0 2` and `1 2`), then `python scripts/select_lr.py --sweep $WS/runs/sweep` | validation only |
| 18 | `tmux new -s m0 'bash scripts/run_matrix.sh "1" "1.0 0.3 0.1" 2>&1 \| tee -a logs/matrix_s1.log'` | then FIRST `BBS=dinov2 METHODS=full bash scripts/run_matrix.sh "2 3" "1.0"` (reference arm of the primary contrast: 3 seeds), then `"2 3" "1.0"` for the other arms, then `"2 3" "0.3 0.1"` |
| 19 | in parallel (CPU): `ssl_build_set.py select --meta $WS/data/metadata/metadata_dermoscopic.csv --source-map configs/source_map.csv --manifest $M --out $WS/data/ssl --smoke 12000`; `download_images.py --cohort $WS/data/ssl/ssl_smoke.csv --out $WS/data/ssl_images`; `resize_cache.py --src $WS/data/ssl_images --dst $WS/data/ssl_images_256 --side 256`; `ssl_dedup_vs_test.py --ssl-ids $WS/data/ssl/ssl_smoke.csv --ssl-images $WS/data/ssl_images_256 --manifest $M --held-images $WS/data/images_256 --out $WS/data/ssl/exclude_smoke.csv`; `ssl_build_set.py list --ids $WS/data/ssl/ssl_smoke.csv --images $WS/data/ssl_images_256 --exclude $WS/data/ssl/exclude_smoke.csv --out $WS/data/ssl/ssl_smoke.txt` | CC-0/CC-BY only; test institutions and every link to validation or test excluded |
| 20 | after the matrix, at most 45 min: `timeout 45m torchrun --nproc_per_node 1 scripts/ssl_dinov2.py --dinov2-repo $WS/dinov2 --hub-weights ~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth --config-file configs/ssl_dinov2_vitl14.yaml --output-dir $WS/runs/ssl_smoke train.dataset_path=FileList:root=$WS/data/ssl_images_256:list=$WS/data/ssl/ssl_smoke.txt`, with `nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -lms 1000 > logs/ssl_gpu.csv &` | then `python scripts/ssl_summarize.py --run $WS/runs/ssl_smoke --batch-per-gpu 48 --gpus 1 --gpu-log logs/ssl_gpu.csv` (ORIGINAL images/s, memory, loss) |

## 5. Wrap-up
| # | Command | Notes |
|---|---|---|
| 21 | `python scripts/aggregate.py --frozen $WS/runs/frozen --matrix $WS/runs/matrix --out results` and `python scripts/resource_report.py --ws $WS --out logs/resource_report.json` | tables, curves, `primary.md` (NOT AVAILABLE in phase A, as expected: `dinov2_cpt` is phase B), resources |
| 22 | optional, needs 2 GPUs: `torchrun --nproc_per_node 1 scripts/ddp_bench.py --img 448 --grad-ckpt --backbone panderm --manifest $M --images $WS/data/images_512`, then `--nproc_per_node 2 ... --ref-ips <1-GPU value>` | scaling; not run in phase A (no 2-GPU pod) |
| 23 | `python scripts/progress.py log "End" "..." "..."`, then stop the pod | |

## Phase B (Leonardo, during the hackathon; not run on the pod)
Ready and tested on synthetic data; to be adapted to the CINECA environment (scheduler, modules, paths).
- Continued pretraining: `torchrun --nproc_per_node 4 scripts/ssl_dinov2.py ...` on the full pool (`ssl_full.csv`), then
  `ssl_summarize.py --gpus 4`; teacher checkpoint in `<output-dir>/eval/<iter>/teacher_checkpoint.pth`.
- Same tuning budget: `BBS=dinov2_cpt EXTRA="--cpt-ckpt <ckpt>" bash scripts/lr_sweep.sh`, then
  `python scripts/select_lr.py --sweep <dir> --backbones dinov2_cpt` (does not change the choices already fixed).
- Primary contrast: `BBS=dinov2_cpt METHODS=full EXTRA="--cpt-ckpt <ckpt>" bash scripts/run_matrix.sh "1 2 3" "1.0"`,
  `frozen_baseline.py --backbone dinov2_cpt --cpt-ckpt <ckpt> ...`, then `aggregate.py` -> `results/primary.md`.
- Fine-tuning scaling: `ddp_bench.py`; continued-pretraining scaling: the same command with 1, 4 and 8 GPUs, fixed total
  workload (strong scaling) and fixed per-GPU workload (weak scaling), recording global batch and any accumulation.

## Choices fixed before the results, and changes made during phase A
- Duplicates: exact SHA-256 plus perceptual difference hash. The pre-specified 64-bit dHash with at most 6 differing bits
  flagged 847,995 pairs of mostly different lesions; before partitioning it was replaced by a 256-bit dHash with at most
  10 differing bits (see `dedup_audit.py`). Validation: 15% of development groups, seed 0; nested 10/30/100% fractions.
- `configs/hparams.json`: epochs per fraction 30/20/10, batch 128, AdamW, weight decay 0.05, 1 warm-up epoch, cosine
  schedule, drop path 0.1, layer decay 0.75 (full), LoRA r16 alpha 32 on qkv and proj, bf16. Initial learning-rate grids:
  full {1e-4, 2e-4, 5e-4}, LoRA {3e-4, 1e-3, 3e-3}; every arm was then extended by one step past its edge (4 values
  per arm). Selected values: `configs/hparams_selected.json`.
- Frozen features: StandardScaler + logistic regression, C=1, class_weight balanced.
- Input normalisation: PanDerm uses the values hard-coded in its fine-tuning script (mean 0.485/0.456/0.406,
  std 0.228/0.224/0.225); DINOv2 uses ImageNet mean and std (`backbones.py`).
- `configs/ssl_dinov2_vitl14.yaml`: values of the short test; to be reviewed with the mentors for phase B.

## Checks completed on the pod
- `test_backbones.py`: LoRA with zero-initialised B identical to the original model, gradients only in the adapters,
  fused attention and 448 px checks passed.
- PanDerm weight loading identical to the original script (414 tensors, difference 0.0;
  `check_panderm_load_vs_original.py`); weight-decay groups aligned with the original script (391 of 391 parameters).
- DALI and torchvision outputs compared by distribution (`bench_loader.py`).
