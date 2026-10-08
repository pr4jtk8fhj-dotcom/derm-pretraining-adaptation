# Ordine dei comandi sul pod

Cartella del progetto: `/workspace/project` (lavora sempre da li'). `WS=/workspace` (default). Con 2 GPU: `export N_GPUS=2`.
Ogni script salva comando, commit, versioni, seed, GPU e hash dei file in JSON. Tutto il lavoro lungo dentro tmux.
Variabili usate sotto: `M=$WS/data/partition/manifest.csv`, `CK=$WS/checkpoints/panderm_ll_data6_checkpoint-499.pth`.

## 0. Avvio e controlli (circa 1 ora, prima di qualsiasi dato vero)
| # | Comando | Note |
|---|---|---|
| 0 | `python scripts/progress.py start` | appena acceso il pod |
| 1 | `bash scripts/00_setup_pod.sh 2>&1 \| tee -a logs/setup.log` | dipendenze con torch congelato, PanDerm, DINOv2, pesi, DALI, isic-cli |
| 2 | `bash scripts/preflight.sh 2>&1 \| tee logs/preflight.log` | sintassi, import, GPU, disco: deve finire con PREFLIGHT PASSED |
| 3 | `bash scripts/smoke/run_smoke.sh 2>&1 \| tee logs/smoke.log` | tutta la pipeline su dati SINTETICI (~15 min). Tutti PASS prima di andare avanti. Non sono risultati |

## 1. Dati
| # | Comando | Note |
|---|---|---|
| 4 | `bash scripts/01_metadata.sh` | metadati: set istologico + tutte le dermoscopiche |
| 5 | `python scripts/build_cohort.py --meta $WS/data/metadata/metadata_all.csv --out $WS/data/cohort` | poi compila `configs/source_map.csv` da `attributions.csv` e rilancia con `--source-map configs/source_map.csv` |
| — | **STOP: proponi le istituzioni di test** da `institutions.md` e aspetta conferma | regola (a) |
| 6 | `python scripts/download_images.py --cohort $WS/data/cohort/cohort.csv --out $WS/data/images --allow-nc` | CC-BY-NC accettate per il set etichettato |
| 7 | `python scripts/resize_cache.py --src $WS/data/images --dst $WS/data/images_256 --side 256` (e `--side 512 --dst $WS/data/images_512`) | tutte le CPU |
| 8 | `python scripts/dedup_audit.py --cohort $WS/data/cohort/cohort.csv --images $WS/data/images_256 --out $WS/data/dedup` | guarda `pairs_sheet.jpg` |
| 9 | `python scripts/make_partitions.py --cohort $WS/data/cohort/cohort.csv --dedup $WS/data/dedup/dedup_decisions.csv --test-sources A B [C] --source-map configs/source_map.csv --out $WS/data/partition` | solo istituzioni verificate; tabella `subsets_table.md`; poi `python controllo_dataset.py $M` |

## 2. Baseline, profilo, benchmark
| # | Comando | Note |
|---|---|---|
| 10 | `python scripts/frozen_baseline.py --backbone panderm --images $WS/data/images_256 --manifest $M --out $WS/runs/frozen` (poi `dinov2`) | tre frazioni |
| 11 | `bash scripts/profile_original.sh help`, scrivi `logs/original_cmd.sh`, poi `bash scripts/profile_original.sh run` | script ORIGINALE, immagini originali, 1 epoca; `summarize_gpu_log.py` |
| 12 | `python scripts/train.py --backbone B --method X --train-frac 1.0 --epochs 1 --no-test --no-save --tag bench --manifest $M --images $WS/data/images_256 --out $WS/runs/bench` per B in {panderm, dinov2}, X in {lora, full} | poi `python scripts/estimate_matrix.py --bench $WS/runs/bench --manifest $M [--gpus 2]` |

## 3. Ingegneria (max 60 min): i "prima/dopo" della proposta
| # | Comando |
|---|---|
| 13 | `python scripts/bench_loader.py --manifest $M --orig $WS/data/images --cache256 $WS/data/images_256 --cache512 $WS/data/images_512` |
| 14 | stesso benchmark del passo 12 (PanDerm full) su `--images $WS/data/images` con `--loader torch` e con `--loader dali`, `--tag dali_cmp` |
| 15 | benchmark con `--sdpa` (equivalenza gia' verificata in `test_backbones.py --sdpa` dal test sintetico) e uno con `--torch-profile` (o `profile_nsys.sh`) |
| 16 | 448 px: `train.py ... --img 448 --images $WS/data/images_512 --train-frac 0.1 --epochs 1 --no-test --no-save --tag bench --out $WS/runs/bench448` per i 4 casi (+ `--grad-ckpt` / `--accum 2` se la memoria non basta); poi `estimate_matrix.py --bench $WS/runs/bench448 --manifest $M` |

## 4. Sweep, matrice, set SSL
| # | Comando | Note |
|---|---|---|
| 17 | `bash scripts/lr_sweep.sh` (2 GPU: `0 2` e `1 2`), poi `python scripts/select_lr.py --sweep $WS/runs/sweep` | solo validazione |
| 18 | `tmux new -s m0 'bash scripts/run_matrix.sh "1" "1.0 0.3 0.1" 2>&1 \| tee -a logs/matrix_s1.log'` | poi PRIMA `BBS=dinov2 METHODS=full bash scripts/run_matrix.sh "2 3" "1.0"` (riferimento del contrasto primario: 3 seed), poi `"2 3" "1.0"` per il resto, poi `"2 3" "0.3 0.1"` se c'e' tempo |
| 19 | in parallelo (CPU): `ssl_build_set.py select --meta $WS/data/metadata/metadata_dermoscopic.csv --source-map configs/source_map.csv --manifest $M --out $WS/data/ssl --smoke 12000`; `download_images.py --cohort $WS/data/ssl/ssl_smoke.csv --out $WS/data/ssl_images`; `resize_cache.py --src $WS/data/ssl_images --dst $WS/data/ssl_images_256 --side 256`; `ssl_dedup_vs_test.py --ssl-ids $WS/data/ssl/ssl_smoke.csv --ssl-images $WS/data/ssl_images_256 --manifest $M --held-images $WS/data/images_256 --out $WS/data/ssl/exclude_smoke.csv`; `ssl_build_set.py list --ids $WS/data/ssl/ssl_smoke.csv --images $WS/data/ssl_images_256 --exclude $WS/data/ssl/exclude_smoke.csv --out $WS/data/ssl/ssl_smoke.txt` | solo CC-0/CC-BY; esclusi test e ogni collegamento con validazione/test |
| 20 | dopo la matrice, max 45 min: `timeout 45m torchrun --nproc_per_node 1 scripts/ssl_dinov2.py --dinov2-repo $WS/dinov2 --hub-weights ~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth --config-file configs/ssl_dinov2_vitl14.yaml --output-dir $WS/runs/ssl_smoke train.dataset_path=FileList:root=$WS/data/ssl_images_256:list=$WS/data/ssl/ssl_smoke.txt`, con `nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -lms 1000 > logs/ssl_gpu.csv &` | poi `python scripts/ssl_summarize.py --run $WS/runs/ssl_smoke --batch-per-gpu 48 --gpus 1 --gpu-log logs/ssl_gpu.csv` (immagini ORIGINALI/s, memoria, loss); se fallisce due volte: "planned" |

## 5. Chiusura
| # | Comando | Note |
|---|---|---|
| 21 | `python scripts/aggregate.py --frozen $WS/runs/frozen --matrix $WS/runs/matrix --out results` e `python scripts/resource_report.py --ws $WS --out logs/resource_report.json` | tabelle, curve, `primary.md` (in fase A risulta NOT AVAILABLE: corretto, manca dinov2_cpt), risorse |
| 22 | opzionale con 2 GPU: `torchrun --nproc_per_node 1 scripts/ddp_bench.py --img 448 --grad-ckpt --backbone panderm --manifest $M --images $WS/data/images_512`, poi `--nproc_per_node 2 ... --ref-ips <valore 1 GPU>` | scalabilita' |
| 23 | `python scripts/progress.py log "Fine" "..." "..."`, avvisa l'utente: **spegnere il pod** | |

## Fase B (Leonardo, all'hackathon: NON si esegue sul pod)
Gia' pronto e provato sui dati sintetici; da adattare all'ambiente CINECA (scheduler, moduli, percorsi).
- Pre-addestramento: `torchrun --nproc_per_node 4 scripts/ssl_dinov2.py ...` sul pool completo (`ssl_full.csv`), poi
  `ssl_summarize.py --gpus 4`; checkpoint del maestro in `<output-dir>/eval/<iter>/teacher_checkpoint.pth`.
- Stesso budget di tuning: `BBS=dinov2_cpt EXTRA="--cpt-ckpt <ckpt>" bash scripts/lr_sweep.sh`, poi
  `python scripts/select_lr.py --sweep <dir> --backbones dinov2_cpt` (non tocca le scelte gia' fissate).
- Contrasto primario: `BBS=dinov2_cpt METHODS=full EXTRA="--cpt-ckpt <ckpt>" bash scripts/run_matrix.sh "1 2 3" "1.0"`,
  `frozen_baseline.py --backbone dinov2_cpt --cpt-ckpt <ckpt> ...`, poi `aggregate.py` -> `results/primary.md`.
- Scalabilita' del fine-tuning: `ddp_bench.py`; del pre-addestramento: stesso comando con 1, 4, 8 GPU, workload fisso
  (strong) e per-GPU fisso (weak), annotando batch globale ed eventuale accumulo.

## Scelte mie, fissate prima dei risultati (da confermare: regola c)
- Duplicati: dHash <= 6 bit. Validazione 15% dei gruppi di sviluppo, seed 0; frazioni annidate 10/30/100%.
- `configs/hparams.json`: epoche per frazione 30/20/10, batch 128, AdamW, wd 0.05, warmup 1 epoca, cosine, drop path 0.1,
  layer decay 0.75 (full), LoRA r16 alpha 32 su qkv e proj, bf16. Griglie lr: full {1e-4, 2e-4, 5e-4}, LoRA {3e-4, 1e-3, 3e-3}.
- Frozen: StandardScaler + regressione logistica C=1, class_weight balanced. Normalizzazione PanDerm: ImageNet, DA VERIFICARE.
- `configs/ssl_dinov2_vitl14.yaml`: valori del test breve, da rivedere con i mentor.

## Da verificare sul pod
- CLI e formato CSV di `run_class_finetuning.py` (per `profile_original.sh`).
- Chiavi del checkpoint PanDerm (lo script si ferma se mancano chiavi diverse da `head.*`/`fc_norm.*`); se serve
  `PANDERM_TRUST_PICKLE=1`, solo dopo averne registrato lo SHA-256.
- Chiavi della configurazione DINOv2 (`ssl_default_config.yaml`) e `student.pretrained_weights`.
