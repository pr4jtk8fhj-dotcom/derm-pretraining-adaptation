# Continued pretraining and controlled adaptation of dermoscopy models on public ISIC data

Research code for a diagnostic proxy (benign vs malignant, histopathology-confirmed) on public ISIC dermoscopic
images, evaluated on institutions excluded from model development.

- **Primary question**: does continued self-supervised pretraining of DINOv2 ViT-L/14 on permissively licensed
  (CC-0, CC-BY) ISIC dermoscopic images improve external discrimination over the original DINOv2, under the same
  downstream protocol? Pre-specified contrast: full fine-tuning, 224 px, 100% of annotations, paired external AUROC.
- **Secondary**: frozen features, LoRA and full fine-tuning with 10/30/100% of annotations; 224 vs 448 px;
  PanDerm ViT-L/16 as an external dermatology reference.
- **Computational**: input pipeline (CPU, cached, NVIDIA DALI), fused attention, activation memory, distributed
  pretraining and scheduling of independent runs, with measured GPU-hours.

Every reported number comes from logged runs (command, commit, seed, hardware, versions, manifest hash).
Not a medical device; no clinical validation is claimed. No hospital data is used.

- Commands, in order: `scripts/README.md`. End-to-end test on synthetic data: `scripts/smoke/run_smoke.sh`.
- Licences: our code is Apache-2.0 (`LICENSE`, `NOTICE`). Third-party code, weights and images are downloaded at run
  time and not redistributed: see `THIRD_PARTY.md`. Weights derived from PanDerm (CC-BY-NC-ND 4.0) are never published.
- CINECA Open Hackathon 2026 proposal: `docs/DermaLensAI_proposal.pdf`.
- Team: Gaia Arienti, Olimpia Cordeschi, Alessandra Fatone, Marco Tarchi and Gianluca Cividini (Politecnico di Milano,
  Executive MBA), supervised by Alessio Ronchini.

## Preliminary results (phase A, 1x H100 on RunPod)

**Hardware and scope.** All numbers below were measured on **one rented NVIDIA H100 80GB HBM3 SXM (RunPod, eu-fr-1)**,
8-9 October 2026, with torch 2.8.0+cu128 and timm 0.9.16. They are **not** measurements on Leonardo (A100 64 GB). The task is a
**diagnostic proxy** (benign vs malignant, histopathology-confirmed), not validated triage. Each run's `results.json` records
the commit it was run with (`project_commit`). Five runs carry `428d6b8-dirty` (the four LR-grid-extension trials and the
PanDerm LoRA 100% seed-1 matrix run) and one benchmark run carries `2eae334-dirty`: the code was at that commit and the
working tree had uncommitted changes.

**Primary contrast** (`dinov2_cpt` − `dinov2`, full fine-tuning, 224 px, 100%): **not available** in phase A, because the continued-pretraining
encoder is phase-B work. No primary result is claimed.

**Data.** Public ISIC Archive only. The development set comes from the non-test institutions: 30,250 training images (2,358 patients) and
5,042 validation images. The external test is **HIBA + MSKCC**: 2,784 images, 1,014 malignant and 1,770 benign, 1,455 patients.

**Learning rate.** Chosen on validation only (30% of annotations, seed 1). Every arm tried the same number of values (4).
Selected values: LoRA r16 3e-4 for both backbones (interior of the grid); full fine-tuning 1e-3 for PanDerm and 5e-5 for DINOv2, both **at the edge of the grid**.

**External test, AUROC** (mean ± SD across seeds; decision threshold locked on validation at 95% sensitivity):

| px | Backbone | Frozen (10/30/100%) | LoRA r16 10% / 30% / 100% | Full 10% / 30% / 100% |
|---|---|---|---|---|
| 224 | DINOv2 ViT-L/14 | 0.697 / 0.763 / 0.802 | 0.855±0.005 / 0.862±0.007 / 0.885±0.002 | 0.848±0.006 / 0.859±0.006 / 0.879±0.002 |
| 224 | PanDerm ViT-L/16 | 0.791 / 0.803 / 0.831 | 0.855±0.005 / 0.863±0.004 / 0.884±0.011 | 0.840±0.007 / 0.846±0.003 / 0.862±0.003 |
| 448 | DINOv2 | – | – | 100%: 0.892±0.001 (3 seeds) |
| 448 | PanDerm | – | – | 100%: 0.848 (1 seed, preliminary) |

Frozen probes have 1 seed. Fine-tuned arms at 224 px have 3 seeds each.

**Paired comparisons** (AUROC differences on the same images, patient-clustered bootstrap, 1,000 resamples):
- LoRA and full fine-tuning beat frozen features in all 36 comparisons; every confidence interval excludes 0.
- Full fine-tuning minus LoRA ranges from −0.033 to 0.000. LoRA was equal or better.
- DINOv2 full at 448 px minus 224 px: +0.011 to +0.016, with all 3 confidence intervals excluding 0. PanDerm (1 seed): −0.013.

**Results per test institution.** MSKCC is lower than HIBA in every configuration. Example: DINOv2 full at 224 px, 100%, seed 1: HIBA 0.912 [0.886–0.935],
MSKCC 0.819 [0.794–0.842].

**Compute** (H100, RunPod):
- **Original PanDerm `run_class_finetuning.py`**, unmodified, 1 epoch on original images: 368 img/s, 23.4% GPU idle, 39.5 GB peak allocated.
- **Our pipeline** (PanDerm full, 224 px, cached images): 1 epoch in 77.1 s at 10.6% GPU idle. The same pipeline takes 124.0 s on original JPEGs and 176.3 s with DALI GPU decoding (slower here; to be investigated). With fused SDPA attention it takes 70.1 s and 33.9 GB.
- **448 px full fine-tuning** (gradient checkpointing): DINOv2 53–55 min/run, 24.1 GB; PanDerm 108 min/run, 35.7 GB. That is 0.90 and 1.80 GPU-h, against 0.19 and 0.21 GPU-h at 224 px.
- **Continued-pretraining smoke test** of DINOv2 (2,000 iterations, batch 48, 1 GPU): 170 original images/s, 47.2 GiB peak. Loss went from 14.03 to 12.88 with no non-finite values. This is engineering evidence only, not a trained model.
- **GPU-hours:** evaluated runs 9.69 GPU-h, LR sweep 2.18 GPU-h.

**Limitations**
- **Test institutions.** MSKCC patient-ID coverage is 63.5% institution-wide, so the test keeps only the 2,045 MSKCC images that have a patient ID. The other 1,174 are excluded everywhere.
- **Overlap with PanDerm.** HIBA is a benchmark of the PanDerm paper. MSKCC patients may appear in ISIC 2024, which is part of PanDerm's pretraining data.
- **Decision threshold.** The threshold locked on validation gives 0.87–0.96 sensitivity on the external test, often below the 95% target.
- **Learning-rate grid.** For full fine-tuning, the best LR is at the grid edge for both backbones even after a one-step extension. The optimum may lie outside the grid, which affects the PanDerm-vs-DINOv2 full comparison. PanDerm is an external reference, and its differences from DINOv2 are not attributable to pretraining alone.
- **448 px.** Hyperparameters were reused from 224 px; there was no 448 px sweep. PanDerm at 448 px has 1 seed. The 224 vs 448 px comparison is not compute-matched.
- **Continued-pretraining pool.** The pool has 45,742 CC-0/CC-BY images. 40% are attributed "anonymous" and could include test-institution images or test patients, beyond the 47 near-duplicates already removed.
- **Not yet done:** DDP scaling and the `dinov2_cpt` encoder (phase B).
