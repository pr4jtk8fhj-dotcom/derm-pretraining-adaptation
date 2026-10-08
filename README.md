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

- Commands, in order: `scripts/README.md` (Italian). End-to-end test on synthetic data: `scripts/smoke/run_smoke.sh`.
- Licences: our code is Apache-2.0 (`LICENSE`, `NOTICE`). Third-party code, weights and images are downloaded at run
  time and not redistributed: see `THIRD_PARTY.md`. Weights derived from PanDerm (CC-BY-NC-ND 4.0) are never published.
