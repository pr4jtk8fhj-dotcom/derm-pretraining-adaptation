# Overlap between PanDerm's pretraining data and our external test

Source: S. Yan et al., "A multimodal vision foundation model for clinical dermatology", Nature Medicine 31,
2691-2702 (2025); PMC version https://pmc.ncbi.nlm.nih.gov/articles/PMC12353815/ and arXiv 2410.15038v3.
First read on 7 October 2026; the quotes below were re-checked on the PMC version on 9 October 2026.

## What the paper says
- Pretraining uses "2,149,706 unlabeled multimodal skin images" from "4 imaging modalities and 11 data sources",
  including 384,441 dermoscopic images.
- Named dermoscopic sources include the MYM and HOP cohorts ("38,110 dermoscopic images") and the in-house MMT
  dataset ("316,399 dermoscopic images"). The in-house cohorts MYM, HOP, NSSI and ACEMID are Australian.
- ISIC 2024 contributes "352,034 tile images" (total-body-photography crops) to pretraining. The paper does not list
  which contributing institutions those tiles come from.
- HIBA is a downstream benchmark in the paper: "The HIBA dataset is curated from the HIBA data from the ISIC archive,
  containing 1,635 dermoscopic images with melanoma and other classes."
- We did not find an explicit image-level duplicate check between the pretraining data and the benchmarks.

## Consequences for our design (external test: HIBA + patient-identified MSKCC)
- Australian institutions are excluded from our external test (SMDC Sydney, UQ Brisbane), because PanDerm's
  in-house pretraining data are Australian.
- HIBA is allowed in the test but is a PanDerm benchmark: PanDerm's HIBA results are not independent of its
  authors' development choices. The released PanDerm checkpoint is the pretrained one, not fine-tuned on HIBA.
- MSKCC patients may appear in the ISIC 2024 tiles used for PanDerm pretraining. This cannot be verified, because
  patient identifiers are not linkable across collections.
- Image-level overlap is therefore not verifiable for PanDerm. DINOv2 (LVD-142M, web images) has the same limitation.
- PanDerm is used only as an external reference: differences between PanDerm and DINOv2 are not attributable to
  domain pretraining alone. The primary contrast (continued-pretrained DINOv2 vs original DINOv2) does not depend on
  PanDerm.
