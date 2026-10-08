# Third-party components and licences

This repository contains only our own code (Apache-2.0, see LICENSE). Everything below is downloaded at run time
by the user from the original source and is NOT included or redistributed here.

| Component | Source | Licence | How we use it |
|---|---|---|---|
| DINOv2 code and ViT-L/14 weights | github.com/facebookresearch/dinov2 | Apache-2.0 | Backbone; official training code used unmodified for continued pretraining |
| PanDerm code and ViT-L/16 weights | github.com/SiyuanYan1/PanDerm | CC-BY-NC-ND 4.0 | Comparator only, non-commercial research. Imported from the user's clone; no modified PanDerm weights (fine-tuned checkpoints, LoRA adapters) are ever published or shared |
| ISIC Archive images and metadata | isic-archive.com | Per image: CC-0, CC-BY or CC-BY-NC | Not included. Per-image licence and attribution are recorded in the manifests the scripts generate |
| timm, PyTorch, torchvision, scikit-learn, pandas, Pillow, NVIDIA DALI, xformers | PyPI | Their own open-source licences | Dependencies |

## Release policy for trained models
- Models derived from PanDerm: never released (ND clause).
- DINOv2 continued-pretrained on CC-0 and CC-BY images only: may be released together with the full attribution
  list of the images used (CC-BY requires attribution). Final licence of the weights to be confirmed with the
  university before any release.
- Any model trained with CC-BY-NC images: non-commercial use only, released separately if at all.
- No clinical data of any hospital is used in this repository.
