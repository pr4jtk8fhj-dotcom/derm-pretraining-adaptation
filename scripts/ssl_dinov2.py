#!/usr/bin/env python3
"""Continued self-supervised pretraining of DINOv2 ViT-L/14 on ISIC dermoscopic images.

Uses the OFFICIAL DINOv2 training code (github.com/facebookresearch/dinov2, Apache-2.0) without modifying it.
This launcher only:
  1. registers a file-list dataset ("FileList:root=<dir>:list=<file with one file name per line>");
  2. converts the released hub weights to the training resolution (pos_embed interpolated 518 -> 224 grid) and
     checks with strict=True that they match the architecture of the config (the DINOv2 trainer loads
     pretrained weights with strict=False, which would silently train from random init on a key mismatch).

Phase A smoke test (1 GPU, ~45 min: throughput, memory, loss trend; NOT a finished model):
  torchrun --nproc_per_node 1 scripts/ssl_dinov2.py --dinov2-repo $WS/dinov2 \
     --hub-weights ~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth \
     --config-file configs/ssl_dinov2_vitl14.yaml --output-dir $WS/runs/ssl_smoke \
     train.dataset_path=FileList:root=$WS/data/ssl_images_256:list=$WS/data/ssl/ssl_smoke.txt
"""
import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbones import _resize_pos_embed  # noqa: E402


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, root, list_file, transform=None, target_transform=None):
        with open(list_file) as f:
            self.files = [os.path.join(root, l.strip()) for l in f if l.strip()]
        self.transform, self.target_transform = transform, target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            x = im.convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        t = self.target_transform(0) if self.target_transform is not None else 0
        return x, t


def convert_and_check(hub_path, out_path, img, patch):
    from dinov2.models.vision_transformer import vit_large
    sd = torch.load(hub_path, map_location="cpu", weights_only=True)
    n = (img // patch) ** 2
    if sd["pos_embed"].shape[1] != n + 1:
        print(f"[ssl] pos_embed {tuple(sd['pos_embed'].shape)} -> grid {img // patch}x{img // patch}")
        sd["pos_embed"] = _resize_pos_embed(sd["pos_embed"], n)
    ref = vit_large(patch_size=patch, img_size=img, init_values=1e-5, ffn_layer="mlp", block_chunks=0)
    ref.load_state_dict(sd, strict=True)  # raises on any key/shape mismatch
    torch.save({"model": sd}, out_path)
    print(f"[ssl] converted weights verified (strict=True) and saved to {out_path}")
    return out_path


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dinov2-repo", required=True)
    pre.add_argument("--hub-weights", required=True)
    known, rest = pre.parse_known_args()
    sys.path.insert(0, known.dinov2_repo)

    import dinov2.data.loaders as L
    import dinov2.train.train as T

    orig = L.make_dataset

    def make_dataset(*, dataset_str, transform=None, target_transform=None):
        if dataset_str.startswith("FileList:"):
            kv = dict(p.split("=", 1) for p in dataset_str.split(":")[1:])
            ds = FileListDataset(kv["root"], kv["list"], transform, target_transform)
            print(f"[ssl] FileListDataset: {len(ds)} images from {kv['list']}")
            return ds
        return orig(dataset_str=dataset_str, transform=transform, target_transform=target_transform)

    L.make_dataset = make_dataset
    T.make_dataset = make_dataset  # train.py imported the name directly

    args = T.get_args_parser(add_help=True).parse_args(rest)
    os.makedirs(args.output_dir, exist_ok=True)
    rank0 = int(os.environ.get("RANK", "0")) == 0
    conv = os.path.join(args.output_dir, "init_vitl14_224.pth")
    if rank0 and not os.path.exists(conv):
        convert_and_check(known.hub_weights, conv, 224, 14)
    if torch.distributed.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import time
        while not os.path.exists(conv):
            time.sleep(2)
    args.opts = list(args.opts or []) + [f"student.pretrained_weights={conv}"]
    T.main(args)


if __name__ == "__main__":
    main()
