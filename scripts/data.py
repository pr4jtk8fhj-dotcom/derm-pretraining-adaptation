"""Dataset over a partition manifest + fixed transforms (identical for both backbones except normalization)."""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from backbones import NORM

CROP_SCALE, CROP_RATIO, JITTER = (0.6, 1.0), (0.85, 1.18), 0.1  # shared with dali_loader.py


def resize_for(img):
    return img * 256 // 224  # 224 -> 256, 448 -> 512 (pre-resized caches: images_256, images_512)


def build_transform(backbone, train, img=224):
    mean, std = NORM[backbone]
    if train:
        tf = [T.Resize(resize_for(img)), T.RandomResizedCrop(img, scale=CROP_SCALE, ratio=CROP_RATIO),
              T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
              T.ColorJitter(JITTER, JITTER, JITTER, 0.0)]
    else:
        tf = [T.Resize(resize_for(img)), T.CenterCrop(img)]
    return T.Compose(tf + [T.ToTensor(), T.Normalize(mean, std)])


def read_manifest(path, train_frac=1.0):
    """Manifest with integer label y; training rows restricted to the nested fraction train_frac (0.1, 0.3, 1.0)."""
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    m["y"] = (m.label == "malignant").astype(int)
    if "train_frac" in m.columns:
        tf = pd.to_numeric(m.train_frac, errors="coerce")
        m = m[(m.split != "train") | (tf <= train_frac + 1e-9)].reset_index(drop=True)
    return m


class ManifestDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.files = [os.path.join(image_dir, f) for f in df.file]
        self.y = torch.tensor(df.y.values, dtype=torch.long)
        self.tf = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.y[i], i
