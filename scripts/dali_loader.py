"""Training input pipeline with JPEG decoding on the GPU (NVIDIA DALI, nvJPEG).

Same sampling as the torchvision path (class-balanced sampling with replacement, seeded), same augmentation
family: random-area crop (area 0.6-1.0, aspect 0.85-1.18) fused with decoding, resize to img x img,
horizontal/vertical flips, brightness/contrast/saturation jitter +-10%, normalization.
Not bit-identical to torchvision (different resampling and crop implementation): any accuracy comparison between
loaders must be declared as such. Validation and test always use the torchvision pipeline.

Install: pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120   (match the pod's CUDA major)
"""
import os

import numpy as np
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from backbones import NORM
from data import CROP_RATIO, CROP_SCALE, JITTER


class _Source:
    """Batch-level external source: draws an epoch of indices (class-balanced), yields encoded bytes + labels."""

    def __init__(self, files, labels, batch, n_samples, seed, weighted):
        self.files, self.labels, self.batch = files, np.asarray(labels), batch
        self.n_batches = n_samples // batch
        w = (1.0 / np.bincount(self.labels)[self.labels]) if weighted else np.ones(len(files))
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        self.idx = self.rng.choice(len(self.files), size=self.n_batches * self.batch, replace=True, p=self.p)
        self.k = 0
        return self

    def __next__(self):
        if self.k >= self.n_batches:
            raise StopIteration
        sel = self.idx[self.k * self.batch:(self.k + 1) * self.batch]
        self.k += 1
        data = [np.fromfile(self.files[i], dtype=np.uint8) for i in sel]
        lab = [np.array([self.labels[i]], dtype=np.int64) for i in sel]
        return data, lab


@pipeline_def
def _train_pipe(source, img, mean, std):
    jpegs, labels = fn.external_source(source=source, num_outputs=2, batch=True, cycle="raise",
                                       dtype=[types.UINT8, types.INT64])
    x = fn.decoders.image_random_crop(jpegs, device="mixed", output_type=types.RGB,
                                      random_area=list(CROP_SCALE), random_aspect_ratio=list(CROP_RATIO))
    x = fn.resize(x, resize_x=img, resize_y=img, antialias=True)
    x = fn.flip(x, horizontal=fn.random.coin_flip(), vertical=fn.random.coin_flip())
    lo, hi = 1 - JITTER, 1 + JITTER
    x = fn.color_twist(x, brightness=fn.random.uniform(range=[lo, hi]), contrast=fn.random.uniform(range=[lo, hi]),
                       saturation=fn.random.uniform(range=[lo, hi]))
    x = fn.crop_mirror_normalize(x, dtype=types.FLOAT, output_layout="CHW",
                                 mean=[m * 255 for m in mean], std=[s * 255 for s in std])
    return x, labels.gpu()


class DaliTrainLoader:
    def __init__(self, df, image_dir, backbone, img, micro_batch, seed, weighted, n_samples, threads=8):
        files = [os.path.join(image_dir, f) for f in df.file]
        self.src = _Source(files, df.y.values, micro_batch, n_samples, seed, weighted)
        mean, std = NORM[backbone]
        dev = torch.cuda.current_device()
        self.pipe = _train_pipe(source=self.src, img=img, mean=mean, std=std, batch_size=micro_batch,
                                num_threads=threads, device_id=dev, seed=seed, prefetch_queue_depth=3)
        self.pipe.build()
        self.it = DALIGenericIterator([self.pipe], ["x", "y"], last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)
        self.started = False

    def epoch(self):
        """Yields exactly n_batches micro-batches. The trainer consumes all of them without hitting StopIteration,
        so the end of the previous epoch is reached here (auto_reset restarts the source with new indices)."""
        if self.started:
            try:
                next(self.it)
                raise RuntimeError("DALI epoch longer than expected")
            except StopIteration:
                pass
        self.started = True
        for _ in range(self.src.n_batches):
            b = next(self.it)
            yield b[0]["x"], b[0]["y"].view(-1).long()
