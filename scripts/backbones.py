"""Backbones (PanDerm ViT-L/16, original DINOv2 ViT-L/14, continued-pretrained DINOv2 "dinov2_cpt"), LoRA,
optional fused attention for PanDerm.

LoRA design: PanDerm's Attention.forward calls F.linear(x, weight=self.qkv.weight, bias=qkv_bias) directly,
which bypasses a peft wrapper on `qkv`. LoRALinear therefore exposes `.weight` as a property returning the
merged weight W + (alpha/r) * B @ A. Any code that reads `.weight` (PanDerm) or calls the module (DINOv2, proj)
sees the adapted weight, and PanDerm's code is NOT modified. With B = 0 (initialization) the merged weight is
exactly W, so outputs are bit-identical to the original (checked in test_backbones.py).
"""
import importlib
import math
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Normalization per backbone. PanDerm: VERIFY against its classification code before the matrix
# (grep -rn "mean" PanDerm/classification) and update here if it differs.
NORM = {"panderm": (IMAGENET_MEAN, IMAGENET_STD), "dinov2": (IMAGENET_MEAN, IMAGENET_STD),
        "dinov2_cpt": (IMAGENET_MEAN, IMAGENET_STD)}
BACKBONES = ("panderm", "dinov2", "dinov2_cpt")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.base_weight = base.weight
        self.base_weight.requires_grad_(False)
        if base.bias is not None:
            self.bias = base.bias
            self.bias.requires_grad_(False)
        else:
            self.register_parameter("bias", None)
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=base.weight.dtype, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base_weight + (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, scaling={self.scaling}"


def apply_lora(model, r, alpha, targets=("qkv", "proj")):
    """Freeze the whole backbone, wrap attn.qkv and attn.proj of every block with LoRA."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for blk in model.blocks:
        for t in targets:
            lin = getattr(blk.attn, t)
            assert isinstance(lin, nn.Linear), f"{t} is {type(lin)}"
            setattr(blk.attn, t, LoRALinear(lin, r, alpha))
            n += 1
    assert n == len(model.blocks) * len(targets)
    return n


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


# ---------------------------------------------------------------- PanDerm
def _import_panderm(repo):
    cls_dir = os.path.join(repo, "classification")
    if cls_dir not in sys.path:
        sys.path.insert(0, cls_dir)
    return importlib.import_module("models.modeling_finetune")


def load_checkpoint_file(path):
    """weights_only=True first (no code execution). The Drive checkpoint may contain pickled objects
    (e.g. argparse.Namespace); then set PANDERM_TRUST_PICKLE=1 explicitly after checking the file's SHA-256."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if os.environ.get("PANDERM_TRUST_PICKLE") != "1":
            raise RuntimeError(f"weights_only load failed ({type(e).__name__}: {e}). "
                               "Set PANDERM_TRUST_PICKLE=1 to allow a full unpickle of this checkpoint.") from e
        return torch.load(path, map_location="cpu", weights_only=False)


def _resize_pos_embed(pe, n_new):
    """(1, 1 + g*g, C) -> (1, 1 + n_new, C), bicubic on the patch grid; the class token is kept."""
    cls, grid = pe[:, :1], pe[:, 1:]
    g_old, g_new = int(math.isqrt(grid.shape[1])), int(math.isqrt(n_new))
    grid = grid.reshape(1, g_old, g_old, -1).permute(0, 3, 1, 2).float()
    grid = F.interpolate(grid, size=(g_new, g_new), mode="bicubic", align_corners=False)
    return torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, g_new * g_new, -1).to(pe.dtype)], 1)


def _resize_rel_pos_table(tab, n_new):
    """BEiT table ((2W-1)^2 + 3, heads): bicubic on the (2W-1)x(2W-1) part, the 3 class-token entries are kept.
    (BEiT uses a geometric-progression interpolation; bicubic is a simpler choice: declare it.)"""
    extra = 3
    s_old = int(math.isqrt(tab.shape[0] - extra))
    s_new = int(math.isqrt(n_new - extra))
    t = tab[:-extra].reshape(s_old, s_old, -1).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(s_new, s_new), mode="bicubic", align_corners=False)
    return torch.cat([t[0].permute(1, 2, 0).reshape(s_new * s_new, -1).to(tab.dtype), tab[-extra:]], 0)


def build_panderm(repo, ckpt, drop_path=0.0, img_size=224, allow_missing=r"^(head|fc_norm)\.", verbose=True):
    """PanDerm_Large_FT with num_classes=0 (head = Identity): forward(x) returns pooled features.
    img_size != 224: position embeddings and relative-position tables are interpolated from the 224 checkpoint."""
    mf = _import_panderm(repo)
    ctor = getattr(mf, "PanDerm_Large_FT")
    kw = dict(pretrained=False, drop_path_rate=drop_path)
    if img_size != 224:
        kw["img_size"] = img_size
    try:
        model = ctor(num_classes=0, **kw)
    except Exception as e:  # some ViT variants do not accept num_classes=0
        print(f"[PanDerm] num_classes=0 failed ({e}); building with 2 classes and replacing head with Identity")
        model = ctor(num_classes=2, **kw)
        model.head = nn.Identity()
    if not isinstance(model.head, nn.Identity):
        raise RuntimeError(f"[PanDerm] expected an Identity head, got {model.head}")
    raw = load_checkpoint_file(ckpt)
    sd = raw
    for k in ("model", "module", "state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            if verbose:
                print(f"[PanDerm] using checkpoint['{k}']")
            break
    if isinstance(raw, dict) and verbose:
        print(f"[PanDerm] top-level checkpoint keys: {list(raw.keys())[:20]}")
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    if sum(k.startswith("encoder.") for k in sd) > len(sd) / 2:
        print("[PanDerm] keeping only 'encoder.*' keys and stripping the prefix")
        sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}

    msd = model.state_dict()
    # shared relative position bias in the checkpoint -> per-block tables in the model (BEiT convention)
    shared = "rel_pos_bias.relative_position_bias_table"
    if shared in sd and shared not in msd:
        for i in range(len(model.blocks)):
            k = f"blocks.{i}.attn.relative_position_bias_table"
            if k in msd:
                sd[k] = sd[shared].clone()
        sd.pop(shared)
        print("[PanDerm] copied shared rel_pos_bias table into every block")
    resized = []
    for k in list(sd):
        if k in msd and sd[k].shape != msd[k].shape:
            if k == "pos_embed":
                sd[k] = _resize_pos_embed(sd[k], msd[k].shape[1] - 1)
            elif k.endswith("relative_position_bias_table"):
                sd[k] = _resize_rel_pos_table(sd[k], msd[k].shape[0])
            elif k.endswith("relative_position_index"):  # buffer rebuilt by the model for the new grid
                del sd[k]
                continue
            else:
                continue
            resized.append(k)
    if resized:
        print(f"[PanDerm] interpolated to img_size={img_size}: {len(resized)} tensors "
              f"({sorted(set(r.split('.')[-1] for r in resized))})")
    shape_mismatch = [k for k in sd if k in msd and sd[k].shape != msd[k].shape]
    if shape_mismatch:
        raise RuntimeError(f"[PanDerm] shape mismatch: "
                           f"{[(k, tuple(sd[k].shape), tuple(msd[k].shape)) for k in shape_mismatch]}")

    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not k.endswith("relative_position_index")]
    print(f"[PanDerm] missing keys ({len(missing)}): {missing}")
    print(f"[PanDerm] unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys}")
    bad = [k for k in missing if not re.match(allow_missing, k)]
    if bad:
        raise RuntimeError(f"[PanDerm] missing keys outside the allow-list {allow_missing!r}: {bad}")
    model.load_report = {"missing": missing, "unexpected": list(res.unexpected_keys), "img_size": img_size,
                         "interpolated": resized}
    model.feat_dim = getattr(model, "num_features", None) or model.embed_dim
    return model


# ---------------------------------------------------------------- DINOv2
def build_dinov2(drop_path=0.0, name="dinov2_vitl14"):
    """CLS token after the final norm; torch.hub code + weights (Apache 2.0).
    Any input size that is a multiple of 14 works: DINOv2 interpolates its position embeddings at run time."""
    model = torch.hub.load("facebookresearch/dinov2", name, drop_path_rate=drop_path)
    model.feat_dim = model.embed_dim
    model.load_report = {"source": f"torch.hub facebookresearch/dinov2 {name}"}
    return model


def load_cpt_backbone_state(path):
    """Backbone weights from a DINOv2 training output: eval/<iter>/teacher_checkpoint.pth ({"teacher": ...}) or a
    trainer checkpoint ({"model": ...}). The TEACHER backbone is used (as in DINOv2 evaluation)."""
    raw = torch.load(path, map_location="cpu", weights_only=False)  # our own training output
    sd = raw.get("teacher", raw.get("model", raw)) if isinstance(raw, dict) else raw
    for prefix in ("teacher.backbone.", "backbone."):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            return sub, prefix
    raise RuntimeError(f"no teacher backbone weights found in {path}; keys: {list(sd)[:10]}")


def build_dinov2_cpt(ckpt, drop_path=0.0):
    """Original DINOv2 ViT-L/14 architecture with weights from our continued pretraining (strict key check)."""
    model = build_dinov2(drop_path=drop_path)
    sd, prefix = load_cpt_backbone_state(ckpt)
    if sd["pos_embed"].shape != model.pos_embed.shape:  # trained at 224 (16x16 grid); hub module stores 37x37
        sd["pos_embed"] = _resize_pos_embed(sd["pos_embed"], model.pos_embed.shape[1] - 1)
    model.load_state_dict(sd, strict=True)
    model.load_report = {"source": f"continued pretraining checkpoint {ckpt}", "prefix": prefix, "strict": True}
    return model


def build_backbone(name, drop_path=0.0, panderm_repo=None, panderm_ckpt=None, img_size=224, cpt_ckpt=None):
    if name == "panderm":
        return build_panderm(panderm_repo, panderm_ckpt, drop_path=drop_path, img_size=img_size)
    if name == "dinov2":
        return build_dinov2(drop_path=drop_path)
    if name == "dinov2_cpt":
        if not cpt_ckpt:
            raise ValueError("dinov2_cpt needs --cpt-ckpt (teacher checkpoint of the continued pretraining)")
        return build_dinov2_cpt(cpt_ckpt, drop_path=drop_path)
    raise ValueError(name)


def apply_grad_checkpointing(model):
    """Recompute each block's activations in backward (less memory, ~30% more compute: measure it)."""
    from torch.utils.checkpoint import checkpoint
    for blk in model.blocks:
        fwd = blk.forward

        def wrapped(*args, _f=fwd, **kw):
            if torch.is_grad_enabled():
                return checkpoint(_f, *args, use_reentrant=False, **kw)
            return _f(*args, **kw)

        blk.forward = wrapped
    return len(model.blocks)


class Classifier(nn.Module):
    def __init__(self, backbone, n_classes=2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feat_dim, n_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------- optimizer groups
def param_groups(model, lr, weight_decay, layer_decay=None):
    """AdamW groups; no weight decay on 1-D params, tokens, pos_embed, rel-pos tables.
    With layer_decay, block i gets lr * decay**(depth + 1 - (i + 1)); embeddings layer 0; head/norms depth+1."""
    depth = len(model.backbone.blocks)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim == 1 or any(s in n for s in ["cls_token", "pos_embed", "mask_token", "register_tokens",
                                                       "relative_position_bias_table", "q_bias", "v_bias"])
        m = re.search(r"blocks\.(\d+)\.", n)
        if m:
            layer = int(m.group(1)) + 1
        elif any(s in n for s in ["patch_embed", "cls_token", "pos_embed", "mask_token", "register_tokens"]):
            layer = 0
        else:
            layer = depth + 1
        scale = layer_decay ** (depth + 1 - layer) if layer_decay else 1.0
        key = (layer, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * scale, "lr_scale": scale,
                           "weight_decay": 0.0 if no_wd else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())


# ---------------------------------------------------------------- fused attention (optimization step)
def patch_panderm_sdpa(model):
    """Replace PanDerm Attention.forward with F.scaled_dot_product_attention + additive bias.
    Reimplements the BEiT-style forward; test_backbones.py --sdpa checks outputs and gradients against the
    original. If the attribute layout differs from what is expected below, this raises instead of guessing."""
    n = 0
    for blk in model.blocks:
        attn = blk.attn
        need = ["qkv", "num_heads", "scale", "proj", "proj_drop", "attn_drop"]
        missing = [k for k in need if not hasattr(attn, k)]
        if missing:
            raise RuntimeError(f"Attention layout unexpected, missing {missing}")

        def fwd(x, rel_pos_bias=None, _a=attn, **kw):
            if kw:
                raise RuntimeError(f"unexpected kwargs in Attention.forward: {list(kw)}")
            B, N, C = x.shape
            qkv_bias = None
            if getattr(_a, "q_bias", None) is not None:
                qkv_bias = torch.cat((_a.q_bias, torch.zeros_like(_a.v_bias, requires_grad=False), _a.v_bias))
            qkv = F.linear(x, _a.qkv.weight, qkv_bias)
            qkv = qkv.reshape(B, N, 3, _a.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            bias = None
            if getattr(_a, "relative_position_bias_table", None) is not None:
                ws = _a.window_size
                rpb = _a.relative_position_bias_table[_a.relative_position_index.view(-1)].view(
                    ws[0] * ws[1] + 1, ws[0] * ws[1] + 1, -1).permute(2, 0, 1).contiguous()
                bias = rpb.unsqueeze(0)
            if rel_pos_bias is not None:
                bias = rel_pos_bias if bias is None else bias + rel_pos_bias
            if bias is not None:
                bias = bias.to(q.dtype).expand(B, -1, -1, -1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                 dropout_p=_a.attn_drop.p if _a.training else 0.0, scale=_a.scale)
            out = out.transpose(1, 2).reshape(B, N, -1)
            return _a.proj_drop(_a.proj(out))

        attn.forward = fwd
        n += 1
    return n
