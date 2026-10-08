#!/usr/bin/env bash
# Pod setup: GPU check, dependencies, PanDerm and DINOv2 repos + weights, isic-cli, DALI.
# Run from the project dir (/workspace/project):  bash scripts/00_setup_pod.sh 2>&1 | tee -a logs/setup.log
# Safe to re-run: every step is skipped if already done.
set -euo pipefail
WS=${WS:-/workspace}
PROJ=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$PROJ/logs" "$WS/checkpoints" "$WS/data" "$WS/runs"
echo "== $(date -u) setup start"
nvidia-smi | tee "$PROJ/logs/nvidia-smi.txt"
df -h "$WS" | tee "$PROJ/logs/disk.txt"

# ---- freeze the pod's torch stack: every pip install below uses these constraints, so nothing can replace it
python - <<'EOF' > "$PROJ/logs/constraints.txt"
import importlib
for m, pkg in [("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy")]:
    try:
        print(f"{pkg}=={importlib.import_module(m).__version__.split('+')[0]}")
    except Exception:
        pass
EOF
cat "$PROJ/logs/constraints.txt"
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")
PIP="pip install -c $PROJ/logs/constraints.txt"

# ---- project under git, so every run records a commit hash
if [ ! -d "$PROJ/.git" ]; then
  git -C "$PROJ" init -q
  git -C "$PROJ" add -A && git -C "$PROJ" -c user.name=pod -c user.email=pod@local commit -qm "scripts as uploaded"
fi

# ---- Python dependencies (no blanket install of third-party requirements files: they may pin old versions)
$PIP "timm==0.9.16" gdown isic-cli scikit-learn pandas pillow nvidia-ml-py omegaconf fvcore iopath submitit einops
CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
$PIP --extra-index-url https://pypi.nvidia.com "nvidia-dali-cuda${CUDA_MAJOR}0" \
  || echo "!! DALI install failed: --loader dali unavailable (log it in PROGRESS.md)"
# xformers is required by DINOv2 multi-crop training (nested tensors); --no-deps so it cannot touch torch
pip install --no-deps xformers || true
python -c "import xformers.ops; print('xformers OK')" \
  || echo "!! xformers not usable with torch $TORCH_BEFORE: SSL test needs a matching wheel (log it)"
command -v nsys >/dev/null && nsys --version || echo "nsys not found: use train.py --torch-profile instead"

# ---- PanDerm (code only; CC-BY-NC-ND) and its checkpoint
[ -d "$WS/PanDerm" ] || git clone https://github.com/SiyuanYan1/PanDerm "$WS/PanDerm"
git -C "$WS/PanDerm" rev-parse HEAD | tee "$PROJ/logs/panderm_commit.txt"
cp "$WS/PanDerm/classification/requirements.txt" "$PROJ/logs/panderm_requirements.txt" || true
CKPT="$WS/checkpoints/panderm_ll_data6_checkpoint-499.pth"
if [ ! -f "$CKPT" ]; then
  gdown 1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE -O "$CKPT" \
    || echo "!! gdown failed (Drive quota?): download it in a browser and upload it to $CKPT"
fi
[ -f "$CKPT" ] && { ls -l "$CKPT"; sha256sum "$CKPT" | tee "$PROJ/logs/panderm_ckpt_sha256.txt"; }
# PanDerm model module must import with the installed packages; if not, install ONLY the missing module with $PIP
(cd "$WS/PanDerm/classification" && python -c "import models.modeling_finetune as m; print('PanDerm import OK:', hasattr(m, 'PanDerm_Large_FT'))") \
  || echo "!! PanDerm model import failed: read the ModuleNotFoundError above and install that package with: $PIP <name>"

# ---- DINOv2 (Apache-2.0): code for hub loading + official training code, and ViT-L/14 weights
[ -d "$WS/dinov2" ] || git clone https://github.com/facebookresearch/dinov2 "$WS/dinov2"
git -C "$WS/dinov2" rev-parse HEAD | tee "$PROJ/logs/dinov2_commit.txt"
mkdir -p ~/.cache/torch/hub/checkpoints
W=~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth
[ -f "$W" ] || wget -q -O "$W" https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
sha256sum "$W" | tee "$PROJ/logs/dinov2_vitl14_sha256.txt"
python -c "import torch; m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14'); print('DINOv2 hub OK', m.embed_dim)"

# ---- final checks
[ "$(python -c 'import torch; print(torch.__version__)')" = "$TORCH_BEFORE" ] || { echo "!! torch changed during setup"; exit 1; }
python -c "import timm; assert timm.__version__ == '0.9.16', timm.__version__; print('timm', timm.__version__)"
isic --version || echo "!! isic-cli not found"
pip freeze > "$PROJ/logs/pip_freeze_setup.txt"
echo "== $(date -u) setup done. Next: bash scripts/preflight.sh"
