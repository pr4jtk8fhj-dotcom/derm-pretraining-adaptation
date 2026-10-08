#!/usr/bin/env bash
# Nsight Systems timeline of ~60 training steps (only if nsys is installed on the pod; otherwise use train.py --torch-profile).
# Usage: bash scripts/profile_nsys.sh <tag> <train.py args...>
#   bash scripts/profile_nsys.sh panderm_full_224_torch --backbone panderm --method full --train-frac 0.1 \
#        --manifest $WS/data/partition/manifest.csv --images $WS/data/images --epochs 1 --no-test --no-save
set -euo pipefail
WS=${WS:-/workspace}
TAG=$1; shift
cd "$(dirname "$0")/.."
mkdir -p logs/nsys
command -v nsys >/dev/null || { echo "nsys not installed"; exit 1; }
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=none --delay=60 --duration=60 \
  --output "logs/nsys/$TAG" --force-overwrite true \
  python scripts/train.py "$@" --out "$WS/runs/nsys" --tag "nsys_$TAG" --force
nsys stats --report cuda_gpu_kern_sum,osrt_sum "logs/nsys/$TAG.nsys-rep" > "logs/nsys/$TAG.stats.txt" || true
echo "timeline: logs/nsys/$TAG.nsys-rep (open in Nsight Systems); summary: logs/nsys/$TAG.stats.txt"
