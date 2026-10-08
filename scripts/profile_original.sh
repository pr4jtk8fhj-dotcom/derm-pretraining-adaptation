#!/usr/bin/env bash
# Profile PanDerm's ORIGINAL, unmodified run_class_finetuning.py for 1 epoch, with GPU utilization logged by nvidia-smi.
#
# Their exact CLI must be taken from PanDerm/classification README / --help (not guessed here):
#   1. bash scripts/profile_original.sh help      -> saves --help and README examples to logs/
#   2. write the command (1 epoch, batch 128, lr 5e-4, weighted sampler, our data/partition/panderm.csv,
#      ORIGINAL full-resolution images in data/images: this is the "before" profile)
#      into logs/original_cmd.sh, then:  bash scripts/profile_original.sh run
# Output: logs/original_profile/{stdout.log, nvidia_smi.csv, time.txt, cmd.sh}. Summarize with:
#   python scripts/summarize_gpu_log.py logs/original_profile/nvidia_smi.csv
set -euo pipefail
WS=${WS:-/workspace}
PROJ=$(cd "$(dirname "$0")/.." && pwd)
CLS="$WS/PanDerm/classification"
OUTD="$PROJ/logs/original_profile"
mkdir -p "$OUTD"

if [ "${1:-}" = help ]; then
  (cd "$CLS" && python run_class_finetuning.py --help) > "$PROJ/logs/panderm_ft_help.txt" 2>&1 || true
  grep -n -A25 -i "run_class_finetuning" "$WS/PanDerm/README.md" "$CLS"/*.md > "$PROJ/logs/panderm_ft_readme_examples.txt" 2>/dev/null || true
  echo "See logs/panderm_ft_help.txt and logs/panderm_ft_readme_examples.txt"; exit 0
fi

[ -f "$PROJ/logs/original_cmd.sh" ] || { echo "write logs/original_cmd.sh first"; exit 1; }
cp "$PROJ/logs/original_cmd.sh" "$OUTD/cmd.sh"
git -C "$WS/PanDerm" status --porcelain > "$OUTD/panderm_git_status.txt"   # must be empty: code unmodified
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,noheader,nounits -lms 500 > "$OUTD/nvidia_smi.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
START=$(date +%s.%N)
(cd "$CLS" && bash "$OUTD/cmd.sh") 2>&1 | tee "$OUTD/stdout.log"
END=$(date +%s.%N)
echo "wall_seconds $(python -c "print(round($END - $START, 1))")" | tee "$OUTD/time.txt"
