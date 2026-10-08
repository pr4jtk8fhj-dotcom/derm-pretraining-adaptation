#!/usr/bin/env bash
# Download ONLY metadata (no images): every dermoscopic, histopathology-confirmed image in the ISIC Archive.
set -euo pipefail
WS=${WS:-/workspace}
OUT="$WS/data/metadata"
mkdir -p "$OUT"
QUERY='image_type:dermoscopic AND diagnosis_confirm_type:histopathology'
# the API can time out on long downloads: retry up to 5 times (each attempt restarts the file)
fetch() { local q=$1 out=$2; for k in 1 2 3 4 5; do isic metadata download --search "$q" -o "$out.part" && { mv "$out.part" "$out"; return 0; }; echo "!! attempt $k failed, retrying in 20 s"; sleep 20; done; return 1; }

isic --version > "$OUT/isic_cli_version.txt" 2>&1 || true
isic metadata download --help > "$OUT/isic_metadata_help.txt" 2>&1 || true
isic collection list > "$OUT/collections_list.txt"
echo "$QUERY" > "$OUT/query.txt"
fetch "$QUERY" "$OUT/metadata_all.csv"
echo "rows: $(($(wc -l < "$OUT/metadata_all.csv") - 1))  (preliminary API counts 7 Oct 2026: 20087 malignant + 19245 benign, plus indeterminate)"
sha256sum "$OUT/metadata_all.csv" | tee "$OUT/metadata_sha256.txt"
head -1 "$OUT/metadata_all.csv" | tr ',' '\n' > "$OUT/columns.txt"
echo "columns: $(wc -l < "$OUT/columns.txt") -> $OUT/columns.txt"
# All dermoscopic images (any diagnosis, labelled or not): pool for continued self-supervised pretraining.
fetch 'image_type:dermoscopic' "$OUT/metadata_dermoscopic.csv"
echo "dermoscopic rows: $(($(wc -l < "$OUT/metadata_dermoscopic.csv") - 1))  (preliminary API count 7 Oct 2026: 124961)"
sha256sum "$OUT/metadata_dermoscopic.csv" | tee -a "$OUT/metadata_sha256.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/snapshot_utc.txt"
