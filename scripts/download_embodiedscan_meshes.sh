#!/usr/bin/env bash
# Download ScanNet meshes for EmbodiedScan val scenes.
# Uses curl with retry for robustness. Same output structure as OpenEQA.
set -euo pipefail

BASE_URL="https://kaldir.vc.in.tum.de/scannet/v2/scans"
OUTPUT_DIR="data/scannetv2"
SCENE_LIST="/tmp/embodiedscan_missing_meshes.txt"
SUFFIX="_vh_clean.ply"
MAX_RETRIES=5
RETRY_DELAY=3

total=$(wc -l < "$SCENE_LIST" | tr -d ' ')
i=0

while IFS= read -r scene_id; do
    i=$((i + 1))
    dest="${OUTPUT_DIR}/${scene_id}/${scene_id}${SUFFIX}"

    if [[ -f "$dest" ]]; then
        size=$(stat -f%z "$dest" 2>/dev/null || stat --printf="%s" "$dest" 2>/dev/null || echo 0)
        if [[ "$size" -gt 1000000 ]]; then
            echo "[$i/$total] SKIP $scene_id (${size} bytes)"
            continue
        fi
    fi

    mkdir -p "${OUTPUT_DIR}/${scene_id}"
    url="${BASE_URL}/${scene_id}/${scene_id}${SUFFIX}"
    echo "[$i/$total] Downloading $scene_id..."

    curl -L --retry "$MAX_RETRIES" --retry-delay "$RETRY_DELAY" \
         -C - -o "$dest" "$url" 2>&1 || {
        echo "  FAILED: $scene_id"
        rm -f "$dest"
    }
done < "$SCENE_LIST"

echo "Done. Checking results..."
ok=0; fail=0
while IFS= read -r scene_id; do
    dest="${OUTPUT_DIR}/${scene_id}/${scene_id}${SUFFIX}"
    if [[ -f "$dest" ]] && [[ $(stat -f%z "$dest" 2>/dev/null || stat --printf="%s" "$dest" 2>/dev/null) -gt 1000000 ]]; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        echo "  MISSING: $scene_id"
    fi
done < "$SCENE_LIST"
echo "OK: $ok  Failed: $fail  Total: $total"
