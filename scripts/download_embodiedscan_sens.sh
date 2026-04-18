#!/bin/bash
# Download ScanNet .sens files for EmbodiedScan val set (243 scenes)
# Run inside tmux: tmux new-session -d -s sens-download "bash scripts/download_embodiedscan_sens.sh 2>&1 | tee /tmp/sens_download.log"

set -e

PROJ_DIR="/home/ysh/codecase/3DVLMReasoning"
SCENE_LIST="$PROJ_DIR/data/embodiedscan/val_scannet_scenes.txt"
OUT_DIR="$PROJ_DIR/data/embodiedscan/scannet"
DOWNLOAD_SCRIPT="/home/ysh/codecase/concept-graphs/tools/scannet/download-scannet.py"

source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph

total=$(wc -l < "$SCENE_LIST")
count=0
failed=0

echo "=== Starting download of $total scenes ==="
echo "=== Output: $OUT_DIR/scans/ ==="
echo "=== $(date) ==="

while IFS= read -r scene_id; do
    count=$((count + 1))
    sens_file="$OUT_DIR/scans/$scene_id/$scene_id.sens"

    if [[ -f "$sens_file" && $(stat -c%s "$sens_file" 2>/dev/null || echo 0) -gt 0 ]]; then
        echo "[$count/$total] SKIP $scene_id (already exists)"
        continue
    fi

    echo "[$count/$total] Downloading $scene_id ..."
    if python "$DOWNLOAD_SCRIPT" -o "$OUT_DIR" --id "$scene_id" --type .sens 2>&1; then
        echo "[$count/$total] OK $scene_id"
    else
        echo "[$count/$total] FAILED $scene_id"
        failed=$((failed + 1))
    fi
done < "$SCENE_LIST"

echo "=== Done: $count total, $failed failed ==="
echo "=== $(date) ==="
