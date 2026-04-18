#!/bin/bash
# Monitor sens-download tmux session; once it finishes,
# automatically run: 1) extract posed images  2) launch multi-GPU conceptgraph pipeline
set -euo pipefail

ROOT_DIR="/home/ysh/codecase/3DVLMReasoning"
ES_SCANNET_ROOT="${ROOT_DIR}/data/embodiedscan/scannet"
LOG="/tmp/es_auto_pipeline.log"

exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date)] Waiting for sens-download to finish ==="

while tmux has-session -t sens-download 2>/dev/null; do
    count=$(ls "${ES_SCANNET_ROOT}/scans/" 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] sens-download running — ${count}/243 scenes downloaded"
    sleep 600  # 10 minutes
done

count=$(ls "${ES_SCANNET_ROOT}/scans/" 2>/dev/null | wc -l)
echo "=== [$(date)] sens-download finished — ${count} scenes downloaded ==="

# Check for failures in download log
if [[ -f /tmp/sens_download.log ]]; then
    fail_count=$(grep -c "FAILED" /tmp/sens_download.log 2>/dev/null || echo 0)
    echo "[INFO] Download failures: ${fail_count}"
fi

# ---- Step 1: Extract posed images ----
echo "=== [$(date)] Step 1: Extracting posed images from .sens files ==="
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph

python "${ROOT_DIR}/data/EmbodiedScan_repo/embodiedscan/converter/generate_image_scannet.py" \
    --dataset_folder "${ES_SCANNET_ROOT}" \
    --fast \
    --nproc 8

posed_count=$(ls -d "${ES_SCANNET_ROOT}/posed_images"/scene* 2>/dev/null | wc -l)
echo "=== [$(date)] Extraction done — ${posed_count} scenes in posed_images/ ==="

# ---- Step 2: Prepare all scenes (raw/ directories) ----
echo "=== [$(date)] Step 2: Preparing raw/ directories for all scenes ==="
while IFS= read -r scene_id; do
    if [[ -d "${ES_SCANNET_ROOT}/posed_images/${scene_id}" ]]; then
        python -m src.scripts.prepare_embodiedscan_scene \
            --scene_id "${scene_id}" \
            --data_root "${ES_SCANNET_ROOT}" 2>&1 | tail -1
    else
        echo "[WARN] No posed_images for ${scene_id}, skipping prepare"
    fi
done < "${ROOT_DIR}/data/embodiedscan/val_scannet_scenes.txt"

prepared_count=$(ls -d "${ES_SCANNET_ROOT}"/scene*/raw 2>/dev/null | wc -l)
echo "=== [$(date)] Preparation done — ${prepared_count} scenes prepared ==="

# ---- Step 3: Launch multi-GPU conceptgraph pipeline ----
echo "=== [$(date)] Step 3: Launching multi-GPU ConceptGraph pipeline ==="
cd "${ROOT_DIR}"
bash bashes/embodiedscan/launch_multi_gpu.sh

echo "=== [$(date)] Multi-GPU pipeline launched. Monitor with: ==="
echo "  tmux ls | grep es_batch"
echo "  for s in \$(tmux ls -F '#{session_name}' | grep es_batch); do echo \"=== \$s ===\"; tmux capture-pane -t \$s -p -S -5; done"
