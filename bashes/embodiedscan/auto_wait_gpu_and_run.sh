#!/bin/bash
# Wait for GPU memory to be available, then launch the conceptgraph pipeline.
# Checks every 5 minutes if at least 4 GPUs have >= 15GB free.
set -euo pipefail

ROOT_DIR="/home/ysh/codecase/3DVLMReasoning"
LOG="/tmp/es_auto_gpu_wait.log"
MIN_FREE_MIB=15000
MIN_GPUS=4

exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date)] Waiting for GPU memory ==="
echo "=== Need ${MIN_GPUS}+ GPUs with >= ${MIN_FREE_MIB} MiB free ==="

while true; do
    # Count GPUs (excluding GPU 1) with enough free memory
    ready=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' -v min="${MIN_FREE_MIB}" '{
            gsub(/ /, "", $1); gsub(/ /, "", $2);
            if ($1 != 1 && $2+0 >= min+0) count++
        } END {print count+0}')

    echo "[$(date '+%H:%M:%S')] GPUs ready: ${ready}/${MIN_GPUS}"

    if (( ready >= MIN_GPUS )); then
        echo "=== [$(date)] GPUs available! Launching pipeline ==="
        break
    fi

    sleep 300  # 5 minutes
done

# Build the GPU list from available GPUs
GPU_LIST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v min="${MIN_FREE_MIB}" '{
        gsub(/ /, "", $1); gsub(/ /, "", $2);
        if ($1 != 1 && $2+0 >= min+0) printf "%s,", $1
    }' | sed 's/,$//')

echo "[INFO] Using GPUs: ${GPU_LIST}"

cd "${ROOT_DIR}"

# Clean old queue state
rm -rf bashes/logs/embodiedscan_batch/queues/

PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
GSA_PATH=/home/ysh/codecase/concept-graphs/Grounded-Segment-Anything \
MIN_FREE_GB=20 \
GPU_LIST="${GPU_LIST}" \
bash bashes/embodiedscan/launch_multi_gpu.sh

echo "=== [$(date)] Pipeline launched ==="
echo "Monitor with: tmux ls | grep es_batch"
