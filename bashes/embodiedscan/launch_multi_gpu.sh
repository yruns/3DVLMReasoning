#!/usr/bin/env bash
# Launch parallel tmux workers across multiple GPUs for EmbodiedScan batch processing.
#
# For each GPU in GPU_LIST, creates a tmux session that runs
# run_batch_pipeline.sh with WORKER_MODE=queue so that scenes
# are claimed from a shared lock-file queue (no duplicated work).
#
# Usage:
#   bash bashes/embodiedscan/launch_multi_gpu.sh
#   GPU_LIST=0,2,3 bash bashes/embodiedscan/launch_multi_gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-${ROOT_DIR}/data/embodiedscan/scannet}"
ES_CONFIG_PATH="${ES_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/embodiedscan.yaml}"
ES_PROCESS_STRIDE="${ES_PROCESS_STRIDE:-2}"
ES_PRUNE_DETECTIONS="${ES_PRUNE_DETECTIONS:-1}"
SCENE_LIST="${SCENE_LIST:-${ROOT_DIR}/data/embodiedscan/val_scannet_scenes.txt}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-15000}"
GPU_WAIT_SECS="${GPU_WAIT_SECS:-30}"
SCENE_LIMIT="${SCENE_LIMIT:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
WORKER_MODE="${WORKER_MODE:-queue}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/bashes/logs/embodiedscan_batch}"
QUEUE_ROOT="${QUEUE_ROOT:-${LOG_DIR}/queues}"
QUEUE_DIR="${QUEUE_DIR:-}"
# Skip GPU 1 (broken)
GPU_LIST="${GPU_LIST:-0,2,3,4,5,6,7}"
SESSION_PREFIX="${SESSION_PREFIX:-es_batch}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${LOG_DIR}" "${QUEUE_ROOT}"

if [[ -z "${QUEUE_DIR}" ]]; then
    QUEUE_DIR="${QUEUE_ROOT}/embodiedscan_${RUN_STAMP}"
fi
mkdir -p "${QUEUE_DIR}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "[ERROR] No GPUs selected."
    exit 1
fi

echo "[INFO] Launching ${#GPUS[@]} workers on GPUs: ${GPUS[*]}"
echo "[INFO] worker_mode=${WORKER_MODE}"
echo "[INFO] queue_dir=${QUEUE_DIR}"
echo "[INFO] scene_list=${SCENE_LIST}"

for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    session_name="${SESSION_PREFIX}_g${gpu}"
    tmux kill-session -t "${session_name}" 2>/dev/null || true
    tmux new-session -d -s "${session_name}" \
        "bash -lc 'cd ${ROOT_DIR} && \
        PYTHONPATH=${ROOT_DIR}:\${PYTHONPATH:-} \
        GSA_PATH=/home/ysh/codecase/concept-graphs/Grounded-Segment-Anything \
        ES_SCANNET_ROOT=${ES_SCANNET_ROOT} \
        ES_CONFIG_PATH=${ES_CONFIG_PATH} \
        ES_PROCESS_STRIDE=${ES_PROCESS_STRIDE} \
        ES_PRUNE_DETECTIONS=${ES_PRUNE_DETECTIONS} \
        SCENE_LIST=${SCENE_LIST} \
        MIN_FREE_GB=${MIN_FREE_GB} \
        FREE_MEM_MIN_MIB=${FREE_MEM_MIN_MIB} \
        GPU_WAIT_SECS=${GPU_WAIT_SECS} \
        SCENE_LIMIT=${SCENE_LIMIT} \
        STOP_ON_ERROR=${STOP_ON_ERROR} \
        WORKER_MODE=${WORKER_MODE} \
        QUEUE_DIR=${QUEUE_DIR} \
        RUN_STAMP=${RUN_STAMP} \
        CUDA_DEVICE=${gpu} \
        WORKER_INDEX=${idx} \
        WORKER_COUNT=${#GPUS[@]} \
        WORKER_TAG=g${gpu} \
        LOG_DIR=${LOG_DIR} \
        bash bashes/embodiedscan/run_batch_pipeline.sh'"
    echo "[LAUNCHED] ${session_name} -> GPU ${gpu}"
done

tmux ls | grep "${SESSION_PREFIX}" || true
