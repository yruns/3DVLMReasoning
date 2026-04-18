#!/usr/bin/env bash
# Batch-process multiple EmbodiedScan scenes through the full pipeline.
#
# Supports two worker modes:
#   - "static": scenes are pre-split across workers by WORKER_INDEX/WORKER_COUNT
#   - "queue":  scenes are claimed one-at-a-time from a shared lock-file queue
#
# Usage (standalone):
#   bash bashes/embodiedscan/run_batch_pipeline.sh
#
# Usually invoked by launch_multi_gpu.sh via tmux sessions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    source "${ROOT_DIR}/env_vars.bash"
fi

ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-${ROOT_DIR}/data/embodiedscan/scannet}"
ES_CONFIG_PATH="${ES_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/embodiedscan.yaml}"
ES_PROCESS_STRIDE="${ES_PROCESS_STRIDE:-2}"
ES_PRUNE_DETECTIONS="${ES_PRUNE_DETECTIONS:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SCENE_LIMIT="${SCENE_LIMIT:-}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-50}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/bashes/logs/embodiedscan_batch}"
FREE_MEM_MIN_MIB="${FREE_MEM_MIN_MIB:-15000}"
GPU_WAIT_SECS="${GPU_WAIT_SECS:-30}"
WORKER_MODE="${WORKER_MODE:-static}"
QUEUE_DIR="${QUEUE_DIR:-}"
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
WORKER_TAG="${WORKER_TAG:-g${CUDA_DEVICE}}"
SCENE_LIST="${SCENE_LIST:-${ROOT_DIR}/data/embodiedscan/val_scannet_scenes.txt}"

mkdir -p "${LOG_DIR}" "${ES_SCANNET_ROOT}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAIN_LOG="${LOG_DIR}/embodiedscan_${RUN_STAMP}_${WORKER_TAG}.log"
FAIL_LOG="${LOG_DIR}/embodiedscan_${RUN_STAMP}_${WORKER_TAG}_failures.log"

load_scene_subset() {
    python - "${SCENE_LIST}" "${SCENE_LIMIT}" "${WORKER_INDEX}" "${WORKER_COUNT}" <<'PY'
import sys
from pathlib import Path

scene_list = Path(sys.argv[1])
limit_raw = sys.argv[2]
worker_index = int(sys.argv[3])
worker_count = int(sys.argv[4])
scenes = [line.strip() for line in scene_list.read_text().splitlines() if line.strip()]
if limit_raw:
    scenes = scenes[: int(limit_raw)]
for scene in scenes[worker_index::worker_count]:
    print(scene)
PY
}

count_scenes() {
    python - "${SCENE_LIST}" "${SCENE_LIMIT}" <<'PY'
import sys
from pathlib import Path

scene_list = Path(sys.argv[1])
limit_raw = sys.argv[2]
scenes = [line.strip() for line in scene_list.read_text().splitlines() if line.strip()]
if limit_raw:
    scenes = scenes[: int(limit_raw)]
print(len(scenes))
PY
}

claim_next_scene() {
    python - "${SCENE_LIST}" "${SCENE_LIMIT}" "${QUEUE_DIR}" <<'PY'
import fcntl
import json
import sys
from pathlib import Path

scene_list_path, limit_raw, queue_dir_raw = sys.argv[1:4]
queue_dir = Path(queue_dir_raw)
queue_dir.mkdir(parents=True, exist_ok=True)
lock_path = queue_dir / "queue.lock"
state_path = queue_dir / "state.json"
scenes = [line.strip() for line in Path(scene_list_path).read_text().splitlines() if line.strip()]
if limit_raw:
    scenes = scenes[: int(limit_raw)]

with lock_path.open("a+", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"next_index": 0}
    next_index = int(state.get("next_index", 0))
    if next_index >= len(scenes):
        print("")
    else:
        scene = scenes[next_index]
        state["next_index"] = next_index + 1
        state["total"] = len(scenes)
        state["last_scene"] = scene
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
        print(scene)
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
PY
}

scene_done() {
    local scene_id="$1"
    local scene_path="${ES_SCANNET_ROOT}/${scene_id}/conceptgraph"
    local pcd_file="${scene_path}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_merge20_masksub_post.pkl.gz"
    local vis_file="${scene_path}/indices/visibility_index.pkl"
    local build_info="${scene_path}/indices/build_info.json"
    [[ -f "${pcd_file}" && -f "${vis_file}" && -f "${build_info}" ]]
}

wait_for_gpu_memory() {
    if [[ -z "${FREE_MEM_MIN_MIB}" || "${FREE_MEM_MIN_MIB}" == "0" ]]; then
        return 0
    fi
    local free_mib
    while true; do
        free_mib="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk -F',' -v target="${CUDA_DEVICE}" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($1 + 0) == (target + 0)) {print $2; exit}}')"
        free_mib="${free_mib:-0}"
        echo "[GPU] cuda=${CUDA_DEVICE} free=${free_mib}MiB threshold=${FREE_MEM_MIN_MIB}MiB" | tee -a "${MAIN_LOG}"
        if (( free_mib >= FREE_MEM_MIN_MIB )); then
            return 0
        fi
        sleep "${GPU_WAIT_SECS}"
    done
}

check_free_space() {
    local free_kb
    free_kb="$(df -Pk "${ES_SCANNET_ROOT}" | awk 'NR==2 {print $4}')"
    local free_gb=$((free_kb / 1024 / 1024))
    echo "[SPACE] free=${free_gb}GB threshold=${MIN_FREE_GB}GB" | tee -a "${MAIN_LOG}"
    if (( free_gb < MIN_FREE_GB )); then
        echo "[STOP] Free space below threshold; stopping batch run." | tee -a "${MAIN_LOG}"
        exit 2
    fi
}

process_scene() {
    local scene_id="$1"
    check_free_space
    echo "================================================" | tee -a "${MAIN_LOG}"
    echo "[SCENE] ${scene_id}" | tee -a "${MAIN_LOG}"
    echo "================================================" | tee -a "${MAIN_LOG}"

    if scene_done "${scene_id}"; then
        echo "[SKIP] scene already completed" | tee -a "${MAIN_LOG}"
        return 0
    fi

    if ! ES_SCANNET_ROOT="${ES_SCANNET_ROOT}" \
        ES_CONFIG_PATH="${ES_CONFIG_PATH}" \
        ES_PROCESS_STRIDE="${ES_PROCESS_STRIDE}" \
        ES_PRUNE_DETECTIONS="${ES_PRUNE_DETECTIONS}" \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        bash "${SCRIPT_DIR}/run_full_pipeline.sh" "${scene_id}" >> "${MAIN_LOG}" 2>&1; then
        echo "[FAIL] ${scene_id}" | tee -a "${MAIN_LOG}" | tee -a "${FAIL_LOG}"
        if [[ "${STOP_ON_ERROR}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
            exit 1
        fi
        return 1
    fi

    echo "[DONE] ${scene_id}" | tee -a "${MAIN_LOG}"
}

TOTAL_SCENES="$(count_scenes)"
echo "[START] Batch EmbodiedScan pipeline" | tee -a "${MAIN_LOG}"
echo "[INFO] scene_list=${SCENE_LIST}" | tee -a "${MAIN_LOG}"
echo "[INFO] scenes=${TOTAL_SCENES}" | tee -a "${MAIN_LOG}"
echo "[INFO] worker_mode=${WORKER_MODE}" | tee -a "${MAIN_LOG}"
echo "[INFO] worker_tag=${WORKER_TAG}" | tee -a "${MAIN_LOG}"
echo "[INFO] output_root=${ES_SCANNET_ROOT}" | tee -a "${MAIN_LOG}"
echo "[INFO] process_stride=${ES_PROCESS_STRIDE} cuda=${CUDA_DEVICE}" | tee -a "${MAIN_LOG}"

if [[ "${WORKER_MODE}" == "queue" ]]; then
    if [[ -z "${QUEUE_DIR}" ]]; then
        echo "[ERROR] WORKER_MODE=queue requires QUEUE_DIR" | tee -a "${MAIN_LOG}"
        exit 1
    fi
    echo "[INFO] queue_dir=${QUEUE_DIR}" | tee -a "${MAIN_LOG}"
    while true; do
        check_free_space
        wait_for_gpu_memory
        scene_id="$(claim_next_scene)"
        if [[ -z "${scene_id}" ]]; then
            break
        fi
        process_scene "${scene_id}" || true
    done
else
    mapfile -t SCENES < <(load_scene_subset)
    for scene_id in "${SCENES[@]}"; do
        wait_for_gpu_memory
        process_scene "${scene_id}" || true
    done
fi

echo "[DONE] Batch EmbodiedScan pipeline finished" | tee -a "${MAIN_LOG}"
echo "[LOG] ${MAIN_LOG}"
echo "[FAILURES] ${FAIL_LOG}"
