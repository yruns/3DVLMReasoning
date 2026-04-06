#!/usr/bin/env bash
# Build visibility index for an EmbodiedScan scene.
#
# Usage:
#   bash bashes/embodiedscan/6b_build_visibility_index.sh <scene_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ID="${1:-}"
if [[ -z "${SCENE_ID}" ]]; then
    echo "Usage: bash bashes/embodiedscan/6b_build_visibility_index.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    source "${ROOT_DIR}/env_vars.bash"
fi

ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-${ROOT_DIR}/data/embodiedscan/scannet}"
ES_PROCESS_STRIDE="${ES_PROCESS_STRIDE:-2}"
SCENE_PATH="${ES_SCANNET_ROOT}/${SCENE_ID}/conceptgraph"
PCD_FILE="$(find "${SCENE_PATH}/pcd_saves" -maxdepth 1 -type f -name '*ram*_post.pkl.gz' | sort | head -n 1)"

if [[ -z "${PCD_FILE}" ]]; then
    echo "[ERROR] Post-processed object map not found under ${SCENE_PATH}/pcd_saves"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python -m conceptgraph.scripts.build_visibility_index \
    --scene_path "${SCENE_PATH}" \
    --pcd_file "${PCD_FILE}" \
    --stride "${ES_PROCESS_STRIDE}"

echo "[DONE] Visibility index built: ${SCENE_PATH}/indices/visibility_index.pkl"
