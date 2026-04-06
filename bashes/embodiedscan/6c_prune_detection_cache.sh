#!/usr/bin/env bash
# Prune the GSA 2D detection cache for an EmbodiedScan scene.
#
# Usage:
#   bash bashes/embodiedscan/6c_prune_detection_cache.sh <scene_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ID="${1:-}"
if [[ -z "${SCENE_ID}" ]]; then
    echo "Usage: bash bashes/embodiedscan/6c_prune_detection_cache.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    source "${ROOT_DIR}/env_vars.bash"
fi

ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-${ROOT_DIR}/data/embodiedscan/scannet}"
SCENE_PATH="${ES_SCANNET_ROOT}/${SCENE_ID}/conceptgraph"
DETECTIONS_DIR="${SCENE_PATH}/gsa_detections_ram_withbg_allclasses"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python -m conceptgraph.scripts.prune_gsa_detection_cache \
    --detections_dir "${DETECTIONS_DIR}"

echo "[DONE] Pruned detection cache: ${DETECTIONS_DIR}"
