#!/usr/bin/env bash
# Prepare a single EmbodiedScan ScanNet scene for the ConceptGraph pipeline.
#
# Converts posed_images/<scene_id>/ into <scene_id>/raw/ with the standard
# ConceptGraph naming convention.
#
# Usage:
#   bash bashes/embodiedscan/1_prepare_scene.sh <scene_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ID="${1:-}"
if [[ -z "${SCENE_ID}" ]]; then
    echo "Usage: bash bashes/embodiedscan/1_prepare_scene.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    source "${ROOT_DIR}/env_vars.bash"
fi

ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-${ROOT_DIR}/data/embodiedscan/scannet}"
ES_PKL="${ES_PKL:-${ROOT_DIR}/data/embodiedscan/embodiedscan_infos_val.pkl}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

python -m src.scripts.prepare_embodiedscan_scene \
    --pkl "${ES_PKL}" \
    --data_root "${ES_SCANNET_ROOT}" \
    --scene_id "${SCENE_ID}"

echo "[DONE] Prepared ${ES_SCANNET_ROOT}/${SCENE_ID}/conceptgraph"
