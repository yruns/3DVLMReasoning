#!/usr/bin/env bash
# Run 2D open-vocabulary detection (GSA) on a single EmbodiedScan scene.
#
# Usage:
#   bash bashes/embodiedscan/1b_extract_2d_segmentation_detect.sh <scene_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ID="${1:-}"
if [[ -z "${SCENE_ID}" ]]; then
    echo "Usage: bash bashes/embodiedscan/1b_extract_2d_segmentation_detect.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    source "${ROOT_DIR}/env_vars.bash"
fi

ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-${ROOT_DIR}/data/embodiedscan/scannet}"
ES_CONFIG_PATH="${ES_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/embodiedscan.yaml}"
ES_PROCESS_STRIDE="${ES_PROCESS_STRIDE:-2}"
SCENE_SPEC="${SCENE_ID}/conceptgraph"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
if [[ -n "${GSA_PATH:-}" ]]; then
    export PYTHONPATH="${GSA_PATH}/GroundingDINO:${PYTHONPATH}"
fi

python "conceptgraph/detection/generate_gsa.py" \
    --dataset_root "${ES_SCANNET_ROOT}" \
    --dataset_config "${ES_CONFIG_PATH}" \
    --scene_id "${SCENE_SPEC}" \
    --class_set ram \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride "${ES_PROCESS_STRIDE}" \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses

SCENE_PATH="${ES_SCANNET_ROOT}/${SCENE_SPEC}"
PREVIEW_DIR="${SCENE_PATH}/checks"
mkdir -p "${PREVIEW_DIR}"
FIRST_VIS="$(find "${SCENE_PATH}/gsa_vis_ram_withbg_allclasses" -maxdepth 1 -type f 2>/dev/null | sort | sed -n '1p')"
if [[ -n "${FIRST_VIS}" ]]; then
    cp -f "${FIRST_VIS}" "${PREVIEW_DIR}/01_detection_preview.png"
fi

echo "[DONE] 2D detections saved under ${SCENE_PATH}/gsa_detections_ram_withbg_allclasses"
