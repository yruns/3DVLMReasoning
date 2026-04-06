#!/usr/bin/env bash
# Extract posed images from ScanNet .sens files for EmbodiedScan.
#
# Uses EmbodiedScan's generate_image_scannet.py with --fast (every 10th frame).
# Input:  data/embodiedscan/scannet/scans/<scene_id>/<scene_id>.sens
# Output: data/embodiedscan/scannet/posed_images/<scene_id>/
#
# Usage:
#   bash bashes/embodiedscan/0_extract_sens.sh [nproc]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-${ROOT_DIR}/data/embodiedscan/scannet}"
NPROC="${1:-8}"
EXTRACT_SCRIPT="${ROOT_DIR}/data/EmbodiedScan_repo/embodiedscan/converter/generate_image_scannet.py"

if [[ ! -f "${EXTRACT_SCRIPT}" ]]; then
    echo "[ERROR] Extraction script not found: ${EXTRACT_SCRIPT}"
    exit 1
fi

if [[ ! -d "${ES_SCANNET_ROOT}/scans" ]]; then
    echo "[ERROR] scans directory not found: ${ES_SCANNET_ROOT}/scans"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

echo "[INFO] Extracting posed images from .sens files"
echo "[INFO] dataset_folder=${ES_SCANNET_ROOT}"
echo "[INFO] nproc=${NPROC}"

python "${EXTRACT_SCRIPT}" \
    --dataset_folder "${ES_SCANNET_ROOT}" \
    --fast \
    --nproc "${NPROC}"

SCENE_COUNT=$(ls -d "${ES_SCANNET_ROOT}/posed_images"/scene* 2>/dev/null | wc -l)
echo "[DONE] Extracted ${SCENE_COUNT} scenes to ${ES_SCANNET_ROOT}/posed_images/"
