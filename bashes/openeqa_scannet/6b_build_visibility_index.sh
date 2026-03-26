#!/usr/bin/env bash
# Build visibility index for an OpenEQA ScanNet clip.
#
# Reads the post-processed 3D object map and constructs a per-frame visibility
# index mapping each keyframe to the set of objects visible in it.
#
# Usage:
#   bash bashes/openeqa_scannet/6b_build_visibility_index.sh <clip_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CLIP_ID="${1:-}"
if [[ -z "${CLIP_ID}" ]]; then
    echo "Usage: bash bashes/openeqa_scannet/6b_build_visibility_index.sh <clip_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/env_vars.bash"
fi

OPENEQA_SCANNET_ROOT="${OPENEQA_SCANNET_ROOT:-${HOME}/Datasets/OpenEQA/scannet}"
OPENEQA_PROCESS_STRIDE="${OPENEQA_PROCESS_STRIDE:-2}"
SCENE_PATH="${OPENEQA_SCANNET_ROOT}/${CLIP_ID}/conceptgraph"
PCD_FILE="$(find "${SCENE_PATH}/pcd_saves" -maxdepth 1 -type f -name '*ram*_post.pkl.gz' | sort | head -n 1)"

if [[ -z "${PCD_FILE}" ]]; then
    echo "[ERROR] Post-processed object map not found under ${SCENE_PATH}/pcd_saves"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

python -m conceptgraph.scripts.build_visibility_index \
    --scene_path "${SCENE_PATH}" \
    --pcd_file "${PCD_FILE}" \
    --stride "${OPENEQA_PROCESS_STRIDE}"

echo "[DONE] Visibility index built: ${SCENE_PATH}/indices/visibility_index.pkl"
