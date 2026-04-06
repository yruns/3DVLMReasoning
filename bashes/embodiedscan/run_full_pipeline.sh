#!/usr/bin/env bash
# Run the full single-scene pipeline (stages 1 through 6c) for EmbodiedScan.
#
# Usage:
#   bash bashes/embodiedscan/run_full_pipeline.sh <scene_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENE_ID="${1:-}"
if [[ -z "${SCENE_ID}" ]]; then
    echo "Usage: bash bashes/embodiedscan/run_full_pipeline.sh <scene_id>"
    exit 1
fi

ES_PRUNE_DETECTIONS="${ES_PRUNE_DETECTIONS:-1}"
ES_PROCESS_STRIDE="${ES_PROCESS_STRIDE:-2}"
ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)/data/embodiedscan/scannet}"
THRESHOLD="${THRESHOLD:-1.2}"
SCENE_PATH="${ES_SCANNET_ROOT}/${SCENE_ID}/conceptgraph"
PCD_POST_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
INDEX_FILE="${SCENE_PATH}/indices/visibility_index.pkl"
BUILD_INFO_FILE="${SCENE_PATH}/indices/build_info.json"

RAW_DIR="${ES_SCANNET_ROOT}/${SCENE_ID}/raw"
if [[ -d "${RAW_DIR}" ]] && ls "${RAW_DIR}"/*-rgb.jpg >/dev/null 2>&1; then
    echo "[SKIP] raw/ already exists for ${SCENE_ID}"
else
    bash "${SCRIPT_DIR}/1_prepare_scene.sh" "${SCENE_ID}"
fi
bash "${SCRIPT_DIR}/1b_extract_2d_segmentation_detect.sh" "${SCENE_ID}"
bash "${SCRIPT_DIR}/2b_build_3d_object_map_detect.sh" "${SCENE_ID}"
bash "${SCRIPT_DIR}/6b_build_visibility_index.sh" "${SCENE_ID}"

if [[ "${ES_PRUNE_DETECTIONS}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    bash "${SCRIPT_DIR}/6c_prune_detection_cache.sh" "${SCENE_ID}"
fi

if [[ ! -f "${PCD_POST_FILE}" ]]; then
    echo "[ERROR] Missing post-processed object map: ${PCD_POST_FILE}"
    exit 1
fi
if [[ ! -f "${INDEX_FILE}" ]]; then
    echo "[ERROR] Missing visibility index: ${INDEX_FILE}"
    exit 1
fi

python - "${BUILD_INFO_FILE}" "${SCENE_ID}" "${ES_PROCESS_STRIDE}" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
scene_id = sys.argv[2]
process_stride = int(sys.argv[3])
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(
        {
            "scene_id": scene_id,
            "pipeline": "embodiedscan-scannet",
            "process_stride": process_stride,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
PY

echo "[DONE] EmbodiedScan full pipeline completed for ${SCENE_ID}"
echo "[SCENE] ${SCENE_PATH}"
