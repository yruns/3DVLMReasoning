#!/usr/bin/env bash
# Build 3D object map via SLAM pipeline for a single EmbodiedScan scene.
#
# Usage:
#   bash bashes/embodiedscan/2b_build_3d_object_map_detect.sh <scene_id>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCENE_ID="${1:-}"
if [[ -z "${SCENE_ID}" ]]; then
    echo "Usage: bash bashes/embodiedscan/2b_build_3d_object_map_detect.sh <scene_id>"
    exit 1
fi

if [[ -f "${ROOT_DIR}/env_vars.bash" ]]; then
    source "${ROOT_DIR}/env_vars.bash"
fi

ES_SCANNET_ROOT="${ES_SCANNET_ROOT:-${ROOT_DIR}/data/embodiedscan/scannet}"
ES_CONFIG_PATH="${ES_CONFIG_PATH:-${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/embodiedscan.yaml}"
ES_PROCESS_STRIDE="${ES_PROCESS_STRIDE:-2}"
THRESHOLD="${THRESHOLD:-1.2}"
SCENE_SPEC="${SCENE_ID}/conceptgraph"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
if [[ -n "${GSA_PATH:-}" ]]; then
    export PYTHONPATH="${GSA_PATH}/GroundingDINO:${PYTHONPATH}"
fi

python "conceptgraph/slam/pipeline.py" \
    dataset_root="${ES_SCANNET_ROOT}" \
    dataset_config="${ES_CONFIG_PATH}" \
    stride="${ES_PROCESS_STRIDE}" \
    scene_id="${SCENE_SPEC}" \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold="${THRESHOLD}" \
    dbscan_eps=0.1 \
    gsa_variant=ram_withbg_allclasses \
    class_agnostic=False \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix="overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub" \
    merge_interval=20 \
    merge_visual_sim_thresh=0.8 \
    merge_text_sim_thresh=0.8

SCENE_PATH="${ES_SCANNET_ROOT}/${SCENE_SPEC}"
PKL_FILE="${SCENE_PATH}/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub_post.pkl.gz"
PREVIEW_DIR="${SCENE_PATH}/checks/02_object_map_preview"

VIS_CMD=(
    python "conceptgraph/visualization/offscreen.py"
    --result_path "${PKL_FILE}"
    --output_dir "${PREVIEW_DIR}"
    --output_format images
    --num_views 1
    --image_width 1600
    --image_height 900
)

"${VIS_CMD[@]}"

echo "[DONE] 3D object map saved to ${PKL_FILE}"
