#!/usr/bin/env bash
# Run SAM 3 ConceptGraph pipeline: detection (sam3 env) + 3D mapping (conceptgraph env)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OPENEQA_ROOT="${OPENEQA_SCANNET_ROOT:-$HOME/Datasets/OpenEQA/scannet}"
CONFIG="${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml"
STRIDE=5
THRESHOLD="${THRESHOLD:-1.2}"

SCENES=(
  "002-scannet-scene0709_00"
  "003-scannet-scene0762_00"
  "012-scannet-scene0785_00"
  "013-scannet-scene0720_00"
  "014-scannet-scene0714_00"
)

# GPU to use (skip GPU 1 — broken)
GPU="${CUDA_VISIBLE_DEVICES:-7}"

cd "${ROOT_DIR}"

for CLIP_ID in "${SCENES[@]}"; do
  SCENE_SPEC="${CLIP_ID}/conceptgraph"
  SCENE_PATH="${OPENEQA_ROOT}/${SCENE_SPEC}"
  echo ""
  echo "================================================================"
  echo "  Scene: ${CLIP_ID}"
  echo "================================================================"

  # ── Step 1: SAM 3 detection (sam3 env) ─────────────────────────────
  DET_DIR="${SCENE_PATH}/gsa_detections_sam3_sn200_withbg"
  if [ -d "${DET_DIR}" ] && [ "$(ls ${DET_DIR}/*.pkl.gz 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[1] SAM 3 detections already exist, skipping."
  else
    echo "[1] SAM 3 detection …"
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate sam3
    CUDA_VISIBLE_DEVICES=${GPU} python "conceptgraph/detection/generate_sam3.py" \
      --scene_dir "${OPENEQA_ROOT}/${CLIP_ID}" \
      --stride "${STRIDE}" \
      --add_bg_classes
  fi

  # ── Step 2: 3D object mapping (conceptgraph env) ───────────────────
  echo "[2] 3D object mapping …"
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
  SAVE_SUFFIX="overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub"
  CUDA_VISIBLE_DEVICES=${GPU} python "conceptgraph/slam/pipeline.py" \
    dataset_root="${OPENEQA_ROOT}" \
    dataset_config="${CONFIG}" \
    scene_id="${SCENE_SPEC}" \
    stride="${STRIDE}" \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold="${THRESHOLD}" \
    dbscan_eps=0.1 \
    gsa_variant=sam3_sn200_withbg \
    class_agnostic=True \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix="${SAVE_SUFFIX}" \
    merge_interval=20 \
    merge_visual_sim=0.8 \
    merge_text_sim=0.8 \
    save_objects_all_frames=True

  # ── Step 3: Visibility index ────────────────────────────────────────
  echo "[3] Building visibility index …"
  PCD_FILE=$(ls -t "${SCENE_PATH}/pcd_saves/"*_post.pkl.gz 2>/dev/null | head -1)
  if [ -z "${PCD_FILE}" ]; then
    echo "  WARNING: No post-processed pcd file found, skipping."
    continue
  fi
  python -m scripts.build_visibility_index \
    --scene_path "${SCENE_PATH}" \
    --pcd_file "${PCD_FILE}" \
    --stride "${STRIDE}"

  echo "[OK] ${CLIP_ID} done."
done

echo ""
echo "================================================================"
echo "  All scenes done!"
echo "================================================================"
