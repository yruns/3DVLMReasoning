#!/usr/bin/env bash
# Run Florence-2 ConceptGraph pipeline on multiple scenes, then compare with v1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OPENEQA_ROOT="${OPENEQA_SCANNET_ROOT:-$HOME/Datasets/OpenEQA/scannet}"
CONFIG="${ROOT_DIR}/conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml"
STRIDE="${OPENEQA_PROCESS_STRIDE:-2}"
THRESHOLD="${THRESHOLD:-1.2}"

SCENES=(
  "002-scannet-scene0709_00"
  "003-scannet-scene0762_00"
  "012-scannet-scene0785_00"
  "013-scannet-scene0720_00"
  "014-scannet-scene0714_00"
)

cd "${ROOT_DIR}"

for CLIP_ID in "${SCENES[@]}"; do
  SCENE_SPEC="${CLIP_ID}/conceptgraph"
  SCENE_PATH="${OPENEQA_ROOT}/${SCENE_SPEC}"
  echo ""
  echo "================================================================"
  echo "  Scene: ${CLIP_ID}"
  echo "================================================================"

  # ── Step 1b: Florence-2 + SAM detection ───────────────────────────
  echo "[1b] Florence-2 + SAM detection …"
  python "conceptgraph/detection/generate_florence2.py" \
    --dataset_root "${OPENEQA_ROOT}" \
    --dataset_config "${CONFIG}" \
    --scene_id "${SCENE_SPEC}" \
    --stride "${STRIDE}" \
    --add_bg_classes

  # ── Step 2b: 3D object mapping ────────────────────────────────────
  echo "[2b] 3D object mapping …"
  SAVE_SUFFIX="overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub"
  python "conceptgraph/slam/pipeline.py" \
    dataset_root="${OPENEQA_ROOT}" \
    dataset_config="${CONFIG}" \
    scene_id="${SCENE_SPEC}" \
    stride="${STRIDE}" \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold="${THRESHOLD}" \
    dbscan_eps=0.1 \
    gsa_variant=florence2_withbg_allclasses \
    class_agnostic=True \
    skip_bg=True \
    max_bbox_area_ratio=0.5 \
    save_suffix="${SAVE_SUFFIX}" \
    merge_interval=20 \
    merge_visual_sim=0.8 \
    merge_text_sim=0.8 \
    save_objects_all_frames=True

  # ── Step 6b: Visibility index ─────────────────────────────────────
  echo "[6b] Building visibility index …"
  PCD_FILE=$(ls -t "${SCENE_PATH}/pcd_saves/"*_post.pkl.gz 2>/dev/null | head -1)
  if [ -z "${PCD_FILE}" ]; then
    echo "  WARNING: No post-processed pcd file found, skipping 6b"
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
echo "  All 5 scenes done. Running comparison …"
echo "================================================================"
python "${ROOT_DIR}/scripts/compare_v1_vs_florence2.py"
