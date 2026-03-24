# Florence-2 ConceptGraph Pipeline — Handoff Document

**Date**: 2026-03-25
**Status**: In Progress — detection done for 2/5 scenes, 3D mapping not yet run

---

## 1. Overall Goal

Replace ConceptGraph's original **RAM + GroundingDINO + SAM** detection frontend with
**Florence-2 + SAM** to eliminate the junk-label problem (verbs, adjectives, color words
like `sit`, `white`, `lead to`, `other item`) that cripples Stage 1 keyframe retrieval.

The end-to-end pipeline is:
1. Florence-2 `<OD>` per-frame detection → clean noun labels + bboxes
2. SAM box-prompted segmentation → instance masks
3. CLIP feature extraction → per-crop + per-text embeddings
4. 3D object mapping (existing `cfslam_pipeline_batch` / `pipeline.py`)
5. Visibility index build
6. **Compare v1 (RAM) vs v2 (Florence-2)** merged scene graphs on 5 scenes

---

## 2. What Has Been Done

### 2.1 Code Migration (COMPLETE)

The entire ConceptGraph pipeline code has been migrated from
`~/codecase/concept-graphs/` into `~/codecase/3DVLMReasoning/conceptgraph/`:

```
3DVLMReasoning/
├── conceptgraph/                    # Scene graph construction pipeline
│   ├── detection/
│   │   ├── generate_gsa.py          # Original RAM+GDINO pipeline (refactored)
│   │   └── generate_florence2.py    # NEW: Florence-2 + SAM pipeline
│   ├── slam/
│   │   ├── models.py               # DetectionList, MapObjectList
│   │   ├── pipeline.py             # 3D object mapping (Hydra)
│   │   ├── mapping.py              # Similarity computation
│   │   └── utils.py                # SLAM utilities
│   ├── dataset/
│   │   ├── loader.py               # GradSLAM dataset loaders
│   │   └── dataconfigs/scannet/openeqa_clip.yaml
│   ├── visualization/
│   │   └── offscreen.py            # 3D scene graph visualization
│   └── utils/
│       ├── general.py, clip.py, ious.py, vis.py
├── bashes/openeqa_scannet/          # 11 bash orchestration scripts
├── scripts/
│   ├── run_florence2_pipeline.sh    # Batch runner for 5 scenes
│   └── compare_v1_vs_florence2.py   # v1 vs v2 comparison script
```

All code has been:
- Reformatted with **black** + **ruff** (0 errors)
- Modernized to Python 3.10+ style (`X | None`, `list[str]`, `pathlib.Path`)
- Import paths updated (e.g., `slam.slam_classes` → `slam.models`)

### 2.2 Directory Layout Restructure (COMPLETE for 5 scenes)

Each scene now uses a `raw/` directory for input data, separating inputs from outputs:

```
OpenEQA/scannet/<clip_id>/
├── raw/                  # RGB, depth, poses, intrinsics, mesh, traj.txt
├── conceptgraph_v1/      # RAM+GDINO pipeline outputs (backup)
│   ├── gsa_detections_ram_withbg_allclasses/
│   ├── pcd_saves/*_post.pkl.gz
│   └── indices/
├── conceptgraph/         # Florence-2 pipeline outputs (in progress)
│   ├── gsa_detections_florence2_withbg_allclasses/
│   ├── pcd_saves/   (NOT YET)
│   └── indices/     (NOT YET)
```

The `ScannetOpenEQADataset` loader in `conceptgraph/dataset/loader.py` auto-detects
`../raw/` and reads frames from there. No symlinks needed in `conceptgraph/`.

`src/scripts/build_visibility_index.py` also falls back to `../raw/traj.txt`.

`src/scripts/prepare_openeqa_scannet_scene.py` now outputs to `raw/` instead of
`conceptgraph/`.

### 2.3 Florence-2 Detection (2/5 scenes COMPLETE)

| Scene | Detection | 3D Mapping | Vis Index |
|-------|-----------|------------|-----------|
| `002-scannet-scene0709_00` | ✅ 300 frames, 28 classes | ❌ | ❌ |
| `003-scannet-scene0762_00` | ✅ 300 frames, 46 classes | ❌ | ❌ |
| `012-scannet-scene0785_00` | ❌ | ❌ | ❌ |
| `013-scannet-scene0720_00` | ❌ | ❌ | ❌ |
| `014-scannet-scene0714_00` | ❌ | ❌ | ❌ |

**Detection output format verified** — identical to `generate_gsa.py`:
```python
{
    "xyxy": np.ndarray (N, 4),
    "confidence": np.ndarray (N,),    # all 1.0 (Florence-2 doesn't give scores)
    "class_id": np.ndarray (N,),
    "mask": np.ndarray (N, H, W),     # SAM masks or bbox rectangles
    "classes": list[str],
    "image_feats": np.ndarray (N, 1024),  # CLIP
    "text_feats": np.ndarray (N, 1024),
    "frame_clip_feat": np.ndarray (1024,),
    "tagging_caption": "florence2",
    "tagging_text_prompt": "label1, label2, ...",
}
```

### 2.4 SAM Issue (FIXED in code, not yet verified in run)

First run used **bbox-rectangle masks** because SAM failed to import. The fix has been
applied in `generate_florence2.py` (added `segment_anything` subdirectory to sys.path),
but the 2 completed scenes were run **without** SAM masks.

---

## 3. What Remains To Do

### 3.1 Re-run detection for scenes 002 and 003 WITH SAM masks

The first 2 scenes used bbox-rectangle masks because SAM wasn't loaded. Either:
- Re-run them with the fixed import (preferred)
- Or proceed with rectangle masks first and re-run later

### 3.2 Run Florence-2 detection on remaining 3 scenes

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
cd ~/codecase/3DVLMReasoning
export GSA_PATH="$HOME/codecase/concept-graphs/Grounded-Segment-Anything"
export PYTHONPATH="${PWD}:${PWD}/src"

for CLIP_ID in 012-scannet-scene0785_00 013-scannet-scene0720_00 014-scannet-scene0714_00; do
  python conceptgraph/detection/generate_florence2.py \
    --dataset_root "$HOME/Datasets/OpenEQA/scannet" \
    --dataset_config "conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml" \
    --scene_id "${CLIP_ID}/conceptgraph" \
    --stride 2 --add_bg_classes
done
```

Each scene takes ~5 minutes on RTX 4090.

### 3.3 Run 3D object mapping on all 5 scenes

```bash
THRESHOLD=1.2
SAVE_SUFFIX="overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1_merge20_masksub"

for CLIP_ID in 002-scannet-scene0709_00 003-scannet-scene0762_00 012-scannet-scene0785_00 013-scannet-scene0720_00 014-scannet-scene0714_00; do
  python conceptgraph/slam/pipeline.py \
    dataset_root="$HOME/Datasets/OpenEQA/scannet" \
    dataset_config="conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml" \
    scene_id="${CLIP_ID}/conceptgraph" \
    stride=2 \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
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
done
```

**Key parameter**: `gsa_variant=florence2_withbg_allclasses` — this tells the pipeline
to look for detection files in `gsa_detections_florence2_withbg_allclasses/`.

### 3.4 Build visibility index on all 5 scenes

```bash
for CLIP_ID in 002-scannet-scene0709_00 003-scannet-scene0762_00 012-scannet-scene0785_00 013-scannet-scene0720_00 014-scannet-scene0714_00; do
  SCENE_PATH="$HOME/Datasets/OpenEQA/scannet/${CLIP_ID}/conceptgraph"
  PCD_FILE=$(ls -t "${SCENE_PATH}/pcd_saves/"*_post.pkl.gz | head -1)
  python -m scripts.build_visibility_index \
    --scene_path "${SCENE_PATH}" \
    --pcd_file "${PCD_FILE}" \
    --stride 2
done
```

### 3.5 Run comparison

```bash
python scripts/compare_v1_vs_florence2.py
```

This loads both `conceptgraph_v1/pcd_saves/*_post.pkl.gz` and
`conceptgraph/pcd_saves/*_post.pkl.gz` for all 5 scenes, and prints a side-by-side
comparison of:
- Total objects, unique categories, junk labels
- Top-20 categories
- Per-scene breakdown

### 3.6 (Optional) Run the full pipeline via batch script

All the above is also wrapped in:
```bash
bash scripts/run_florence2_pipeline.sh
```

---

## 4. Risk Points

### 4.1 SAM mask quality with bbox prompts

Florence-2 bboxes may be less tight than GroundingDINO bboxes. SAM's mask quality
depends on the box prompt quality. If masks are loose, 3D point clouds will be noisier.

**Mitigation**: Check per-object point cloud quality after 3D mapping. If needed,
use SAM's `multimask_output=True` and pick by IoU score (already implemented).

### 4.2 Florence-2 confidence = 1.0 for all detections

Florence-2 `<OD>` doesn't output confidence scores. All detections are set to 1.0.
The 3D mapping pipeline uses `mask_conf_threshold=0.25` to filter, which now has no
effect. This may increase false positive detections.

**Mitigation**: Could use Florence-2 `<OPEN_VOCABULARY_DETECTION>` which may give scores,
or post-filter by CLIP similarity.

### 4.3 Florence-2 lower recall than RAM+GDINO

Previous frame-level comparison showed Florence-2 `<OD>` detects fewer objects per frame
(6.2 vs 23.1 avg). After 3D merging across 300 frames this may be less severe, but
small/occluded objects could still be missed.

**Mitigation**: Consider `<DENSE_REGION_CAPTION>` mode for higher recall (see below).

### 4.4 Hydra config path for `pipeline.py`

`pipeline.py` uses Hydra for config. When run from `3DVLMReasoning/`, the working
directory matters. Ensure `PYTHONPATH` includes both `${PWD}` and `${PWD}/src`.
The Hydra config directory is inferred from the script location.

### 4.5 Rectangle masks (scenes 002, 003)

The first 2 scenes were run WITHOUT SAM (used bbox rectangles as masks). The 3D mapping
will include all depth pixels within the bbox, causing objects to "bleed" into each
other. These should be re-run with SAM for a fair comparison.

---

## 5. Environment Setup

```bash
# Conda env
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph

# Working directory
cd ~/codecase/3DVLMReasoning

# Required env vars
export GSA_PATH="$HOME/codecase/concept-graphs/Grounded-Segment-Anything"
export PYTHONPATH="${PWD}:${PWD}/src"

# Key packages (in conceptgraph conda env)
# - transformers 4.45.2 (for Florence-2)
# - torch 2.0.1 + CUDA (RTX 4090)
# - open_clip (ViT-H-14 for CLIP features)
# - segment_anything (from GSA_PATH/segment_anything/)
# - hydra-core (for pipeline.py config)
```

---

## 6. Key File Locations

| Purpose | Path |
|---------|------|
| Florence-2 detection script | `conceptgraph/detection/generate_florence2.py` |
| Original RAM detection script | `conceptgraph/detection/generate_gsa.py` |
| 3D mapping pipeline | `conceptgraph/slam/pipeline.py` |
| Dataset loader (raw/ auto-detect) | `conceptgraph/dataset/loader.py` |
| Visibility index builder | `src/scripts/build_visibility_index.py` |
| Scene preparation | `src/scripts/prepare_openeqa_scannet_scene.py` |
| Batch pipeline runner | `scripts/run_florence2_pipeline.sh` |
| v1 vs v2 comparison | `scripts/compare_v1_vs_florence2.py` |
| Bash orchestration scripts | `bashes/openeqa_scannet/` (11 scripts) |
| Dataset config | `conceptgraph/dataset/dataconfigs/scannet/openeqa_clip.yaml` |
| Scene data | `~/Datasets/OpenEQA/scannet/<clip_id>/{raw,conceptgraph,conceptgraph_v1}/` |

---

## 7. Background Context

### Why Florence-2?

See `docs/eval_analysis_20260324.md` Section 3.1: RAM's 4584-tag vocabulary produces
garbage labels (verbs: `sit`, `connect`; colors: `white`, `black`; nonsense:
`lead to`, `other item`). 67% of Stage 1 queries fail to find target objects.

Florence-2 `<OD>` mode directly outputs noun+bbox via autoregressive generation,
eliminating the junk label problem by architecture (not post-hoc filtering).

### Earlier experiments (in concept-graphs repo)

Frame-level comparison run from `concept-graphs/scripts/compare_florence2_vs_ram.py`
and `concept-graphs/scripts/compare_scenelevel_florence2_vs_ram.py` showed:
- Florence-2 `<OD>`: 0% junk labels (vs 9.1% RAM)
- Florence-2 `<OD>`: lower per-frame recall (6.2 vs 23.1 dets/frame)
- Florence-2 `<DENSE_REGION_CAPTION>`: high recall but needs noun extraction + dedup

### Future directions

If Florence-2 `<OD>` recall proves insufficient after 3D merging:
1. **Hybrid**: Florence-2 `<DENSE_REGION_CAPTION>` → extract nouns → feed to GroundingDINO
2. **SAM 3**: Built-in detection + segmentation (released Nov 2025)
3. **DINO-X**: Prompt-free universal detection (API-only)
