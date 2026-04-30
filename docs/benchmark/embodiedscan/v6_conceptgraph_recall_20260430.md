# v6 ConceptGraph Recall-Only Detector Study

**Date:** 2026-04-30
**Branch:** `feat/explore_3dbbox`
**Run-time tip:** `406b52d` plus dirty recall, merge, ConceptGraph extractor, pack, and doc changes
**Eval scale:** v3 frozen 2000-sample scope, 220 ScanNet scenes
**Proposal sources:** `pack_conceptgraph` and `pack_3way` (`pack_vdetr` + `pack_bip3d` + `pack_conceptgraph`)
**Judge:** none; detector recall is computed programmatically with 9-DoF oriented 3D IoU
**SQLite:** no row inserted; this is a detector-pool recall property, not a Stage 2 benchmark score

## Scope

This version adds a ConceptGraph proposal pool from the prepared
`conceptgraph/pcd_saves/full_pcd_*_post.pkl.gz` assets already present under
`data/embodiedscan/scannet/<scene>/`. The run evaluates ConceptGraph alone and
then a 3-way no-NMS union with the existing V-DETR and BIP3D pools.

The first extractor implementation used a PCA OBB from `pcd_np`, but that was
superseded during the run because `pcd_np` contains only observed surface
points and badly underestimates object depth. The final v6 numbers below use
ConceptGraph's own `bbox_np` 8-corner box, emitted as an axis-aligned 9-DoF
box with zero Euler angles and `raw_corners = bbox_np`.

## What Changed

- Added `src/evaluation/scripts/extract_conceptgraph_proposals.py`.
- Added focused tests in
  `src/evaluation/scripts/tests/test_extract_conceptgraph_proposals.py`.
- Extracted 220 ConceptGraph scene-level detector records from fused post-pcd
  pickles.
- Materialized `pack_conceptgraph` for the v3 frozen 2000-question fold.
- Rebuilt `pack_3way` from `pack_vdetr`, `pack_bip3d`, and
  `pack_conceptgraph`.
- Evaluated keyframe-visible and full-scene recall for ConceptGraph and 3-way
  pools.

## Headline Metrics

Keyframe-visible pool, matching the current Stage 2 visible proposal set:

| pool | Recall@0.25 | Recall@0.50 | mean max IoU | median max IoU | proposals/sample |
|---|---:|---:|---:|---:|---|
| `pack_vdetr` | 0.4080 | 0.3305 | 0.2950 | 0.0679 | min 0, median 126, max 256 |
| `pack_bip3d` | 0.7250 | 0.4330 | 0.4219 | 0.4452 | min 7, median 126, max 256 |
| `pack_conceptgraph` | 0.0035 | 0.0000 | 0.0088 | 0.0000 | min 0, median 14, max 175 |
| `pack_vdetr_bip3d` | 0.7735 | 0.5320 | 0.4928 | 0.5242 | min 18, median 253, max 508 |
| `pack_3way` | 0.7735 | 0.5320 | 0.4929 | 0.5242 | min 32, median 277, max 636 |

Full-scene proposal-pool upper bound:

| pool | Recall@0.25 | Recall@0.50 | mean max IoU | median max IoU | proposals/sample |
|---|---:|---:|---:|---:|---|
| `pack_vdetr` | 0.4080 | 0.3305 | 0.2950 | 0.0679 | min 36, median 256, max 256 |
| `pack_bip3d` | 0.7250 | 0.4335 | 0.4222 | 0.4459 | min 256, median 256, max 256 |
| `pack_conceptgraph` | 0.0035 | 0.0000 | 0.0088 | 0.0000 | min 4, median 65, max 236 |
| `pack_vdetr_bip3d` | 0.7735 | 0.5320 | 0.4929 | 0.5244 | min 292, median 512, max 512 |
| `pack_3way` | 0.7735 | 0.5320 | 0.4930 | 0.5244 | min 313, median 576, max 748 |

ConceptGraph remains below 1% Recall@0.25 after the `bbox_np` correction. The
3-way pool therefore increases proposal count but does not change headline
Recall@0.25 or Recall@0.50 relative to the v5 V-DETR+BIP3D merged pool.

## Raw Artifacts

Recall JSONs:

- ConceptGraph keyframe:
  `/home/ysh/.super-orchestrator/3d/artifacts/W7_recall_pack_conceptgraph_keyframe.json`
- ConceptGraph full-scene:
  `/home/ysh/.super-orchestrator/3d/artifacts/W7_recall_pack_conceptgraph_scene.json`
- 3-way keyframe:
  `/home/ysh/.super-orchestrator/3d/artifacts/W7_recall_pack_3way_keyframe.json`
- 3-way full-scene:
  `/home/ysh/.super-orchestrator/3d/artifacts/W7_recall_pack_3way_scene.json`

Detector and pack artifacts:

- ConceptGraph detector records:
  `tmp/embodiedscan_conceptgraph_outputs_full_20260430/merged/detector_records.jsonl`
- ConceptGraph packs: `data/embodiedscan/<scene>/pack_conceptgraph/`
- 3-way merged packs: `data/embodiedscan/<scene>/pack_3way/`
- Scene list:
  `/home/ysh/.super-orchestrator/3d/artifacts/B_vdetr_scene_list.json`
- Frozen sample IDs:
  `/home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json`
- Pack chunk IDs:
  `/home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_chunks/chunk_00.json` ...
  `chunk_07.json`
- Logs:
  `/tmp/w7-cg-extract-bboxnp.log`,
  `/tmp/w7-pack-cg-bboxnp-chunk00.log` ...
  `/tmp/w7-pack-cg-bboxnp-chunk07.log`,
  `/tmp/w7-recall-cg-bboxnp-keyframe.log`,
  `/tmp/w7-recall-cg-bboxnp-scene.log`,
  `/tmp/w7-merge-3way-bboxnp.log`,
  `/tmp/w7-recall-3way-bboxnp-keyframe.log`,
  `/tmp/w7-recall-3way-bboxnp-scene.log`

## Commands

ConceptGraph extraction:

```bash
tmux new-session -d -s w7-cg-extract-bboxnp 'cd /home/ysh/codecase/3DVLMReasoning && source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph && PYTHONPATH=src python src/evaluation/scripts/extract_conceptgraph_proposals.py --data-root data/embodiedscan --scene-list /home/ysh/.super-orchestrator/3d/artifacts/B_vdetr_scene_list.json --output-jsonl tmp/embodiedscan_conceptgraph_outputs_full_20260430/merged/detector_records.jsonl 2>&1 | tee /tmp/w7-cg-extract-bboxnp.log'
```

ConceptGraph pack conversion, run once for each `XX in 00..07`:

```bash
tmux new-session -d -s "w7-pack-cg-bboxnp-chunk${XX}" "cd /home/ysh/codecase/3DVLMReasoning && source .venv-agents/bin/activate && export PYTHONPATH=src && python src/evaluation/scripts/prepare_detector_pack_inputs.py --detector-records tmp/embodiedscan_conceptgraph_outputs_full_20260430/merged/detector_records.jsonl --infos-pkl data/embodiedscan/embodiedscan_infos_val.pkl --vg-json data/embodiedscan/embodiedscan_val_vg.json --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_chunks/chunk_${XX}.json --data-root data/embodiedscan --pack-name pack_conceptgraph --split val 2>&1 | tee /tmp/w7-pack-cg-bboxnp-chunk${XX}.log"
```

ConceptGraph recall:

```bash
source .venv-agents/bin/activate && export PYTHONPATH=src
python src/evaluation/scripts/eval_proposal_pool_recall.py \
  --data-root data/embodiedscan \
  --pack-name pack_conceptgraph \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json \
  --output-json /home/ysh/.super-orchestrator/3d/artifacts/W7_recall_pack_conceptgraph_keyframe.json \
  --use-keyframe-pool true

python src/evaluation/scripts/eval_proposal_pool_recall.py \
  --data-root data/embodiedscan \
  --pack-name pack_conceptgraph \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json \
  --output-json /home/ysh/.super-orchestrator/3d/artifacts/W7_recall_pack_conceptgraph_scene.json \
  --use-keyframe-pool false
```

3-way merge:

```bash
source .venv-agents/bin/activate && export PYTHONPATH=src
python src/evaluation/scripts/merge_proposal_packs.py \
  --data-root data/embodiedscan \
  --source-packs pack_vdetr pack_bip3d pack_conceptgraph \
  --output-pack pack_3way \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json \
  --max-proposals-per-scene 2000
```

3-way recall:

```bash
source .venv-agents/bin/activate && export PYTHONPATH=src
python src/evaluation/scripts/eval_proposal_pool_recall.py \
  --data-root data/embodiedscan \
  --pack-name pack_3way \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json \
  --output-json /home/ysh/.super-orchestrator/3d/artifacts/W7_recall_pack_3way_keyframe.json \
  --use-keyframe-pool true

python src/evaluation/scripts/eval_proposal_pool_recall.py \
  --data-root data/embodiedscan \
  --pack-name pack_3way \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json \
  --output-json /home/ysh/.super-orchestrator/3d/artifacts/W7_recall_pack_3way_scene.json \
  --use-keyframe-pool false
```

## Prepared Artifact Counts

| artifact | count |
|---|---:|
| `pack_conceptgraph` directories | 220 |
| `pack_conceptgraph/proposals.jsonl` | 220 |
| `pack_conceptgraph/visibility.json` | 220 |
| `pack_conceptgraph/samples/*.json` | 2000 |
| `pack_3way` directories | 220 |
| `pack_3way/proposals.jsonl` | 220 |
| `pack_3way/visibility.json` | 220 |
| `pack_3way/samples/*.json` | 2000 |

Proposal counts:

| pack | scenes | total proposals | min proposals | median proposals | max proposals |
|---|---:|---:|---:|---:|---:|
| `pack_conceptgraph` | 220 | 11890 | 4 | 46 | 236 |
| `pack_3way` | 220 | 118970 | 313 | 551 | 748 |

## Caveats

- The metric is recall-only. It does not measure mAP, precision, or Stage 2
  selection accuracy.
- The final extractor uses `bbox_np` as an axis-aligned box. This intentionally
  avoids the failed PCA-from-`pcd_np` path, but it may over-enclose if future
  ConceptGraph assets store oriented corners.
- `prepare_detector_pack_inputs.py` still writes `"source": "vdetr"` in the
  scene-level pack JSON for detector-derived packs. The per-proposal metadata
  correctly records `detector = "ConceptGraph"` and
  `box_format = "aabb_from_conceptgraph_bbox_np"`.
- No SQLite row was inserted. The canonical `runs.sqlite` table tracks
  benchmark run scores, while this document records detector-pool recall before
  Stage 2.
