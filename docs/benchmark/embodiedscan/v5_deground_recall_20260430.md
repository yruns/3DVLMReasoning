# v5 DEGround/BIP3D Recall-Only Detector Study

**Date:** 2026-04-30
**Branch:** `feat/explore_3dbbox`
**Run-time tip:** `406b52d` plus dirty recall evaluator, merge-pool, pack, and doc changes
**Eval scale:** v3 frozen 2000-sample scope, 220 ScanNet scenes
**Proposal sources:** `pack_vdetr`, `pack_bip3d`, and `pack_vdetr_bip3d`
**Judge:** none; detector recall is computed programmatically with 9-DoF oriented 3D IoU
**SQLite:** no row inserted; this is a detector-pool recall property, not a Stage 2 benchmark score

## Scope

This version compares detector proposal-pool recall only. Stage 2 was not run
in this round because the backend `aidp-i18ntt-sg.tiktok-row.net` was
unreachable from this host during v4 smoke attempts. The metric here is the
oracle upper bound available to a later VLM selector: for each target, compute
the maximum 3D IoU over proposals in the candidate pool, then aggregate
Recall@0.25, Recall@0.50, and max-IoU statistics.

DEGround was the original target provider, but its repository did not expose
public runnable code. BIP3D is used here as the closest available substitute.

## What Changed

- Added BIP3D as a second detector-backed proposal provider after V-DETR.
- Materialized `pack_bip3d` for the same v3 frozen 2000-question fold.
- Added `src/evaluation/scripts/merge_proposal_packs.py`.
- Built `pack_vdetr_bip3d` as a no-NMS union of V-DETR and BIP3D proposals.
- Evaluated keyframe-visible and full-scene recall for all detector pools.

## Headline Metrics

Keyframe-visible pool, matching the current Stage 2 visible proposal set:

| pool | Recall@0.25 | Recall@0.50 | mean max IoU | median max IoU | proposals/sample |
|---|---:|---:|---:|---:|---|
| `pack_vdetr` | 0.4080 | 0.3305 | 0.2950 | 0.0679 | min 0, median 126, max 256 |
| `pack_bip3d` | 0.7250 | 0.4330 | 0.4219 | 0.4452 | min 7, median 126, max 256 |
| `pack_vdetr_bip3d` | 0.7735 | 0.5320 | 0.4928 | 0.5242 | min 18, median 253, max 508 |
| delta vs `pack_vdetr` | +0.3655 | +0.2015 | +0.1978 | +0.4562 | - |

Full-scene proposal-pool upper bound:

| pool | Recall@0.25 | Recall@0.50 | mean max IoU | median max IoU | proposals/sample |
|---|---:|---:|---:|---:|---|
| `pack_vdetr` | 0.4080 | 0.3305 | 0.2950 | 0.0679 | min 36, median 256, max 256 |
| `pack_bip3d` | 0.7250 | 0.4335 | 0.4222 | 0.4459 | min 256, median 256, max 256 |
| `pack_vdetr_bip3d` | 0.7735 | 0.5320 | 0.4929 | 0.5244 | min 292, median 512, max 512 |
| delta vs `pack_vdetr` | +0.3655 | +0.2015 | +0.1979 | +0.4565 | - |

The merged pool is also above BIP3D alone on the keyframe-visible metric:
`+0.0485` Recall@0.25, `+0.0990` Recall@0.50, and `+0.0709` mean max-IoU.

## Raw Artifacts

Recall JSONs:

- V-DETR keyframe: `/home/ysh/.super-orchestrator/3d/artifacts/W1_recall_pack_vdetr_keyframe.json`
- V-DETR full-scene: `/home/ysh/.super-orchestrator/3d/artifacts/W1_recall_pack_vdetr_scene.json`
- BIP3D keyframe: `/home/ysh/.super-orchestrator/3d/artifacts/W3_recall_pack_bip3d_keyframe.json`
- BIP3D full-scene: `/home/ysh/.super-orchestrator/3d/artifacts/W3_recall_pack_bip3d_scene.json`
- Merged keyframe: `/home/ysh/.super-orchestrator/3d/artifacts/W5_recall_pack_vdetr_bip3d_keyframe.json`
- Merged full-scene: `/home/ysh/.super-orchestrator/3d/artifacts/W5_recall_pack_vdetr_bip3d_scene.json`

Detector and pack artifacts:

- V-DETR detector records: `tmp/embodiedscan_vdetr_fullscene_outputs_full_20260429/merged/detector_records.jsonl`
- BIP3D detector records: `tmp/embodiedscan_bip3d_outputs_full_20260430/merged/detector_records.jsonl`
- BIP3D detector summary: `tmp/embodiedscan_bip3d_outputs_full_20260430/merged/detector_records.summary.json`
- V-DETR packs: `data/embodiedscan/<scene>/pack_vdetr/`
- BIP3D packs: `data/embodiedscan/<scene>/pack_bip3d/`
- Merged packs: `data/embodiedscan/<scene>/pack_vdetr_bip3d/`
- Merge log: `/tmp/W5_merge_pack_vdetr_bip3d.log`
- Recall logs: `/tmp/W5_recall_pack_vdetr_bip3d_keyframe.log`, `/tmp/W5_recall_pack_vdetr_bip3d_scene.log`

## Commands

V-DETR pool prep and detector execution are documented in
[v4_vdetr_pool_prepared_20260430.md](v4_vdetr_pool_prepared_20260430.md).

BIP3D inference, as dispatched in W3:

```bash
mkdir -p tmp/embodiedscan_bip3d_outputs_full_20260430
for gpu in 0 2 3 4 5 6 7; do
  tmux new-session -d -s "bip3d-full-gpu${gpu}" "cd /home/ysh/codecase/3DVLMReasoning/external/BIP3D && \
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate bip3d && \
    export CUDA_VISIBLE_DEVICES=${gpu} CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:\$PATH PYTHONPATH=./ && \
    python /tmp/w2b_bip3d_smoke.py \
      --config configs/bip3d_det.py \
      --checkpoint checkpoints/bip3d_det_rgbd_dat.pth \
      --ann-file data/embodiedscan/shards/v3_2k_gpu${gpu}.pkl \
      --output-dir /home/ysh/codecase/3DVLMReasoning/tmp/embodiedscan_bip3d_outputs_full_20260430/gpu${gpu} \
      2>&1 | tee /tmp/bip3d-full-gpu${gpu}.log"
done
```

BIP3D pack conversion used the existing detector-pack converter. The initial
single-process converter was too slow, so W3 finished with 8 disjoint chunks
under `/home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_chunks/`:

```bash
source .venv-agents/bin/activate && export PYTHONPATH=src && \
python src/evaluation/scripts/prepare_detector_pack_inputs.py \
  --detector-records tmp/embodiedscan_bip3d_outputs_full_20260430/merged/detector_records.jsonl \
  --infos-pkl data/embodiedscan/embodiedscan_infos_val.pkl \
  --vg-json data/embodiedscan/embodiedscan_val_vg.json \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_chunks/chunk_XX.json \
  --data-root data/embodiedscan \
  --pack-name pack_bip3d \
  --split val
```

Merge V-DETR and BIP3D packs:

```bash
source .venv-agents/bin/activate && export PYTHONPATH=src && \
python src/evaluation/scripts/merge_proposal_packs.py \
  --data-root data/embodiedscan \
  --source-packs pack_vdetr pack_bip3d \
  --output-pack pack_vdetr_bip3d \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json \
  --max-proposals-per-scene 2000
```

Recall evaluation:

```bash
source .venv-agents/bin/activate && export PYTHONPATH=src && \
python src/evaluation/scripts/eval_proposal_pool_recall.py \
  --data-root data/embodiedscan \
  --pack-name pack_vdetr_bip3d \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json \
  --output-json /home/ysh/.super-orchestrator/3d/artifacts/W5_recall_pack_vdetr_bip3d_keyframe.json \
  --use-keyframe-pool true

python src/evaluation/scripts/eval_proposal_pool_recall.py \
  --data-root data/embodiedscan \
  --pack-name pack_vdetr_bip3d \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json \
  --output-json /home/ysh/.super-orchestrator/3d/artifacts/W5_recall_pack_vdetr_bip3d_scene.json \
  --use-keyframe-pool false
```

## Prepared Artifact Counts

| artifact | count |
|---|---:|
| `pack_vdetr` directories | 220 |
| `pack_bip3d` directories | 220 |
| `pack_vdetr_bip3d` directories | 220 |
| `pack_vdetr_bip3d/proposals.jsonl` | 220 |
| `pack_vdetr_bip3d/visibility.json` | 220 |
| `pack_vdetr_bip3d/samples/*.json` | 2000 |
| `pack_vdetr_bip3d/README.md` | 220 |
| non-empty merged `annotated/` directories | 0 |

Merged proposal counts:

| scenes | total proposals | min proposals | median proposals | max proposals |
|---:|---:|---:|---:|---:|
| 220 | 107080 | 292 | 512 | 512 |

## Caveats

- The metric is recall-only. It does not measure mAP, precision, or Stage 2
  selection accuracy.
- 9-DoF oriented IoU is computed by
  `benchmarks.embodiedscan_eval.compute_oriented_iou_3d`.
- Raw BIP3D detector records contain 1000 proposals per scene; the pack
  converter capped `pack_bip3d` at 256 proposals per scene to match the current
  detector-pack workflow. The merged pack therefore has 292-512 proposals per
  scene, not 1256 proposals per scene.
- BIP3D outputs include duplicate-class winners on the same anchor because the
  postprocessor flattens `num_anchor x num_classes`. These inflate the raw
  detector proposal count without harming recall, which is the only metric in
  this study.
- BIP3D used checkpoint `det_rgbd_dat`
  (`external/BIP3D/checkpoints/bip3d_det_rgbd_dat.pth`, AP@0.25=23.24
  detection), not a grounding checkpoint.
- DEGround was the original target, but only paper-level materials were
  available locally; BIP3D is the available substitute used for this recall
  comparison.
- No SQLite row was inserted. The canonical `runs.sqlite` table tracks
  benchmark run scores, while this document records detector-pool recall before
  Stage 2.
