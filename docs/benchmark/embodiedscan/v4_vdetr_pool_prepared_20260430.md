# v4 V-DETR pool prepared

**Date:** 2026-04-30
**Branch:** `feat/explore_3dbbox`
**Run-time tip:** `4cd6b51` plus dirty V-DETR, pack-prep, runner, and test changes
**Eval scale:** v3 frozen 2000-sample scope, 220 ScanNet scenes
**Proposal source:** V-DETR full-scene detector pool, capped at 256 proposals per scene
**Judge:** none; Stage 2 did not run because the ModelHub endpoint was unreachable
**Raw detector artifacts:** `tmp/embodiedscan_vdetr_fullscene_outputs_full_20260429/merged/`
**Prepared meshes:** `data/scannet_meshes_v3/`
**Prepared pack:** `data/embodiedscan/<scene>/pack_vdetr/`
**Visual validation:** `tmp/embodiedscan_visual_validation_pack_vdetr_20260429/`
**SQLite:** no row inserted; metrics are intentionally deferred until Stage 2 runs

## What Changed

This version moves EmbodiedScan visual grounding from the GT oracle proposal
pool toward a detector-backed proposal pool.

- P0-A generalized the pack loader and side-by-side runner with `--pack-name`,
  preserving `pack_v1` defaults while allowing `pack_vdetr` sample loading and
  non-colliding checkpoints under `per_sample/pack_v1__pack_vdetr/`.
- P0-B materialized full-scene V-DETR inputs for the 220 scenes in the v3 frozen
  2k scope and ran V-DETR across GPUs `0,2,3,4,5,6,7`.
- P0-C added `src/evaluation/scripts/prepare_detector_pack_inputs.py`, which
  converts detector records into per-scene `pack_vdetr` artifacts:
  `proposals.jsonl`, `visibility.json`, `annotated/frame_<id>.png`, and
  `samples/<target_id>.json`.
- P0-D prepared the v3 frozen 2k `pack_vdetr` scope, generated a 20-sample
  visual validation grid, and preserved the 50-sample Stage 2 fold for a future
  rerun.

Stage 2 metrics are not reported in this document. The internal ModelHub
endpoint `aidp-i18ntt-sg.tiktok-row.net` was unreachable from this host with
both the local proxy and proxy variables unset. The SQLite ingest was skipped
because there is no valid metric set.

## Commands

Materialize ScanNet-style PLY meshes from local `.npy` arrays:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph && \
python scripts/build_scannet_ply_from_npy.py \
  --db docs/benchmark/embodiedscan/runs.sqlite \
  --run-id v3_projectable_2k_adaptive \
  --source-root /data1/dyj_dataset/scannet/train \
  --output-root data/scannet_meshes_v3 \
  --summary-json /home/ysh/.super-orchestrator/3d/artifacts/B_vdetr_materialization_summary.json \
  --min-success 220 \
  2>&1 | tee /tmp/vdetr-materialize-v3.log
```

Prepare full-scene detector inputs:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph && \
export PYTHONPATH=src && \
python -m benchmarks.embodiedscan_bbox_feasibility.cli prepare-inputs \
  --data-root data/embodiedscan \
  --scene-data-root data/embodiedscan \
  --scannet-root data/scannet_meshes_v3 \
  --output-dir tmp/embodiedscan_vdetr_fullscene_inputs_full_20260429 \
  --conditions scannet_full \
  --scene-level-full \
  --max-points 40000 \
  --scene-ids $(python - <<'PY'
import json
print(" ".join(json.load(open("/home/ysh/.super-orchestrator/3d/artifacts/B_vdetr_scene_list.json"))))
PY
) \
  2>&1 | tee /tmp/vdetr-full-prepare.log
```

Run V-DETR over seven GPU shards:

```bash
for gpu in 0 2 3 4 5 6 7; do
  tmux new-session -d -s "vdetr-full-gpu${gpu}" \
    "cd /home/ysh/codecase/3DVLMReasoning && \
     source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph && \
     export PYTHONPATH=src && \
     python -m benchmarks.embodiedscan_bbox_feasibility.cli run-detector \
       --inputs-jsonl tmp/embodiedscan_vdetr_fullscene_inputs_full_20260429/shards/shard_gpu${gpu}.jsonl \
       --output-dir tmp/embodiedscan_vdetr_fullscene_outputs_full_20260429/shard_gpu${gpu} \
       --method 3d-vdetr \
       --detector-profile vdetr \
       --vdetr-repo-dir external/V-DETR \
       --vdetr-checkpoint external/V-DETR/checkpoints/scannet_540ep.pth \
       --vdetr-python /home/ysh/miniconda3/envs/vdetr/bin/python \
       --vdetr-num-points 40000 \
       --vdetr-conf-thresh 0.05 \
       --vdetr-top-k 256 \
       --cwd /home/ysh/codecase/3DVLMReasoning \
       --cuda-device ${gpu} \
       2>&1 | tee /tmp/vdetr-full-gpu${gpu}.log"
done
```

Merge detector records:

```bash
mkdir -p tmp/embodiedscan_vdetr_fullscene_outputs_full_20260429/merged
cat tmp/embodiedscan_vdetr_fullscene_outputs_full_20260429/shard_gpu*/detector_records.jsonl \
  > tmp/embodiedscan_vdetr_fullscene_outputs_full_20260429/merged/detector_records.jsonl
```

Recover the v3 frozen 2k sample scope:

```bash
python - <<'PY'
import json, os, sqlite3
db = sqlite3.connect("docs/benchmark/embodiedscan/runs.sqlite")
cur = db.cursor()
cur.execute("""
  SELECT scene_id||'::'||target_id
  FROM samples
  WHERE run_id='v3_projectable_2k_adaptive'
  ORDER BY scene_id,target_id
""")
ids = [r[0] for r in cur.fetchall()]
out = "/home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_sample_ids.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(ids, f, indent=2)
print(out, len(ids))
PY
```

Convert detector records into `pack_vdetr` in eight disjoint scene-balanced
chunks:

```bash
for idx in 00 01 02 03 04 05 06 07; do
  tmux new-session -d -s "p0d-pack-vdetr-${idx}" \
    "cd /home/ysh/codecase/3DVLMReasoning && \
     source .venv-agents/bin/activate && export PYTHONPATH=src && \
     python src/evaluation/scripts/prepare_detector_pack_inputs.py \
       --detector-records tmp/embodiedscan_vdetr_fullscene_outputs_full_20260429/merged/detector_records.jsonl \
       --infos-pkl data/embodiedscan/embodiedscan_infos_val.pkl \
       --vg-json data/embodiedscan/embodiedscan_val_vg.json \
       --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_v3_full_2k_chunks/chunk_${idx}.json \
       --data-root data/embodiedscan \
       --pack-name pack_vdetr \
       --split val \
       2>&1 | tee /tmp/pack-vdetr-full-chunk${idx}.log"
done
```

Prepare the deterministic 50-sample smoke list for future Stage 2:

```bash
python - <<'PY'
import json, os, sqlite3
db = sqlite3.connect("docs/benchmark/embodiedscan/runs.sqlite")
cur = db.cursor()
cur.execute("""
  SELECT scene_id || '::' || target_id AS sample_id
  FROM samples
  WHERE run_id='v3_projectable_2k_adaptive'
  ORDER BY scene_id, target_id
  LIMIT 50
""")
ids = [r[0] for r in cur.fetchall()]
out = "/home/ysh/.super-orchestrator/3d/artifacts/D_smoke50_sample_ids.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(ids, f, indent=2)
print(out, len(ids))
PY
```

Generate the visual validation page:

```bash
source .venv-agents/bin/activate && \
python - <<'PY'
# Inline validation helper used in P0-D. It reads
# /home/ysh/.super-orchestrator/3d/artifacts/D_smoke50_sample_ids.json,
# copies the first 20 samples' raw RGB, pack_v1 annotated frame, and
# pack_vdetr annotated frame into
# tmp/embodiedscan_visual_validation_pack_vdetr_20260429/thumbs/, then writes
# tmp/embodiedscan_visual_validation_pack_vdetr_20260429/index.html.
PY
```

Attempted Stage 2 smoke command, preserved only as failure evidence:

```bash
source .venv-agents/bin/activate && export PYTHONPATH=src && \
python src/evaluation/scripts/run_embodiedscan_vg_side_by_side.py \
  --sample-ids /home/ysh/.super-orchestrator/3d/artifacts/D_smoke50_sample_ids.json \
  --data-root data/embodiedscan \
  --output-dir tmp/embodiedscan_eval_vdetr_smoke50_20260429 \
  --pack-name pack_vdetr \
  --workers 4 \
  --sample-retries 2 \
  2>&1 | tee /tmp/eval-vdetr-smoke50.log
```

## Prepared Artifact Counts

| Artifact | Count |
|----------|------:|
| V-DETR detector input rows | 220 |
| Merged detector records | 220 |
| `pack_vdetr` directories | 220 |
| `pack_vdetr/proposals.jsonl` | 220 |
| `pack_vdetr/visibility.json` | 220 |
| `pack_vdetr/samples/*.json` | 2000 |
| Visual validation cards | 20 |
| Visual validation thumbnails | 60 |

## V-DETR Proposal Stats

| scenes | failures | empty scenes | min proposals | median proposals | max proposals |
|-------:|---------:|-------------:|--------------:|-----------------:|--------------:|
| 220 | 0 | 0 | 36 | 256 | 256 |

Top class distribution from the merged detector records:

| Class | Proposals |
|-------|----------:|
| door | 4680 |
| cabinet | 4368 |
| garbagebin | 4361 |
| picture | 3873 |
| table | 3661 |
| chair | 3316 |
| window | 3154 |
| desk | 2963 |
| curtain | 2603 |
| refrigerator | 2528 |
| sofa | 2427 |
| bookshelf | 2407 |
| sink | 2302 |
| counter | 1810 |
| toilet | 1761 |
| showercurtrain | 1633 |
| bed | 1631 |
| bathtub | 1282 |

## Visual Validation

The validation grid is:

```text
tmp/embodiedscan_visual_validation_pack_vdetr_20260429/index.html
```

It contains the first 20 deterministic smoke samples from the v3 frozen 2k
ordering. Each card shows the same frame across raw RGB, GT `pack_v1`
annotations, and V-DETR `pack_vdetr` annotations, plus proposal counts.

Manual spot check on `scene0006_00::2`:

- raw RGB rendered correctly;
- GT annotations are dense and include many small/semantic objects;
- V-DETR annotations are sparse and closed-set, with observed label drift
  such as a lamp-like object labeled as `picture`.

This is consistent with detector recall/class coverage becoming the ceiling
once the oracle GT proposal pool is replaced.

## Stage 2 Status

Stage 2 was not completed in this session. The 4-worker smoke produced
transport failures before any valid sample completed:

```text
tmp/embodiedscan_eval_vdetr_smoke50_20260429_w4_transportfail
/tmp/eval-vdetr-smoke50-w4-transportfail.log
```

Observed partial checkpoints:

```text
16 checkpoints, all status="error"
error_type="APIConnectionError"
error="Connection error."
```

A one-sample probe with the normal proxy environment also failed with
`APIConnectionError`. A no-proxy one-sample probe changed the failure mode to
backend timeouts:

```text
/tmp/eval-vdetr-smoke1-probe.log
/tmp/eval-vdetr-smoke1-noproxy.log
```

The 50-sample fold artifact is preserved for rerun:

```text
/home/ysh/.super-orchestrator/3d/artifacts/D_smoke50_sample_ids.json
```

## SQLite

No row was inserted into `docs/benchmark/embodiedscan/runs.sqlite`.

This is intentional: the run has no valid Stage 2 metrics. Insert a SQLite row
only after `run_embodiedscan_vg_side_by_side.py` completes against
`pack_vdetr` and produces a valid `side_by_side.json`.

Verification query:

```sql
SELECT count(*)
FROM runs
WHERE run_id IN (
  'v4_vdetr_pool_prepared_20260430',
  'v4_vdetr_smoke50_20260429'
);
```

Expected result for this document: `0`.

## Cross-Version Position

| Version | Fold | Proposal source | Stage 2 metrics | Notes |
|---------|------|-----------------|-----------------|-------|
| v3 | frozen 2k | GT `pack_v1` oracle | available | selection over annotated object boxes |
| v4 | same frozen 2k scene scope | V-DETR `pack_vdetr` detector pool | pending | engineering complete; endpoint access blocked Stage 2 |

v4 is not directly comparable to v3 yet because Stage 2 did not complete on a
matched fold. The durable result here is the detector-backed proposal pool and
visual validation artifacts.

## Caveats

- V-DETR is closed-set and cannot cover the long-tail EmbodiedScan referring
  vocabulary. Thin objects and categories outside the detector label set can be
  missed or mapped to nearby classes.
- Detector bboxes are AABB-only (`box_format=aabb_from_vdetr_corners`) with
  zero Euler angles, so future Acc@0.50 will be biased low for rotated and thin
  objects.
- Keyframe selection still uses the GT target visibility path inherited from
  pack preparation. This keeps the visibility upper bound GT-driven; only the
  proposal pool itself is detector-backed.
- `pack_vdetr` proposal ids are local detector ids, not EmbodiedScan GT
  `bbox_id`s.
- The working tree was dirty at doc time. Preserve the A/B/C/D source changes
  before expecting this to reproduce from a clean checkout.
