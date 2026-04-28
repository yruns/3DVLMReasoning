# EmbodiedScan VG Evaluation Results

This directory tracks EmbodiedScan visual-grounding evaluations for the
Stage-2 task-pack pipeline.

**Benchmark:** EmbodiedScan ScanNet visual grounding
**Metric:** 9-DoF oriented 3D IoU, reported as mean IoU, Acc@0.25, Acc@0.50
**Judge:** none; scoring is programmatic IoU against the annotation bbox

## Version Timeline

| Version | Date | Acc@0.25 | Acc@0.50 | mean IoU | Eval Scale | Key Change |
|---------|------|---------:|---------:|---------:|------------|------------|
| [v1](v1_gt_pool_20260428.md) | 2026-04-28 | 91.18 | 91.18 | 91.18 | 68Q smoke | GT-pool oracle pack_v1, per-scene artifacts |

## Current Interpretation

The v1 run is a pipeline smoke, not a headline benchmark result. It verifies
that the pack_v1 visual-grounding agent can consume per-scene GT proposal pools,
inspect marked keyframes, submit a selected EmbodiedScan `bbox_id`, and be
scored by oriented 3D IoU.

Do not compare this 68-question smoke directly to public detector-based
EmbodiedScan numbers:

- the proposal pool is oracle GT, not vDETR or another detector;
- two of the frozen 70 targets were skipped because the annotation reported no
  visible frames;
- n=68 is below the repo's n>=200 benchmark-claim threshold.

## Reproduction Pattern

Prepare a frozen list into per-scene pack_v1 artifacts:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs.py \
    --sample-ids tmp/embodiedscan_artifacts/v1_70_sample_ids.json \
    --data-root data/embodiedscan \
    --split val
```

Run the side-by-side pack_v1 agent:

```bash
PYTHONPATH=src python src/evaluation/scripts/run_embodiedscan_vg_side_by_side.py \
    --sample-ids tmp/embodiedscan_artifacts/v1_68_prepared_sample_ids.json \
    --output-dir tmp/embodiedscan_eval_v1_smoke_20260428_1749 \
    --data-root data/embodiedscan \
    --sample-retries 2
```

Ingest the output:

```bash
PYTHONPATH=src python scripts/ingest_embodiedscan_run.py \
    --output-dir tmp/embodiedscan_eval_v1_smoke_20260428_1749 \
    --run-id v1_gt_pool_smoke_68 \
    --branch feat/explore_3dbbox \
    --commit 3e4d108 \
    --notes "GT-pool pack_v1 70-sample smoke; 68 prepared after filtering zero-visible targets" \
    --db docs/benchmark/embodiedscan/runs.sqlite
```

## SQLite

Canonical DB: `docs/benchmark/embodiedscan/runs.sqlite`

```sql
SELECT run_id, n,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50
FROM runs;
```
