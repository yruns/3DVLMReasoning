# EmbodiedScan VG Evaluation Results

This directory tracks EmbodiedScan visual-grounding evaluations for the
Stage-2 task-pack pipeline.

**Benchmark:** EmbodiedScan ScanNet visual grounding
**Metric:** 9-DoF oriented 3D IoU, reported as mean IoU, Acc@0.25, Acc@0.50
**Judge:** none; scoring is programmatic IoU against the annotation bbox

## Version Timeline

| Version | Date | Acc@0.25 | Acc@0.50 | mean IoU | Eval Scale | Key Change |
|---------|------|---------:|---------:|---------:|------------|------------|
| [v6](v6_conceptgraph_recall_20260430.md) | 2026-04-30 | n/a | n/a | n/a | 2000Q recall study | ConceptGraph bbox_np proposal pool added; recall only 0.35%@0.25; 3-way union unchanged vs v5 |
| [v5](v5_deground_recall_20260430.md) | 2026-04-30 | n/a | n/a | n/a | 2000Q recall study | BIP3D detector pool added; recall +31.7pp@0.25 vs V-DETR; merged pool ready for Stage 2 |
| [v4](v4_vdetr_pool_prepared_20260430.md) | 2026-04-30 | n/a | n/a | n/a | 220 scenes / 2000Q prepared | V-DETR detector pool engineering-complete; metrics pending Stage 2 endpoint access |
| [v3](v3_projectable_2k_20260429.md) | 2026-04-29 | 89.15 | 89.15 | 89.20 | 2000Q smoke | Projectable-only unique-target sweep with checkpointed adaptive concurrency |
| [v2](v2_projectable_20260429.md) | 2026-04-29 | 94.03 | 94.03 | 94.08 | 67Q smoke | Projectable-only corrected GT-pool prep, 12-worker run |
| [v1](v1_gt_pool_20260428.md) | 2026-04-28 | 91.18 | 91.18 | 91.18 | 68Q smoke | GT-pool oracle pack_v1, per-scene artifacts |

## Current Interpretation

The v1/v2 runs are pipeline smokes, and v3 is the first broad 2k smoke. They
verify that the pack_v1 visual-grounding agent can consume per-scene GT proposal
pools, inspect marked keyframes, submit a selected EmbodiedScan `bbox_id`, and
be scored by oriented 3D IoU. v3 is the current large smoke because it uses the
corrected projectable-only preparation path at 2000 samples.

Do not compare these small smokes directly to public detector-based
EmbodiedScan numbers:

- the proposal pool is oracle GT, not vDETR or another detector;
- the frozen sample counts changed as the preparation filter was corrected;
- v1/v2 are below the repo's n>=200 benchmark-claim threshold; v3 is large
  enough for a broad smoke but still not detector-comparable.

The v4-v6 runs are detector-pool engineering and recall studies. They report
oracle proposal-pool recall, not Stage 2 selected-box accuracy. v6 shows that
ConceptGraph is not useful on this fold by itself: after the `bbox_np`
correction it reaches only 0.35% Recall@0.25, and the 3-way union leaves the
v5 V-DETR+BIP3D headline recall unchanged at 77.35% Recall@0.25.

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
    --sample-ids tmp/embodiedscan_artifacts/v1_prepared_projectable_20260429_sample_ids.json \
    --output-dir tmp/embodiedscan_eval_projectable_w12_20260429_150000 \
    --data-root data/embodiedscan \
    --workers 12 \
    --sample-retries 2
```

Ingest the output:

```bash
PYTHONPATH=src python scripts/ingest_embodiedscan_run.py \
    --output-dir tmp/embodiedscan_eval_v3_projectable_2k_w32_20260429_1730 \
    --run-id v3_projectable_2k_adaptive \
    --branch feat/explore_3dbbox \
    --commit 370b12e \
    --backend pack_v1 \
    --judge-model gpt-5.4-2026-03-05 \
    --notes "Projectable GT-pool pack_v1 2k unique-target smoke" \
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
