# NR3D VG Evaluation Results

This directory tracks NR3D visual-grounding evaluations for the Stage-2
task-pack pipeline.

**Benchmark:** NR3D ScanNet visual grounding (test split now has Phase 8 GT-CG bboxes for local 9-DoF scoring)
**Metric:** 9-DoF oriented 3D IoU (matching EmbodiedScan v3 setup), reported as Acc@0.25, Acc@0.50, mean IoU
**Judge:** none; scoring is programmatic IoU against EmbodiedScan PKL bbox

## Version Timeline

| Version | Date | Acc@0.25 | Acc@0.50 | mean IoU | Eval Scale | Key Change |
|---------|------|---------:|---------:|---------:|------------|------------|
| [v1_phase8_smoke](v1_phase8_smoke_20260430.md) | 2026-04-30 | - | - | - | 20Q smoke | Phase 8 GT-CG bbox source, NR3D pack prep, runner reached Stage 2 call; endpoint unreachable on Linux |
| [v1_phase8_smoke20_mac](v1_phase8_smoke20_mac_20260430.md) | 2026-04-30 | 0.7000 | 0.6500 | 0.6701 | 20Q smoke (same fold) | First green pack_v1 numbers on Mac (PCA-aligned 9-DoF, endpoint reachable). 13/20 IoU=1.0 (GT-pool inflation). |

## Current Interpretation

The first NR3D pack-v1 smoke (v1_phase8_smoke) reached the first Stage 2 model
call on Linux but the internal ModelHub endpoint returned `[SSL:
UNEXPECTED_EOF_WHILE_READING]` on every retry, so no metric was produced.

The same 20-sample fold was re-run on Mac (v1_phase8_smoke20_mac, 2026-04-30)
on tip `9115fd7` (commits since the original Linux smoke include the loader's
PCA-aligned 9-DoF OBB recovery for Phase 8 corners). The Mac endpoint is
reachable and produced the first green pack_v1 numbers: **Acc@0.25 = 70.0 %**,
**Acc@0.50 = 65.0 %**, **mean IoU = 0.6701** on 19 completed + 1 failed
samples. Of the 14 IoU ≥ 0.25 hits, 13 are exactly IoU = 1.0 because the
GT-pool setup includes every aggregation instance (including the GT itself) as
a candidate; a detector-pool variant is tracked separately.

The Phase-2 plumbing path loads the canonical NR3D CSV, derives train/test
membership from upstream scene lists, filters bad contexts and clothing rows by
default, and can still score train split predictions against EmbodiedScan PKL
boxes. The Phase-8 path adds local test split GT boxes from
`data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz`.

## Reproduction Pattern

Download the raw NR3D annotation files:

```bash
bash scripts/download_nr3d.sh data/nr3d
```

Load test samples with Phase 8 GT-CG boxes:

```bash
PYTHONPATH=src python -c "
from benchmarks.nr3d_loader import Nr3dDataset
ds = Nr3dDataset.from_path(
    data_root='data/nr3d',
    split='test',
    bbox_source='phase8_gt_cg',
    phase8_data_root='data/nr3d/scannet',
    max_samples=200,
)
print(f'loaded={len(ds)} stats={ds.stats}')
"
```

Ingest a future side-by-side output:

```bash
PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_<run> \
    --run-id <run> \
    --branch feat/nr3d-vg-benchmark \
    --commit <short_sha> \
    --backend pack_v1 \
    --judge-model none \
    --notes "NR3D pack_v1 run"
```

Prepare a Phase 8 pack:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --split test
```

Run the pack-v1 Stage 2 runner:

```bash
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --output-dir tmp/nr3d_eval_v1_smoke20 \
    --workers 1
```

## SQLite

Canonical DB: `docs/benchmark/nr3d/runs.sqlite`

```sql
SELECT run_id, n,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50
FROM runs;
```

## Caveats

- EmbodiedScan PKL boxes remain available through
  `bbox_source="embodiedscan_pkl"` for the train split.
- NR3D test split local scoring now uses Phase 8 GT-CG boxes through
  `bbox_source="phase8_gt_cg"`.
- Phase 8 boxes are axis-aligned 8-corner boxes; the current conversion emits
  zero Euler angles.
- View-dep / view-indep breakdown not implemented (no canonical word list).
- Easy / Hard breakdown not implemented (data is on the sample; aggregator is
  a follow-up).
