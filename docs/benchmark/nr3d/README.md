# NR3D VG Evaluation Results

This directory tracks NR3D visual-grounding evaluations for the Stage-2
task-pack pipeline.

**Benchmark:** NR3D ScanNet visual grounding (train split for local 9-DoF scoring; test split inference-only until GT source is wired)
**Metric:** 9-DoF oriented 3D IoU (matching EmbodiedScan v3 setup), reported as Acc@0.25, Acc@0.50, mean IoU
**Judge:** none; scoring is programmatic IoU against EmbodiedScan PKL bbox

## Version Timeline

| Version | Date | Acc@0.25 | Acc@0.50 | mean IoU | Eval Scale | Key Change |
|---------|------|---------:|---------:|---------:|------------|------------|
| _none yet_ | - | - | - | - | - | Loader, evaluator, adapter, ingester, and docs scaffolded; first eval run pending |

## Current Interpretation

No NR3D pipeline version has been evaluated yet. This scaffold exists so the
first evaluation can leave a durable process record immediately, including
the raw artifact directory, SQLite ingestion, exact fold, and metric query.

The Phase-2 plumbing path loads the canonical NR3D CSV, derives train/test
membership from upstream scene lists, filters bad contexts and clothing rows by
default, and scores predictions against 9-DoF oriented boxes sourced from local
EmbodiedScan PKLs. On this checkout, only EmbodiedScan train+val PKLs expose
GT `instances`; the EmbodiedScan test PKL withholds instances.

## Reproduction Pattern

Download the raw NR3D annotation files:

```bash
bash scripts/download_nr3d.sh data/nr3d
```

Load samples with EmbodiedScan PKLs as the bbox oracle:

```bash
PYTHONPATH=src python -c "
from benchmarks.nr3d_loader import Nr3dDataset
ds = Nr3dDataset.from_path(
    data_root='data/nr3d',
    embodiedscan_data_root='data/embodiedscan',
    split='train',
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

The runner and pack-prep entrypoints are intentionally deferred until a
follow-up branch.

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

- GT bboxes are sourced from EmbodiedScan train + val PKLs (which hold
  `instances`), not ScanNet raw aggregation. The EmbodiedScan test PKL
  withholds GT instances.
- As a consequence: NR3D **train** split (511 scenes) is fully evaluable
  against our pipeline. Every NR3D-train scene has a bbox in either ES train
  or ES val PKL.
- NR3D **test** split (130 scenes) parses cleanly from `nr3d.csv`, but
  `Nr3dDataset.from_path(split="test")` currently yields zero samples because
  every test scene's GT bbox lookup fails through
  `skipped_missing_or_ambiguous_bbox`. An inference-only mode (yielding samples
  with `gt_bbox_3d=None`) is a follow-up; until then the headline NR3D metric
  we report is the **train**-split number.
- View-dep / view-indep breakdown not implemented (no canonical word list).
- Easy / Hard breakdown not implemented (data is on the sample; aggregator is
  a follow-up).
