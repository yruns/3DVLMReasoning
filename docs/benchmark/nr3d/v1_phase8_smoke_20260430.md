# v1 Phase8 Smoke - 2026-04-30

## Run Identity

- Branch: `feat/nr3d-vg-benchmark`
- Tip commit at run start: `3c491a9`
- Worktree: `/home/ysh/codecase/3DVLMReasoning/.worktrees/nr3d-vg-benchmark`
- Internal version: `v1_phase8_smoke`
- Outcome: `endpoint-unreachable, plumbing-verified-up-to-Stage-2-call`

## What Changed

This smoke uses the new Phase 8 GT-ConceptGraph path for NR3D test scenes:

- `Nr3dDataset.from_path(..., bbox_source="phase8_gt_cg")`
- `prepare_pack_v1_inputs_nr3d.py`
- `run_nr3d_vg_side_by_side.py`

The Phase 8 GT-CG producer output was already present under
`data/nr3d/scannet/<scene>/conceptgraph/`.

## Exact Commands

Loader sanity:

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
print(f'sample[0]: scan_id={ds[0].scan_id} target={ds[0].target} bbox={[round(x,3) for x in ds[0].gt_bbox_3d]}')
"
```

Smoke sample selection:

```text
tmp/nr3d_artifacts/v1_smoke20_sample_ids.json
```

Pack prep:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --split test
```

Runner:

```bash
. .venv-agents/bin/activate
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --output-dir tmp/nr3d_eval_v1_smoke20 \
    --workers 1 \
    --sample-retries 0
```

## Raw Artifacts

- Sample IDs: `tmp/nr3d_artifacts/v1_smoke20_sample_ids.json`
- Prepared packs:
  - `data/nr3d/scannet/scene0011_00/pack_nr3d_v1`
  - `data/nr3d/scannet/scene0015_00/pack_nr3d_v1`
  - `data/nr3d/scannet/scene0019_00/pack_nr3d_v1`
  - `data/nr3d/scannet/scene0025_00/pack_nr3d_v1`
  - `data/nr3d/scannet/scene0030_00/pack_nr3d_v1`
- Runner output dir: `tmp/nr3d_eval_v1_smoke20`
- Logs:
  - `/tmp/nr3d_loader_sanity.log`
  - `/tmp/nr3d_smoke_prep.log`
  - `/tmp/nr3d_smoke_run.log`

## Fold

- Split: NR3D `test`
- Size: 20 utterances
- Scenes: `scene0011_00`, `scene0015_00`, `scene0019_00`, `scene0025_00`, `scene0030_00`
- Selection: first 4 non-blacklisted, non-clothing test utterances per selected scene
  with an existing Phase 8 GT-CG package.
- Judge model: none; scoring is programmatic 3D IoU when predictions exist.

## Results

Loader sanity:

```text
loaded=200 stats={'total_loaded': 200, 'skipped_missing_scene': 0, 'skipped_missing_or_ambiguous_bbox': 0, 'skipped_blacklist': 0, 'skipped_clothes': 0, 'skipped_correct_guess_filter': 0, 'skipped_mentions_target_class_filter': 0}
sample[0]: scan_id=scannet/scene0164_00 target=kitchen cabinet bbox=[-0.797, -1.207, 1.683, 0.691, 0.166, 0.84, 0.0, 0.0, 0.0]
```

Pack prep:

```text
Built 20 NR3D samples (split=test, stats={'total_loaded': 20, 'skipped_missing_scene': 0, 'skipped_missing_or_ambiguous_bbox': 0, 'skipped_blacklist': 0, 'skipped_clothes': 0, 'skipped_correct_guess_filter': 0, 'skipped_mentions_target_class_filter': 0})
wrote 20 sample artifacts under data/nr3d/scannet/<scene>/pack_nr3d_v1/
```

Runner:

```text
[Stage2DeepResearchAgent] task=visual_grounding plan_mode=brief keyframes=5 max_turns=6
[ModelHubHttpClient] transport failure on attempt 1/5: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)
[ModelHubHttpClient] transport failure on attempt 2/5: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)
[ModelHubHttpClient] transport failure on attempt 3/5: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)
[ModelHubHttpClient] transport failure on attempt 4/5: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1016)
openai.APIConnectionError: Connection error.
```

No `side_by_side.json` was written because the first Stage 2 model call failed
before a prediction existed.

## Metrics

No Acc@0.25, Acc@0.50, or mean IoU is available for this run. The smoke reached
the Stage 2 model call and failed on backend transport.

## SQLite

Not ingested. The canonical ingester requires `side_by_side.json` with non-empty
`per_sample`; this endpoint-unreachable smoke produced neither because it failed
before the first sample completed.

## Caveats

- Phase 8 pkl files are large gzip pickles. The loader supports a `sample_ids`
  filter so smoke prep touches only requested targets.
- Phase 8 pkl name and producer metadata state the boxes are axis-aligned. The
  loader converts 8 corners to `[cx, cy, cz, dx, dy, dz, 0, 0, 0]` directly.
  This intentionally avoids Open3D OBB/qhull conversion, which was too slow for
  real Phase 8 gzip payloads during the smoke.
- The conda `conceptgraph` env is Python 3.10 and cannot import `deepagents`.
  The runner smoke used `.venv-agents` Python 3.11.

## Comparison

No prior NR3D pack-v1 pipeline run exists in this directory. This v1 smoke
establishes that NR3D test GT boxes, pack prep, and runner I/O are wired through
the first Stage 2 call.
