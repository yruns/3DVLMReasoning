# NR3D v16 select_by_text bbox-spatial strat600 coverage audit

This is a tool-coverage audit, not an agent leaderboard run. It measures
`select_by_text` / `KeyframeSelector.select_keyframes_v2` after the bbox-aware
spatial checker and hard quick-filter fallthrough repair.

Question:

> If `select_by_text` is called once with the raw NR3D query and `k=3`, do the
> returned keyframes include at least one frame where the GT `target_id` is
> visible?

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `analysis/nr3d-select-by-text-coverage` |
| Head commit at launch | `43aed24` |
| Run-time code commit | `43aed24` - no worktree drift |
| Relevant code change | `43aed24` adds bbox-aware `SpatialRelationChecker`, all-pair `between`, hard quick-filter fallthrough, and compact `execution_trace` metadata |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold size | 600 canonical stratified samples |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_select_by_text_audit_bbox_spatial_strat600_20260524/` |
| Merged summary | `tmp/nr3d_select_by_text_audit_bbox_spatial_strat600_20260524/merged_summary.json` |
| Durable summary copy | `docs/benchmark/nr3d/assets/v16_select_by_text_bbox_spatial_strat600_coverage_20260524_merged_summary.json` |
| Started | `2026-05-24T23:23:44+08:00` |
| Completed | `2026-05-24T23:39:00+08:00` |
| Workers | 20 shard processes |
| Tool-equivalent call | `select_by_text(query=<raw NR3D query>, k=3, hidden_categories=[], use_visual_context=False, viewpoint_aware=True)` |
| Judge model | none - deterministic coverage against visibility indices |

## Exact Command

The 600 samples were split into 20 shard files under:

```bash
tmp/nr3d_select_by_text_audit_bbox_spatial_strat600_20260524/shards/
```

Each shard ran:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
.venv/bin/python scripts/audit_select_by_text_nr3d.py \
  --sample-ids tmp/nr3d_select_by_text_audit_bbox_spatial_strat600_20260524/shards/shard_XX.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --k 1,2,3 \
  --output tmp/nr3d_select_by_text_audit_bbox_spatial_strat600_20260524/outputs/shard_XX.json \
  --max-selector-cache-size 2 \
  --viewpoint-aware \
  --resume
```

The tmux runner was:

```bash
tmp/nr3d_select_by_text_audit_bbox_spatial_strat600_20260524/run_parallel.sh
```

All 20 shard processes completed with `ALL_SHARDS_DONE status=0`.

## Headline Coverage

| Slice | n | hit@1 | hit@2 | hit@3 | empty frames | viewpoint-context samples |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 600 | 56.50 | 68.50 | **71.17** | 13.50 | 97 |
| Easy | 290 | 66.90 | 76.21 | **78.28** | 9.31 | 46 |
| Hard | 310 | 46.77 | 61.29 | **64.52** | 17.42 | 51 |
| View-Dep | 211 | 56.40 | 67.77 | **70.14** | 14.69 | 95 |
| View-Indep | 389 | 56.56 | 68.89 | **71.72** | 12.85 | 2 |

Headline answer: **427 / 600 = 71.17 %** of strat600 samples have at least one
returned keyframe containing the target id.

## Comparison To v15

Previous record: [`v15_select_by_text_viewpoint_prompt_strat600_coverage_20260524.md`](v15_select_by_text_viewpoint_prompt_strat600_coverage_20260524.md),
runtime `9c62e51`, same fold and same `viewpoint_aware=True` call shape, before
the bbox-aware spatial checker repair.

| Slice | v15 hit@3 | v16 hit@3 | Delta | v15 empty | v16 empty | Empty delta |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 69.17 | 71.17 | +2.00 pp | 106 | 81 | -25 |
| Easy | 73.79 | 78.28 | +4.49 pp | 45 | 27 | -18 |
| Hard | 64.84 | 64.52 | -0.32 pp | 61 | 54 | -7 |
| View-Dep | 73.93 | 70.14 | -3.79 pp | 29 | 31 | +2 |
| View-Indep | 66.58 | 71.72 | +5.14 pp | 77 | 50 | -27 |

Sample-level hit@3 transitions vs v15:

| Transition | Count |
|---|---:|
| hit -> hit | 373 |
| miss -> miss | 131 |
| miss -> hit | 54 |
| hit -> miss | 42 |

Empty/non-empty transitions:

| Transition | Count |
|---|---:|
| empty -> empty | 56 |
| empty -> non-empty | 50 |
| non-empty -> empty | 25 |
| non-empty -> non-empty | 469 |

Parse/execution status also shifted in the expected direction:

| Status | v16 count |
|---|---:|
| `direct_grounded` | 453 |
| `proxy_grounded` | 52 |
| `context_only` | 14 |
| `no_evidence` | 81 |

`no_evidence` drops from **106** in v15 to **81** in v16.

## Interpretation

The bbox-aware spatial repair is a positive selector-coverage change overall:
hit@3 improves by **+12 samples** and empty returned frames drop by **25
samples** on the canonical strat600 fold.

The gain is not uniform. It is concentrated in view-independent and easy cases,
which matches the repaired relations (`inside`, `on`, `above/below`,
`near/next_to`, `between`) being mostly world-frame geometric constraints.
View-dependent hit@3 regresses by **-3.79 pp**, despite nearly unchanged
viewpoint-context emission. That likely comes from different hard world-frame
spatial hypotheses becoming executable and changing the target object set before
keyframe ranking, not from the viewpoint parser itself.

Next diagnostic should sample:

1. `empty -> non-empty` recoveries to confirm they are genuine spatial false
   negative fixes;
2. `hit -> miss` regressions, especially view-dependent rows, to check whether
   bbox-aware world-frame relations are over-accepting anchors/targets before
   viewer-frame ranking can help;
3. remaining 81 `no_evidence` rows using the new executor trace path.
