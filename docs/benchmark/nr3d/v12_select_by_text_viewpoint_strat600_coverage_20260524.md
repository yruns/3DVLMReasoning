# NR3D v12 select_by_text viewpoint strat600 coverage audit

This is a tool-coverage audit, not an agent leaderboard run. It measures the
latest `select_by_text` / `KeyframeSelector.select_keyframes_v2` behavior after
the viewpoint-hypothesis schema and strict no-drift executor changes.

Question:

> If `select_by_text` is called once with the raw NR3D query and `k=3`, do the
> returned keyframes include at least one frame where the GT `target_id` is
> visible?

The production tool caps `k` at 3, so the headline is `hit@3`.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `analysis/nr3d-select-by-text-coverage` |
| Head commit at launch | `b8582d6` |
| Run-time code commit | `b8582d6` - no worktree drift |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold size | 600 canonical stratified samples |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_select_by_text_audit_viewpoint_latest_strat600_20260524/` |
| Merged summary | `tmp/nr3d_select_by_text_audit_viewpoint_latest_strat600_20260524/merged_summary.json` |
| Started | `2026-05-24T18:06:27+08:00` |
| Completed | `2026-05-24T18:23:01+08:00` |
| Workers | 16 shard processes |
| Tool-equivalent call | `select_by_text(query=<raw NR3D query>, k=3, hidden_categories=[], use_visual_context=False)` |
| Judge model | none - deterministic coverage against visibility indices |

## Exact Command

The 600 samples were split into 16 shard files under:

```bash
tmp/nr3d_select_by_text_audit_viewpoint_latest_strat600_20260524/shards/
```

Each shard ran:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
.venv/bin/python scripts/audit_select_by_text_nr3d.py \
  --sample-ids tmp/nr3d_select_by_text_audit_viewpoint_latest_strat600_20260524/shards/shard_XX.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --k 1,2,3 \
  --output tmp/nr3d_select_by_text_audit_viewpoint_latest_strat600_20260524/outputs/shard_XX.json \
  --max-selector-cache-size 2 \
  --resume
```

The tmux runner was:

```bash
tmp/nr3d_select_by_text_audit_viewpoint_latest_strat600_20260524/run_parallel.sh
```

All 16 shard processes completed with `ALL_SHARDS_DONE status=0`.

## Headline Coverage

| Slice | n | hit@1 | hit@2 | hit@3 | empty frames | hit@3 when non-empty |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 600 | 47.50 | 57.83 | **60.50** | 27.17 | 83.07 |
| Easy | 290 | 52.76 | 61.72 | **63.10** | 26.55 | 85.92 |
| Hard | 310 | 42.58 | 54.19 | **58.06** | 27.74 | 80.36 |
| View-Dep | 211 | 35.55 | 45.97 | **47.87** | 39.81 | 79.53 |
| View-Indep | 389 | 53.98 | 64.27 | **67.35** | 20.31 | 84.52 |

Headline answer: **363 / 600 = 60.50 %** of strat600 samples have at least one
returned keyframe containing the target id.

## Status Breakdown

| Field | Count |
|---|---:|
| Samples | 600 |
| Missing samples after merge | 0 |
| Duplicate sample rows | 0 |
| Script errors | 0 |
| Empty `pred_top_k` / no returned frames | 163 |
| Non-empty responses | 437 |

Parser / executor statuses:

| Status | Count |
|---|---:|
| `direct_grounded` | 356 |
| `no_evidence` | 163 |
| `proxy_grounded` | 50 |
| `context_only` | 31 |

## Comparison To Previous Coverage Audit

Previous record: [`v11_select_by_text_strat600_coverage_20260524.md`](v11_select_by_text_strat600_coverage_20260524.md),
runtime `45a2dae`.

| Metric | v11 `45a2dae` | v12 `b8582d6` | Delta |
|---|---:|---:|---:|
| Overall hit@3 | 62.17 | 60.50 | -1.67 pp |
| hit@3 when non-empty | 83.07 | 83.07 | ~0.00 pp |
| Empty returned frames | 25.17 | 27.17 | +2.00 pp |
| View-Dep hit@3 | 50.71 | 47.87 | -2.84 pp |
| View-Indep hit@3 | 68.38 | 67.35 | -1.03 pp |

The non-empty quality is unchanged. The aggregate drop is from more empty /
`no_evidence` results after the latest viewpoint-hypothesis prompt/schema
changes.

## Interpretation

For the latest code, `select_by_text` is still strong when it returns frames:
`363 / 437 = 83.07 %` non-empty hit@3. The production-facing probability is
lower, **60.50 %**, because `163 / 600` raw queries return no keyframes.

The weak slice remains view-dependent NR3D: **47.87 %** hit@3 with **39.81 %**
empty responses. The viewpoint-aware schema work added the representation
needed to reason about these cases, but production `select_by_text` still runs
with `viewpoint_aware=False` for strict no-drift, so this audit does not yet
benefit from viewer-frame execution.
