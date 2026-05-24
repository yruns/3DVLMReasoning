# NR3D v13 select_by_text viewpoint-aware strat600 coverage audit

This is a tool-coverage audit, not an agent leaderboard run. It measures
`select_by_text` / `KeyframeSelector.select_keyframes_v2` with
`viewpoint_aware=True` on the canonical NR3D strat600 fold.

Question:

> If `select_by_text` is called once with the raw NR3D query and `k=3`, do the
> returned keyframes include at least one frame where the GT `target_id` is
> visible?

The production tool still does not pass `viewpoint_aware=True`; this is an
opt-in diagnostic of the new viewpoint policy path.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `analysis/nr3d-select-by-text-coverage` |
| Head commit at launch | `f8e6d22` |
| Run-time code commit | `f8e6d22` - no worktree drift |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold size | 600 canonical stratified samples |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_select_by_text_audit_viewpoint_aware_fixed_strat600_20260524/` |
| Merged summary | `tmp/nr3d_select_by_text_audit_viewpoint_aware_fixed_strat600_20260524/merged_summary.json` |
| Started | `2026-05-24T18:51:18+08:00` |
| Completed | `2026-05-24T19:07:10+08:00` |
| Workers | 16 shard processes |
| Tool-equivalent call | `select_by_text(query=<raw NR3D query>, k=3, hidden_categories=[], use_visual_context=False, viewpoint_aware=True)` |
| Judge model | none - deterministic coverage against visibility indices |

## Exact Command

The 600 samples were split into 16 shard files under:

```bash
tmp/nr3d_select_by_text_audit_viewpoint_aware_fixed_strat600_20260524/shards/
```

Each shard ran:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
.venv/bin/python scripts/audit_select_by_text_nr3d.py \
  --sample-ids tmp/nr3d_select_by_text_audit_viewpoint_aware_fixed_strat600_20260524/shards/shard_XX.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --k 1,2,3 \
  --output tmp/nr3d_select_by_text_audit_viewpoint_aware_fixed_strat600_20260524/outputs/shard_XX.json \
  --max-selector-cache-size 2 \
  --viewpoint-aware \
  --resume
```

The tmux runner was:

```bash
tmp/nr3d_select_by_text_audit_viewpoint_aware_fixed_strat600_20260524/run_parallel.sh
```

All 16 shard processes completed with `ALL_SHARDS_DONE status=0`.

## Headline Coverage

| Slice | n | hit@1 | hit@2 | hit@3 | empty frames | hit@3 when non-empty |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 600 | 54.00 | 65.83 | **69.00** | 17.83 | 83.98 |
| Easy | 290 | 61.72 | 72.07 | **74.14** | 14.48 | 86.69 |
| Hard | 310 | 46.77 | 60.00 | **64.19** | 20.97 | 81.22 |
| View-Dep | 211 | 53.55 | 66.82 | **69.67** | 17.54 | 84.48 |
| View-Indep | 389 | 54.24 | 65.30 | **68.64** | 17.99 | 83.70 |

Headline answer: **414 / 600 = 69.00 %** of strat600 samples have at least one
returned keyframe containing the target id.

## Status Breakdown

| Field | Count |
|---|---:|
| Samples | 600 |
| Missing samples after merge | 0 |
| Duplicate sample rows | 0 |
| Script errors | 0 |
| Empty `pred_top_k` / no returned frames | 107 |
| Non-empty responses | 493 |

Parser / executor statuses:

| Status | Count |
|---|---:|
| `direct_grounded` | 409 |
| `no_evidence` | 107 |
| `proxy_grounded` | 65 |
| `context_only` | 19 |

## Comparison To v12 Legacy-Hard Audit

Previous record: [`v12_select_by_text_viewpoint_strat600_coverage_20260524.md`](v12_select_by_text_viewpoint_strat600_coverage_20260524.md),
runtime `b8582d6`, `viewpoint_aware=False`.

| Metric | v12 `viewpoint_aware=False` | v13 `viewpoint_aware=True` | Delta |
|---|---:|---:|---:|
| Overall hit@3 | 60.50 | 69.00 | +8.50 pp |
| hit@3 when non-empty | 83.07 | 83.98 | +0.91 pp |
| Empty returned frames | 27.17 | 17.83 | -9.33 pp |
| View-Dep hit@3 | 47.87 | 69.67 | +21.80 pp |
| View-Indep hit@3 | 67.35 | 68.64 | +1.29 pp |

Sample-level transition vs v12:

| Transition | Count |
|---|---:|
| hit -> hit | 346 |
| miss -> miss | 169 |
| miss -> hit | 68 |
| hit -> miss | 17 |
| empty -> non-empty | 70 |
| non-empty -> empty | 14 |

## Interpretation

The gain comes mainly from reducing the hard `no_evidence` wall. Empty responses
drop from 163 to 107, while non-empty hit@3 changes only slightly. This confirms
that the new path is improving recall by keeping more candidate sets alive, not
by dramatically improving frame ranking after evidence is found.

The improvement is concentrated in view-dependent NR3D: hit@3 rises from 47.87
to 69.67 and empty responses fall from 84 to 37. Easy view-dependent improves
from 53.26 to 81.52, and hard view-dependent improves from 43.70 to 60.50.

Important caveat: the parser logs still emitted no non-empty
`viewpoint_contexts` and no `reference_frame="viewer"` rows in this run. The
executor logs show 156 samples with `policy=rank_only`. Therefore this v13 gain
is mostly the Phase-1 viewpoint policy floor - demoting ambiguous directional
constraints to rank-only - rather than true viewer-frame geometric execution
from resolved viewpoint contexts.

## Aborted Attempt

An earlier diagnostic run at HEAD `b731986` was stopped after 54 completed rows
because `viewpoint_aware=True` exposed
`AttributeError: 'SelectConstraint' object has no attribute 'relation'` in
`_normalize_viewpoint_policies()`. The valid run above used HEAD `f8e6d22`,
which adds a regression test and handles `SpatialConstraint` and
`SelectConstraint` separately. Verification before relaunch:

```bash
.venv/bin/python -m pytest src/query_scene/tests -q
```

Result: `183 passed`.
