# NR3D v15 select_by_text viewpoint prompt strat600 coverage audit

This is a tool-coverage audit, not an agent leaderboard run. It measures
`select_by_text` / `KeyframeSelector.select_keyframes_v2` after the active
`src/query_scene/parsing` prompt and dynamic schema were updated to emit
viewpoint contexts.

Question:

> If `select_by_text` is called once with the raw NR3D query and `k=3`, do the
> returned keyframes include at least one frame where the GT `target_id` is
> visible, and does the active parser now emit usable viewpoint fields?

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `analysis/nr3d-select-by-text-coverage` |
| Head commit at launch | `9c62e51` |
| Run-time code commit | `9c62e51` - no worktree drift |
| Relevant code change | `0c6cb85` ports viewpoint prompt/schema to active parser; `3688096` records viewpoint stats in audit output |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold size | 600 canonical stratified samples |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_select_by_text_audit_viewpoint_prompt_strat600_20260524/` |
| Merged summary | `tmp/nr3d_select_by_text_audit_viewpoint_prompt_strat600_20260524/merged_summary.json` |
| Started | `2026-05-24T20:15:11+08:00` |
| Completed | `2026-05-24T20:32:01+08:00` |
| Workers | 20 shard processes |
| Tool-equivalent call | `select_by_text(query=<raw NR3D query>, k=3, hidden_categories=[], use_visual_context=False, viewpoint_aware=True)` |
| Judge model | none - deterministic coverage against visibility indices |

## Exact Command

The 600 samples were split into 20 shard files under:

```bash
tmp/nr3d_select_by_text_audit_viewpoint_prompt_strat600_20260524/shards/
```

Each shard ran:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
.venv/bin/python scripts/audit_select_by_text_nr3d.py \
  --sample-ids tmp/nr3d_select_by_text_audit_viewpoint_prompt_strat600_20260524/shards/shard_XX.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --k 1,2,3 \
  --output tmp/nr3d_select_by_text_audit_viewpoint_prompt_strat600_20260524/outputs/shard_XX.json \
  --max-selector-cache-size 2 \
  --viewpoint-aware \
  --resume
```

The tmux runner was:

```bash
tmp/nr3d_select_by_text_audit_viewpoint_prompt_strat600_20260524/run_parallel.sh
```

All 20 shard processes completed with `ALL_SHARDS_DONE status=0`.

## Headline Coverage

| Slice | n | hit@1 | hit@2 | hit@3 | empty frames | viewpoint-context samples |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 600 | 55.67 | 66.17 | **69.17** | 17.67 | 98 |
| Easy | 290 | 63.79 | 72.07 | **73.79** | 15.52 | 45 |
| Hard | 310 | 48.06 | 60.65 | **64.84** | 19.68 | 53 |
| View-Dep | 211 | 59.24 | 71.09 | **73.93** | 13.74 | 96 |
| View-Indep | 389 | 53.73 | 63.50 | **66.58** | 19.79 | 2 |

Headline answer: **415 / 600 = 69.17 %** of strat600 samples have at least one
returned keyframe containing the target id.

## Viewpoint Emission

Unlike v13, the active parser now emits actual viewpoint fields.

| Metric | Count |
|---|---:|
| Samples with non-empty `viewpoint_contexts` | 98 / 600 |
| View-dependent samples with non-empty `viewpoint_contexts` | 96 / 211 |
| View-independent samples with non-empty `viewpoint_contexts` | 2 / 389 |
| `reference_frame="viewer"` constraints/selectors | 117 |
| `reference_frame="object_local"` constraints/selectors | 11 |
| `reference_frame="ambiguous"` constraints/selectors | 142 |
| `execution_policy="soft"` constraints/selectors | 123 |
| `execution_policy="rank_only"` constraints/selectors | 154 |

The two view-independent context rows are not necessarily wrong labels: the
canonical view-dep flag is a token heuristic, while the utterances include
phrases such as "closest to you" or "nearest you" that imply observer position.

## Comparison To v13

Previous record: [`v13_select_by_text_viewpoint_aware_strat600_coverage_20260524.md`](v13_select_by_text_viewpoint_aware_strat600_coverage_20260524.md),
runtime `f8e6d22`, same `viewpoint_aware=True` switch but before the active
parser prompt/schema port.

| Slice | v13 hit@3 | v15 hit@3 | Delta |
|---|---:|---:|---:|
| Overall | 69.00 | 69.17 | +0.17 pp |
| Easy | 74.14 | 73.79 | -0.35 pp |
| Hard | 64.19 | 64.84 | +0.65 pp |
| View-Dep | 69.67 | 73.93 | +4.27 pp |
| View-Indep | 68.64 | 66.58 | -2.06 pp |

Empty returned frames are effectively unchanged: 107 -> 106.

Sample-level hit@3 transitions vs v13:

| Transition | Count |
|---|---:|
| hit -> hit | 385 |
| miss -> miss | 156 |
| miss -> hit | 30 |
| hit -> miss | 29 |

Empty/non-empty transitions:

| Transition | Count |
|---|---:|
| empty -> empty | 92 |
| empty -> non-empty | 15 |
| non-empty -> empty | 14 |
| non-empty -> non-empty | 479 |

## Interpretation

The prompt/schema port worked: v13 had zero parser-emitted viewpoint contexts,
while v15 emits them on 98 samples, almost entirely in view-dependent queries.

Metric-wise, this is not an overall breakthrough yet. The net overall gain is
only +1 correct sample on strat600, inside noise. The positive signal is
concentrated in view-dependent queries (+4.27 pp hit@3 and fewer empty
responses), but it is offset by a view-independent drop. For samples where the
parser emitted a viewpoint context, v15 has 14 miss->hit transitions and 7
hit->miss transitions vs v13, so true viewpoint emission is locally positive;
the non-context portion is slightly negative.

Next diagnostic should sample the 7 context regressions and the 14 context
recoveries to distinguish:

1. wrong viewpoint-anchor parsing;
2. correct parser output but wrong viewer-frame geometry;
3. query parser drift unrelated to viewpoint;
4. cases still only helped by ambiguous `rank_only` demotion.
