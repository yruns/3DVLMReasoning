# NR3D v14 select_by_text viewpoint prompt smoke50 audit

This is a tool-coverage smoke audit, not an agent leaderboard run. It measures
`select_by_text` / `KeyframeSelector.select_keyframes_v2` after porting the
viewpoint contract into the active `src/query_scene/parsing` prompt and dynamic
schema.

Question:

> After the prompt/schema port, does the parser actually emit
> `viewpoint_contexts` / `reference_frame="viewer"`, and what is the quick
> hit@3 signal on 50 NR3D samples?

The 50 samples are the first 50 rows from the canonical strat600 fold. This is
only a smoke check; use strat600 or full-set reruns for decision-grade claims.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `analysis/nr3d-select-by-text-coverage` |
| Head commit at launch | `3688096` |
| Run-time code commit | `3688096` - no worktree drift |
| Relevant code change | `0c6cb85` ports viewpoint prompt/schema to active parser; `3688096` records viewpoint stats in the audit output |
| Fold | `tmp/nr3d_select_by_text_audit_viewpoint_prompt_smoke50_20260524/sample_ids_50.json` |
| Fold source | first 50 rows of `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold composition | 50 samples; 18 view-dependent; 24 easy; 38 scenes |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_select_by_text_audit_viewpoint_prompt_smoke50_20260524/` |
| Merged summary | `tmp/nr3d_select_by_text_audit_viewpoint_prompt_smoke50_20260524/merged_summary.json` |
| Started | `2026-05-24T20:02:46+08:00` |
| Completed | `2026-05-24T20:06:17+08:00` |
| Workers | 10 shard processes |
| Tool-equivalent call | `select_by_text(query=<raw NR3D query>, k=3, hidden_categories=[], use_visual_context=False, viewpoint_aware=True)` |
| Judge model | none - deterministic coverage against visibility indices |

## Exact Command

The smoke samples were split into 10 shard files under:

```bash
tmp/nr3d_select_by_text_audit_viewpoint_prompt_smoke50_20260524/shards/
```

Each shard ran:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
.venv/bin/python scripts/audit_select_by_text_nr3d.py \
  --sample-ids tmp/nr3d_select_by_text_audit_viewpoint_prompt_smoke50_20260524/shards/shard_XX.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --k 1,2,3 \
  --output tmp/nr3d_select_by_text_audit_viewpoint_prompt_smoke50_20260524/outputs/shard_XX.json \
  --max-selector-cache-size 2 \
  --viewpoint-aware \
  --resume
```

The tmux runner was:

```bash
tmp/nr3d_select_by_text_audit_viewpoint_prompt_smoke50_20260524/run_parallel.sh
```

All 10 shard processes completed with exit code 0.

## Headline Coverage

| Slice | n | hit@1 | hit@2 | hit@3 | empty frames |
|---|---:|---:|---:|---:|---:|
| Overall | 50 | 52.00 | 64.00 | **66.00** | 20.00 |
| Easy | 24 | 58.33 | 75.00 | **79.17** | 8.33 |
| Hard | 26 | 46.15 | 53.85 | **53.85** | 30.77 |
| View-Dep | 18 | 55.56 | 61.11 | **61.11** | 22.22 |
| View-Indep | 32 | 50.00 | 65.62 | **68.75** | 18.75 |

Headline answer: **33 / 50 = 66.00 %** of this smoke slice have at least one
returned keyframe containing the target id.

## Viewpoint Emission

The active parser now emits actual viewpoint fields on this smoke slice:

| Metric | Count |
|---|---:|
| Samples with non-empty `viewpoint_contexts` | 7 / 50 |
| View-dependent samples with non-empty `viewpoint_contexts` | 7 / 18 |
| View-independent samples with non-empty `viewpoint_contexts` | 0 / 32 |
| `reference_frame="viewer"` constraints/selectors | 8 |
| `reference_frame="ambiguous"` constraints/selectors | 10 |
| `execution_policy="soft"` constraints/selectors | 8 |
| `execution_policy="rank_only"` constraints/selectors | 10 |

The seven samples with emitted viewpoint contexts were all view-dependent:

| Sample | Query | hit@3 | Viewpoint stats |
|---|---|---:|---|
| `scene0644_00::43::2885` | `As you enter the room... right hand side.` | 0 | viewer=1, world=1 |
| `scene0549_00::3::7602` | `Facing the windows... coffee table to your left` | 0 | viewer=1 |
| `scene0663_00::16::21361` | `facing the door, left trash can` | 1 | viewer=1 |
| `scene0329_00::27::40883` | `looking at the wall... far right trash can` | 1 | viewer=1 |
| `scene0580_00::17::2010` | `standing at the foot of the bed... pillow on the left` | 1 | viewer=1 |
| `scene0338_00::21::35740` | `Facing the boxes... back on the left side` | 1 | viewer=2 |
| `scene0607_00::16::35347` | `facing the sinks... one on the right` | 1 | viewer=1 |

## Same-50 Comparison To v13

Previous record: [`v13_select_by_text_viewpoint_aware_strat600_coverage_20260524.md`](v13_select_by_text_viewpoint_aware_strat600_coverage_20260524.md),
runtime `f8e6d22`, same `viewpoint_aware=True` switch but before the prompt was
ported into the active parser path.

| Slice | v13 same 50 hit@3 | v14 smoke50 hit@3 | Delta |
|---|---:|---:|---:|
| Overall | 68.00 | 66.00 | -2.00 pp |
| View-Dep | 66.67 | 61.11 | -5.56 pp |
| View-Indep | 68.75 | 68.75 | +0.00 pp |

Same-50 empty returned frames moved from 9 / 50 to 10 / 50. The changed rows
are small enough that this is noise-level for a 50-sample smoke. The important
signal from this run is not an accuracy gain, but that the active parser path
now produces real `viewpoint_contexts` and `reference_frame="viewer"` fields.

## Interpretation

The prompt/schema port worked mechanically: viewpoint contexts are no longer
zero, and they appear only on view-dependent queries in this slice. However,
this 50-case smoke does **not** show an aggregate hit@3 improvement over v13 on
the same rows. The likely next diagnostic is to run the full strat600 again and
separate:

1. cases helped by true viewer-frame execution;
2. cases still helped only by ambiguous `rank_only` demotion;
3. cases regressed because the parser emitted an explicit viewer frame but the
   resolved viewpoint geometry ranked the wrong side.
