# Transcrib3D Transfer Notes - 2026-05-14

This note starts the branch-local investigation for borrowing useful ideas from
`/Users/bytedance/project/Transcrib3D` into the 3DVLMReasoning NR3D agent.

## Branch

- Branch: `feat/nr3d-transcrib3d-first300`
- Starting commit: `101acaa`
- Goal: compare against Transcrib3D on its NR3D first300-valid fold, then
  iterate until the 3DVLMReasoning pipeline clearly exceeds that text-only
  baseline.

## Transcrib3D Reference Results

Source files:

- `/Users/bytedance/project/Transcrib3D/runs/modelhub_sharded/nr3d_full_results_20260514.md`
- `/Users/bytedance/project/Transcrib3D/runs/modelhub_sharded/20260513-175704_nr3d_nr3d_first300_valid/metrics.json`
- `/Users/bytedance/project/Transcrib3D/runs/modelhub_sharded/20260513-175704_nr3d_nr3d_first300_valid/manifest.json`

Reported metrics:

| Scope | Model pool | Valid n | Overall | Easy | Hard | View-dep | View-indep |
|---|---|---:|---:|---:|---:|---:|---:|
| first300 valid | GPT-4o ModelHub pool | 281 | 74.02 | 78.72 | 69.29 | 57.47 | 81.44 |
| full valid | GPT-4o ModelHub pool | 6910 | 77.61 | 84.08 | 70.92 | 70.84 | 81.14 |

The full-valid row filters NR3D to `correct_guess=true` and
`mentions_target_class=true`. The first300-valid row applies the same filter
to `data/referit3d/nr3d_test_sampled1000.csv` lines 2..301, leaving 281
valid rows.

## Fixed Comparison Fold

The Transcrib3D sampled CSV does not include `assignmentid`, while the local
3DVLMReasoning sample id format requires it:

`scannet/<scene_id>::<target_id>::<assignmentid>`

I mapped each first300-valid row back to `data/nr3d/raw/nr3d.csv` using the
tuple `(stimulus_id, utterance, speaker_id, listener_id)`.

Validation:

- Rows mapped: 281
- Missing mappings: 0
- Ambiguous mappings: 0
- Local `Nr3dDataset.from_path(..., bbox_source="phase8_gt_cg")` loaded:
  281/281
- Loader missing samples: 0

Tracked fold file:

- `docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json`

Temporary working copy:

- `tmp/nr3d_artifacts/transcrib3d_first300_valid_sample_ids_20260514.json`

## What Transcrib3D Does Well

The strong result is not from visual perception. It is a text-only structured
object-selection pipeline over GT ScanNet boxes:

1. It gives the LLM a compact object table with class label, object id, center,
   size, HSL color, and sometimes front/left/right/behind direction vectors.
2. It frames NR3D as target-instance classification over GT boxes, matching the
   public NR3D classification protocol.
3. It asks the model to identify the focal object first, then use object class
   similarity to form a candidate set before applying secondary constraints.
4. It tells the model to reason quantitatively and use a local Python
   interpreter when distances, colors, left/right, or vertical relations need
   calculation.
5. It has explicit rules that avoid common NR3D mistakes:
   - if only one category-compatible candidate exists, answer it directly;
   - do not use raw x/y comparisons for left/right;
   - treat front/behind as relative distance when object orientation is absent;
   - use HSL lightness for black/white and hue for other colors;
   - be lenient with vertical and color constraints.
6. It runs with deterministic temperature/top-p and sharded ModelHub retries.

## Transfer Hypotheses

The most promising ideas to port are:

1. A structured text-first candidate reasoning path before visual evidence is
   requested. Our current agent often spends turns inspecting frames before it
   has a crisp candidate set.
2. A compact object table in the VG prompt/tool output: `id`, category, center,
   size, extent/minmax, HSL color, and direction vectors when available.
3. Explicit NR3D constraint policy in the playbook, especially focal-object
   parsing, category-compatible single-candidate short-circuiting, left/right
   cross-product logic, and HSL color comparison.
4. A bounded deterministic geometry helper/tool rather than asking the model
   to invent calculations in free text.

## Immediate Next Step

Run the current v7 3DVLMReasoning agent on the fixed 281-sample fold:

- Pack name: `pack_nr3d_v8_transcrib3d_first300_baseline`
- Output dir: `tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514`
- Sample ids:
  `docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json`

This establishes the true gap before porting Transcrib3D-style prompt/tools.

## Baseline Pack Preparation

The v8 baseline pack was prepared with query-driven Stage1 keyframes, not
GT-target-visible keyframes.

Command shape:

```bash
./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
  bash tmp/nr3d_artifacts/run_v8_first300_prep_shards.sh
```

The shard runner split the 281 samples into four scene-disjoint shards:

| Shard | Samples | Scenes |
|---|---:|---:|
| `shard_00.json` | 71 | 27 |
| `shard_01.json` | 70 | 27 |
| `shard_02.json` | 70 | 27 |
| `shard_03.json` | 70 | 27 |

Prepared artifacts:

- Pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_transcrib3d_first300_baseline/`
- Main log:
  `tmp/nr3d_v8_transcrib3d_first300_prep4_20260514.log`
- Shard logs:
  `tmp/nr3d_v8_transcrib3d_first300_prep_shard_{00..03}_20260514.log`
- Earlier single-process warmup log:
  `tmp/nr3d_v8_transcrib3d_first300_prep_20260514.log`

Validation:

- Sample artifacts present: 281/281
- `keyframe_mode`: `query_driven` for 281/281
- `keyframe_selection_uses_gt_target`: 0/281
- Keyframes per sample: 3/281
- `keyframe_selection_used_fallback`: 79/281
- Log checks across warmup + shard logs:
  - `Traceback`: 0
  - `ERROR`: 0
  - `rss_guard`: 0
  - retryable `429`: 0
  - retryable `503`: 0
  - missing depth-aware visibility: 0
  - projection fallback visibility: 0

The next step is Stage2 evaluation into:

`tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/`

## Baseline Stage2 Result

Run ID:

`v8_transcrib3d_first300_baseline_20260514`

Artifacts:

- Output dir:
  `tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/`
- Metrics:
  `tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/leaderboard_metrics.json`
- Comparison:
  `tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/comparison_vs_transcrib3d.json`

Matched comparison:

| Method | Correct / Total | Overall | Easy | Hard | View-dep | View-indep |
|---|---:|---:|---:|---:|---:|---:|
| Transcrib3D GPT-4o text-only | 208/281 | 74.02 | 78.72 | 69.29 | 57.47 | 81.44 |
| Ours v8 baseline | 209/281 | 74.38 | 78.72 | 70.00 | 59.76 | 80.40 |

This is not a clear win: it is only +1 sample overall. Per-sample agreement:

- both correct: 166
- ours correct, Transcrib3D wrong: 43
- Transcrib3D correct, ours wrong: 42
- both wrong: 30

The next transfer step should target the 42 Transcrib3D-only wins, especially
candidate ranking by category-compatible object tables, distance/superlative
relations, color/size comparison, and left/right viewpoint language.
