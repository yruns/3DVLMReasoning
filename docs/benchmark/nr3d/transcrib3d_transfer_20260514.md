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

## v9 Transfer Step: Compact Proposal Inventory

Implemented first because it is the lowest-risk Transcrib3D idea:
Transcrib3D gives GPT-4o a compact full-scene object table before reasoning,
while our v8 prompt only exposed the proposal pool through tools. v9 injects a
compact `Scene Proposal Inventory` into the VG user message directly from
`bundle.extra_metadata["vg_proposal_pool"]`.

Injected fields per proposal:

- `proposal_id`
- category label
- 3D center `(cx, cy, cz)`
- 3D size `(dx, dy, dz)`
- visible view count

Prompt/playbook changes:

- The VG protocol now tells the agent to use the inventory as a text-first
  candidate prior before spending turns on more images.
- The playbook now mirrors Transcrib3D's candidate discipline: identify focal
  category, form a category-compatible candidate set, use size/center for
  simple superlatives, and then use marked frames for ambiguous visual checks.
- `chassis_tools_version` was bumped from 18 to 19 so prompt-cache keys do not
  mix v8 and v9 prompts.

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/tests/test_agent_config_flags.py \
  src/agents/tests/test_stage2_deep_agent.py::test_vg_prompt_formats_proposal_inventory_from_pack_pool \
  -q
```

Result: 9 passed.

The first300 evaluation should reuse the same prepared v8 pack and only rerun
Stage2:

- Output dir:
  `tmp/nr3d_eval_v9_inventory_first300_20260514/`
- Pack:
  `pack_nr3d_v8_transcrib3d_first300_baseline`

### Random100 Sanity Rerun

Before rerunning the Transcrib3D first300-valid fold, v9 was rerun on the fixed
depth-aware NR3D random100 fold to check that the prompt change did not break
the established pilot:

- Version doc: [v9_inventory_random100_20260514](v9_inventory_random100_20260514.md)
- Branch / commit: `feat/nr3d-transcrib3d-first300` / `9eaf05d`
  (code change: `db95169`)
- Fold / pack:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json` +
  `pack_nr3d_v6_inline_labels_depth_visible`
- Result: Overall 71.00, Easy 85.37, Hard 61.02, View-Dep 61.76,
  View-Indep 75.76
- Comparison: tied with v7.1 overall; eight v7.1 misses became correct and
  eight v7.1 correct samples regressed.

Interpretation: the compact proposal inventory is stable and memory-safe on
random100, but it is not an accuracy win by itself. This made first300-valid
the necessary direct comparison before deciding whether to invest in stronger
deterministic candidate-ranking helpers.

### First300-Valid Rerun

v9 was then run on the Transcrib3D first300-valid matched fold:

- Version doc: [v9_inventory_first300_20260514](v9_inventory_first300_20260514.md)
- Branch / commit: `feat/nr3d-transcrib3d-first300` / `db95169`
- Fold / pack:
  `docs/benchmark/nr3d/assets/transcrib3d_first300_valid_sample_ids_20260514.json` +
  `pack_nr3d_v8_transcrib3d_first300_baseline`
- Result: 211/281 = 75.09 overall; Easy 80.14, Hard 70.00, local View-Dep
  62.20, local View-Indep 80.40
- Comparison:
  - v8 baseline: 209/281 = 74.38
  - Transcrib3D GPT-4o first300-valid: 208/281 = 74.02
  - v9 is +2 samples over v8 and +3 samples over Transcrib3D

This is a real improvement, but the margin is still small. The remaining
Transcrib3D-only wins point to the next transfer target: deterministic helper
logic for same-category superlatives, under/above/left/right relations, and
attribute disambiguation, rather than only asking the LLM to infer these from
the inventory table.

## v10 Transfer Step: Deterministic Geometry Ranking

Implemented next because v9 still lost Transcrib3D-only cases involving
same-category geometry and superlatives:

- line 5: larger chair near octagonal table
- line 75: pillow farthest from door
- line 111: larger curved desk
- line 46: lower set of pipes

v10 adds a new VG tool:

- `rank_proposals_by_geometry(candidate_ids, criterion)`
- supported criteria:
  `largest`, `smallest`, `tallest`, `shortest`, `highest`, `lowest`,
  `widest`, `narrowest`
- returned evidence:
  ranked proposal ids, volumes, footprint areas, heights, center z values,
  sizes, centers, and per-proposal rows

The playbook now instructs the agent to call this tool for deterministic
same-category size/height superlatives after building a candidate shortlist.
For combined expressions such as "the larger chair near the octagonal table",
the intended workflow is: find chair candidates, use spatial/visual evidence to
narrow chairs near the table, then rank that shortlist by `largest`.

`chassis_tools_version` was bumped from 19 to 20 because the VG tool surface
changed.

Verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/agents/packs/vg_embodiedscan/tests/test_tools.py \
  src/agents/tests/test_agent_config_flags.py \
  src/agents/tests/test_stage2_deep_agent.py::test_pack_v1_vg_tool_list_snapshot \
  src/agents/tests/test_stage2_deep_agent.py::test_vg_prompt_formats_proposal_inventory_from_pack_pool \
  -q
```

Result: 42 passed.

Next evaluation should reuse the same first300 pack and only rerun Stage 2:

- Output dir:
  `tmp/nr3d_eval_v10_geometry_first300_20260514/`
- Pack:
  `pack_nr3d_v8_transcrib3d_first300_baseline`
