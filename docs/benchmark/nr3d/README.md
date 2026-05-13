# NR3D VG Benchmark Archive

This directory is the permanent process archive for NR3D visual grounding
evaluations in this repo. Keep the version docs immutable; use this README as
the current human-facing index.

## Entry Points

| File | Purpose |
|---|---|
| [leaderboard.md](leaderboard.md) | Public SOTA context plus our invalidated v5.1 audit row and historical rows. |
| [protocol.md](protocol.md) | Consolidated protocol notes: metric family, fold/filter rules, candidate-pool equivalence, fairness boundary. |
| [depth_visibility_rebuild_20260513.md](depth_visibility_rebuild_20260513.md) | Root-cause record and rebuild summary for depth-aware NR3D visibility indices. |
| [depth_visibility_spotcheck_20260513.html](depth_visibility_spotcheck_20260513.html) | Visual spotcheck frames rendered from rebuilt depth-aware `view_to_objects`. |
| [v9_inventory_random100](v9_inventory_random100_20260514.md) | Proposal-inventory prompt rerun on the fixed depth-aware random100 fold. |
| [v7p1_callbacks_noclip_random100_rerun](v7p1_callbacks_noclip_random100_rerun_20260514.md) | Current-code rerun of the v7 depth-aware callback-wired random100 pilot. |
| [v7_stage1_callbacks_noclip_random100](v7_stage1_callbacks_noclip_random100_20260513.md) | Initial depth-aware random100 callback-wired pilot. |
| [v6_inline_labels_depth_visible_random100](v6_inline_labels_depth_visible_random100_20260513.md) | Depth-aware random100 no-NMS rerender pilot before NR3D callbacks were wired. |
| [v6_random100_case_studies_20260513.html](v6_random100_case_studies_20260513.html) | Full Stage1+Stage2 visual walkthrough for 2 correct and 2 failed v6 random100 cases. |
| [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md) | Latest full-test audit record; invalidated pending depth-aware rerun. |
| [v5p1_case_studies_20260513.html](v5p1_case_studies_20260513.html) | Visual stage1+stage2 reasoning walkthrough for selected correct and failed cases. |
| [runs.sqlite](runs.sqlite) | Queryable per-run and per-sample metrics. |
| [transcrib3d_transfer_20260514](transcrib3d_transfer_20260514.md) | Branch-local transfer notes and first300-valid fold mapping. |

## Current Result Status

The latest recorded full-test row is now **invalidated pending rerun**. The
v5/v5.1 packs were built from NR3D `visibility_index.pkl` files whose metadata
records `use_depth=False`, so `view_to_objects` / `object_to_views` encoded
projection/frustum candidates rather than depth-occlusion visibility. Those
indices are not valid as agent-visible evidence.

Latest recorded, invalidated fair-view full-test row:

- Version: `v5p1_failed_rerun_full_20260513`
- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Run-time code commit: `c404536`
- Documentation commit introducing the v5.1 record: `6000e2b`
- Fold: 8584 NR3D test utterances; n_filtered=7805 after canonical
  `mentions_target_class=True`
- Evidence selection: query-driven fair keyframes; target id / GT bbox used
  only for scoring
- Stage 2 backend: `gpt-5.4-2026-03-05`
- Raw merged artifacts: `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- SQLite run id: `v5p1_failed_rerun_full_20260513`
- Case-study HTML:
  [v5p1_case_studies_20260513.html](v5p1_case_studies_20260513.html)

| Metric | v5.1 recorded, invalidated | v5 before failed-rerun | v3 recorded, invalidated | UniVLG public SOTA |
|---|---:|---:|---:|---:|
| Overall | **68.48** | 66.53 | 80.79 | 65.20 |
| Easy | **78.43** | 76.09 | 86.06 | 73.30 |
| Hard | **59.18** | 57.59 | 75.87 | 57.00 |
| View-Dep | **57.38** | 55.74 | 72.46 | 55.10 |
| View-Indep | **74.53** | 72.41 | 85.34 | 69.90 |

The v5.1 run reran all 241 failed sentinels from v5. Of those, 240 recovered
to completed outputs and 1 persisted as no-match. That persistent row is
filtered out of the 7805Q headline because `mentions_target_class=false`.
The rerun adds 152 correct samples inside the filtered fold and moves overall
from 66.53 to 68.48.

Interpretation:

- v5.1 is no longer a valid fair-view NR3D result because the underlying
  object-frame visibility index did not use depth occlusion.
- Do not compare v5.1 against UniVLG or any other SOTA row until the depth-aware
  visibility indices are rebuilt, packs regenerated, and the full run rerun.
- v3 is also invalidated as a benchmark claim because its GT-target-visible
  shortcut used the same projection-only visibility source.

Latest depth-aware Transcrib3D-matched pilot:

- Version: `v8_transcrib3d_first300_baseline`
- Branch / commit: `feat/nr3d-transcrib3d-first300` / `f90be6d`
- Scope: Transcrib3D `nr3d_first300_valid` fold, 281 samples
- Result: Overall 74.38, Easy 78.72, Hard 70.00, View-Dep 59.76,
  View-Indep 80.40
- Matched baseline: Transcrib3D GPT-4o text-only first300-valid is 74.02
  overall, so v8 is only +1 sample and is not a clear win.
- Raw artifacts:
  `tmp/nr3d_eval_v8_transcrib3d_first300_baseline_20260514/`

Latest depth-aware random100 partial pilot rerun:

- Version: `v9_inventory_random100`
- Branch / commit: `feat/nr3d-transcrib3d-first300` / `db95169`
- Scope: same v4 random100 fold; preserved v4 keyframe frame ids; no selector
  NMS
- Result: Overall 71.00, Easy 85.37, Hard 61.02, View-Dep 61.76,
  View-Indep 75.76
- Raw artifacts:
  `tmp/nr3d_eval_v9_inventory_random100_20260514/`
- Note: v9 injects a compact proposal inventory into the VG prompt. The final
  result has 100/100 completed checkpoints after rerunning 2 initial
  `invalid_prompt` sentinels. It is tied with v7.1 overall and -2pp versus the
  original v7 random100 result.

## Version Timeline

| Version | Date | Branch / commit | Headline | Scope | Status |
|---|---|---|---:|---|---|
| [v1_phase8_smoke20_mac](v1_phase8_smoke20_mac_20260430.md) | 2026-04-30 | `feat/nr3d-vg-benchmark` / `9115fd7` | Acc@0.50=65.00 | 20Q smoke | Invalidated: projection-only visibility |
| [v2_phase8_full](v2_phase8_full_20260501.md) | 2026-05-01 | `feat/nr3d-vg-benchmark` / `ad8d439` plus in-flight fixes | Acc@0.50=77.62 | 8584Q full | Invalidated: projection-only visibility |
| [v3_referit3d_track](v3_referit3d_track_20260501.md) | 2026-05-01 | `feat/nr3d-vg-benchmark` / `b6f211a` | Overall=80.79 | 8584Q / 7805Q filtered | Invalidated: projection-only GT-visible source |
| [v4_agent_guards_fair_views](v4_agent_guards_fair_views_20260512.md) | 2026-05-12 | `feat/nr3d-v4-agent-guards-fair-views` / `3f1c6d8` | Overall=71.00 | 100Q pilot | Invalidated: projection-only visibility |
| [v5_agent_guards_fair_views_full](v5_agent_guards_fair_views_full_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `7c996ad` | Overall=66.53 | 8584Q / 7805Q filtered | Invalidated: projection-only visibility |
| [v5p1_failed_rerun_full](v5p1_failed_rerun_full_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `c404536` | Overall=68.48 | 8584Q / 7805Q filtered | Invalidated: projection-only visibility |
| [v6_inline_labels_depth_visible_random100](v6_inline_labels_depth_visible_random100_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `41253ad` | Overall=71.00 | 100Q pilot | Depth-aware partial, no NMS |
| [v7_stage1_callbacks_noclip_random100](v7_stage1_callbacks_noclip_random100_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `690cbf6` | Overall=73.00 | 100Q pilot | Depth-aware partial, callbacks wired, no NMS |
| [v7p1_callbacks_noclip_random100_rerun](v7p1_callbacks_noclip_random100_rerun_20260514.md) | 2026-05-14 | `feat/nr3d-transcrib3d-first300` / `101acaa` | Overall=71.00 | 100Q pilot | Depth-aware partial rerun, callbacks wired, no NMS |
| [v8_transcrib3d_first300_baseline](v8_transcrib3d_first300_baseline_20260514.md) | 2026-05-14 | `feat/nr3d-transcrib3d-first300` / `f90be6d` | Overall=74.38 | 281Q Transcrib3D first300-valid fold | Depth-aware partial, +1 sample vs Transcrib3D GPT-4o, not a clear win |
| [v9_inventory_random100](v9_inventory_random100_20260514.md) | 2026-05-14 | `feat/nr3d-transcrib3d-first300` / `db95169` | Overall=71.00 | 100Q pilot | Depth-aware partial, proposal inventory prompt, tied with v7.1 |

## Protocol Summary

- Public NR3D leaderboard headline is classification accuracy:
  `selected_object_id == target_id`.
- Canonical local full fold: 8584 utterances, 130 scenes.
- Canonical filtered fold: 7805 utterances after `mentions_target_class=True`.
- Easy/Hard split: `n_objects <= 2`.
- View-Dep split: canonical 10-token literal set from ReferIt3D.
- Candidate pool: full-scene GT segmented / object proposals; the old
  "target-type-only public pool" assumption is retracted.
- Fairness boundary: v3 uses GT-target-visible keyframes; v5/v5.1 use
  query-driven keyframes but are invalidated because their object-frame
  visibility was projection-only. v6/v7 are depth-aware but partial and
  preserve v4 keyframe choices to isolate Stage2/runtime changes before NMS.
  v7.1 is a same-code rerun of v7 and shows a -2pp drift on the 100Q fold.
  v9 adds a proposal-inventory prompt prior but ties v7.1 overall on the same
  fold.

See [protocol.md](protocol.md) for the consolidated evidence and caveats.

## Reproduction

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

For high-concurrency query-driven prep, build lightweight ConceptGraph object
caches once, then enable the ensure flag during prep. If a cache is still
missing, it is built under a cross-process lock and the sample continues:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --build-lightweight-cache-only

PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v4_agent_guards_fair_views \
    --split test \
    --keyframe-mode query_driven \
    --ensure-lightweight-cache
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

Run the v5.1 reproduction path from the version doc:

- Failed rerun subset:
  `tmp/nr3d_artifacts/v5_failed241_sample_ids_20260513.json`
- Base full output:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/`
- Failed rerun output:
  `tmp/nr3d_eval_v5_failed_rerun_20260513/`
- Merged output:
  `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- Detailed CLI and ingest commands:
  [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md)

## SQLite

Canonical DB: `docs/benchmark/nr3d/runs.sqlite`

```sql
SELECT run_id, n,
       printf('%.4f', classification_acc_filtered) AS acc_filtered,
       n_filtered,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50
FROM runs;
```

Current headline query:

```sql
SELECT run_id, n, n_filtered,
       printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind
FROM runs
WHERE run_id='v5p1_failed_rerun_full_20260513';
-- v5p1_failed_rerun_full_20260513|8584|7805|0.6848|0.7843|0.5918|0.5738|0.7453
```

## Retained / Deleted Docs

Retained:

- README, leaderboard, protocol, runs.sqlite
- one version doc per evaluated run: v1, v2, v3, v4, v5, v5.1

Deleted during consolidation:

- `protocol_audit_20260501.md`
- `paper_crosscheck_20260501.md`
- `pool_equivalence_log_20260501.md`

Their durable conclusions were merged into [protocol.md](protocol.md),
[leaderboard.md](leaderboard.md), and the current-result section above.
