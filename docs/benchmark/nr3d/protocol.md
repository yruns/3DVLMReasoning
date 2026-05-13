# NR3D Protocol Notes

This file is the consolidated protocol note for the local NR3D benchmark
archive. It replaces the older one-off audit files:

- `protocol_audit_20260501.md`
- `paper_crosscheck_20260501.md`
- `pool_equivalence_log_20260501.md`

Those files were useful while debugging the protocol, but their durable
conclusions now live here and in the versioned run docs.

## Metric Families

There are two NR3D metric families in this repository:

| Track | Meaning | Used by |
|---|---|---|
| Classification / GT-track | `selected_object_id == target_id`, with Easy/Hard and View-Dep/View-Indep slices | Public ReferIt3D leaderboard, v3, v4, v5, v5.1 |
| IoU proxy on GT pool | 9-DoF oriented box IoU, reported as Acc@0.25 / Acc@0.50 / mean IoU | v1/v2 historical runs |

Under the GT object pool, IoU and classification collapse to the same main
event: picking the correct object id yields IoU near 1.0; picking the wrong
object usually yields IoU near 0.0. For public NR3D comparisons, use the
classification track.

## Canonical Fold and Filters

The local full test fold is the canonical NR3D test split with Phase 8 GT-CG
boxes available for scoring:

| Check | Value |
|---|---:|
| Test scenes | 130 / 130 |
| Full utterances after baseline filters | 8584 |
| Canonical `mentions_target_class=True` utterances | 7805 |
| Retain rate | 90.92% |
| Easy / Hard | 3773 / 4032 |
| View-Dep / View-Indep | 2752 / 5053 |

Baseline filters remove the bad-context blacklist and clothing rows. The
leaderboard-track filter additionally keeps `mentions_target_class=True`.
`correct_guess=True` is not applied for the canonical ReferIt3D-style rows in
this archive. UniVLG's public repo applies a stricter NR3D-side filter, but
that is treated as a method-specific implementation detail rather than the
canonical ReferIt3D filter chain.

## Candidate Pool Equivalence

The earlier assumption that the public NR3D leaderboard uses a target-type-only
candidate pool was wrong. The durable conclusion is:

- Public GT-track NR3D is target-object classification over GT segmented /
  object proposals.
- The local Phase 8 GT-CG pool is structurally equivalent for the dimensions
  used by the leaderboard:
  - scene set: 130 / 130 NR3D test scenes covered;
  - utterance set: 8584 after baseline filters;
  - target indexing: 8584 / 8584 rows have `phase8.objects[target_id]` class
    matching the NR3D `instance_type`;
  - candidate source: full-scene ScanNet aggregation instances.
- Some scenes have more than ReferIt3D's historical `max_test_objects=88` cap.
  Keeping all objects slightly increases clutter for those scenes and should
  not inflate our score.

## Fairness Boundary

The main fairness distinction in this archive is not the candidate pool; it is
the visual evidence selection path:

| Row | Evidence selection | Interpretation |
|---|---|---|
| v3 | Historical GT-target-visible keyframe path, but visibility index was projection-only | Invalidated historical row |
| v5 / v5.1 | Query-driven keyframes; target id and GT bbox used only for scoring, but visibility index was projection-only | Invalidated pending depth-aware rerun |

The latest recorded v5.1 row is retained as an invalidated audit record:

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Run-time code commit: `c404536`
- Run ID: `v5p1_failed_rerun_full_20260513`
- Headline: 68.48 overall on n_filtered=7805
- Raw artifacts: `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- Version doc: `v5p1_failed_rerun_full_20260513.md`
- Invalidation reason: NR3D `visibility_index.pkl` metadata has
  `use_depth=false`; `view_to_objects` / `object_to_views` were not
  depth-occlusion visibility.

## Public Reference Points

The official ReferIt3D benchmark page reports NR3D as classification accuracy:
https://referit3d.github.io/benchmarks.html

The current top public NR3D row retained for context here is UniVLG. The local
v5.1 row below is invalidated and must not be used as a SOTA comparison until
depth-aware visibility is rebuilt and the full fold is rerun:

| Method | Overall | Easy | Hard | View-Dep | View-Indep |
|---|---:|---:|---:|---:|---:|
| UniVLG | 65.20 | 73.30 | 57.00 | 55.10 | 69.90 |
| Ours v5.1 recorded, invalidated | 68.48 | 78.43 | 59.18 | 57.38 | 74.53 |

Detection-mode NR3D numbers such as Acc@0.25 / Acc@0.50 from recent papers are
separate from the public classification leaderboard and should not be mixed
with the headline rows in `leaderboard.md`.

## Reproducibility Pointers

- Loader: `src/benchmarks/nr3d_loader.py`
- Leaderboard metric script: `src/evaluation/scripts/nr3d_leaderboard_metrics.py`
- Runner: `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
- Ingester: `scripts/ingest_nr3d_run.py`
- SQLite DB: `docs/benchmark/nr3d/runs.sqlite`
- Full sample-id list: `tmp/nr3d_artifacts/full_test_sample_ids.json`
