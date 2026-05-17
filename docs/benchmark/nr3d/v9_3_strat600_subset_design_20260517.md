# NR3D 600-case stratified subset — design + validation

This is **not** a Stage-2 evaluation run; it is a *fold design* doc. The goal:
pick a 600-sample subset of the canonical NR3D filtered fold (7805) such that
running any future agent on the 600 cases yields the same 5-column leaderboard
metrics (Overall / Easy / Hard / View-Dep / View-Indep) as running the full
7805 — to within ~±0.2 pp on a fixed-policy reference and ~±2.3 pp 90 %
band for a randomly-permuted realization.

## Pre-run checklist (per CLAUDE.md MANDATORY rule)

| Item | Value |
|---|---|
| Pending changes committed before design | yes |
| **Head commit at fold construction** | `4728205` |
| Run-time code commit | same (`4728205`) — no worktree drift |
| Salt-search reference run | `v9_1_fix_FULL_REPRO_20260516` |
| Output sample-ids (tmp, regenerable) | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Output sample-ids (durable copy) | `docs/benchmark/nr3d/assets/v9_3_strat600_sample_ids_20260517.json` |
| Output summary | `tmp/nr3d_artifacts/v9_3_strat600_summary.json` |
| Output summary (durable copy) | `docs/benchmark/nr3d/assets/v9_3_strat600_summary_20260517.json` |
| Validation artifact | `docs/benchmark/nr3d/assets/v9_3_strat600_validation_20260517.json` |
| Salt-search artifact | `docs/benchmark/nr3d/assets/v9_3_strat600_salt_search_20260517.json` |
| Byte-stable sample_ids MD5 | `12a69d8d14a81519024bbe00d6334434` |

## TL;DR

**Pool**: canonical NR3D filtered test fold (7805 samples after
`mentions_target_class=True`, 130 scenes, 72 instance categories).

**Subset**: 600 samples, salt-locked to `nr3d_v9_3_strat600_v291` (chosen by
salt search across 4096 candidates to minimize max |Δ| against
v9.1_fix FULL on all 5 leaderboard metrics).

**Stratification cells**: `(is_easy, is_view_dep)` 2 × 2 cross, proportional
allocation via the largest-remainder (Hamilton) method:

| Cell | Pool n | Pool % | Subset n | Subset % |
|---|---:|---:|---:|---:|
| Easy × View-Indep | 2570 | 32.93 % | **198** | 33.00 % |
| Easy × View-Dep   | 1203 | 15.41 % |  **92** | 15.33 % |
| Hard × View-Indep | 2483 | 31.81 % | **191** | 31.83 % |
| Hard × View-Dep   | 1549 | 19.85 % | **119** | 19.83 % |
| **Total** | **7805** | 100 % | **600** | 100 % |

**Diversity**: 119 / 130 unique scenes (91.5 %), 65 / 72 unique categories
(90.3 %), per-scene min / max / mean = 1 / 17 / 5.04.

## Headline (salt-locked subset vs FULL filtered 7805, on v9.1_fix predictions)

| Metric | FULL filtered 7805 | strat600 (salt v291) | **Δ** |
|---|---:|---:|---:|
| `classification_acc_filtered` (Overall) | 0.8295 | **0.8300** | **+0.05 pp** |
| `acc_easy`       | 0.8836 | 0.8828 | −0.09 pp |
| `acc_hard`       | 0.7788 | 0.7806 | +0.19 pp |
| `acc_view_dep`   | 0.7823 | 0.7820 | −0.03 pp |
| `acc_view_indep` | 0.8551 | 0.8560 | +0.09 pp |

**Max |Δ| across all 5 metrics = 0.19 pp.** This is below the per-sample
granularity (1 / 600 ≈ 0.167 pp), so the subset is calibrated to the limit of
fold resolution.

## Bootstrap validation (1000 random salts, n = 600, on v9.1_fix predictions)

The Monte-Carlo bootstrap demonstrates the **design itself is unbiased** — the
salt-locked outcome above is just one realization, but the *expected* realization
matches the FULL filtered fold on every metric to within the precision of 1000
trials.

| Metric | Bootstrap mean | Bootstrap std | 5–95 % band | **bias** vs FULL | half-width |
|---|---:|---:|---:|---:|---:|
| `classification_acc_filtered` | 0.8291 | 0.0140 | [0.8050, 0.8517] | **−0.0004** | **±2.3 pp** |
| `acc_easy`       | 0.8830 | 0.0184 | [0.8552, 0.9138] | −0.0007 | ±2.9 pp |
| `acc_hard`       | 0.7787 | 0.0217 | [0.7452, 0.8161] | −0.0000 | ±3.5 pp |
| `acc_view_dep`   | 0.7809 | 0.0267 | [0.7346, 0.8246] | −0.0015 | ±4.5 pp |
| `acc_view_indep` | 0.8553 | 0.0168 | [0.8278, 0.8817] | +0.0001 | ±2.7 pp |

Interpretation:

- **Unbiased design**: |bias| < 0.0015 on every metric. The stratified sampler
  is statistically equivalent to evaluating the full filtered 7805 in
  expectation.
- **Calibrated single realization**: by picking the salt `v291`, we landed at a
  realization where all 5 metrics fall within ±0.19 pp of the FULL mean — well
  inside even the tightest (overall) 5–95 % band of ±2.3 pp.
- **Variance budget for a new agent**: a future agent run on this subset will
  see Overall accuracy that differs from its true (full-7805) value by at
  most about ±2.3 pp 90 % of the time. Cell metrics (esp. View-Dep n=119)
  have ±3.5–4.5 pp bands. Read multi-pp comparisons with that in mind.

## Design

### Algorithm

1. Load 7805 filtered NR3D test rows via `Nr3dDataset.from_path(..., mentions_target_class_only=True)`.
2. Bin by `(is_easy, is_view_dep)` where
   - `is_easy ⟺ n_objects ≤ 2` — source: `referit3d/analysis/deepnet_predictions.py:34-36`
   - `is_view_dep ⟺ tokens ∩ {front, behind, back, right, left, facing, leftmost, rightmost, looking, across} ≠ ∅` — source: `referit3d/analysis/utterances.py:103-105`
3. Allocate 600 slots across cells by Hamilton's largest-remainder method on
   proportional share, with deterministic tie-break by `(is_easy, is_view_dep)`.
4. Within each cell, sort rows by `sha1(sample_id + SELECTION_SALT).hexdigest()`
   and take the top `n_cell` rows. This is the same deterministic-ordering
   trick used by `scripts/build_nr3d_v4_random100_fold.py`.
5. Final ordering: globally re-sort the selected 600 by the same salted-hash
   key for byte-stable file output.

### Why these strata

The NR3D leaderboard reports 5 columns: `Overall / Easy / Hard / View-Dep /
View-Indep`. The Easy/Hard and View-Dep/View-Indep dimensions partition the
filtered fold and are the source of all per-tier comparisons. A 2 × 2 cross
on those two binaries is therefore the natural minimal stratification —
controlling for both columns simultaneously is strictly stronger than
controlling for each marginally (which would let intersection cells drift).

### Why not also stratify on scene / category

- 130 scenes is comparable to the 119 the subset already covers; further
  stratifying inflates per-cell granularity without measurable benefit on
  bootstrap variance.
- 72 categories, very long tail (chair, table, window, door, trash can
  dominate). Forcing category-balanced sampling would *distort* the
  population mean away from the leaderboard quantity. Population-faithful
  proportional sampling is correct here.

### Why salt search is methodologically clean

The salt is a **fold design parameter**, fixed before any new agent is run.
Picking the salt that minimizes the deviation on a *reference* run (v9.1_fix)
controls one source of fold-construction noise without leaking any signal from
the model under test — for any future agent with a different correct/wrong
pattern, the chosen salt is still a draw from the same bootstrap-predicted
distribution, just one with a calibration head start.

Per-metric weighting in the search loss:
- `classification_acc_filtered` weight = 2.0 (the headline)
- All four sub-metrics weight = 1.0

Search space: `f"nr3d_v9_3_strat600_v{i}"` for `i ∈ [0, 4096)`. Winner: `v291`,
loss = 0.0051.

## Files

| Path | Purpose |
|---|---|
| `scripts/build_nr3d_strat600_fold.py` | Build the salt-locked 600-case fold |
| `scripts/validate_nr3d_strat600_subset.py` | Compute salt-locked metrics + bootstrap on v9.1_fix |
| `scripts/search_nr3d_strat600_salt.py` | Salt search across N candidates |
| `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` | The 600 sample-ids (durable fold) |
| `tmp/nr3d_artifacts/v9_3_strat600_summary.json` | Per-cell + diversity summary |
| `docs/benchmark/nr3d/assets/v9_3_strat600_validation_20260517.json` | Bootstrap + salt-locked validation numbers |
| `docs/benchmark/nr3d/assets/v9_3_strat600_salt_search_20260517.json` | Salt-search artifact (winning salt + ranked-1 metrics) |
| `docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md` | **This document** |

## Reproduce

```bash
# Build the fold (writes tmp/nr3d_artifacts/v9_3_strat600_{sample_ids,summary}.json)
PYTHONPATH=src python scripts/build_nr3d_strat600_fold.py

# Validate against v9.1_fix FULL (salt-locked metrics + 1000-trial bootstrap)
PYTHONPATH=src python scripts/validate_nr3d_strat600_subset.py

# (Optional) re-search the salt space (e.g., if the reference run changes)
PYTHONPATH=src python scripts/search_nr3d_strat600_salt.py --n-search 4096
```

## How to use this fold in a new evaluation

```bash
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_<your_run>/ \
    --workers 20
```

Then aggregate the canonical metrics restricted to the 600 ids:

```bash
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
    --side-by-side tmp/nr3d_eval_<your_run>/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --output docs/benchmark/nr3d/assets/<your_run>_strat600_leaderboard.json
```

`run_nr3d_vg_side_by_side.py` already filters to the provided
`--sample-ids`, and `nr3d_leaderboard_metrics.py` supports
`--sample-ids` for restricted aggregation (added in `c2c52d0`).

## Limitations

- **Salt-locked Δ above is measured on v9.1_fix predictions only.** A new agent
  with a very different per-sample correctness pattern will see deviations
  drawn from the (unbiased) bootstrap distribution, not from the calibrated
  realization. Treat ±2.3 pp (90 %) as the variance budget on Overall acc.
- **Cell n is small enough that View-Dep (n=119) has ~±4.5 pp 90 % band.** Do
  not over-interpret <5 pp View-Dep deltas between agent versions evaluated on
  this fold. For tighter View-Dep comparisons, use the FULL 7805 (or scale n
  up to ~1500).
- **`--sample-ids` filtering is inner-join.** If the agent doesn't emit a
  prediction for any of the 600 ids, the aggregator raises rather than
  silently dropping; this is correct behavior.
- **Calibration is on v9.1_fix only.** The salt was picked specifically so
  that the v9.1_fix subset metrics ≈ v9.1_fix full metrics; it does *not*
  guarantee that any new agent's subset metrics ≈ its own full metrics by
  the same margin. The bootstrap gives the right tolerance for that case
  (±2.3 pp 90 % on Overall).
