# NR3D v9.1_fix FULL REPRO — Stage-1 broken catalog-only run on full 8584

> **⚠️ INVALIDATED (2026-05-17) — Stage-1 seed-keyframe drain leak.**
> Subsequent investigation (`v9_4d_strat600_force_error_seed_drain_20260517.md`)
> isolated the source of this row's +16-19 pp advantage as the Stage-1
> seed-keyframe drain leak: at commit `d5f40ba`,
> `DeepAgentsStage2Runtime.build_evidence_update_message` silently
> auto-injected the 5 GT-target-visible RGB seed keyframes from
> `bundle.keyframes` into agent context on every evidence-update turn.
> The agent never asked for those frames; the leak gave it "the answer
> for free". The leak was fixed in commit `8ebf701` (post-`d5f40ba`).
>
> Cleanly reproducing the leak on v9.3 code (via the v9.4-D experimental
> flag `--restore-stage1-seed-keyframe-drain` plus
> `--force-stage1-text-retrieval-to-error`) gives 83.33 % on the strat600
> calibrated fold — within +0.33 pp of this row's calibrated baseline
> (83.00 %), and **0 / 6** vs **6 / 6** magic-case recovery proves the
> mechanism is the leak, not policy / cadence / playbook prose.
>
> **Status: this row is now in the same category as `v3_referit3d_track`
> and `v5p1_failed_rerun_full` — a depth-aware NR3D row whose number
> depended on GT information leaking into the model's input. The 82.95 %
> should not be quoted against public NR3D SOTA.**
>
> The fair v9.3 baseline is ~65 % (Overall) which matches public UniVLG
> 65.2 %. See `v9_4d_strat600_force_error_seed_drain_20260517.md` for the
> full audit trail.

## Pre-run checklist (per the MANDATORY rule in CLAUDE.md)

| Item | Value |
|---|---|
| Pending changes committed before launch | yes |
| **Head commit at launch** | `e750d8b` |
| **Run-time code commit** | `d5f40ba` (worktree at `/var/folders/fy/wxx8yt6156qf6y7knc9nhr5m0000gn/T/v91fix-worktree.Q3nOdbvG0K`) |
| run-id (REPRO-tagged per rule) | `v9_1_fix_FULL_REPRO_20260516` |
| Source run id being reproduced | `v9_1_fix_random100` (random100, 86.0 %) and its REPRO `v9_1_fix_random100_REPRO_20260516` (random100, 84.0 %) |
| Worktree drift reason | reproducing the v9.1_fix wrapper-bypass bug at the original commit `d5f40ba` |

## Headline (NR3D canonical filtered fold, n=7805)

| Metric          | **v9.1_fix FULL REPRO** | v9.2 text-first full | v9.2 no-text full | Δ vs text-first | Δ vs no-text |
|-----------------|---:|---:|---:|---:|---:|
| **Overall**     | **82.95 %** | 65.20 % | 64.65 % | **+17.75 pp** | **+18.30 pp** |
| Easy (n=3773)   | 88.36 % | 76.01 % | 76.09 % | +12.35 | +12.27 |
| **Hard (n=4032)** | **77.88 %** | 55.08 % | 53.94 % | **+22.80** | **+23.94** |
| **V-Dep (n=2752)** | **78.23 %** | 56.83 % | 58.14 % | **+21.40** | **+20.09** |
| V-Indep (n=5053)| 85.51 % | 69.76 % | 68.20 % | +15.75 | +17.31 |
| failed (no-match `-1`) | 25 / 8584 | 16 / 8584 | 30 / 8584 | — | — |
| Python errors   | **0** | 0 | 0 | — | — |
| Wall clock      | 8h32m | 15h28m (parallel) | (paired) | — | — |

This is the **highest depth-aware NR3D test result in the project's SQLite history**, ahead of:
- `v3_referit3d_track` 80.79 % (invalidated due to projection-only visibility)
- `v5p1_failed_rerun_full` 68.48 % (also invalidated)
- All other v9.x runs

## What changed vs random100

The v9.1_fix random100 number was **86.00 %** (original) / **84.00 %** REPRO. At full scale:
- **classification_acc_full** = 79.49 % (vs 84-86 random100 → −5 to −7 pp from random100 to full)
- **classification_acc_filtered** = 82.95 % (vs 84-86 random100 → −1 to −3 pp from random100 to full)

So random100 modestly overstated the true mean (~3 pp), but the headline policy genuinely sits in the **80+ band on the full test set**.

## Why this is a 17-23 pp swing vs v9.2

v9.1_fix and v9.2 share the same:
- v9.1+ tool surface (`mark_frame_with_bbox`-required, no `view_keyframe`)
- Three guards on (TADG / no-match / evidence-frame)
- Same fold (full 8584 / filtered 7805)
- Same pack (`pack_nr3d_v9_catalog_first`)
- Same Stage 2 backend (`gpt-5.4-2026-03-05`)

What differs:

|                              | v9.1_fix (this) | v9.2 text-first | v9.2 no-text |
|------------------------------|---|---|---|
| Code commit                  | `d5f40ba` (wrapper bypass bug) | `c2c52d0` (working) | `c2c52d0` (working) |
| `select_by_text` registered  | yes (returns ERROR every call) | yes (works) | NO |
| `runtime.keyframe_selector`  | None (bug → broken) | KeyframeSelector (working) | None |
| Playbook bodies              | original text-first | original text-first | `_no_text` catalog-first |
| Effective behavior           | catalog-only, agent learns to fall through `list_scene_proposals` | text-first prior, Stage-1 leads | catalog-first jumps to `select_by_proposal` |

Two-step regression decomposition:

1. **Stage-1 actually working = −17.75 pp at full** (v9.1_fix → v9.2 text-first). Same code surface, same playbook prose; only the wrapper-fix commit `de8225f` changed Stage-1 from "always ERROR" to "really runs". This re-confirms the random100 finding: making `select_by_text` the prompted first move is actively harmful on NR3D.

2. **Dropping the tool + switching playbook = −0.55 pp additional** (v9.2 text-first → v9.2 no-text). At full scale my hand-written `_no_text` playbook is essentially neutral vs the original playbook with broken Stage-1. (Random100 had +5 pp gap — that turns out to be small-fold variance.)

The dominant signal is **Stage-1 working as first move = −17 pp**.

## What the agent actually does in v9.1_fix (sanity, full set)

Tool-trace aggregate over 8584 samples:

| Tool                          | Calls   | Per-sample avg |
|-------------------------------|--------:|---:|
| `inspect_proposal`            | ~30k    | ~3.5 |
| `mark_frame_with_bbox`        | ~28k    | ~3.3 |
| `select_by_proposal`          | ~19k    | ~2.2 |
| `submit_final`                | ~18k    | ~2.1 |
| `load_skill`                  | ~17k    | ~2.0 |
| `view_bev`                    | ~9k     | ~1.0 |
| `list_scene_proposals`        | ~8k     | ~1.0 |
| **`select_by_text`**          | **~8.6k** | **~1.0** (every sample tries it once and gets ERROR) |
| `compare_proposals_spatial`   | ~3.5k   | ~0.4 |

The behaviour pattern is:

```
load_skill → load_skill → select_by_text → ERROR
  → list_scene_proposals (or view_bev) → select_by_proposal → mark_frame_with_bbox
  → inspect_proposal × N → submit_final
```

That broken-Stage-1-as-tripwire pattern is what turned out to be the strongest NR3D policy on this surface. The agent treats the ERROR as "the query has no clean catalog handle yet, enumerate first" and naturally walks the catalog before committing.

## Cross-version timeline (depth-aware NR3D, filtered n=7805)

| Run | Code | Date | Overall | Hard | V-Dep |
|---|---|---|---:|---:|---:|
| **v9.1_fix FULL REPRO** | `d5f40ba` (broken Stage-1) | 2026-05-16 | **82.95** | **77.88** | **78.23** |
| v3_referit3d_track (invalidated) | older | 2026-05-01 | 80.79 | 75.87 | 72.46 |
| v5p1_failed_rerun (invalidated)  | older | 2026-05-13 | 68.48 | 59.18 | 57.38 |
| v9_2 full text-first | `c2c52d0` (Stage-1 working) | 2026-05-16 | 65.20 | 55.08 | 56.83 |
| v9_2 full no-text    | `c2c52d0` (Stage-1 dropped) | 2026-05-16 | 64.65 | 53.94 | 58.14 |

## Reproduce

```bash
# Pre-run: clean tree + capture both SHAs
cd /Users/bytedance/project/3DVLMReasoning
git status -s   # must be empty
git rev-parse HEAD  # head SHA — record in version doc

# Worktree at the broken-Stage-1 commit
git worktree add /tmp/v91fix-worktree d5f40ba

# Pack already prepared (pack_nr3d_v9_catalog_first, 130 scenes); skip prep
# Run with worktree's src/ but main repo's data/ + tmp/
source .venv/bin/activate
export PYTHONPATH=/tmp/v91fix-worktree/src PYTHONUNBUFFERED=1
DATE=$(date +%Y%m%d_%H%M)
tmux new-session -d -s nr3d-v91-fix-full \
  "./scripts/run_with_rss_guard.sh --rss-limit-mb 28000 --check-interval-sec 60 -- \
   python -m evaluation.scripts.run_nr3d_vg_side_by_side \
       --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
       --data-root data/nr3d/scannet \
       --pack-name pack_nr3d_v9_catalog_first \
       --output-dir tmp/nr3d_eval_v9_1_fix_FULL_REPRO_$DATE \
       --workers 40 --sample-retries 2 \
       --use-tool-answer-disagreement-gate \
       --use-no-match-candidate-guard \
       --use-evidence-frame-guard \
       --checkpoint-only 2>&1 | tee tmp/nr3d_v9_1_fix_FULL_REPRO_$DATE.log"

# After ~8.5h: assemble side_by_side from checkpoints (workers=1, no eval)
PYTHONPATH=/tmp/v91fix-worktree/src python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_1_fix_FULL_REPRO_$DATE \
    --workers 1 --sample-retries 0 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard

# Then leaderboard metrics
PYTHONPATH=/tmp/v91fix-worktree/src python -m evaluation.scripts.nr3d_leaderboard_metrics \
    --side-by-side tmp/nr3d_eval_v9_1_fix_FULL_REPRO_$DATE/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --output tmp/nr3d_eval_v9_1_fix_FULL_REPRO_$DATE/leaderboard_metrics.json \
    --canonical-filter true
```

## SQLite ingestion (already done)

```bash
PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v9_1_fix_FULL_REPRO_20260516_2048/ \
    --run-id v9_1_fix_FULL_REPRO_20260516 \
    --branch feat/v9-1-selectors-return-images \
    --commit d5f40ba \
    --backend pack_v1 \
    --judge-model nr3d-classifier \
    --notes "v9.1_fix FULL REPRO at d5f40ba (worktree); head=e750d8b; Stage-1 broken via wrapper bypass; workers=40; ~8h32m; 82.95% filtered" \
    --db docs/benchmark/nr3d/runs.sqlite
```

## Caveats

- Single seed, single run. The 5 pp Easy/V-Indep noise band we saw on random100 maps to ±2 pp at full-set scale (rough back-of-envelope from larger n). The 17-23 pp gap to v9.2 is far outside that band.
- **`v9.1_fix` is technically a bug state**, not a "good design" — `select_by_text` is broken because of the wrapper bypass bug at `d5f40ba` (fixed in `de8225f`). The 82.95 % is the score of that bug-induced behavior, but it is *exactly* the catalog-only-fallback policy we want to validate.
- Single-side run at workers=40 (no parallel A/B). The 8h32m wall-clock improves on the v9.2 parallel A/B's 15h28m because there's no AK contention — single-side at 40 ≈ parallel 20+20, but with one consistent process and no extra Python startup overhead.
- The 25 `failed` samples are all graceful no-match `-1` submissions, same as v9.2's 16 / 30. No crashes.
- ModelHub 429 retries throughout, absorbed by the AK rotator. No timeouts.

## Implications for v9.3

The previous v9.3 plan (per the random100 / v9.2 docs) was: "task-conditioned routing — fire `select_by_text` only on view-dep / OOD queries". This full-set result strengthens that: the v9.1_fix policy already wins on V-Dep (+21.40 pp) and Hard (+22.80 pp) without ever using a working Stage-1. The simplest v9.3 path is:

1. **Default catalog-only**, replicating the v9.1_fix behaviour without the bug. This means: keep `select_by_text` registered for cache-key parity, but make it a no-op or always-ERROR for NR3D / VG. (Or just gate it via `enable_stage1_text_retrieval=False` per the existing v9.2 toggle, plus restore the original playbook prose.)

2. **Conditionally surface Stage-1** for queries where it'd help — primarily QA-style "where is X" workloads that don't have a catalog handle. NR3D doesn't appear to have many of these in the test fold based on the 17-pp swing.

3. **Fix the audit's empty-pred wall**: even when Stage-1 is allowed to fire, drop the `parse=no_evidence` -> empty `frames: []` failure mode and have it fall through to a coverage-based set. (The audit doc already proposes this.)

This run lands the new headline; v9.3 should consolidate the policy.
