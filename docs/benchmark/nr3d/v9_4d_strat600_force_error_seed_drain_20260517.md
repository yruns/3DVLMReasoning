# v9.4 Experiment D — Restore the Stage-1 seed-keyframe drain leak. **Hypothesis confirmed: leak = +16 pp**.

> 📎 **Part of the v9.4 GT-leak investigation** —
> consolidated narrative in
> [**v9_4_gt_leak_postmortem_20260518.md**](v9_4_gt_leak_postmortem_20260518.md).
> This doc is **Step 6** (final isolation). The 83.33 % here is the
> **GT-visible upper bound**, NOT a fair NR3D number — both diagnostic
> flags (`force_stage1_text_retrieval_to_error` +
> `restore_stage1_seed_keyframe_drain`) are test-time-only. Read the
> post-mortem for the unified narrative and the implications for
> v9.4's next direction.

This is **Experiment D** from
[v9_4a_strat600_force_error_20260517.md §Recommended next experiments](v9_4a_strat600_force_error_20260517.md).
v9.4-A (force-ERR only) refuted the simple cadence hypothesis. Experiment D
adds the second flag: restore the Stage-1 seed-keyframe drain leak that was
present at commit `d5f40ba` (v9.1_fix FULL REPRO) and fixed in `8ebf701`.

**Result: hypothesis confirmed**. v9.4-D scores **83.33 % Overall** — within
**+0.33 pp** of the v9.1_fix strat600 calibrated baseline (83.00 %). Hard
**+4.20 pp better**. **6 / 6 magic audit cases recovered**. The v9.1_fix
+16 pp advantage is fully isolated to the leak, **not** the deliberation
cadence.

**This means v9.1_fix's 82.95 % is an information leak, not a fair NR3D
result**. The honest v9.3 baseline is ~65 % (matches public UniVLG SOTA at
65.2 %). The leak silently injects 5 GT-target-visible RGB frames into agent
context every turn — the agent doesn't know it's seeing the answer.

## Pre-run checklist

| Item | Value |
|---|---|
| Pending changes committed before launch | yes (HEAD `a0d802b`) |
| **Head commit at launch** | `a0d802b` |
| Run-time code commit | `a0d802b` — no worktree drift |
| Branch | `feat/v9-1-selectors-return-images` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` (MD5 `12a69d8d…`, identical to all paired runs) |
| Pack | `pack_nr3d_v9_catalog_first` (unchanged across the 5-way A/B/C/D/E) |
| Output dir | `tmp/nr3d_eval_v9_4d_strat600_force_error_seed_drain_20260517_2129/` |
| Eval log | `tmp/nr3d_v9_4d_strat600_force_error_seed_drain_20260517_2129.log` |
| Workers | 40 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame (all on) |
| `--force-stage1-text-retrieval-to-error` | **YES** (v9.4-A flag) |
| `--restore-stage1-seed-keyframe-drain` | **YES** (v9.4-D flag, the new variable) |
| RSS limit | 15 000 MB |
| Backend | `gpt-5.4-2026-03-05` |
| Wall clock | **34.5 min** (21:29:55 → 22:04:22 local) |
| Tracebacks in log | 0 |
| `failed` samples | 1 / 600 (graceful no-match) |

## Headline — 5-way comparison on strat600

| Metric | v9.1_fix calibrated baseline | **v9.4-D**<br>force-ERR + restore-drain | v9.4-A force-ERR | v9.3 text-first | v9.3 no-text |
|---|---:|---:|---:|---:|---:|
| **Overall (classification_acc_filtered)** | **83.00 %** | **83.33 %** | 63.17 % | 66.67 % | 64.17 % |
| Easy (n=290) | 88.28 % | 84.48 % | 71.03 % | 73.45 % | 73.10 % |
| Hard (n=310) | 78.06 % | **82.26 %** | 55.81 % | 60.32 % | 55.81 % |
| V-Dep (n=211) | 78.20 % | 76.78 % | 55.45 % | 54.98 % | 56.87 % |
| V-Indep (n=389) | 85.60 % | 86.89 % | 67.35 % | 73.01 % | 68.12 % |
| bbox Acc@0.50 | (≈ 0.83) | 0.8333 | 0.6317 | 0.6667 | 0.6417 |

### Δ vs v9.1_fix calibrated baseline (the target)

| Metric | Δ |
|---|---:|
| Overall  | **+0.33 pp** ← within ±2.3 pp strat600 90 % band |
| Easy     | −3.80 pp |
| Hard     | **+4.20 pp** ← v9.4-D BEATS v9.1_fix on Hard |
| V-Dep    | −1.42 pp |
| V-Indep  | +1.29 pp |

v9.4-D essentially matches v9.1_fix on Overall and per-tier within fold
variance. The +4.20 pp Hard improvement is just outside the per-tier band
and may be a real edge (v9.4-D inherits v9.3's working evidence-frame guard
and improved mark_frame_with_bbox labelling, which v9.1_fix lacked).

### Δ vs v9.4-A (which had only the force-ERR flag, not the leak)

| Metric | Δ |
|---|---:|
| Overall  | **+20.16 pp** |
| Easy     | +13.45 pp |
| Hard     | **+26.45 pp** |
| V-Dep    | +21.33 pp |
| V-Indep  | +19.54 pp |

Adding the `restore-stage1-seed-keyframe-drain` flag on top of v9.4-A
lifts Overall by +20 pp and Hard by +26 pp. **The cadence-anchor hypothesis
contributed essentially 0 pp; the leak contributed essentially all 16-19 pp**.

## Magic-6 audit recovery

The 6 ids the audit isolated as "v9.1_fix uniquely right, both v9.3 variants wrong":

| Sample | v9.1_fix | v9.3-txt | v9.3-no | v9.4-A | **v9.4-D** |
|---|:-:|:-:|:-:|:-:|:-:|
| `scene0015_00::47` longer whiteboard farthest from doorway | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0086_00::10` stall door to the disabled bathroom | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0231_00::45` picture on the wall near the kitchen | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0338_00::21` facing the boxes, box in back-left | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0648_00::10` plant closest to the mirror | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0697_00::7` nightstand with blue striped snake | ✓ | ✗ | ✗ | ✗ | **✓** |
| **Recovery rate** | — | 0 % | 0 % | 0 / 6 | **6 / 6 (100 %)** |

Clean separation between v9.4-A (cadence-anchor only, 0 / 6) and v9.4-D
(cadence-anchor + leak, 6 / 6). The leak is the mechanism.

## Wiring verification

Both v9.4 flags fired exactly as designed:

| Stat | v9.4-D (this) | v9.4-A | v9.3 text-first |
|---|---:|---:|---:|
| `select_by_text` calls | 177 | 174 | 185 |
| `select_by_text` ERROR returns | **177 (100 %)** | 174 (100 %) | 38 (21 %) |
| of which `force-disabled` signature | **177** | 174 | 0 |
| `KeyframeSelector` invocations from `select_by_text` | 0 | 0 | 147 |
| Seed-keyframe injections per turn | **6 (1 BEV + 5 seeds)** ✓ | (1 BEV) | (1 BEV) |
| Tracebacks | 0 | 0 | 0 |
| Per-sample completed | 599 / 600 | 599 / 600 | 599 / 600 |
| Per-sample graceful no-match | 1 | 1 | 1 |

The smoke-test log (5 magic cases) showed every evidence-update injecting
**6 new images** consistently — that's the 1 BEV + 5 seed-keyframe pattern
the v9.1_fix bug-state had. Smoke 4 / 5 correct (vs v9.4-A on the same 5
which was 1 / 5).

## Tool histogram — what does the agent do with the leaked seeds?

Across all 600 samples (totals + per-sample rates):

| Tool | v9.4-D (this) | v9.1_fix audit per-sample | v9.4-A | v9.3 text-first |
|---|---:|---:|---:|---:|
| `mark_frame_with_bbox` | **2 088 (3.48)** | (2.90) | 1 483 (2.47) | 1 504 (2.51) |
| `inspect_proposal` | 1 737 (2.90) | (3.07) | 1 793 (2.99) | 1 685 (2.81) |
| `select_by_proposal` | 1 395 (2.33) | — | 1 491 (2.48) | 1 319 (2.20) |
| `submit_final` | 1 338 (2.23) | (2.20) | 1 373 (2.29) | 1 366 (2.28) |
| `load_skill` | 1 336 (2.23) | — | 1 363 (2.27) | 1 347 (2.25) |
| `view_bev` | 608 (1.01) | — | 639 (1.06) | 600 (1.00) |
| `list_scene_proposals` | 323 (0.54) | — | 297 (0.49) | 209 (0.35) |
| `compare_proposals_spatial` | 286 (0.48) | — | 289 (0.48) | 284 (0.47) |
| `list_frame_proposals` | 233 (0.39) | — | 109 (0.18) | 139 (0.23) |
| `select_by_text` | 177 (0.29) | (1.00 forced) | 174 (0.29) | 185 (0.31) |
| `select_by_frame_neighbor` | 157 (0.26) | — | 162 (0.27) | 167 (0.28) |
| `request_crops` | 80 (0.13) | — | 69 (0.12) | 69 (0.12) |
| **Total** | **9 799 (16.33)** | (16.1) | 8 587 (14.31) | 8 915 (14.86) |

Notable per-tier rate shifts in v9.4-D vs v9.4-A:

- **`mark_frame_with_bbox` +1.01 / sample (2.47 → 3.48)** — the agent
  marks 41 % more frames after seeing the leaked seeds. This is what the
  audit predicted ("v9.1_fix marks 2.90 / sample vs v9.3-no's 2.13"). The
  leak triggers the cadence; the cadence isn't intrinsic to the
  force-ERR signature alone.
- `list_scene_proposals` +0.05 — agent still enumerates.
- `list_frame_proposals` +0.21 — agent now asks more about *seen* frames
  (the leaked ones), confirming it's using the seeds as anchors.
- `inspect_proposal` −0.09 — slightly less catalog reading; the seeds
  already show the agent what to focus on.

The increased marking + reduced raw inspection is the v9.1_fix cadence,
**but it only emerges with the leak in place**. v9.4-A had the same
text-first playbook and tool-side ERR signature but never showed this
cadence because there was nothing in the agent's visual context to anchor
the per-candidate marking around.

## The leak — what is it, exactly?

`bundle.keyframes` is populated by `prepare_pack_v1_inputs_nr3d.py` at
pack-prep time. The 5 default entries are GT-target-visible RGB frames —
i.e., frames the pack-prep step **knows** contain the answer (because it
has access to the target ground-truth bbox during prep). These are written
into the pack so that legacy v6-era runners had something to seed visual
evidence with.

In the v9 (catalog-first) era, those seed keyframes were not supposed to
flow into the agent's context — the agent is supposed to do its own
catalog-first reasoning. But at commit `d5f40ba`,
`build_evidence_update_message` (in `agents/runtime/deepagents_agent.py`)
drained **every** keyframe in `runtime.bundle.keyframes` on every
evidence-update turn. So after the agent's first deferred `submit_final`
attempt (which the evidence-frame guard typically defers once), the 5 GT-
visible seeds silently entered the context.

Commit `8ebf701` plugged this by snapshotting the pack-prep keyframe paths
into `Stage2RuntimeState.initial_keyframe_paths` at runtime construction
and filtering them out of the evidence-update drain.

v9.4-D uses
`Stage2DeepAgentConfig.restore_stage1_seed_keyframe_drain = True` to
bypass that filter. Wiring: 4 files, 14 LoC + 5 tests
(commit `a0d802b`).

## Implications

### The v9.1_fix 82.95 % FULL row is invalidated as a fair NR3D claim

This puts `v9_1_fix_FULL_REPRO_20260516` in the same status as
`v3_referit3d_track_20260501` (GT-target-visible projection-only
visibility) and `v5p1_failed_rerun_full_20260513` (projection-only
visibility index): a benchmark-grade number that turned out to depend on
GT information leaking into the model's input.

The leaderboard.md and per-version README rows for v9.1_fix should be
re-tagged with a "fair-view caveat" note. The number is still a useful
upper bound for how well the agent can perform when given target-visible
context, but it cannot be quoted against public NR3D SOTA.

### The honest v9.3 baseline is ~65 %, matching public UniVLG SOTA 65.2 %

The clean v9.3 results (66.67 % text-first, 64.17 % no-text) are the actual
state-of-the-art-fair numbers for this agent on NR3D filtered fold. They
are competitive with public UniVLG (65.2 %), within 1.5 pp.

Per-tier:

| Tier | UniVLG SOTA | v9.3 text-first | Δ |
|---|---:|---:|---:|
| Overall | 65.2 | 66.67 | +1.47 |
| Easy | 73.3 | 73.45 | +0.15 |
| Hard | 57.0 | 60.32 | +3.32 |
| V-Dep | 55.1 | 54.98 | −0.12 |
| V-Indep | 69.9 | 73.01 | +3.11 |

So the agent is **competitive with public SOTA** and even slightly ahead
on Hard / V-Indep, with zero information leak. The headline narrative
should be "v9.3 matches UniVLG", not "v9.1_fix beats UniVLG by 17 pp".

### v9.4 next steps need to lift accuracy without the leak

Recovering even ~70 % cleanly would put the agent solidly above UniVLG.
Concrete options:

1. **Per-query routing** (Experiment C from the audit doc) — view-dep
   queries to `_no_text` playbook, rare-target queries to text-first.
   Cost: ~1 day. v9.2 FULL A/B suggests the gap is small (+0.55 pp on
   matched-fold), but per-tier routing might give +2-3 pp.
2. **Stronger playbook prose** (Experiment B from the audit doc) —
   explicit cadence + per-candidate marking instructions. The v9.4-D
   tool histogram shows the agent CAN follow this pattern when anchored;
   the question is whether prose alone (without the leak) is enough to
   trigger it. Cost: ~2 h.
3. **Improve the BEV / catalog signal** — the chassis turn-0 panel
   currently shows a clean BEV + Cat-B text. The audit shows agents are
   confidently making wrong picks before seeing first-person frames.
   Stronger turn-0 evidence (e.g., per-proposal "what does this look
   like" thumbnails, not just dots) could lift the catalog-only path
   directly. Cost: ~2-3 days.

### The leak as a research tool

`--restore-stage1-seed-keyframe-drain` should be retained as a
benchmark-time flag for **diagnostic** purposes:

- Establishing an upper bound for the agent's task-solving capability
  when given GT-visible visual context (useful for ablations of system
  prompt / playbook prose: if v9.4-D-style runs don't lift with prose
  changes, the prose isn't the bottleneck).
- Estimating the cost-of-no-leak (run v9.4-D and v9.3 on the same fold;
  the gap is the value-add of better visual evidence retrieval).

But it cannot be used to produce a published number.

## SQLite 5-way reproduction

```sql
SELECT run_id, n, n_filtered,
       ROUND(classification_acc_filtered*100, 2) AS overall,
       ROUND(acc_easy*100, 2) AS easy,
       ROUND(acc_hard*100, 2) AS hard,
       ROUND(acc_view_dep*100, 2) AS vdep,
       ROUND(acc_view_indep*100, 2) AS vindep
FROM runs WHERE run_id IN (
  'v9_4d_strat600_force_error_seed_drain_20260517',
  'v9_4a_strat600_force_error_20260517',
  'v9_3_strat600_20260517',
  'v9_3_strat600_notext_20260517',
  'v9_1_fix_FULL_REPRO_20260516'
) ORDER BY classification_acc_filtered DESC;
```

| run_id | n | n_filt | overall | easy | hard | vdep | vindep |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`v9_4d_strat600_force_error_seed_drain_20260517` (this)** | 600 | 600 | **83.33** | 84.48 | **82.26** | 76.78 | 86.89 |
| `v9_1_fix_FULL_REPRO_20260516` | 8 584 | 7 805 | 82.95 | 88.36 | 77.88 | 78.23 | 85.51 |
| `v9_3_strat600_20260517` | 600 | 600 | 66.67 | 73.45 | 60.32 | 54.98 | 73.01 |
| `v9_3_strat600_notext_20260517` | 600 | 600 | 64.17 | 73.10 | 55.81 | 56.87 | 68.12 |
| `v9_4a_strat600_force_error_20260517` | 600 | 600 | 63.17 | 71.03 | 55.81 | 55.45 | 67.35 |

## Files

| Path | Purpose |
|---|---|
| `tmp/nr3d_eval_v9_4d_strat600_force_error_seed_drain_20260517_2129/` | Per-sample checkpoints + side_by_side.json (14 MB) + leaderboard_metrics.json |
| `tmp/nr3d_v9_4d_strat600_force_error_seed_drain_20260517_2129.log` | tee'd eval log (0 tracebacks) |
| `docs/benchmark/nr3d/assets/v9_4d_strat600_force_error_seed_drain_20260517_leaderboard.json` | Durable leaderboard JSON |
| `docs/benchmark/nr3d/assets/v9_4d_strat600_force_error_seed_drain_20260517_side_by_side_metadata.json` | Slim 525 KB no-tool-trace copy |
| `docs/benchmark/nr3d/runs.sqlite` row `v9_4d_strat600_force_error_seed_drain_20260517` | SQLite ingestion |
| `docs/benchmark/nr3d/v9_4d_strat600_force_error_seed_drain_20260517.md` | **This document** |

## Reproduce

```bash
cd /Users/bytedance/project/3DVLMReasoning
source .venv/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1

./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_4d_strat600_force_error_seed_drain_<DATE> \
    --workers 40 --sample-retries 2 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard \
    --force-stage1-text-retrieval-to-error \
    --restore-stage1-seed-keyframe-drain
```

Run-time commit must include both v9.4 flags — anything ≥ `a0d802b` on
`feat/v9-1-selectors-return-images` works.

## Caveats

- **n=600 has ±2.3 pp 90 %-band on Overall.** The +0.33 pp gap to
  v9.1_fix calibrated is well within the band — the runs are
  statistically indistinguishable on Overall.
- **Per-tier bands are wider** (Hard ±3.5, V-Dep ±4.5, V-Indep ±2.7).
  The +4.20 pp Hard improvement vs v9.1_fix is just outside the per-tier
  band and is positive — likely a real edge from v9.3's working
  evidence-frame guard + better mark_frame_with_bbox labels.
- **Single seed.** Run-to-run variance is typically ~1 pp on Overall.
- **The flag is permanent infrastructure** but should NEVER be enabled
  in any run that produces a published number. The doc string and CLI
  help both call out test-time-only usage.
- **`bundle.keyframes` is itself problematic** — even with v9.4-D off
  (default), the keyframes are still on disk and could be re-introduced
  by any future code path that iterates `bundle.keyframes` without the
  `initial_keyframe_paths` filter. A follow-up should consider stopping
  pack-prep from writing the seeds at all when they aren't needed (the
  runner-side `keyframes must be non-empty` assertion mentioned in
  `memory/v9_catalog_first_branch.md` Open followups).
