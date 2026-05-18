# v9.4 Experiment A — Force `select_by_text → ERR` cleanly. **Refutes the simple cadence hypothesis.**

> 📎 **Part of the v9.4 GT-leak investigation** —
> consolidated narrative in
> [**v9_4_gt_leak_postmortem_20260518.md**](v9_4_gt_leak_postmortem_20260518.md).
> This doc is **Step 5** (cadence-only ablation, the negative control).
> 63.17 % is essentially "v9.3 honest baseline minus a turn of LLM
> compute"; recovering the v9.1_fix +16 pp required also restoring the
> seed-keyframe drain leak — see Step 6 (v9.4-D) in the post-mortem.
> The `force_stage1_text_retrieval_to_error` flag wired up here remains
> useful as a permanent diagnostic toggle.

This is **Experiment A** from the audit doc
[v9_1_fix_vs_v9_3_audit30_20260517.md](v9_1_fix_vs_v9_3_audit30_20260517.md):
the cheapest hypothesis (1 LoC + 30 min) for closing the 16 pp gap between
v9.1_fix's 82.95 % and v9.3 text-first's 66.67 % on NR3D filtered fold.

The hypothesis: if the cadence pattern that v9.1_fix exhibited was triggered
by the forced `select_by_text → ERROR` first-turn anchor, then cleanly
reproducing that anchor on v9.3 should recover the 83 %.

**Result: refuted.** v9.4-A scores **63.17 %** — even **lower** than the
v9.3 no-text run (64.17 %) and 19.83 pp below the v9.1_fix calibrated
baseline on the same fold. The flag is wired correctly (all 174
`select_by_text` calls returned the new force-disabled ERROR, 0 reached
the selector). The agent simply doesn't act like the bug-state agent.

The 16 pp gap is **not** a simple consequence of the tool ERR signature.

## Pre-run checklist

| Item | Value |
|---|---|
| Pending changes committed before launch | yes (HEAD `5180d60`) |
| **Head commit at launch** | `5180d60` (the v9.4-A flag commit itself) |
| Run-time code commit | `5180d60` — no worktree drift |
| Branch | `feat/v9-1-selectors-return-images` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` (MD5 `12a69d8d…`) |
| Pack | `pack_nr3d_v9_catalog_first` (unchanged across the v9.3 ↔ v9.4-A comparison) |
| Output dir | `tmp/nr3d_eval_v9_4a_strat600_force_error_20260517_2024/` |
| Eval log | `tmp/nr3d_v9_4a_strat600_force_error_20260517_2024.log` |
| Workers | 40 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame (all on) |
| `--force-stage1-text-retrieval-to-error` | **YES** (the variable under test) |
| `--disable-stage1-text-retrieval` | NO (kept text-first playbook + system prompt + tool registration) |
| RSS limit | 15 000 MB |
| Backend | `gpt-5.4-2026-03-05` (same as paired runs) |
| Wall clock | **32.8 min** (20:24:05 → 20:56:52 local) |
| Tracebacks in log | 0 |
| `failed` samples | 1 / 600 (graceful no-match on OOD query) |

## Headline — 4-way comparison on the same strat600 fold

| Metric | v9.1_fix on strat600<br>(calibrated baseline) | **v9.4-A force-ERR**<br>(this run) | v9.3 text-first | v9.3 no-text |
|---|---:|---:|---:|---:|
| **Overall (classification_acc_filtered)** | **83.00 %** | **63.17 %** | 66.67 % | 64.17 % |
| Easy (n=290) | 88.28 % | 71.03 % | 73.45 % | 73.10 % |
| Hard (n=310) | 78.06 % | 55.81 % | 60.32 % | 55.81 % |
| V-Dep (n=211) | 78.20 % | 55.45 % | 54.98 % | 56.87 % |
| V-Indep (n=389) | 85.60 % | 67.35 % | 73.01 % | 68.12 % |
| bbox Acc@0.50 | (≈ 0.83) | 0.6317 | 0.6667 | 0.6417 |
| Mean IoU | — | 0.635 | 0.669 | 0.645 |

### Δ vs each comparator

| Comparator | Δ Overall | Δ Easy | Δ Hard | Δ V-Dep | Δ V-Indep |
|---|---:|---:|---:|---:|---:|
| v9.4-A − v9.3 text-first | **−3.50 pp** | −2.42 | −4.51 | +0.47 | **−5.66** |
| v9.4-A − v9.3 no-text | **−1.00 pp** | −2.07 | 0.00 | −1.42 | −0.77 |
| **v9.4-A − v9.1_fix calibrated** | **−19.83 pp** | −17.25 | −22.25 | **−22.75** | −18.25 |

v9.4-A is essentially indistinguishable from v9.3 no-text (within strat600's
±2.3 pp 90 % band on Overall), and **further** from v9.1_fix than either
clean v9.3 policy. Hard is exactly identical to v9.3 no-text (55.81 %),
which is the strongest single piece of evidence that the force-ERR setup
collapses into the same behaviour profile as catalog-only on the cells where
the audit said the cadence should matter most.

## Tool-level contract: the wiring works

The force-ERR path is firing exactly as designed:

| Stat | v9.4-A | v9.3 text-first | v9.3 no-text |
|---|---:|---:|---:|
| `select_by_text` calls | 174 | 185 | 0 |
| `select_by_text` ERROR returns | **174 (100 %)** | 38 (21 %) | — |
| of which **`force-disabled` signature** (new in v9.4-A) | **174 (100 %)** | 0 | — |
| of which `runtime.keyframe_selector is None` (the legacy bug signature) | 0 | 0 | — |
| of which `Stage-1 parse/exec failed` (real parser misses) | 0 | 36 | — |
| `KeyframeSelector` invocations through `select_by_text` | **0** | 147 | — |

This pins three things:

1. **The flag short-circuits before `KeyframeSelector`** — the unit test
   guarantee holds at scale (0 invocations from this path).
2. **The new ERROR signature is distinguishable** — future audits can grep
   `force-disabled` vs the legacy `runtime.keyframe_selector is None`
   signature to tell v9.4-A traces from v9.1_fix worktree traces.
3. **The agent still chooses whether to call `select_by_text`** — 174 / 600
   calls is **basically the same call rate as v9.3 text-first** (185 / 600).
   The agent rationally avoids the tool when prior turns indicate the
   catalog is fully readable; the text-first playbook prose alone does not
   force every sample to try `select_by_text` first.

This is the key behavioural insight: v9.1_fix bug-state had **every
sample** try `select_by_text` once (30/30 in the audit), not 29 % of
samples. Some other property of the v9.1_fix code path was making the
agent attempt the call every time. The text-first playbook prose by
itself is not that property.

## Tool histogram: v9.4-A vs the three references

Counts are totals across 600 samples; per-sample rates in parentheses.

| Tool | v9.4-A (this) | v9.3 text-first | v9.3 no-text |
|---|---:|---:|---:|
| `inspect_proposal` | **1 793 (2.99)** | 1 685 (2.81) | 1 770 (2.95) |
| `select_by_proposal` | 1 491 (2.48) | 1 319 (2.20) | 1 469 (2.45) |
| `mark_frame_with_bbox` | 1 483 (2.47) | 1 504 (2.51) | 1 449 (2.42) |
| `submit_final` | 1 373 (2.29) | 1 366 (2.28) | 1 412 (2.35) |
| `load_skill` | 1 363 (2.27) | 1 347 (2.25) | 1 245 (2.08) |
| `view_bev` | 639 (1.06) | 600 (1.00) | 568 (0.95) |
| `list_scene_proposals` | **297 (0.49)** | 209 (0.35) | 44 (0.07) |
| `compare_proposals_spatial` | 289 (0.48) | 284 (0.47) | 303 (0.51) |
| `select_by_text` | 174 (0.29) | 185 (0.31) | 0 |
| `select_by_frame_neighbor` | 162 (0.27) | 167 (0.28) | 199 (0.33) |
| `list_frame_proposals` | 109 (0.18) | 139 (0.23) | 23 (0.04) |
| `request_crops` | 69 (0.12) | 69 (0.12) | 40 (0.07) |
| **Total** | **9 286 (15.48)** | 8 915 (14.86) | 8 587 (14.31) |

Notable shifts in v9.4-A relative to v9.3 text-first:

- `list_scene_proposals` +88 (0.35 → 0.49 / sample) — the agent does take
  the playbook's "fall through to enumeration" hint after the forced ERR
  and reads more inventory, just not enough to change outcomes.
- `inspect_proposal` +108 (2.81 → 2.99 / sample) — more candidates
  inspected, matching the v9.1_fix per-sample rate of 3.07 (essentially
  parity now).
- `mark_frame_with_bbox` -21 — **not** the +0.4-frame lift the audit said
  v9.1_fix needs. This is the most direct refutation of the cadence
  hypothesis: the v9.1_fix advantage manifests in per-candidate marking,
  and v9.4-A does not reproduce that increase.

## Magic-6 audit on the same 6 ids (carry-over from audit doc)

The 6 ids where v9.1_fix bug-state was uniquely right while both v9.3
variants were wrong:

| Sample | v9.1_fix | v9.3 text-first | v9.3 no-text | **v9.4-A** |
|---|:-:|:-:|:-:|:-:|
| `scene0015_00::47` longer whiteboard furthest from doorway | ✓ | ✗ | ✗ | ✗ |
| `scene0086_00::10` stall door to the disabled bathroom | ✓ | ✗ | ✗ | ✗ |
| `scene0231_00::45` picture on the wall near the kitchen | ✓ | ✗ | ✗ | ✗ |
| `scene0338_00::21` facing the boxes, box in back-left | ✓ | ✗ | ✗ | ✗ |
| `scene0648_00::10` plant closest to the mirror | ✓ | ✗ | ✗ | ✗ |
| `scene0697_00::7` nightstand with blue striped snake | ✓ | ✗ | ✗ | ✗ |
| **v9.4-A magic-case recovery** | — | — | — | **0 / 6** |

Not a single magic case recovered. The cadence pattern the audit isolated
(per-candidate mark + evidence-frame-guard re-mark + corrected re-submit)
is not reproduced by the force-ERR tool short-circuit alone.

## What this rules out and what it doesn't

**Ruled out** by this run:

- The 16 pp gap is *not* explained by "agent receives ERROR on first
  `select_by_text` call and falls through to a more careful catalog
  walk". The agent does fall through but the additional inspection
  doesn't translate to additional marking, and the additional reading
  doesn't translate to better selections.
- The 16 pp gap is *not* explained by "text-first playbook prose makes
  the agent try `select_by_text` first every sample". The playbook prose
  is the same as the v9.3 text-first run; the call rate is essentially
  the same (29 % vs 31 %).

**Still on the table** (these are the candidates for future experiments):

1. **Chassis turn-0 / image-injection differences between `d5f40ba` and
   `aa11922`.** The "Stage-1 seed-keyframe drain leak" was fixed at
   commit `8ebf701` (after `d5f40ba` but before any v9.3 run). The
   v9.1_fix runs at `d5f40ba` had 5 GT-target-visible seed keyframes
   silently injected into context on every turn. v9.3 runs do not. Even
   though the v9_catalog_first leak-fix doc says fixing the leak
   **improved** random100 by 5 pp (81 → 86) in the **catalog-only**
   policy, the same leak in a **text-first** policy may have helped:
   the text-first agent already requests its own frames via
   `select_by_text`, so the GT-visible seeds become bonus context
   pointing at the right answer.
2. **System prompt / playbook-injection structure changes.** The
   `selector_lines` block in `base.py` and the playbook bodies have
   been edited multiple times between `d5f40ba` and `aa11922`. Diff is
   tractable but ~50 commits.
3. **`view_bev` / `mark_frame_with_bbox` UI redesigns** (commits
   `2cd5e6d`, `7696625`). The v9.3 default BEV is sparser and the
   bounding-box label style is different. Either could shift the
   agent's spatial reasoning. Worth a small ablation: run v9.3-text on
   strat600 with `SceneBEVConfig` reverted to v9.1_fix defaults.
4. **Different `Stage2DeepResearchAgent.run` chassis loop semantics.**
   The wrapper at `d5f40ba` and the wrapper at `aa11922` differ in how
   they handle `submit_final` retries and evidence-frame-guard
   blocks. This is fiddly but on the critical path of the v9.1_fix
   re-mark cycle (audit found 3 / 6 magic cases involved a "first
   submit wrong → re-mark → second submit right" trajectory that the
   evidence-frame guard drives).

## Recommended next experiments

The audit doc proposed Experiments B (stronger playbook prose) and C
(per-query routing + cadence-forced playbook). With Experiment A as a
clear negative, the priority order shifts:

1. **Experiment D (NEW): Restore the Stage-1 seed-keyframe drain leak.**
   This is the highest-leverage hypothesis remaining. One config flag:
   `Stage2DeepAgentConfig.restore_stage1_seed_keyframe_drain: bool =
   False`. When True, `DeepAgentsStage2Runtime.build_evidence_update_message`
   skips the `initial_keyframe_paths` filter, so the 5 GT-target-visible
   seed keyframes flow into context every turn — exactly what the bug
   at `d5f40ba` did. Run on strat600 with `--force-stage1-text-retrieval-to-error`
   (to keep text-first playbook + tool registered) plus this new flag.
   Cost: ~30 min for the change + 32 min wall clock.
2. **Experiment E: BEV UI revert.** Add a config flag to disable the
   v9.3 BEV default-no-labels behaviour and `_v93` cache tag. Run on
   strat600 with the rest of the v9.4-A flags. Cost: ~1 h.
3. **Experiment B (deferred):** Strong playbook prose mandating
   "always call `select_by_text` first, treat any response as data".
   Cost: ~2 h prose + 32 min run.
4. **Experiment C (deferred):** Per-query routing on top of the
   winning policy from D / E / B.

If neither D nor E recovers the gap, the next step is a bisection
between `d5f40ba` and `aa11922` over the ~50 intermediate commits —
expensive but mechanical.

## Files

| Path | Purpose |
|---|---|
| `tmp/nr3d_eval_v9_4a_strat600_force_error_20260517_2024/` | Per-sample checkpoints + side_by_side.json (14 MB) + leaderboard_metrics.json |
| `tmp/nr3d_v9_4a_strat600_force_error_20260517_2024.log` | tee'd eval log (0 tracebacks) |
| `docs/benchmark/nr3d/assets/v9_4a_strat600_force_error_20260517_leaderboard.json` | Durable leaderboard copy |
| `docs/benchmark/nr3d/assets/v9_4a_strat600_force_error_20260517_side_by_side_metadata.json` | Slim 525 KB metadata copy |
| `docs/benchmark/nr3d/runs.sqlite` row `v9_4a_strat600_force_error_20260517` | SQLite |
| `docs/benchmark/nr3d/v9_4a_strat600_force_error_20260517.md` | **This document** |

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
    --output-dir tmp/nr3d_eval_v9_4a_strat600_force_error_<DATE> \
    --workers 40 --sample-retries 2 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard \
    --force-stage1-text-retrieval-to-error
```

Run-time commit must include the v9.4-A flag wiring — anything ≥ `5180d60`
on `feat/v9-1-selectors-return-images` works.

## Caveats

- **n=600 has ±2.3 pp 90 %-band on Overall on the strat600 fold** (per
  the fold-design doc bootstrap). The −1.00 pp gap to v9.3 no-text and
  the −3.50 pp gap to v9.3 text-first both sit inside or just outside
  that band. The −19.83 pp gap to v9.1_fix calibrated is far outside.
- **Per-tier bands are wider** (Hard ±3.5 pp, V-Dep ±4.5 pp). The
  v9.4-A vs v9.3-no exact tie on Hard (55.81 % both) is a stronger
  qualitative signal than the Overall delta.
- **Only one seed**. We don't have a v9.4-A reproduction yet. The
  variance between independent runs of the same code on the same fold
  is typically ~1 pp on Overall (per the v9.1_fix random100 reproduction
  doc).
- **The flag is now permanent infrastructure**. Even though Experiment
  A failed, the wiring is useful for Experiment D (combining force-ERR
  + drain-leak-restore) and any future cadence-isolation experiments.
