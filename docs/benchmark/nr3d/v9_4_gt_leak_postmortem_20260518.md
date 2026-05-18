# NR3D GT-leak post-mortem — `v9.1_fix` 82.95 % is an information leak, not SOTA

**Date**: 2026-05-17 / 2026-05-18 (investigation concluded 2026-05-17 22:04 local)
**Branch**: `feat/v9-1-selectors-return-images`
**Investigation commits (oldest → newest)**:
- `cdca81f` — strat600 canonical fold (the workhorse of the investigation)
- `aa11922` — v9.3 strat600 text-first 66.67 %
- `b3bce7b` — v9.3 strat600 catalog-only A/B 64.17 %
- `eb66846` — 30-case audit (v9.1_fix vs v9.3 per-sample)
- `5180d60` — `force_stage1_text_retrieval_to_error` flag (v9.4-A wiring)
- `ab8ad07` — v9.4-A run 63.17 % (refutes the cadence hypothesis)
- `a0d802b` — `restore_stage1_seed_keyframe_drain` flag (v9.4-D wiring)
- `2b896d6` — **v9.4-D run 83.33 % (isolates the leak)**

## TL;DR

The `v9_1_fix_FULL_REPRO_20260516` row at **82.95 %** Overall on NR3D
filtered fold is **not a fair NR3D result**. The agent silently received
5 GT-target-visible RGB seed keyframes per evidence-update turn — a
benign-looking implementation bug at commit `d5f40ba`
(`build_evidence_update_message` not filtering `bundle.keyframes`), fixed
in `8ebf701`.

Cleanly reproducing the bug on v9.3 code via the new diagnostic flag
`--restore-stage1-seed-keyframe-drain` recovers **83.33 %** on the
strat600 calibrated fold — within +0.33 pp of the v9.1_fix calibrated
baseline, and **6 / 6** of the audit-isolated "magic" cases. The
mechanism is fully attributed to the leak; not the cadence, not the
playbook prose, not the tool surface.

| Status | Number | Caveat |
|---|---:|---|
| ❌ **Not fair** | 82.95 % (v9.1_fix FULL, the prior "best") | Information leak: 5 GT-visible RGBs injected/turn |
| ❌ **Not fair** | 83.33 % (v9.4-D leak reproduction) | Same leak via explicit diagnostic flag |
| ✅ **Fair** | **66.67 %** (v9.3 text-first, strat600) | Honest baseline — matches public UniVLG SOTA |
| ✅ **Fair** | 64.17 % (v9.3 no-text, strat600) | Catalog-only honest baseline |

The honest v9.3 number (~65 %) matches **public UniVLG SOTA 65.2 %** on
Overall — competitive with published work without any information leak.

The investigation's parting narrative should be **"v9.3 matches UniVLG"**,
NOT "v9.1_fix beats UniVLG by 17 pp".

## How we got here (the timeline)

### Step 0 — The puzzle (2026-05-16)

`v9_1_fix_FULL_REPRO_20260516.md` reported the highest depth-aware NR3D
score in the project's history (82.95 % filtered), beating v3
(invalidated, 80.79 %), v5p1 (invalidated, 68.48 %), and v9.2's clean
runs (text-first 65.20 %, no-text 64.65 %) by 17–18 pp.

The doc itself flagged this as suspect: *"v9.1_fix is technically a bug
state — `select_by_text` is broken because of the wrapper bypass bug at
`d5f40ba`."* So the 82.95 % was somehow tied to that bug, but at the
time we didn't understand the mechanism.

The naive hypothesis at the start of this investigation: when
`select_by_text` always errors, the agent falls through to
`select_by_proposal` / `list_scene_proposals` and that catalog-only
fallback policy is what scores 83 %. The plan: replicate "clean
catalog-only" on the working v9.3 code and check.

### Step 1 — Build a comparable fold (`cdca81f`)

random100 (the old fold) had ±15 pp 90 %-band on Overall — too noisy.
Built **strat600**, a 600-case stratified subset (Easy × V-Dep cells,
salt-locked to v9.1_fix calibrated within 0.19 pp on all 5 leaderboard
columns). 1000-trial bootstrap gives ±2.3 pp 90 %-band on Overall — fast
(32 min vs 8 h FULL) and statistically calibrated. See
[v9_3_strat600_subset_design_20260517.md](v9_3_strat600_subset_design_20260517.md).

### Step 2 — Baseline v9.3 (`aa11922`)

Ran v9.3 HEAD (working `select_by_text`, working chassis) text-first on
strat600. Got **66.67 %**, −16.33 pp below v9.1_fix calibrated 83.00 %.
At this point I argued the gap was the "Stage-1 working = −17 pp"
finding from v9.2 — exactly as the v9.1 audit predicted. See
[v9_3_strat600_20260517.md](v9_3_strat600_20260517.md).

### Step 3 — A/B catalog-only on v9.3 (`b3bce7b`)

The "easy" hypothesis: maybe just dropping `select_by_text` and switching
to the `_no_text` playbook reproduces v9.1_fix's catalog-only behaviour.
Ran with `--disable-stage1-text-retrieval`. Result: **64.17 %** — same
ballpark as v9.3 text-first, and v9.2 no-text FULL. **−18.78 pp** vs
v9.1_fix; not even moving in the right direction. The simple "clean
catalog-only = 83" hypothesis is refuted. See
[v9_3_strat600_notext_20260517.md](v9_3_strat600_notext_20260517.md).

### Step 4 — Per-sample audit on 30 ids (`eb66846`)

If neither clean v9.3 policy reproduces 83 %, what does v9.1_fix do
*differently* on each sample? Built a 30-case stratified audit subset of
strat600, ran v9.1_fix bug-state worktree (`d5f40ba`) and both v9.3
variants on the same 30. v9.1_fix won 6 cases that both v9.3 variants
lost — the "magic 6". Trace audit found a recurring pattern:

```
select_by_text → ERROR → catalog walk → per-candidate mark_frame_with_bbox
  → first submit → evidence-frame-guard re-mark → corrected re-submit
```

Aggregate stat: v9.1_fix marks **+36 %** more frames per sample
(2.90 vs 2.13). `inspect_proposal` rate is identical (3.07/sample).
**Hypothesis**: the 16 pp is a **deliberation cadence** — forcing `select_by_text → ERR`
triggers the agent to walk the text-first playbook's documented
fallback chain (catalog walk + per-candidate marking + re-mark cycle).
See [v9_1_fix_vs_v9_3_audit30_20260517.md](v9_1_fix_vs_v9_3_audit30_20260517.md).

### Step 5 — Experiment A: force `select_by_text → ERR` on v9.3 (`5180d60`, `ab8ad07`)

Added `force_stage1_text_retrieval_to_error` config flag. `select_by_text`
stays registered (text-first playbook + system prompt unchanged), but
the tool body short-circuits to ERROR before touching `KeyframeSelector`.
Ran on strat600.

Result: **63.17 %**. **Worse** than both clean v9.3 baselines.
**0 / 6** magic cases recovered. The cadence hypothesis is **refuted**.

Wiring is verified correct: 174/174 calls return the new force-disabled
ERROR signature, 0 reach the selector. But the agent only invoked
`select_by_text` on 29 % of samples (vs v9.1_fix audit's 100 % of
samples) — so the text-first playbook prose alone does not force the
"always try select_by_text first" cadence. Something else does. See
[v9_4a_strat600_force_error_20260517.md](v9_4a_strat600_force_error_20260517.md).

### Step 6 — Experiment D: restore the seed-keyframe drain leak (`a0d802b`, `2b896d6`)

Re-read the `8ebf701` commit message. It plugged a leak in
`build_evidence_update_message`: at commit `d5f40ba`, the function
drained **every** keyframe in `runtime.bundle.keyframes` every
evidence-update turn — including the 5 GT-target-visible seed keyframes
that pack-prep writes (using GT bbox during prep). After the agent's
first deferred `submit_final` (which the evidence-frame guard typically
defers once), those seeds silently entered context.

Added `restore_stage1_seed_keyframe_drain` config flag that bypasses the
`initial_keyframe_paths` filter. Combined with `--force-stage1-text-retrieval-to-error`,
this should cleanly reproduce the v9.1_fix runtime behaviour.

Result on strat600: **83.33 %** Overall — within +0.33 pp of v9.1_fix
calibrated 83.00 %, **Hard +4.20 pp ABOVE** v9.1_fix, **6 / 6 magic
cases recovered**. Each evidence-update turn now injects 6 images
(1 BEV + 5 GT-visible seed keyframes) — exactly the v9.1_fix bug-state
shape. `mark_frame_with_bbox` shoots from 2.47/sample (v9.4-A) to
3.48/sample (v9.4-D). The cadence IS triggered, but only when the
leak gives the agent anchored visual context to mark.

The 16 pp is fully isolated as the leak. See
[v9_4d_strat600_force_error_seed_drain_20260517.md](v9_4d_strat600_force_error_seed_drain_20260517.md).

## The decisive table

5-way on strat600 fold, same pack, same guards, same backend, same fold
calibration:

| Run | Code commit | Stage-1 state | Seed-drain state | Overall | Easy | Hard | V-Dep | V-Indep |
|---|---|---|---|---:|---:|---:|---:|---:|
| **v9.1_fix FULL REPRO (calibrated to strat600)** | `d5f40ba` (worktree) | broken (bug) | LEAK ON (bug) | **83.00** | 88.28 | 78.06 | 78.20 | 85.60 |
| **v9.4-D leak-restore** (this investigation) | `a0d802b` | force-ERR (flag) | LEAK ON (flag) | **83.33** | 84.48 | **82.26** | 76.78 | 86.89 |
| v9.4-A force-ERR (no leak) | `5180d60` | force-ERR (flag) | filtered (clean) | 63.17 | 71.03 | 55.81 | 55.45 | 67.35 |
| v9.3 no-text (clean catalog-only) | `aa11922` | disabled | filtered (clean) | 64.17 | 73.10 | 55.81 | 56.87 | 68.12 |
| **v9.3 text-first (honest baseline)** | `ae83fea` | working | filtered (clean) | **66.67** | 73.45 | 60.32 | 54.98 | 73.01 |

### Magic-6 case recovery (the cleanest visualisation)

These 6 ids were uniquely won by v9.1_fix bug-state and lost by both v9.3
variants in the audit:

| Sample | v9.1_fix (bug) | v9.3 text-first | v9.3 no-text | **v9.4-A** (force-ERR only) | **v9.4-D** (force-ERR + leak) |
|---|:-:|:-:|:-:|:-:|:-:|
| `scene0015_00::47` longer whiteboard farthest from doorway | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0086_00::10` stall door to the disabled bathroom | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0231_00::45` picture on the wall near the kitchen | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0338_00::21` facing the boxes, box in back-left | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0648_00::10` plant closest to the mirror | ✓ | ✗ | ✗ | ✗ | **✓** |
| `scene0697_00::7` nightstand with blue striped snake | ✓ | ✗ | ✗ | ✗ | **✓** |
| **Recovery** | — | 0 % | 0 % | **0 / 6** | **6 / 6** |

This is the cleanest mechanistic separator. Adding `restore-drain` on
top of force-ERR flips every single magic case from wrong to right.
Cadence forcing without the leak (v9.4-A) recovers none.

## What is the leak, mechanically?

```
┌────────────────────────────────────────────────────────────────────┐
│  pack-prep (offline)                                               │
│    src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py           │
│    -> reads GT bbox from NR3D referit3d annotations                │
│    -> finds 5 RGB frames whose depth-aware visibility shows the    │
│       GT target                                                    │
│    -> writes these 5 paths into pack/<sample>.json[keyframes]      │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  bundle build  (run_nr3d_vg_side_by_side.py)                       │
│    -> Stage2EvidenceBundle(keyframes=<those 5 paths>, ...)         │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  build_user_message  (chassis turn-0)         <-- correctly         │
│    -> agent sees: BEV + SceneCatalog text         catalog-first.    │
│    -> agent does NOT see: bundle.keyframes        Seeds NOT used.   │
└────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  build_evidence_update_message  (every turn)                       │
│                                                                    │
│    at commit d5f40ba (the leak):                                   │
│      for kf in runtime.bundle.keyframes:                           │
│          if Path(kf.image_path).exists() and kf not in seen:       │
│              new_images.append(kf.image_path)                      │
│      -> all 5 GT-visible seeds get auto-injected silently          │
│      -> agent now "sees" the target without asking for it          │
│                                                                    │
│    after commit 8ebf701 (the fix):                                 │
│      initial_seeds = runtime.initial_keyframe_paths                │
│      for kf in runtime.bundle.keyframes:                           │
│          if kf.image_path in initial_seeds:                        │
│              continue   # <-- skips the 5 pack-prep seeds          │
│          ...                                                       │
│      -> only tool-pushed keyframes (request_crops crops etc) drain │
└────────────────────────────────────────────────────────────────────┘
```

The fix in `8ebf701` snapshots the pack-prep seed paths into
`Stage2RuntimeState.initial_keyframe_paths` at runtime construction,
then filters them out of the evidence-update drain pass. Tool-produced
keyframes (e.g. `request_crops` appending new crop paths) are not in
the snapshot and still drain correctly.

The `v9_catalog_first_20260515.md` doc (commit `8ebf701`'s companion
write-up) shows the fix improved random100 from 81 → 86 in catalog-only
policy. But it ran the leaky vs clean comparison in catalog-only mode,
where the leak hurts (because it biases the agent toward GT-target-visible
seeds that show ONE of N candidates, starving disambiguation). It did
NOT test text-first mode, where the leak turns out to help massively:
the text-first agent always requests its own frames, and when
`select_by_text` is broken (v9.1_fix's bug), the GT-visible seeds
become bonus context pointing at the correct answer.

This is why `v9_1_fix_FULL_REPRO_20260516` got 82.95 %: it was a
**double bug** (broken `select_by_text` + active seed-drain leak) and
the second bug compensated for the first while also leaking the answer.

## Why the cadence intuition was wrong

The 30-case audit found that v9.1_fix marked +36 % more frames per
sample than v9.3-no, and that magic-case wins involved a "first submit
wrong, second submit right after re-mark" trajectory. The natural
inference: forcing `select_by_text → ERR` triggers the playbook's
fallback chain, which triggers the per-candidate marking, which triggers
the re-mark + re-submit nature of evidence-frame-guard correction.

That story was wrong. v9.4-A confirmed it:

- `select_by_text` ERROR'd on every call (174/174, 100 %).
- The text-first playbook + system prompt were loaded.
- The text-first playbook's "fallback chain" prose was still there.
- → The agent only called `select_by_text` on 29 % of samples,
  not 100 % (compared to v9.1_fix audit's 100 %).
- → `mark_frame_with_bbox` rate was 2.47 / sample, not 2.90.
- → 0 / 6 magic cases recovered.

What v9.4-D revealed: the cadence is **downstream** of the leak, not
of the ERR signature. Once the agent has GT-visible anchors in
context, the playbook prose makes it mark each candidate; without
anchors, the playbook prose is too weak to force per-candidate marking
as a routine.

In other words: the deliberation cadence is something the agent does
*when it has reason to*. The leak provides that reason for free. The
agent doesn't even know the reason is "the answer is right here".

## What this means for our NR3D leaderboard claims

### `v9_1_fix_FULL_REPRO_20260516` is invalidated

The v9.1_fix FULL row joins **v3_referit3d_track_20260501** (80.79 %,
GT-target-visible projection-only visibility shortcut) and
**v5p1_failed_rerun_full_20260513** (68.48 %, projection-only visibility
index) in the "depth-aware NR3D row whose number depended on GT
information leaking into the model input" category.

The status was added at the top of `v9_1_fix_FULL_REPRO_20260516.md`
in commit `2b896d6`.

### The honest "best" depth-aware NR3D result is v9.3 text-first at ~66.7 %

Per-tier comparison vs UniVLG (current public SOTA):

| Tier | UniVLG | v9.3 text-first (strat600) | Δ |
|---|---:|---:|---:|
| Overall | 65.2 | 66.67 | **+1.47** |
| Easy | 73.3 | 73.45 | +0.15 |
| Hard | 57.0 | 60.32 | **+3.32** |
| V-Dep | 55.1 | 54.98 | −0.12 |
| V-Indep | 69.9 | 73.01 | **+3.11** |

The honest agent is competitive with public SOTA on Overall, ahead on
Hard and V-Indep, dead even on V-Dep — without information leakage.
This is the number to cite. A FULL 7805-case v9.3 text-first run would
nail the strat600 → FULL bias correction; current strat600 is
calibrated to ±0.19 pp of FULL on the v9.1_fix reference, with ±2.3 pp
90 %-band on a fresh agent's full-fold realization.

### Two diagnostic flags are now permanent infrastructure

Both flags are opt-in (default False), wired through
`Stage2DeepAgentConfig` → `Stage2RuntimeState`, documented in
docstrings + CLI help as **test-time-only**, and covered by 8 new unit
tests.

| Flag | Effect | Use case |
|---|---|---|
| `--force-stage1-text-retrieval-to-error` | `select_by_text` returns ERROR before touching `KeyframeSelector`, but the tool is still registered. text-first playbook + system prompt unchanged. | Ablate the cadence hypothesis without re-introducing the broken-Stage-1 bug. **Safe to ship as a research toggle.** |
| `--restore-stage1-seed-keyframe-drain` | `build_evidence_update_message` skips the `initial_keyframe_paths` filter; all 5 pack-prep seed keyframes get auto-injected. **GT-target-visible RGBs flow into agent context the agent didn't ask for.** | Diagnostic upper bound only. Quantifies how much of the v9.3 ↔ v9.1_fix gap is "seeing the answer" vs "actually reasoning". **NEVER produce a published number with this flag on.** |

The flags can be combined to cleanly recreate the v9.1_fix runtime
behaviour for ablation comparisons.

## Implications for future evaluations

### 1. `bundle.keyframes` is itself a latent risk

Even with the v9.4-D flag default-off, the 5 GT-target-visible seeds
are still on disk inside the pack. Any future code path that iterates
`bundle.keyframes` without consulting `initial_keyframe_paths` would
re-introduce the leak silently. Hardening options:

- (A) Stop pack-prep from writing seed keyframes when they aren't
  needed. This blocks the leak at source. Risk: legacy code paths
  asserting `bundle.keyframes` is non-empty need updates (see
  `memory/v9_catalog_first_branch.md` Open followups).
- (B) Add a runtime invariant check: if any image in
  `runtime.seen_image_paths` ever ends up matching an entry in
  `initial_keyframe_paths`, log a loud WARNING and raise in CI.
- (C) Pack-prep audit script: scan the pack and flag any seed-keyframe
  paths that include the GT target's pixels (proxy: per-frame visibility
  > 0). This re-asserts the boundary at every pack rebuild.

We should probably do all three. (A) is the root fix; (B) is the cheap
safety net; (C) catches regressions on the data side.

### 2. Per-benchmark fair-view audit checklist

The lesson from v3 → v5.1 → v9.1_fix is that "fair view" in this
codebase is a multi-component contract:

| Component | Leak risk |
|---|---|
| Visibility index (`use_depth=True`) | Used by Stage-1 + selectors; if `use_depth=False`, projection-only frames can claim to "see" the target even when occluded. v3 + v5.1 broke here. |
| Pack-prep `keyframes` field | Populated using GT bbox. Must NOT auto-inject. v9.1_fix broke here. |
| Pack-prep `bev_image_path` content | Currently only labels public categories; no per-instance GT bbox annotations. Safe. |
| Catalog generation | Uses `proposals.jsonl` produced by detection model (no GT). Safe. |
| `request_crops` | Crops are based on agent-chosen bboxes. Safe. |
| `select_by_text` Stage-1 parser | Reads scene categories only, no GT. Safe. |

For each new benchmark added, run a checklist: which fields in the
pack derive from GT? Which of those flow into the agent context? Is
there an explicit filter at the runtime boundary?

### 3. Where does v9.4 actually go from here?

The "honest" upper bound for the agent on this code is now v9.3 at
~66.7 %. Closing more of the 16 pp gap without information leakage
requires lifting genuine perception / reasoning. Three concrete
candidate experiments (ordered by cost):

1. **Per-query routing** (Experiment C from earlier docs) —
   view-dependent queries → `_no_text` playbook + catalog-first;
   rare-target queries → text-first + Stage-1. Costs ~1 day. Existing
   v9.2 FULL A/B suggests the gap is small (+0.55 pp on matched-fold),
   but per-tier routing may give +2–3 pp.
2. **Stronger playbook prose** (Experiment B) — explicit cadence +
   per-candidate marking instructions in the playbook prose. Costs
   ~2 h prose edit + 32 min run. v9.4-D tool histogram shows the agent
   CAN follow the per-candidate-marking pattern when anchored; question
   is whether prose alone is enough to trigger it without the leak.
3. **Richer turn-0 evidence panel** — currently the chassis seeds the
   agent with BEV + Cat-B text. Adding per-proposal thumbnails (the
   single best-visible non-GT-leaking RGB per proposal, computed via
   the visibility index) would give the agent grounded visual anchors
   for catalog-first reasoning without any GT info. Costs ~2–3 days.

Option 3 is the most aligned with "what the leak does, without the
leak": give the agent real visual context up front, but cap it at
fair-view boundaries.

## Files in this investigation (read in any order)

| Doc | Role |
|---|---|
| [v9_3_strat600_subset_design_20260517.md](v9_3_strat600_subset_design_20260517.md) | Step 1 — fold design + bootstrap |
| [v9_3_strat600_20260517.md](v9_3_strat600_20260517.md) | Step 2 — v9.3 text-first 66.67 % |
| [v9_3_strat600_notext_20260517.md](v9_3_strat600_notext_20260517.md) | Step 3 — v9.3 catalog-only A/B 64.17 % |
| [v9_1_fix_vs_v9_3_audit30_20260517.md](v9_1_fix_vs_v9_3_audit30_20260517.md) | Step 4 — 30-case audit, found the cadence hypothesis |
| [v9_4a_strat600_force_error_20260517.md](v9_4a_strat600_force_error_20260517.md) | Step 5 — Experiment A 63.17 %, **refuted** the cadence hypothesis |
| [v9_4d_strat600_force_error_seed_drain_20260517.md](v9_4d_strat600_force_error_seed_drain_20260517.md) | Step 6 — Experiment D 83.33 %, **isolated** the leak |
| **`v9_4_gt_leak_postmortem_20260518.md`** | **This document — unified narrative** |
| [v9_1_fix_FULL_REPRO_20260516.md](v9_1_fix_FULL_REPRO_20260516.md) | The invalidated v9.1_fix FULL row (deprecation header added in `2b896d6`) |

Per CLAUDE.md MANDATORY rule, all of these are kept immutable; this
post-mortem doc is the human-readable index and consolidator.

## Reproduction (the whole investigation in 5 commands)

```bash
cd /Users/bytedance/project/3DVLMReasoning
source .venv/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1

# A) v9.3 honest baseline (text-first, the row to cite)
python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_3_REPRO \
    --workers 40 --sample-retries 2 \
    --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard

# B) v9.3 honest catalog-only baseline (the A/B partner)
python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_3_notext_REPRO \
    --workers 40 --sample-retries 2 \
    --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard \
    --disable-stage1-text-retrieval

# C) Cadence-only ablation (negative control — should land ~63 %)
python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_4a_REPRO \
    --workers 40 --sample-retries 2 \
    --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard \
    --force-stage1-text-retrieval-to-error

# D) Leak-restore upper bound (positive control — should land ~83 %)
python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_4d_REPRO \
    --workers 40 --sample-retries 2 \
    --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard \
    --force-stage1-text-retrieval-to-error \
    --restore-stage1-seed-keyframe-drain

# Aggregate any of them
python -m evaluation.scripts.nr3d_leaderboard_metrics \
    --side-by-side tmp/nr3d_eval_<dir>/side_by_side.json \
    --nr3d-data-root data/nr3d --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --output tmp/nr3d_eval_<dir>/leaderboard_metrics.json \
    --canonical-filter true
```

Each run takes ~32 min wall on workers=40 on macOS. Expected Overall
metrics: A ≈ 66.7, B ≈ 64.2, C ≈ 63.2, D ≈ 83.3 (all ±2.3 pp 90 %-band).

## SQLite — current state of the 5 runs

```sql
SELECT run_id,
       n,
       ROUND(classification_acc_filtered * 100, 2) AS overall,
       ROUND(acc_easy * 100, 2)       AS easy,
       ROUND(acc_hard * 100, 2)       AS hard,
       ROUND(acc_view_dep * 100, 2)   AS vdep,
       ROUND(acc_view_indep * 100, 2) AS vindep
FROM runs
WHERE run_id IN (
  'v9_1_fix_FULL_REPRO_20260516',           -- invalidated; 8584 fold
  'v9_3_strat600_20260517',                 -- honest text-first baseline
  'v9_3_strat600_notext_20260517',          -- honest catalog-only baseline
  'v9_4a_strat600_force_error_20260517',    -- cadence ablation (negative control)
  'v9_4d_strat600_force_error_seed_drain_20260517' -- leak-restore (positive control)
)
ORDER BY classification_acc_filtered DESC;
```

| run_id | n | overall | easy | hard | vdep | vindep |
|---|---:|---:|---:|---:|---:|---:|
| **v9_4d_strat600_force_error_seed_drain_20260517** (leak ON) | 600 | **83.33** | 84.48 | 82.26 | 76.78 | 86.89 |
| v9_1_fix_FULL_REPRO_20260516 (leak ON, **invalidated**) | 8 584 | 82.95 | 88.36 | 77.88 | 78.23 | 85.51 |
| **v9_3_strat600_20260517** (honest, cite this) | 600 | **66.67** | 73.45 | 60.32 | 54.98 | 73.01 |
| v9_3_strat600_notext_20260517 (honest catalog-only) | 600 | 64.17 | 73.10 | 55.81 | 56.87 | 68.12 |
| v9_4a_strat600_force_error_20260517 (cadence-only) | 600 | 63.17 | 71.03 | 55.81 | 55.45 | 67.35 |

## Acknowledgements

The leak fix in `8ebf701` was authored before this investigation began;
the credit for plugging the bug belongs there. This investigation
re-discovered the leak from its consequences and quantified it.
