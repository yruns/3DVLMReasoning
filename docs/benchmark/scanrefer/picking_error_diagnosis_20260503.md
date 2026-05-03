# ScanRefer v3.x Picking-Error Diagnosis (2026-05-03 22:00)

> **Status**: open investigation. v3.1 (commit `a6b8606` + working-tree X1
> patch) is stuck at Acc@0.25 = 39.0% / Acc@0.50 = 15.0% on the random100
> fold, ~28-45pp behind the v2 GT-view-oracle upper bound (67.0 / 59.0)
> on the **same fold**. This document enumerates what we know, how to
> reproduce, and the live hypotheses for closing the gap.

---

## 1. The problem in one paragraph

The ScanRefer Camp-A pipeline (Mask3D pool, gpt-5.4 Stage 2 agent,
3D IoU evaluation) shows a **28-45pp accuracy gap** between two
keyframe-selection strategies that share the same proposals, the
same agent prompt, the same evaluator, and the same 100-utt random
val fold:

- **v2 (controlled upper bound):** initial KFs are the top-5 frames
  where the GT `target_id` is most visible (Phase 8 visibility).
  → Acc@0.25 = 67.0% / Acc@0.50 = 59.0% on random100.
- **v3 / v3.1 (zero-shot):** initial KFs are picked from the query
  alone — v3 via Phase 8 hypothesis-parser, v3.1 via Mask3D-CG
  candidate visibility for parser-extracted categories.
  → Acc@0.25 = 39.0% / Acc@0.50 = 15.0% on random100 (both versions).

X1 (v3 → v3.1) eliminated the visibility-source mismatch — pack-prep
fallback rate dropped from 38% to **0%** — but downstream metrics
moved by less than the noise floor (+0 / +1pp). So **the gap is not
caused by initial-keyframe coverage**, even though every prior diagnosis
pointed there.

The fundamental open question:

> **What does v2's GT-view-oracle KF give the agent that v3.1's
> Mask3D-aligned same-category KF does not?**

---

## 2. Quantitative landscape

### 2.1 Headline (random100, frozen seed=20260503)

| Variant | KF source | Pack fallback | Acc@0.25 | Acc@0.50 | mean IoU |
|---|---|---:|---:|---:|---:|
| v2 GT-target visibility (oracle) | Phase 8 visibility of GT `target_id` | n/a | **67.0** | **59.0** | **0.566** |
| v3 query-driven (Phase 8) | `select_keyframes_v2(query)` Phase 8 | 38% | 39.0 | 14.0 | 0.193 |
| v3.1 query-driven (Mask3D, X1) | Mask3D-CG candidate visibility | **0%** | 39.0 | 15.0 | 0.198 |

### 2.2 Per-IoU-bucket distribution on v3.1 (n=100)

| IoU bucket | Count | % |
|---|---:|---:|
| ≥ 0.90 | 0 | 0 |
| 0.70 – 0.90 | 2 | 2 |
| 0.50 – 0.70 | 13 | 13 |
| 0.25 – 0.50 | 24 | 24 |
| 0.10 – 0.25 | 8 | 8 |
| 0.001 – 0.10 | 11 | 11 |
| **0.0** | **40** | **40** |

40% of samples land at IoU = 0 — the agent submitted a proposal that
does not overlap the GT bbox at all (wrong proposal entirely). Only
2% reach high-quality (≥ 0.70) IoU, even though v2 oracle hits 59%
at IoU ≥ 0.50.

### 2.3 Agent termination pattern (v3.1)

| Stat | Value |
|---|---:|
| samples terminated at turn 1 via chassis `submit_final` | **97 / 97 non-error** |
| samples terminated at turn ≥ 2 | **0** |
| samples errored (mesh missing) | 3 |
| tool calls per sample (range) | [10, 39] |
| tool calls per sample (median ~) | 16 |
| Stage 1 callback firings (total across 100 samples) | 64 |
| ↳ `request_more_views` | 36 |
| ↳ `request_crops` | 25 |
| ↳ `switch_or_expand_hypothesis` | 3 |
| Stage 1 callback path-resolve warnings | **107** |

**Key observations:**

1. **No agent ever runs more than one ReAct turn.** It plans, fires
   10-39 tools internally, calls `submit_final`, done. The project's
   claimed innovation — multi-turn Stage 2 ↔ Stage 1 iteration — is
   **not behaviourally happening on ScanRefer** (vs. OpenEQA where
   it is intended to fire).
2. **Most "tools" are intra-pack (VG pack + chassis), not Stage 1
   callbacks.** Average ~16 tools per sample but only ~0.64 Stage 1
   callbacks per sample. The agent prefers `view_keyframe_marked`
   / `inspect_proposal` / `find_proposals_by_category` over
   `request_more_views` / `request_crops` / `switch_or_expand_hypothesis`.
3. **Many callbacks return unusable frames.** 107 path-resolve
   warnings: `[Stage1Callback] Could not resolve path for view_id=N`.
   This means the callback's `KeyframeSelector` (Phase 8) returns
   view_ids that don't map to any raw RGB file the agent can load.
   When fired, callbacks are partially silent.
4. **`switch_or_expand_hypothesis` is a dead tool.** 3 calls in 100
   samples (3% utilization) — even though it's the only tool that
   can re-parse the description into new categories.

### 2.4 Sample-level patterns

Picked at random across the IoU spectrum:

| sample_id | iou | conf | query (truncated) |
|---|---:|---:|---|
| `scene0030_00::5::4` | **0.00** | 0.58 | "there is a brown chair near the center of the room at a brown table. its left side faces the window" |
| `scene0095_00::25::1` | **0.00** | 0.72 | "this is a white keyboard. it is near the corner at the end of the table." |
| `scene0164_00::14::3` | **0.00** | 0.82 | "this is a wooden kitchen cabinet. it is above the water bottle." |
| `scene0187_00::6::4` | **0.00** | 0.76 | "this is a white table. it is to the left of a coffee table." |
| `scene0406_00::6::4` | 0.730 | 0.91 | "there is a white porcelean sink in the bathroom. it is next to some personal care products." |
| `scene0139_00::13::0` | 0.719 | 0.91 | "a shelf, at the bottom of which are two washing machines. on the left is a paper shelf..." |
| `scene0025_00::7::0` | 0.668 | 0.71 | "there is a door in the middle of the northern wall next to a metal cabinet..." |

**Pattern:** zero-IoU samples have descriptions that resolve a
**multi-instance category** by spatial relation to other objects
("near the center", "above the water bottle", "to the left of a
coffee table"). High-IoU samples either describe a near-unique
instance ("the white porcelean sink in the bathroom") or carry rich
co-occurring context ("two washing machines" + "paper shelf" — a
multi-object anchor).

The picture: when the description requires **discriminating among
same-category candidates**, the agent fails. When the description
implicitly identifies a unique instance, it succeeds.

---

## 3. Reproduction (deterministic)

### 3.1 Environment

```bash
cd /Users/bytedance/project/3DVLMReasoning
source .venv/bin/activate          # macOS: uv-managed .venv
python -c "import open_clip; print(open_clip.__version__)"  # 3.3.0
git rev-parse --short HEAD          # a6b8606
git status --short                  # working tree should have X1 patch
```

If the X1 patch is missing, the `mask3d_query_driven` mode below
will be rejected by `argparse`. The patch lives in the working tree
of the same branch and adds a `mask3d_query_driven` choice to
`--keyframe-mode` of `prepare_pack_v1_inputs_scanrefer.py`.

### 3.2 Frozen 100-utt fold

```bash
# Idempotent — re-running produces the same file (seed=20260503).
python scripts/build_scanrefer_random100_fold.py
ls tmp/scanrefer_artifacts/random100_sample_ids.json   # 100 entries × 66 scenes
```

### 3.3 Pack-prep (Mask3D-driven KFs)

```bash
PYTHONPATH=src python -m evaluation.scripts.prepare_pack_v1_inputs_scanrefer \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --pack-name pack_scanrefer_v3p_iterative \
  --keyframe-mode mask3d_query_driven 2>&1 | tee /tmp/v3p_logs/random100_pack.log
```

Expected end-of-log line:

```
mask3d_query_driven mode: 0/100 samples used Mask3D-density fallback (0.00%)
wrote 100 sample artifacts under data/scanrefer/scannet/<scene>/pack_scanrefer_v3p_iterative/
```

Wall time: ~55 min on local Mac with the gemini-2.5-pro pool. Per-sample
artifacts have shape `{sample_id, scene_id, target_id, ann_id,
gt_bbox_3d_9dof, keyframe_mode, keyframes: [{keyframe_idx, image_path,
frame_id} × 3]}`.

### 3.4 Agent run (8 workers, all 3 callbacks wired)

```bash
PYTHONPATH=src python -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p_random100_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 8 --sample-retries 1 2>&1 | tee /tmp/v3p_logs/random100_agent.log
```

Wall time: ~14 min (8 concurrent agents on local model-hub endpoint).
Output:
- `tmp/v3p_random100_eval/side_by_side.json` — per-backend metrics + per_sample list
- `tmp/v3p_random100_eval/per_sample/pack_scanrefer_v3p_iterative/*.json` — checkpoints
- `/tmp/v3p_logs/random100_agent.log` — loguru console log

### 3.5 Headline extraction

```bash
PYTHONPATH=src python -c "
import json
r = json.load(open('tmp/v3p_random100_eval/side_by_side.json'))['pack_v1']
print(f\"n={r['n']} mean={r['mean_iou']:.3f} \"
      f\"Acc@0.25={r['Acc@0.25']*100:.1f} Acc@0.50={r['Acc@0.50']*100:.1f}\")"
# expected: n=100 mean=0.198 Acc@0.25=39.0 Acc@0.50=15.0
```

### 3.6 SQLite ingest (already done for `v3p_smoke_random100_20260503`)

```bash
python scripts/ingest_scanrefer_run.py \
    --output-dir tmp/v3p_random100_eval \
    --run-id v3p_smoke_random100_20260503 \
    --branch feat/scanrefer-v3-query-driven \
    --commit a6b8606 \
    --backend pack_v1 \
    --keyframe-mode mask3d_query_driven \
    --notes "X1 smoke ..." \
    --db docs/benchmark/scanrefer/runs.sqlite
```

Run-row fields are NULL'd by default unless `--leaderboard-metrics` is
passed; `samples` table is correctly populated. To repair the run-level
columns from the per-sample data:

```sql
UPDATE runs
SET n_total = (SELECT COUNT(*) FROM samples WHERE run_id = runs.run_id),
    acc25_overall = (SELECT AVG(acc25) FROM samples WHERE run_id = runs.run_id),
    acc50_overall = (SELECT AVG(acc50) FROM samples WHERE run_id = runs.run_id),
    mean_iou_overall = (SELECT AVG(iou) FROM samples WHERE run_id = runs.run_id)
WHERE run_id = 'v3p_smoke_random100_20260503';
```

### 3.7 Reproduction sanity check

| Check | Expected |
|---|---|
| Pack fallback rate (v3.1) | **0%** |
| Agent terminations at turn 1 via chassis | **97 / 97 non-error** |
| Stage 1 callback firings | 60-70 across 100 samples |
| Path-resolve warnings | ≥ 100 |
| Acc@0.25 | **39 ± 2 pp** (LLM stochasticity) |
| Acc@0.50 | **15 ± 2 pp** |
| n errors | ≤ 3 (mesh-missing scenes; infra noise) |

Per-LLM-call durability is incomplete — the `tool_calls` and `llm_calls`
SQLite tables remain empty. To reproduce tool-call statistics today,
parse the loguru console log:

```bash
grep -oE "tool_calls=[0-9]+" /tmp/v3p_logs/random100_agent.log | sort | uniq -c
grep "Stage1Callback" /tmp/v3p_logs/random100_agent.log | awk '{print $7}' | sort | uniq -c
```

---

## 4. Hypothesis tree

### Hypothesis A — Multi-distractor picking error (PRIMARY SUSPECT)

**Claim**: When 3-7 same-category Mask3D candidates appear across the
initial KFs, the VLM cannot reliably discriminate which one matches
the natural-language spatial constraints in the description.

**Mechanism**: ScanRefer descriptions for `is_unique=False` cases are
of the form "the X near/above/next to Y". The agent must:
1. Identify all visible X candidates (Mask3D marks in annotated PNG).
2. Identify Y in the same frame (or nearby frame).
3. Compute geometric relation between each X and Y.
4. Select the unique X for which the relation holds.

Step 4 is the hard one. With Mask3D `proposal_id` overlay and a
single text prompt asking the VLM to "pick the one closest to Y", the
model frequently picks **a** chair without verifying the spatial
constraint.

**Evidence FOR**:
- Zero-IoU samples *predominantly* contain descriptions with relational
  modifiers ("near the center", "above the water bottle", "to the left
  of the coffee table" — see §2.4).
- v2 oracle's 67/59 lift is structurally explainable: the GT-target
  visibility-best frames present the target up-close and unambiguous,
  bypassing step 4 entirely.
- 97% of agent runs submit on turn 1 with median 16 tool calls — the
  agent looks at proposals but doesn't loop back through Stage 1 to
  refine its understanding when picking is hard.

**Evidence AGAINST**:
- Some zero-IoU descriptions are not multi-distractor ("this is a
  white keyboard" → if there's only one keyboard, why fail?). Need
  scene-level Mask3D candidate counts per category to confirm.
- v2's Multiple-class number is 65/57 — even the oracle drops 18-19pp
  vs Unique-class. So multi-distractor IS hard for v2 too. But v3.x
  is much worse than v2's 65/57 floor.

**Falsification test**: split random100 by `is_unique` and recompute
v3.1 metrics. If v3.1 ≈ v2 on Unique and v3.1 << v2 on Multiple,
hypothesis A is confirmed.

**Predicted fix family**:
- **X2** — playbook hardening: force `find_proposals_by_category +
  inspect_proposal[K].frames_appeared + view_keyframe_marked(M)` on
  ALL viable K before submit. ~30 lines of playbook + worked example.
  Cheap, attacks step 4 directly.
- **A2** — auxiliary spatial-relation tool: add a `compare_proposals_spatial`
  tool that programmatically computes the geometric predicate
  ("nearest to anchor X") in 3D from proposal centroids. The VG pack
  already has one (#13 in the tool list); make the playbook **mandate**
  its use whenever a relational modifier is parsed.
- **A3** — re-prompt with a same-category shortlist: after the agent
  identifies the candidate set, re-issue a single VLM call with all
  same-category crops as a 1-of-K choice question. Different VLM call
  topology, larger surgery.

### Hypothesis B — Agent fast-submits at turn 1 (BEHAVIOURAL ROOT CAUSE)

**Claim**: The Stage 2 agent calls `submit_final` after one ReAct turn
on virtually every sample (97/97 non-error). The "iterative ReAct
loop" innovation that's the project's main differentiator vs. ZSVG3D /
SeeGround does not actually fire on ScanRefer.

**Mechanism**: With `max_turns=6`, the agent has the budget to try a
hypothesis, observe outcomes, and re-plan. Empirically it doesn't.
Either:
1. The system prompt / VG playbook gives the agent strong "submit
   when confident" pressure.
2. The agent *is* confident (median confidence on zero-IoU is 0.7+)
   and the playbook doesn't tell it to verify before submitting.
3. The chassis `submit_final` tool is too easy to call; there's no
   gating logic.

**Evidence FOR**:
- 97/97 turn-1 submissions; 0/100 turn-2 escalations.
- Mean confidence on zero-IoU samples ~0.72 (agent thinks it's right).
- 64 Stage 1 callback firings across 100 samples, mostly within turn 1.

**Evidence AGAINST**:
- v2 also runs the same agent and presumably also fast-submits, yet
  it scores 67/59. So fast-submit alone isn't the gap — it's
  fast-submit + bad initial KFs.

**Falsification test**: increase `max_turns` to 1, force the agent to
submit immediately. If metrics drop sharply, multi-turn is helping
*somewhat*; if metrics are unchanged, multi-turn is irrelevant on
this benchmark.

**Predicted fix family**:
- **X2 with explicit submit gates**: playbook adds "before
  `submit_final`, if more than one same-category proposal exists in
  any KF, you must inspect at least 2 candidates and articulate why
  you rejected each of the others". 
- **B2** — agent runtime gate: wrap `submit_final` to refuse
  submissions when there exists `find_proposals_by_category(target)`
  > 1 and the agent has not called `inspect_proposal` on at least 2
  IDs in the result. Code-side enforcement, ~50 lines in the chassis.

### Hypothesis C — Stage 1 callbacks return useless frames

**Claim**: When the agent does invoke `request_more_views` or
`switch_or_expand_hypothesis`, the returned frames are often
unloadable (the runner's KeyframeSelector uses Phase 8 with
`stride=10` while the saved visibility index was built with
`stride=1`, producing wrong view-id → frame-id translations).

**Mechanism**: The KeyframeSelector logs
`Stride mismatch: saved=1, current=10. View indices may be
incorrect.` and then issues 107 `Could not resolve path for view_id=N`
warnings during the run. If a callback returns 5 view_ids and 3 of
them resolve to no path, the agent gets degraded evidence and may
silently fall back to its prior belief.

**Evidence FOR**:
- 107 path-resolve warnings across 64 callback firings ≈ 1.7
  unresolvable frames per callback.
- `Stride mismatch` warning fires once per scene KeyframeSelector
  build (~30+ scenes).

**Evidence AGAINST**:
- Runner returns successfully — agents got *some* frames per
  callback. Hard to quantify how often the returned set was empty.
- v2 uses no callbacks (all initial KFs picked deterministically) and
  this hypothesis is irrelevant to v2 — but v2 is the one that scores
  high. So this hypothesis is at most a *secondary* gap, not the
  primary 28-pp driver.

**Falsification test**: instrument `Stage1Callback._resolve_views_to_keyframes`
to record `n_returned` and `n_failed` per call into the per-sample JSON;
then check whether samples whose callbacks returned 0 usable frames
correlate with zero-IoU outcomes.

**Predicted fix family**:
- **C1** — fix stride-mismatch by rebuilding visibility indices at
  current stride, OR by translating saved-stride view-ids to
  current-stride view-ids in the KeyframeSelector loader. Pure
  infrastructure, ~50 lines.
- **C2** — extend X1 into the runner's callback KeyframeSelector:
  use Mask3D-CG visibility there too, mirroring pack-prep. Same
  amount of work as X1, low priority given X1's no-op result.

### Hypothesis D — Annotated-PNG mark interpretability

**Claim**: The numeric overlay (proposal_id label rendered onto the
RGB image) is not visually salient enough for the VLM to reliably
distinguish 3-7 closely-spaced marks (especially when they overlap
geometrically — e.g., 5 chairs at one table).

**Mechanism**: Marks are drawn as small numeric text labels next to
bbox corners. When proposals overlap or cluster (ScanNet aggregation
often produces multiple proposals per chair-row), the overlay becomes
visually dense and the VLM may misread the number.

**Evidence FOR**:
- Indirect: zero-IoU samples often have several same-category
  proposals in the same frame; if marks were perfectly readable, the
  agent would at least read the right number.
- Cross-cite: SeeGround's main contribution is generating
  **synthetic per-candidate views** to remove this clutter.

**Evidence AGAINST**:
- Hard to falsify without VLM reasoning trace.
- `inspect_proposal` exists and presents per-candidate crops, so the
  agent can disambiguate without reading the mark — but it doesn't
  always use it (median 16 tool calls but `inspect_proposal` may be
  only 2-3 of those).

**Falsification test**: extract the agent's intermediate VLM responses
(via instrumented runtime) and check whether errors stem from
"misread mark" vs. "couldn't apply spatial relation". Requires per-LLM-call
durability that we don't have today.

**Predicted fix family**:
- **D1** — bigger / higher-contrast marks. UI-level change.
- **D2** — render per-candidate "single-mark" crops in addition to
  the multi-mark frame. Heavier infra; partially overlaps SeeGround's
  approach.

### Hypothesis E — Mask3D recall ceiling

**Claim**: Some random100 utterances correspond to GT bboxes for
which Mask3D simply did not produce any candidate with IoU ≥ 0.25 to
the GT. Those samples can never be solved and contribute ~10-15% of
the failure rate.

**Mechanism**: Mask3D is a 3D instance segmenter; its recall on
ScanNet200 categories isn't 100%. ZSVG3D / SeeGround report Mask3D
upper-bound at Acc@0.25 ≈ 84% (per handoff §2.2).

**Evidence FOR**:
- A theoretical 16% of random100 (~16 samples) cannot be solved by
  any picker.

**Evidence AGAINST**:
- v2 oracle on the same fold is 67% — it's nowhere near 84% either,
  so the ceiling isn't binding even for the oracle.
- v3.1's 39% is way below the 84% ceiling, so this hypothesis cannot
  account for more than a fraction of the gap.

**Falsification test**: run `oracle_picker` (pick the Mask3D candidate
with maximum 3D IoU vs GT) on random100. If oracle hits 84%, recall
is 84% on this fold; the remaining 39 → 84 gap is picking error.

**Predicted fix family**: not actionable for the agent; would require
swapping detector.

### Hypothesis F — Stochastic Stage 2 LLM (NOISE, NOT GAP)

**Claim**: gpt-5.4 generations are non-deterministic. v3 vs v3.1
showing ±1pp Acc@0.50 (14 → 15) is within run-to-run variance.

**Status**: TRUE for v3 vs v3.1 noise, but irrelevant to the v3.x vs
v2 28-pp gap which is far beyond LLM stochasticity.

**Falsification test**: run v3.1 a second time on the same pack. If
both runs land within ±2pp of each other, variance bound established.

### Hypothesis G — Random100 fold is unrepresentative

**Claim**: The 100-utt fold is over-sampled from `is_unique=False`
descriptions, biasing v3.x downward vs the full 9508 val.

**Status**: PARTIALLY TRUE — v2 on random100 is 67/59 vs full val
69.92 / 62.79. So the fold runs ~3pp lower across the board. The 28-pp
v3 vs v2 gap on random100 should reproduce within ±3pp on full val.

**Falsification test**: after we settle on a v3.x candidate, run it
on the full 9508 once. Don't pre-emptively burn the budget.

### Hypothesis H — Stage 2 prompt drift over the day

**Claim**: ModelHub's gpt-5.4 endpoint silently shifted between v2's
morning run (~12:23) and v3.1's evening run (~21:43).

**Status**: LOW PRIOR. Anthropic / internal model-hub doesn't usually
hot-swap. But possible. Hard to falsify without v2 re-runs.

**Predicted fix**: re-run v2 once tomorrow on random100 to confirm
67/59 reproduces. If it doesn't, this hypothesis becomes leading.

---

## 5. Recommended next-step ordering

Cheap → expensive, attacking the most-likely root cause first:

1. **X2 — playbook hardening (estimated +5-15pp Acc@0.25)**.
   Hypothesis A1 + B1 fix. ~30 lines of `vg_grounding_playbook.md`.
   Mandate `find_proposals_by_category` → `inspect_proposal[K]` →
   `view_keyframe_marked(M)` for each viable K before submitting,
   when target category has > 1 proposal in any KF. Worked anti-example.
2. **`is_unique` split on existing v3.1 data (zero-cost diagnostic)**.
   Falsifies / confirms hypothesis A1: if Unique@0.25 ≥ 65 and Multiple@0.25
   ~30, A1 is confirmed.
3. **Oracle-picker 3D-IoU (zero-LLM-cost diagnostic)**. Falsifies / confirms
   hypothesis E: gives the actual Mask3D recall ceiling on random100.
4. **Re-run v2 on random100 (smoke; ~14 min)**. Falsifies hypothesis H
   (LLM drift). Should reproduce 67/59.
5. **Increase max_turns and observe** (zero-code, just rerun with config
   override). Falsifies hypothesis B (multi-turn is happening or not).
6. **C1: fix stride-mismatch in saved visibility index (~50 LOC)**. Removes
   the 107 path-resolve warnings; reveals whether callbacks were partially
   silent.
7. **A2: mandatory `compare_proposals_spatial` for relational descriptions**.
   ~80 LOC playbook + python.
8. **D2 / per-candidate "single-mark" crop view** (heavier infra).
9. **X3 — accept v3.1 as the honest baseline and write paper**. Story:
   "Mask3D-aligned, OpenEQA-style iterative ReAct, 39% Acc@0.25 ≈ ZSVG3D's
   36.4%; we trade SOTA for zero-shot honesty and report the v2 oracle
   as a controlled upper bound."

If after X2 + A2 the headline doesn't cross 50% Acc@0.25 on random100,
default to X3.

---

## 6. Files / artifacts referenced

- This doc: `docs/benchmark/scanrefer/picking_error_diagnosis_20260503.md`
- v3.1 version doc: `docs/benchmark/scanrefer/v3p1_mask3d_query_driven_20260503.md`
- v2 reference: `docs/benchmark/scanrefer/v2_aggregation_gt_track_20260503.md`
- README + leaderboard: `docs/benchmark/scanrefer/{README,leaderboard}.md`
- SQLite: `docs/benchmark/scanrefer/runs.sqlite` (run_id `v3p_smoke_random100_20260503`)
- Code (X1 patch): `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`
- Tests: `src/evaluation/scripts/tests/test_prepare_pack_v1_inputs_scanrefer.py`
- Pack-prep artifacts: `data/scanrefer/scannet/<scene>/pack_scanrefer_v3p_iterative/`
- Eval artifacts: `tmp/v3p_random100_eval/` (local-only, not git-tracked)
- Console logs: `/tmp/v3p_logs/random100_pack.log`,
  `/tmp/v3p_logs/random100_agent.log` (local-only)
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json` —
  rebuild via `python scripts/build_scanrefer_random100_fold.py`
- Originating handoff: `docs/handoff_2026-05-03_2020.md`
- Playbook (current): `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`
- Stage 1 callbacks: `src/agents/stage1_callbacks.py`
- Runner: `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`

---

*Authoritative until X2 lands or a hypothesis is falsified.*
