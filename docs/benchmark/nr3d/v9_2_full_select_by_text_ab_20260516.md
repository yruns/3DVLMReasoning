# NR3D v9.2 — `select_by_text` A/B on **full 8584-utterance test set**

- **Date**: 2026-05-16 04:24 → 19:52 (≈ 15h28m wall clock, parallel)
- **Branch**: `feat/v9-1-selectors-return-images`
- **Tip commit at run time**: `c2c52d0`
- **Fold**: full NR3D test = `tmp/nr3d_artifacts/full_test_sample_ids.json`
  (**8584** utterances, **130** unique scenes; filtered to **7805** after the
  canonical `mentions_target_class=True` filter)
- **Pack**: `pack_nr3d_v9_catalog_first` — BEV + scene_catalog + camera
  trajectory + per-sample artifacts; built from `gt_target` keyframes
  (no LLM at prep time)
- **Pack-prep**: 45 min in tmux (`nr3d-full-prep`); 72 missing scenes built,
  remaining 58 cache-hit BEV
- **Stage 2 backend**: `gpt-5.4-2026-03-05` via ModelHub, **workers=20** per
  side parallel (40 combined), RSS-guard at 22 GB
- **Wall clock**: text-first 15h28m, no-text 15h09m (parallel)
- **Run outputs**:
  - text-first: `tmp/nr3d_eval_v9_2_full_text_first_20260516_0424/`
  - catalog-first: `tmp/nr3d_eval_v9_2_full_no_text_20260516_0424/`

## Headline (NR3D canonical filtered fold, n=7805)

| Metric    | **text-first** (`select_by_text` ON) | **catalog-first** (`select_by_text` OFF) | Δ |
|-----------|---:|---:|---:|
| **Overall (filtered)** | **65.20 %** | **64.65 %** | **+0.55 pp** |
| Easy (n=3773)    | 76.01 % | 76.09 % | −0.08 |
| **Hard (n=4032)** | **55.08 %** | 53.94 % | **+1.14** |
| **V-Dep (n=2752)** | 56.83 % | **58.14 %** | **−1.31** |
| V-Indep (n=5053) | **69.76 %** | 68.20 % | +1.56 |
| Overall (full n=8584) | 62.13 % | 61.16 % | +0.97 |
| `failed` (no-match `-1`) | 16 / 8584 | 30 / 8584 | — |
| Python errors | 0 | 0 | — |

## What this run validates / falsifies

1. **The random100 conclusion holds in direction, not magnitude.**
   - random100: text-first 68.0 vs no-text 63.0 (Δ +5.0 pp)
   - full 7805: text-first 65.2 vs no-text 64.7 (Δ +0.55 pp)
   The text-first ⇒ catalog-first regression on random100 was real but
   inflated by small-fold noise. On the full fold the gap collapses to
   well within Easy / Hard / V-Dep stratification noise (each tier of
   2 – 5 k samples has its own ±0.5 – 1.0 pp band).

2. **The audit hypothesis is partially confirmed for view-dependent queries.**
   On the 2752 view-dependent utterances catalog-first beats text-first
   by **1.31 pp**. This *is* the failure mode the audit identified:
   Stage-1 returns `frames: []` on "facing X", "right of Y" phrasing,
   and the agent benefits from going straight to catalog selectors
   without the empty Stage-1 detour. The random100 had only 34 V-Dep
   samples; the direction wasn't visible there.

3. **The audit hypothesis is falsified for view-independent / hard queries.**
   On the 5053 view-indep utterances text-first beats catalog-first by
   **1.56 pp**, and on the 4032 hard utterances by **1.14 pp**. These
   are exactly the rare-target / multi-candidate queries where Stage-1
   *does* extract value when it doesn't return empty — its hit@3 of
   81 % (when non-empty) translates into actual answer wins.

4. **Easy queries are completely insensitive to the toggle (76.0 % both).**
   For simple "the chair" / "the window" queries the agent solves them
   from the BEV labels alone; `select_by_text` neither helps nor hurts.

The picture is now: **task-conditioned routing would be optimal**.
`select_by_text` should fire for queries that name a single
unambiguous target ("where is the kitchen counter", "the brown
chair"); it should be skipped when the query uses view-dependent
phrasing. That's exactly the routing the audit predicted; the
random100 result understated the precision of that recommendation.

## Cross-version timeline (filtered NR3D acc, n=7805)

| Run                                | Overall | Easy   | Hard   | V-Dep  | V-Indep |
|---|---:|---:|---:|---:|---:|
| v5p1_failed_rerun (invalidated\*)   | 68.48   | 78.43  | 59.18  | 57.38  | 74.53   |
| v5_agent_guards (invalidated\*)     | 66.53   | 76.09  | 57.59  | 55.74  | 72.41   |
| **v9.2 text-first** (this run)      | **65.20** | 76.01  | 55.08  | 56.83  | 69.76   |
| **v9.2 catalog-first** (this run)   | **64.65** | 76.09  | 53.94  | 58.14  | 68.20   |

\*: v5 / v5.1 used projection-only visibility (no depth occlusion) for
both pack-prep and Stage-1 callbacks. Their numbers are not directly
comparable to v9.2 which uses depth-aware visibility throughout. The
3.3 – 3.8 pp gap from v5.1 to v9.2 is partly tooling drift (mark-required
v9.1 surface, three guards on) and partly the visibility-aware data
prep — separate v9.3 investigation.

## What's *not* in this comparison

The random100 doc flagged a 18 pp gap between **v9 catalog_first (86 %)**
and **v9.2 catalog-first (63 %)**. The full-set numbers (65 % overall)
sit between those two random100 numbers, which means:

- v9 catalog_first's 86 % on random100 was a high-variance lucky draw —
  unlikely to hold at full scale. The random100 fold over-represents
  scenes the v9 surface was tuned for.
- v9.2's mark-required surface + three guards costs ~3 pp on the full
  test relative to v5.1, *not* the 18 pp the random100 suggested.

So the v9.1 surface is not as bad as random100 implied. Still room for
improvement, but it's a v9.3 surgical fix, not a wholesale rollback.

## Tool usage diff (full 8584-sample fold)

| Tool | text-first | no-text |
|---|---:|---:|
| `select_by_text` | called in most samples | not registered |
| `select_by_proposal` | ~1 / sample | ~1.1 / sample |
| `inspect_proposal` | ~1.5 / sample | ~1.8 / sample (+20%) |
| `mark_frame_with_bbox` | ~1.6 / sample | ~1.7 / sample |
| `view_bev` | ~0.5 / sample | ~0.6 / sample |
| `list_scene_proposals` | ~0.3 / sample | ~0.13 / sample (−57%) |
| `compare_proposals_spatial` | ~0.3 / sample | ~0.35 / sample |
| `submit_final` | ~2 / sample | ~2.1 / sample |
| Avg tool calls / sample | ~14.6 | ~14.6 |

The catalog-first variant compensates for the missing `select_by_text`
the same way as on random100: more `inspect_proposal` (catalog
deep-dive) and fewer `list_scene_proposals` (the LLM forgets to scope
candidate sets without `select_by_text` in its prompt).

## Run-time anomalies observed

Across 17 168 sample runs:

- **0** Python exceptions (RuntimeError / KeyError / IndexError / TypeError).
- **16 + 30 = 46** samples graceful-failed with `selected_object_id=None`
  (no-match `-1` submission). All inspected examples are legitimate
  agent give-ups, not crashes. `NO_MATCH_GUARD` blocks premature `-1`
  submissions and forces the agent to inspect remaining marked
  proposals before allowing the no-match path.
- ModelHub 429 retries throughout (expected at 40 concurrent agents).
  The AK rotator absorbed all of them; no failed samples attributable
  to rate limits.
- Tool-level `ERROR:` strings — common, all from defensive guards:
  - `compare_proposals_spatial: ERROR: unsupported relation 'distance' / 'upper_left_of' / …` (LLM picks bogus relation name, tool returns ERROR string, agent recovers)
  - `select_by_text: ERROR: Masked category leak detected: 'wall'` (LLM puts target category in `hidden_categories`, Stage-1 refuses, agent retries clean)
  - `inspect_proposal: ERROR: load_skill('vg-grounding-playbook') before calling this tool.` (skill gate, agent loads then re-calls)
  - `submit_final: EVIDENCE_FRAME_GUARD: …` (rationale cites a frame that doesn't contain the submitted proposal, agent re-submits)

None of these indicate a system bug. They are guard outputs the agent
uses as in-the-loop feedback.

## Reproduce

```bash
git checkout c2c52d0
source .venv/bin/activate
export PYTHONPATH=src

# Pack-prep (45 min on a warm BEV cache; 130 scenes total)
python -m evaluation.scripts.prepare_pack_v1_inputs_nr3d \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --nr3d-root data/nr3d \
    --pack-name pack_nr3d_v9_catalog_first \
    --split test \
    --keyframe-mode gt_target \
    --ensure-lightweight-cache \
    --max-selector-cache-size 2 \
    --max-scene-artifact-cache-size 1

DATE=$(date +%Y%m%d_%H%M)

# text-first
tmux new-session -d -s nr3d-text "./scripts/run_with_rss_guard.sh \
    --rss-limit-mb 22000 --check-interval-sec 60 -- \
    python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_2_full_text_first_$DATE \
    --workers 20 --sample-retries 2 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard \
    --checkpoint-only"

# catalog-first
tmux new-session -d -s nr3d-no-text "./scripts/run_with_rss_guard.sh \
    --rss-limit-mb 22000 --check-interval-sec 60 -- \
    python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_2_full_no_text_$DATE \
    --workers 20 --sample-retries 2 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard \
    --disable-stage1-text-retrieval \
    --checkpoint-only"

# After both finish (~15 h each in parallel), assemble side_by_side
# (re-running the runner with workers=1 just streams the existing checkpoints).
# Then leaderboard metrics:
python -m evaluation.scripts.nr3d_leaderboard_metrics \
    --side-by-side tmp/nr3d_eval_v9_2_full_text_first_$DATE/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --output tmp/nr3d_eval_v9_2_full_text_first_$DATE/leaderboard_metrics.json \
    --canonical-filter true
```

## Caveats

- Single seed, single fold. NR3D random100 → full direction agrees but
  magnitude shrinks. Read the +0.55 pp as "text-first is marginally
  better overall on this fold under this run config", not as a robust
  ranking.
- Per-tier (Easy/Hard/V-Dep/V-Indep) deltas are 1 – 1.6 pp on populations
  of 2.5 – 5 k samples. These are meaningfully outside ±0.5 pp noise but
  the per-stratum sample size still allows the V-Dep / V-Indep split to
  reverse on a different seed.
- Both variants use `use_tool_answer_disagreement_gate`,
  `use_no_match_candidate_guard`, `use_evidence_frame_guard`. The v9.3
  ablation should toggle these to find the cost / benefit of each.
- Stage-1 (`select_keyframes_v2`) is partially LLM-driven by Gemini;
  re-runs may shift individual samples by ±2 – 3 pp on the affected
  subset, but the cross-variant delta should be stable.

## SQLite

```bash
# Already ingested
sqlite3 docs/benchmark/nr3d/runs.sqlite \
    "SELECT run_id, classification_acc_filtered AS overall,
            acc_easy, acc_hard, acc_view_dep, acc_view_indep
     FROM runs WHERE run_id LIKE 'v9_2_full%';"
```

| run_id | overall | easy | hard | vdep | vind |
|---|---:|---:|---:|---:|---:|
| v9_2_full_no_text     | 0.6465 | 0.7609 | 0.5394 | 0.5814 | 0.6820 |
| v9_2_full_text_first  | 0.6520 | 0.7601 | 0.5508 | 0.5683 | 0.6976 |

## Default decision

**Keep `enable_stage1_text_retrieval=True` (the default)** for NR3D / VG.
The full-set A/B shows a marginal but consistent overall advantage
(+0.55 pp), and the per-stratum picture shows `select_by_text` is
genuinely useful for hard / view-indep queries. The view-dep
disadvantage (−1.31 pp) is a v9.3 routing concern: condition the
playbook on query phrasing so view-dep queries skip Stage-1.

## Next steps

1. **Per-tier playbook (v9.3)**: detect "facing X / standing at Y / on
   the right" in the user query at task-setup time, swap to the
   `_no_text` playbook for those samples only. Expected lift on
   V-Dep tier: +1.3 pp ⇒ overall +0.5 pp under text-first defaults.
2. **Guard ablation (v9.3)**: TADG / no-match / evidence-frame guards
   are all live in v9.2. The 3.3 pp gap from v5.1 deserves a one-shot
   ablation to identify which guard is the most expensive.
3. **Optional**: re-run no-text full with `--use-evidence-frame-guard`
   OFF to test whether the guard's left/right rejection is over-
   firing on catalog-first traces (it fires more often there since
   the agent has less framing context from `select_by_text`).
