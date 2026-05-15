# NR3D v9.2 — `select_by_text` A/B (text-first vs catalog-first)

- **Date**: 2026-05-16 02:23 → 02:42 UTC+8 (eval + leaderboard)
- **Branch**: `feat/v9-1-selectors-return-images`
- **Tip commit at run time**: `c2c52d0`
- **Fold**: `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
  (100 utterances, 58 scenes — same fold used by v9 / v9.1 / v9.1_real)
- **Pack**: `pack_nr3d_v9_catalog_first` (BEV-cache warm; pack-prep skipped, ~0 s)
- **Stage 2 backend**: `gpt-5.4-2026-03-05` via ModelHub, `workers=12` each
  (parallel tmux sessions, total 24 concurrent agents)
- **Wall clock**: ~13 min each, ran fully in parallel; ~13 min total
- **Run outputs**:
  - text-first: `tmp/nr3d_eval_v9_2_text_first_20260516_0223/`
  - catalog-first (no_text): `tmp/nr3d_eval_v9_2_no_text_20260516_0223/`

## Why this run exists

The `select_by_text` reliability audit
([`v9_1_select_by_text_audit_20260516.md`](v9_1_select_by_text_audit_20260516.md))
found that on NR3D random100 the Stage-1 language→frame retrieval tool
returns `parse=no_evidence` for **32 / 100** queries — a hard wall that
caps the tool's top-K hit rate at 55 %. The hypothesis going in:

> Drop `select_by_text` from the tool list; let the agent rely solely on
> the catalog-driven selectors (`select_by_proposal`, `select_by_region`,
> …). Expected: ≥ +5 pp because the 32 empty-pred queries no longer waste
> a turn and the simple-target queries (where catalog is sufficient) stop
> being shadowed by an unreliable first move.

This run flips the new `Stage2DeepAgentConfig.enable_stage1_text_retrieval`
toggle (added in commit `c2c52d0`) which simultaneously:

- removes `select_by_text` from `build_selector_tools` output;
- rewrites the system-prompt tool-family enumeration so
  `select_by_proposal` becomes "primary entry";
- swaps `scene_exploration_playbook.md` and `vg_grounding_playbook.md`
  for their `_no_text.md` siblings via `load_skill`'s sibling-resolution
  logic.

Same fold, same pack, same backend, same prompts apart from the toggle.

## Headline

| Metric | **v9.2 text-first** (`select_by_text` ON) | **v9.2 catalog-first** (`select_by_text` OFF) | Δ |
|---|---:|---:|---:|
| Overall (n=100)  | **68.00 %** | **63.00 %** | **−5.0 pp** ⬇ |
| Easy (n=41)      | 78.05 % | 70.73 % | −7.3 |
| Hard (n=59)      | 61.02 % | 57.63 % | −3.4 |
| View-dep (n=34)  | 64.71 % | 64.71 % | **0.0** |
| View-indep (n=66)| 69.70 % | 62.12 % | −7.6 |

**The audit's catalog-first hypothesis is falsified.** Removing
`select_by_text` from the tool surface *hurts* accuracy on the same fold,
not helps. Net per-sample diff:

- Both correct: **54**
- Both wrong:   **23**
- text-first wins: **14**
- catalog-first wins: **9**
- Net: **+5 samples** for text-first → matches the −5 pp delta exactly.

## Cross-version timeline

| Run | Date | Stage-1 wiring | Playbook first move | Overall | Hard | V-dep |
|---|---|---|---|---:|---:|---:|
| v9 catalog_first (clean) | 2026-05-15 14:42 | not wired | catalog selectors | **86.00** | 81.36 | 82.35 |
| v9.1_fix                 | 2026-05-15 23:52 | wired but broken (errors) | text-first (in prompt, broken in practice) | 86.00 | 84.75 | 82.35 |
| v9.1_real                | 2026-05-16 00:35 | wired and live | text-first | 69.00 | 57.63 | 58.82 |
| **v9.2 text-first**      | 2026-05-16 02:23 | wired and live | text-first + audit-informed routing | **68.00** | 61.02 | 64.71 |
| **v9.2 catalog-first**   | 2026-05-16 02:23 | tool dropped | catalog-first (no `select_by_text` mention) | **63.00** | 57.63 | 64.71 |

Two clean observations from the timeline:

1. **`select_by_text` in the agent's reach is net positive (+5 pp)**, even
   with the 32 % empty-pred rate the audit measured. Counterintuitive
   but real on this fold.
2. **The audit-informed routing tweak in the v9.2 text-first playbook
   (`use select_by_text only for rare-target queries; skip on
   view-dependent / spatial-anchor phrasing`) buys nothing**
   (68 % vs 69 % v9.1_real ≈ same fold variance). The LLM appears to
   ignore the steering text and call `select_by_text` whenever the
   system prompt suggests it.

## What looked surprising and why

The 86 → 63 pp gap between v9 catalog_first and v9.2 catalog-first is
the most striking number in the table, and it deserves a caveat:

- **v9 catalog_first** ran on the old (pre-v9.1) tool surface:
  `view_keyframe(mode='auto')` instead of `mark_frame_with_bbox`,
  `view_keyframe(mode='marked')` for verification, no
  `select_by_text` at all in the prompt. The agent ran a simpler
  3-tool loop (`select_by_proposal` → `view_keyframe` → submit).
- **v9.2 catalog-first** runs on the v9.1+ tool surface
  (`mark_frame_with_bbox` for annotated verification, no
  `view_keyframe`, plus the audit caveat in the playbook). It also has
  every guard enabled (`use_tool_answer_disagreement_gate`,
  `use_no_match_candidate_guard`, `use_evidence_frame_guard`).

So **v9.2 catalog-first ≠ v9 catalog_first**. The right
intra-run comparison is **v9.2 text-first vs v9.2 catalog-first**:
same code, same prompts, same guards, same pack — the only difference
is the toggle. That comparison shows **−5 pp without `select_by_text`**.

The right inter-run comparison for "is the v9.1 mark-based tool surface
worse than the v9 view_keyframe one" is **v9 catalog_first (86 %) vs
v9.2 catalog-first (63 %)** — and that shows a 23 pp regression that has
nothing to do with `select_by_text`. That belongs in a separate
diagnostic. (Likely culprits: guard cost; mark-as-required-evidence
forces a verification step even when the catalog ID was obvious from
the BEV; the v9.1 playbook prose now emphasises mark+verify which
encourages more turns and more chances to flip the answer.)

## Tool usage diff (same 100-sample fold)

|                    | text-first (1562 calls) | catalog-first (1552 calls) |
|---|---:|---:|
| `select_by_text`     | n/a (no per-call count without inspecting traces) | 0 (not registered) |
| `select_by_proposal` | 231 | 246 |
| `select_by_frame_neighbor` | 34 | 44 |
| `inspect_proposal`   | 312 | **352** |
| `mark_frame_with_bbox` | 253 | 262 |
| `list_scene_proposals` | 46 | 20 |
| `view_bev`           | 99 | 107 |
| `load_skill`         | 235 | 207 |
| `submit_final`       | 227 | 234 |
| `compare_proposals_spatial` | 47 | 51 |

The catalog-first variant compensates for the missing `select_by_text`
by:

- +6 % `inspect_proposal` calls (catalog deep-dive)
- +6 % `select_by_proposal` calls (manual candidate enumeration)
- +29 % `select_by_frame_neighbor` calls (expand around one good frame)
- −57 % `list_scene_proposals` calls (the LLM seems to forget to use
  this when the playbook stops naming `select_by_text` as the
  default entry — counter-intuitive)

The drop in `list_scene_proposals` is interesting: removing
`select_by_text` from the prompt evidently also removes "catalog-first"
phrasing the LLM was using to orient itself, so it falls back to
opportunistic enumeration via `select_by_proposal` without first
scoping the candidate set. That's a plausible cause of the −7 pp drop
on Easy / View-indep where a well-scoped candidate list would have
locked the answer.

## Per-sample anecdotes (first 5 of each)

text-first wins (catalog-first fails):

| sample_id | query |
|---|---|
| scannet/scene0629_00::6::19188 | The white cabinet closes to the closed door. |
| scannet/scene0565_00::0::3757 | pick the grey chair next to the green one. |
| scannet/scene0084_00::44::13751 | Choose the horizontal rail under the shower head |
| scannet/scene0221_00::22::33555 | **facing the beds, right picture** ← view-dep, audit predicted catalog should win |
| scannet/scene0549_00::5::24557 | The round table with a plant on it |

catalog-first wins (text-first fails):

| sample_id | query |
|---|---|
| scannet/scene0621_00::0::5230 | The table in the corner next to the chairs that are pushed up against the wall. |
| scannet/scene0095_00::24::39202 | The mouse close to the two chairs in the corner |
| scannet/scene0084_00::35::19742 | **the toilet paper roll on the right side of the room.** ← right/left, audit predicted catalog should win |
| scannet/scene0568_00::15::20548 | chair in the corner, closest to the black shelf. |
| scannet/scene0652_00::13::34162 | Standing at the foot of the bed, there are five pillows. Select the one in front. |

The first hypothesis-falsifying observation: "facing the beds, right
picture" — the view-dep / right-left query the audit specifically
flagged as a case Stage-1 cannot ground — *text-first got it right and
catalog-first did not*. The audit was correct that Stage-1 returns
`no_evidence` for this kind of phrasing on the keyframe-hit metric, but
**that does not mean the agent benefits from `select_by_text` being
removed**. The empty `frames: []` response from `select_by_text` is
itself a useful signal — it tells the agent "this query doesn't have a
clean catalog handle; reach for the spatial reasoning tools instead",
and the agent then often picks the right disambiguation path.
Removing the tool removes that fork in the road.

## What this run validates

1. **Keep `select_by_text` in the tool set as a default** for NR3D /
   VG. The audit's empty-pred wall is real but does not justify
   removing the tool; the agent extracts value even from its failures.
2. **The audit-informed routing prose in the text-first playbook is
   inert.** Either rewrite it as a stricter rule the agent can follow
   ("if select_by_text returns `frames: []`, immediately call
   `select_by_proposal`") or remove it. Right now it costs prompt
   tokens for no observable behavioural change.
3. **The v9.1 → v9.2 mark-required tool surface is not optimal for the
   v9 catalog-first ceiling.** The 86 → 63 / 68 drop is on the v9.1
   surface, not the toggle. That's a separate v9.3 investigation:
   either (a) loosen `evidence_frame_guard` so a marked frame is no
   longer compulsory when the catalog signal is unambiguous, or (b)
   restore an unfiltered `view_keyframe` for simple "I just want to
   look at this frame" cases.
4. **`use_no_match_candidate_guard` + `use_evidence_frame_guard` +
   `use_tool_answer_disagreement_gate` may be over-constraining on the
   v9.1 surface.** The v9 catalog_first 86 % run had all three off (per
   the version doc); v9.1_fix and later had all three on. Worth an
   ablation.

## Reproduce

```bash
git checkout c2c52d0
source .venv/bin/activate
export PYTHONPATH=src

DATE=$(date +%Y%m%d_%H%M)

# text-first
tmux new-session -d -s nr3d-text "python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_2_text_first_$DATE \
    --workers 12 --sample-retries 2 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard \
    2>&1 | tee tmp/nr3d_v9_2_text_first_$DATE.log"

# catalog-first
tmux new-session -d -s nr3d-no-text "python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_v9_2_no_text_$DATE \
    --workers 12 --sample-retries 2 \
    --use-tool-answer-disagreement-gate \
    --use-no-match-candidate-guard \
    --use-evidence-frame-guard \
    --disable-stage1-text-retrieval \
    2>&1 | tee tmp/nr3d_v9_2_no_text_$DATE.log"

# After both finish (~13 min each, parallel):
python -m evaluation.scripts.nr3d_leaderboard_metrics \
    --side-by-side tmp/nr3d_eval_v9_2_text_first_$DATE/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
    --output tmp/nr3d_eval_v9_2_text_first_$DATE/leaderboard_metrics.json \
    --canonical-filter true

python -m evaluation.scripts.nr3d_leaderboard_metrics \
    --side-by-side tmp/nr3d_eval_v9_2_no_text_$DATE/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
    --output tmp/nr3d_eval_v9_2_no_text_$DATE/leaderboard_metrics.json \
    --canonical-filter true
```

## Caveats

- Single seed, single fold, single run per variant. Reading the 86/68/63
  series, run-to-run variance on this fold is in the **±2 pp** range.
  A 5 pp delta survives that noise; a 3 pp delta does not.
- The fold is `random100`, not the full 7805 / 8584 NR3D test set.
  Direction should generalise but absolute numbers will move.
- Stage-1 (`select_keyframes_v2`) uses Gemini for query parsing. Backend
  outages, key rotations, or temperature drift can flip a handful of
  samples between runs.
- No LLM judge needed — accuracy is programmatic
  (`selected_object_id == target_id`).

## SQLite ingestion

```bash
python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v9_2_text_first_20260516_0223/ \
    --run-id v9_2_text_first \
    --branch feat/v9-1-selectors-return-images \
    --commit c2c52d0 \
    --backend pack_v1 \
    --judge-model nr3d-classifier \
    --notes "v9.2 A/B: select_by_text ON (text-first, audit-informed routing in playbook)" \
    --db docs/benchmark/nr3d/runs.sqlite

python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v9_2_no_text_20260516_0223/ \
    --run-id v9_2_no_text \
    --branch feat/v9-1-selectors-return-images \
    --commit c2c52d0 \
    --backend pack_v1 \
    --judge-model nr3d-classifier \
    --notes "v9.2 A/B: select_by_text OFF (catalog-first, no_text playbook variant)" \
    --db docs/benchmark/nr3d/runs.sqlite
```
