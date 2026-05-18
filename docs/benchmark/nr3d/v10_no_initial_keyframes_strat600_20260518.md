# NR3D v10 no-initial-keyframes strat600

This run validates the new spec-cleanup branch where Stage 2 no longer receives
pack-prepared initial keyframe evidence. The only path to first-person evidence
is now active agent tool use through the selector / marking tool surface.

Result: the pipeline runs through the canonical 600-case fold cleanly, but the
headline score is **64.33 %**, lower than the current honest v9.3 text-first
baseline (**66.67 %**) and essentially tied with the v9.3 catalog-only partner
(**64.17 %**).

## Pre-run checklist

| Item | Value |
|---|---|
| Pending changes committed before launch | yes |
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `220f128` |
| Run-time code commit | `220f128` - no worktree drift |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold MD5 | `12a69d8d14a81519024bbe00d6334434` |
| Fold design | `docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md` |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/` |
| Run log | `tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/run.log` |
| Launch / exit log | `tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/launch_info.log` |
| Leaderboard metrics | `tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/leaderboard_metrics.json` |
| Side-by-side JSON | `tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/side_by_side.json` |
| Workers | 20 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |
| Diagnostic leak flags | none |
| `--restore-stage1-seed-keyframe-drain` | not set |
| `--force-stage1-text-retrieval-to-error` | not set |
| Eval exit status | `0` |
| Metrics exit status | `0` |
| Per-sample status | 600 / 600 `completed` |
| No-match / non-completed | 0 / 600 |
| Tracebacks in run log | 0 |
| ModelHub 429 retries | 305 at attempt 1/5; 0 at attempt 2/5 or later |
| SQLite run id | `v10_no_initial_keyframes_strat600_20260518` |

## Exact commands

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128 \
  --workers 20 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --output tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128 \
  --run-id v10_no_initial_keyframes_strat600_20260518 \
  --branch feat/remove-initial-keyframes \
  --commit 220f128 \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/leaderboard_metrics.json \
  --notes "Remove all initial keyframe evidence; keyframes only reachable via select_by_* tools; canonical NR3D strat600 run."
```

## Headline metrics

| Metric | v10 no-initial-keyframes | v9.3 text-first | v9.3 catalog-only |
|---|---:|---:|---:|
| Overall | **64.33** | 66.67 | 64.17 |
| Easy | 72.41 | 73.45 | 73.10 |
| Hard | 56.77 | 60.32 | 55.81 |
| V-Dep | 53.55 | 54.98 | 56.87 |
| V-Indep | 70.18 | 73.01 | 68.12 |
| bbox Acc@0.25 | 64.33 | 66.83 | 64.17 |
| bbox Acc@0.50 | 64.33 | 66.67 | 64.17 |
| Mean IoU | 0.647 | 0.669 | 0.645 |

Delta vs v9.3 text-first: Overall **-2.34 pp**, Easy -1.04, Hard -3.55,
V-Dep -1.42, V-Indep -2.83. The Overall delta is right at the strat600
variance budget, so it should be treated as a borderline regression signal
rather than a full-set claim.

Delta vs v9.3 catalog-only: Overall **+0.16 pp**, Easy -0.69, Hard +0.96,
V-Dep -3.32, V-Indep +2.06. This confirms the no-initial-keyframes cleanup runs
closer to the catalog-only policy than to the previous text-first strat600
baseline.

## Log and trace findings

No script-level failures were observed:

- `RUN_EXIT_STATUS=0`
- `METRICS_EXIT_STATUS=0`
- 600 checkpoint JSON files written
- every checkpoint has `status=completed`
- 0 Tracebacks, 0 logged `Exception`
- 0 no-match / non-completed samples
- ModelHub 429s were only first-attempt retry events and did not exhaust retries

Case-level failures are ordinary metric misses: 214 / 600 samples selected a
proposal id different from the target id.

Trace-level failure patterns:

| Pattern | Count | Affected cases | Wrong among affected | Notes |
|---|---:|---:|---:|---|
| `compare_proposals_spatial` unsupported relation | 105 tool errors | 93 | 24 | Agent often emits aliases like `closer_to`, `farther_from`, `in_front_of`, `behind`, `between`, `beside`, `across_from`. The tool accepts only `closest_to`, `near`, `next_to`, `farthest_from`, `above`, `below`, `left_of`, `right_of`. Many cases recover by retrying with `near`, but this is a real tool-contract mismatch. |
| `select_by_text` masked-category guard | 39 tool errors | 33 | 19 | Most common leaks: `wall` (20), `floor` (6), `bed` (3), `door` (2). These are non-fatal but remove a text-retrieval path in spatial / support queries. |
| Evidence-frame guard block | 135 tool blocks | 89 | 54 | Often catches rationale / cited-frame mismatch, but repeated guard turns can still steer the agent to a wrong final id. |
| TADG block | 15 tool blocks | 12 | 7 | Mostly anchor / target role conflicts. One audited case shows TADG blocking the visually correct proposal and forcing the wrong relation-ranked proposal. |
| No-match guard block | 2 tool blocks | 2 | 1 | Rare and not a run-stability issue. |

Representative audited traces:

- `scannet/scene0696_00::21::29050` -> target `21`, final `32`. The agent first
  selected `21` and cited marked frames where `21` is left of `22`, but TADG
  classified `21` as the spatial anchor and repeatedly rejected it. The accepted
  final answer `32` follows the guard's relation ranking, not the human reading
  of the marked evidence.
- `scannet/scene0704_00::10::7262` -> target `10`, final `22`. The trace has
  `select_by_text(... hidden_categories=['wall'])` blocked by masked-category
  guard, then the agent inspected marked frames with four picture proposals and
  grouped `#11/#10/#22` as the "three together" set. This made it choose `22`
  as center, while the target id is `10`.
- `scannet/scene0699_00::26::40486` -> target `26`, final `26`. This is a
  positive recovery example: `mark_frame_with_bbox(frame_id=184, ids=[6])`
  failed because the table was not visible, and `closer_to` was rejected as an
  unsupported relation. The agent retried with `near`; the spatial ranking
  recovered `26`.

## Tool histogram

| Tool | Calls |
|---|---:|
| `inspect_proposal` | 1 798 |
| `mark_frame_with_bbox` | 1 586 |
| `load_skill` | 1 475 |
| `submit_final` | 1 353 |
| `select_by_proposal` | 1 331 |
| `view_bev` | 590 |
| `compare_proposals_spatial` | 317 |
| `list_scene_proposals` | 286 |
| `select_by_frame_neighbor` | 178 |
| `select_by_text` | 174 |
| `list_frame_proposals` | 151 |
| `request_crops` | 75 |
| `retrieve_object_context` | 31 |
| `select_by_coverage` | 17 |
| `select_by_region` | 10 |

## SQLite reproduction query

```sql
SELECT run_id, n, n_filtered,
       ROUND(classification_acc_filtered * 100, 2) AS overall,
       ROUND(acc_easy * 100, 2) AS easy,
       ROUND(acc_hard * 100, 2) AS hard,
       ROUND(acc_view_dep * 100, 2) AS vdep,
       ROUND(acc_view_indep * 100, 2) AS vindep
FROM runs
WHERE run_id = 'v10_no_initial_keyframes_strat600_20260518';
-- v10_no_initial_keyframes_strat600_20260518 | 600 | 600 | 64.33 | 72.41 | 56.77 | 53.55 | 70.18
```

## Reading

The no-initial-keyframes cleanup is runnable and removes the legacy evidence
path without introducing run-level failures. It does not improve NR3D accuracy
on strat600. The most actionable trace findings are the relation-vocabulary
gap in `compare_proposals_spatial` and the masked-category behavior of
`select_by_text`; both are tool-contract issues exposed by the run, not process
stability problems.

## Failed-case hardening replay slice

The follow-up hardening spec/plan uses a durable replay slice:

- `docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json`

Run after guard/tool-flow changes:

```bash
COMMIT=$(git rev-parse --short HEAD)
tmux new-session -d -s nr3d_v10_hardening_audit40 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   PYTHONPATH=src .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_hardening_audit40_${COMMIT} \
     --workers 8 \
     --sample-retries 2 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard 2>&1 | tee /tmp/nr3d_v10_hardening_audit40_${COMMIT}.log"
```

Use this slice only as behavioral replay, not as a leaderboard claim. The file
name keeps `audit40` because the audit was planned as 40 cases; the durable
list contains 41 ids due to keeping positive recovery example
`scene0699_00::26::40486` as a guardrail.

### Hardening replay result: 8cae30d

Follow-up guard/tool-flow hardening was replayed on the durable slice after
commits through `8cae30d`:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids docs/benchmark/nr3d/assets/v10_failed_case_audit40_sample_ids_20260518.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v10_hardening_audit40_8cae30d_w20 \
  --workers 20 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Artifacts:

- Side-by-side: `tmp/nr3d_eval_v10_hardening_audit40_8cae30d_w20/side_by_side.json`
- Metrics: `tmp/nr3d_eval_v10_hardening_audit40_8cae30d_w20/leaderboard_audit40.json`
- Console log: `/tmp/nr3d_v10_hardening_audit40_8cae30d_w20.log`

Run stability:

- 41 / 41 samples completed.
- 0 `status=failed`.
- 0 completed samples with missing `selected_object_id`.
- 0 Tracebacks / logged Exceptions.
- 0 ModelHub `retryable status=403` lines in this replay.

Audit-slice metric, for sanity only:

```text
n_full=41 n_filtered=41
classification_acc_filtered = 0.3171
Easy   (n=13): 0.3077
Hard   (n=28): 0.3214
V-Dep  (n=26): 0.3077
V-Ind  (n=15): 0.3333
```

Previously failed cases now terminate normally:

| Sample | Final status | Selected | IoU | Note |
|---|---:|---:|---:|---|
| `scannet/scene0030_00::18::30641` | completed | 19 | 0.0000 | No longer blocked into no-match by wall-vs-shelf target parsing; still an ordinary metric miss. |
| `scannet/scene0378_00::41::26109` | completed | 41 | 1.0000 | Nested relation case recovered after allowing same-category intermediate anchor evidence. |
| `scannet/scene0496_00::29::10819` | completed | 12 | 0.0000 | No longer blocked into no-match by orientation phrase `wall of windows`; still an endpoint-ordering metric miss. |
| `scannet/scene0565_00::23::30788` | completed | 22 | 0.1089 | No longer blocked into no-match by `green wall ... this cart`; still confuses the two cart candidates. |
| `scannet/scene0629_00::14::17248` | completed | 14 | 1.0000 | `wall painting` target-category regression is fully recovered. |

This replay validates run stability of the hardening changes, not an accuracy
claim. Accuracy misses remain on this deliberately failure-heavy slice and
should be treated as the next prompt/tool-flow improvement pool.

### Old strat600 failure replay result: f4c03f7

To estimate whether the hardening changes recover failures from the original
600-case run, the 214 metric-miss samples from
`tmp/nr3d_eval_no_initial_keyframes_strat600_20260518_220f128/leaderboard_metrics.json`
were extracted to:

- `docs/benchmark/nr3d/assets/v10_no_initial_strat600_failed214_sample_ids_20260518.json`
- Fold MD5: `a537f518ef4f63d90d1d319df636218a`

Launch state:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `f4c03f7` |
| Run-time code commit | `f4c03f7` - no worktree drift |
| Workers | 20 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |

Command:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids docs/benchmark/nr3d/assets/v10_no_initial_strat600_failed214_sample_ids_20260518.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v10_failed214_rerun_20260518_f4c03f7_w20 \
  --workers 20 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_v10_failed214_rerun_20260518_f4c03f7_w20/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids docs/benchmark/nr3d/assets/v10_no_initial_strat600_failed214_sample_ids_20260518.json \
  --output tmp/nr3d_eval_v10_failed214_rerun_20260518_f4c03f7_w20/leaderboard_failed214.json \
  --canonical-filter true
```

Artifacts:

- Side-by-side: `tmp/nr3d_eval_v10_failed214_rerun_20260518_f4c03f7_w20/side_by_side.json`
- Metrics: `tmp/nr3d_eval_v10_failed214_rerun_20260518_f4c03f7_w20/leaderboard_failed214.json`
- Console log: `tmp/nr3d_eval_v10_failed214_rerun_20260518_f4c03f7_w20/run.log`
- Side-by-side MD5: `e3c64f3b7e9453d87d553149e5b6c0a0`
- Metrics MD5: `0bc11172b2308b670c4013d6a00d5bac`

Run stability:

- 214 / 214 samples emitted side-by-side rows.
- 210 / 214 completed.
- 4 / 214 final `status=failed`.
- 0 completed samples with missing `selected_object_id`.
- 0 Tracebacks / logged Exceptions.
- 97 ModelHub 429 retry lines, all at attempt 1/5; 0 attempt 2/5 or later.

Recovery result:

| Metric | Value |
|---|---:|
| Old metric misses rerun | 214 |
| Recovered to correct target | 81 |
| Recovery rate on old misses | 37.85 % |
| Still incorrect or failed | 133 |
| Completed but still wrong, same selected id as original miss | 87 |
| Completed but still wrong, changed selected id | 42 |
| Final `status=failed` | 4 |

Slice breakdown:

| Column | n | Recovery |
|---|---:|---:|
| Easy | 80 | 38.75 % |
| Hard | 134 | 37.31 % |
| View-Dep | 98 | 39.80 % |
| View-Indep | 116 | 36.21 % |

The four non-completed samples were:

- `scannet/scene0338_00::18::25573`
- `scannet/scene0629_00::30::28923`
- `scannet/scene0647_00::8::3986`
- `scannet/scene0144_00::17::37726`

Interpretation: this is a one-sided recovery probe, not a valid replacement
for the full strat600 score. If the original 386 correct samples did not
regress, the arithmetic upper projection would be `(386 + 81) / 600 = 77.83 %`.
That number is not claimable until the full 600 is rerun on the same code.

### Full strat600 rerun after tool-trace image metadata: cfee0ef

After removing the runtime / bundle pending-image side channel, the full
canonical 600-case fold was rerun from commit `cfee0ef`. This commit changes
tool-produced images to flow through `Stage2ToolObservation.image_metadata`
only; `build_evidence_update_message()` scans tool trace metadata and no longer
reads bundle-side image queues.

Launch state:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `cfee0ef987368404cb31a6ddf90706cc327fdcef` |
| Run-time code commit | `cfee0ef987368404cb31a6ddf90706cc327fdcef` - no worktree drift |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold MD5 | `12a69d8d14a81519024bbe00d6334434` |
| Output dir | `tmp/nr3d_eval_v10_tool_trace_images_strat600_20260518_cfee0ef/` |
| Workers | 20 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |
| Diagnostic leak flags | none |
| Eval exit status | `0` |
| Metrics exit status | `0` |
| SQLite run id | `v10_tool_trace_images_strat600_20260518` |

Command:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v10_tool_trace_images_strat600_20260518_cfee0ef \
  --workers 20 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_v10_tool_trace_images_strat600_20260518_cfee0ef/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --output tmp/nr3d_eval_v10_tool_trace_images_strat600_20260518_cfee0ef/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_tool_trace_images_strat600_20260518_cfee0ef \
  --run-id v10_tool_trace_images_strat600_20260518 \
  --branch feat/remove-initial-keyframes \
  --commit cfee0ef \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_tool_trace_images_strat600_20260518_cfee0ef/leaderboard_metrics.json \
  --notes "Route tool images through Stage2 tool_trace image_metadata; remove bundle/runtime pending image queues; canonical NR3D strat600 rerun."
```

Run stability:

- 600 / 600 per-sample JSON files written.
- 600 / 600 side-by-side rows written.
- 590 / 600 final `status=completed`.
- 10 / 600 final `status=failed`.
- 0 Tracebacks / logged Exceptions.
- 295 ModelHub 429 retry lines, all at attempt 1/5; 0 attempt 2/5 or later.
- Output artifact scan found 0 occurrences of the removed image side-channel
  keys.
- 596 / 600 samples contain tool-trace image metadata; total
  `image_metadata` rows: 6 830.

Headline metrics:

| Metric | 220f128 v10 baseline | cfee0ef tool-trace images | Delta |
|---|---:|---:|---:|
| Overall | 64.33 | **61.00** | -3.33 |
| Easy | 72.41 | 67.93 | -4.48 |
| Hard | 56.77 | 54.52 | -2.26 |
| V-Dep | 53.55 | 52.13 | -1.42 |
| V-Indep | 70.18 | 65.81 | -4.37 |

Matched-sample churn vs the original `220f128` full strat600 run:

| Count | Value |
|---|---:|
| `220f128` correct samples | 386 |
| `cfee0ef` correct samples | 366 |
| Old positives preserved | 293 |
| Old positives now wrong / failed | 93 |
| Old misses recovered | 73 |
| Net correct delta | -20 |

The 10 non-completed samples were:

| Sample | Primary failure shape |
|---|---|
| `scannet/scene0552_00::30::40217` | Target-category guard repeatedly parsed the anchor `wall` as target; agent kept trying the box. |
| `scannet/scene0629_00::30::28923` | Target-category guard repeatedly parsed anchor `mirror` as target; agent kept trying the chair. |
| `scannet/scene0665_00::19::10095` | Agent concluded no round table was in the pool after inspecting table candidates. |
| `scannet/scene0144_00::17::37726` | Pack prediction missing status; payload had `status=None`, `selected_object_id=None`. |
| `scannet/scene0700_00::34::40232` | Pack prediction missing status; payload had `status=None`, `selected_object_id=None`. |
| `scannet/scene0019_00::19::19821` | Pack prediction missing status; payload had `status=None`, `selected_object_id=None`. |
| `scannet/scene0164_00::24::20371` | Pack prediction missing status; payload had `status=None`, `selected_object_id=None`. |
| `scannet/scene0648_00::24::25259` | Target-category guard repeatedly parsed anchor `plant` as target; agent kept trying the supporting shelf. |
| `scannet/scene0606_00::32::9837` | Target-category guard repeatedly parsed anchor `clock` as target; agent kept trying the storage bin. |
| `scannet/scene0598_00::2::4382` | No-match guard plus target-category guard deadlock around monitor vs bookshelf anchor. |

Reading: removing the pending-image side channel is architecturally cleaner and
keeps artifacts leak-free, but the exact `cfee0ef` rerun is not an accuracy
improvement. It introduces 10 final failed statuses and a net -20 correct
samples vs `220f128`. The largest actionable failure mode is still target /
anchor parsing in guards, now exposed more sharply because the runtime no
longer has any hidden bundle image queue to lean on.

### Target-category guard recovery probe: 2ce81df

After the `cfee0ef` full strat600 run, three independent read-only case audits
covered 15 base cases: 10 final failed statuses plus 5 old-positive regressions.
The dominant actionable bucket was target-category guard target/anchor
inversion. The guard was interpreting anchor/support nouns as the target head in
queries like `chair closest to the mirror` or `shelf that has a plant`.

Code change:

- `02a2c0c` updates `target_category_guard` head-noun parsing so relation cues
  (`closest to`, `farthest from`, `against`, etc.) truncate the candidate head
  before anchor nouns.
- It treats `that/which/who` as relative-clause cues rather than demonstrative
  target heads.
- It handles discourse-set queries like `When looking at the three storage
  bins, it...` by using the set category as the target.
- The change uses only query text and proposal categories; it does not inspect
  GT targets, GT boxes, target-visible frames, or metrics.

Validation:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `2ce81df09aac8a008be82bb69563f46c2efda618` |
| Run-time code commit | `2ce81df09aac8a008be82bb69563f46c2efda618` - no worktree drift |
| Sample ids | `docs/benchmark/nr3d/assets/v10_target_category_guard5_sample_ids_20260519.json` |
| Output dir | `tmp/nr3d_eval_v10_target_category_guard5_20260519_2ce81df/` |
| SQLite run id | `v10_target_category_guard5_20260519` |
| Workers | 5 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |
| Eval exit status | `0` |
| Metrics exit status | `0` |

Commands:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids docs/benchmark/nr3d/assets/v10_target_category_guard5_sample_ids_20260519.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v10_target_category_guard5_20260519_2ce81df \
  --workers 5 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_v10_target_category_guard5_20260519_2ce81df/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids docs/benchmark/nr3d/assets/v10_target_category_guard5_sample_ids_20260519.json \
  --output tmp/nr3d_eval_v10_target_category_guard5_20260519_2ce81df/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_target_category_guard5_20260519_2ce81df \
  --run-id v10_target_category_guard5_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 2ce81df \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_target_category_guard5_20260519_2ce81df/leaderboard_metrics.json \
  --notes "Target-category guard head-noun fix recovery probe on 5 prior guard-deadlock failures; no GT inputs, standard guards."
```

Probe result:

| Sample | `cfee0ef` status / selected | `2ce81df` status / selected | Guard outcome |
|---|---|---|---|
| `scannet/scene0552_00::30::40217` | failed / `null` | completed / `30` | expected `box`, submitted `box`, not blocked |
| `scannet/scene0629_00::30::28923` | failed / `null` | completed / `30` | expected `chair`, submitted `chair`, not blocked |
| `scannet/scene0648_00::24::25259` | failed / `null` | completed / `24` | expected `bookshelf`, submitted `bookshelf`, not blocked |
| `scannet/scene0606_00::32::9837` | failed / `null` | completed / `32` | expected `storage bin`, submitted `storage bin`, not blocked |
| `scannet/scene0598_00::2::4382` | failed / `null` | completed / `2` | expected `monitor`, submitted `monitor`, not blocked |

Metrics on this diagnostic subset:

| Metric | Value |
|---|---:|
| n | 5 |
| Overall | 100.00 |
| Easy | 100.00 |
| Hard | 100.00 |
| V-Dep | 100.00 |
| V-Indep | 100.00 |

Reading: this is not a benchmark-grade accuracy claim; it is a targeted
regression probe over five previously audited guard-deadlock failures. It
confirms that the head-noun fix removes the target-category guard blockage on
the exact failure pattern. Remaining high-impact buckets from the 15-case audit
are missing-status result extraction, unsupported `behind` / `in_front_of`
spatial relations, evidence-frame guard overreach, and candidate coverage before
no-match.

### Missing-status extractor recovery probe: e5617ac

The same 15-case audit found four `cfee0ef` failures that were not visual
grounding failures. The agent produced a direct structured VG response with a
`proposal_id` and confidence, but the NR3D side-by-side runner only accepted
`selected_object_id` + `bbox_3d` / `status`. The result was a synthetic
`prediction missing status` failure and the side-by-side row lost the query and
trace.

Code change:

- `179edcd` brings the NR3D pack-v1 extractor to parity with ScanRefer.
- `extract_pack_v1_prediction()` now accepts `proposal_id` as a synonym for
  `selected_object_id`.
- When the payload lacks `bbox_3d`, it resolves the 9-DoF bbox from
  `result.final_bundle.extra_metadata["vg_proposal_pool"]`.
- If a bbox is resolved and `status` is absent, status is inferred as
  `completed`.
- This is an evaluation/output-normalization fix only; the agent still receives
  no GT target id, GT bbox, target-visible seed frames, or metric information.

Validation:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `e5617ac1360363dbd5381a5dbaf5249cb867b44d` |
| Run-time code commit | `e5617ac1360363dbd5381a5dbaf5249cb867b44d` - no worktree drift |
| Sample ids | `docs/benchmark/nr3d/assets/v10_missing_status4_sample_ids_20260519.json` |
| Output dir | `tmp/nr3d_eval_v10_missing_status4_20260519_e5617ac/` |
| SQLite run id | `v10_missing_status4_20260519` |
| Workers | 4 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |
| Eval exit status | `0` |
| Metrics exit status | `0` |

Probe result:

| Sample | `cfee0ef` failure shape | `e5617ac` status / selected | Query / trace preserved |
|---|---|---|---|
| `scannet/scene0144_00::17::37726` | missing status, no query/trace row | completed / `17` | yes / 14 tool records |
| `scannet/scene0700_00::34::40232` | missing status, no query/trace row | completed / `34` | yes / 14 tool records |
| `scannet/scene0019_00::19::19821` | missing status, no query/trace row | completed / `19` | yes / 13 tool records |
| `scannet/scene0164_00::24::20371` | missing status, no query/trace row | completed / `24` | yes / 14 tool records |

Metrics on this diagnostic subset:

| Metric | Value |
|---|---:|
| n | 4 |
| Overall | 100.00 |
| Easy | 100.00 |
| Hard | n/a |
| V-Dep | 100.00 |
| V-Indep | 100.00 |

The run log contained no `prediction missing status`, Traceback, or Exception
matches. Reading: this confirms the extractor failure bucket is fixed for the
audited shape. This probe is still not a benchmark-grade accuracy claim.

### Full strat600 rerun after guard/extractor fixes: a6f6077

This is the standard canonical 600-case rerun after the target-category guard
head-noun fix and NR3D proposal-only extractor fix. It keeps the v10 evidence
contract: no initial keyframes, no bundle/runtime pending-image side channel,
and first-person images only via active tool calls recorded in
`tool_trace.image_metadata`. BEV remains the only non-first-person image seeded
into the initial context.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `a6f6077b7142e0d28918c44671dad1801e196c5e` |
| Run-time code commit | `a6f6077b7142e0d28918c44671dad1801e196c5e` - no worktree drift |
| Current branch note | Later test-only commit `f887917` was made after launch to ban pending-image channels in CI; it did not affect this run. |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold MD5 | `12a69d8d14a81519024bbe00d6334434` |
| Output dir | `tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077/` |
| Run log | `tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077/run.log` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077/leaderboard_metrics.json` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077/side_by_side.json` |
| SQLite run id | `v10_no_gt_fixes_strat600_20260519` |
| Workers | 20 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |
| Eval exit status | `0` |
| Metrics exit status | `0` |

Commands:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077 \
  --workers 20 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --output tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077 \
  --run-id v10_no_gt_fixes_strat600_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit a6f6077 \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_no_gt_fixes_strat600_20260519_a6f6077/leaderboard_metrics.json \
  --notes "Full canonical strat600 after target-category guard head-noun fix and NR3D proposal-only extractor fix; no GT inputs, standard guards."
```

Run stability:

- 600 / 600 per-sample JSON files written.
- 600 / 600 side-by-side rows written.
- 597 / 600 final `status=completed`.
- 3 / 600 final `status=failed`.
- 0 Tracebacks / logged Exceptions.
- 0 `prediction missing status` matches.
- 0 ModelHub retries at attempt 2/5 or later.
- 273 ModelHub retry lines at attempt 1/5, including one `status=403` retry.
- Source/artifact scan found 0 occurrences of `pending_image_paths`,
  `pending_image_metadata`, `vg_pending_images`, or `queue_pending`.

Headline metrics:

| Metric | cfee0ef tool-trace images | a6f6077 fixes full rerun | Delta |
|---|---:|---:|---:|
| Overall | 61.00 | **62.67** | +1.67 |
| Easy | 67.93 | 71.03 | +3.10 |
| Hard | 54.52 | 54.84 | +0.32 |
| V-Dep | 52.13 | 53.55 | +1.42 |
| V-Indep | 65.81 | 67.61 | +1.80 |
| bbox Acc@0.25 | 61.00 | 62.67 | +1.67 |
| bbox Acc@0.50 | 61.00 | 62.67 | +1.67 |
| Mean IoU | n/a | 0.632 | n/a |

Matched-sample churn vs `cfee0ef`:

| Count | Value |
|---|---:|
| `cfee0ef` correct samples | 366 |
| `a6f6077` correct samples | 376 |
| Old misses recovered | 79 |
| Old positives regressed | 69 |
| Net correct delta | +10 |

The 3 non-completed samples were:

| Sample | Primary failure shape |
|---|---|
| `scannet/scene0490_00::13::36559` | Target-category guard correctly preserves the demonstrative target `this white board` (`target_id=13`); the agent misread chair-context evidence as the referent and then submitted `-1`. |
| `scannet/scene0629_00::2::19124` | Target-category guard treats generic `object` as target in `The object is a fully closed door.` and blocks the evidence-backed door proposal. |
| `scannet/scene0651_00::7::6342` | Unsupported `same side as` spatial semantics; agent resolves visually and submits `-1`, but target is a chair. |

Reading: the guard/extractor fixes recover part of the `cfee0ef` regression
without changing the no-GT evidence contract. The full rerun is +10 correct vs
`cfee0ef`, and final failed statuses drop from 10 to 3. It is still below the
original `220f128` no-initial-keyframes score (64.33 %) and below the honest
v9.3 text-first baseline (66.67 %). The next improvement bucket remains
guard/tool semantics, especially generic target nouns (`object`), agent
compliance after a correct target-category block, and unsupported relations such
as `same side as`, `behind`, `in_front_of`, and `across from`.

### Guard target-semantics probe: 2f9afe4

This probe follows a 15-case subagent audit. The code change is intentionally
small and no-GT: target-category extraction now treats generic shell heads such
as `object is a/an X` as category `X`, while evidence-frame left/right checks
are scoped to the original query text rather than direction words introduced
only in the agent rationale. The probe sample-id file used only sample-id
strings, not target/category metadata.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `2f9afe4` |
| Run-time code commit | `2f9afe4` - no worktree drift |
| Probe IDs | `/tmp/nr3d_guard_tight_probe_ids_2f9afe4.json` |
| Output dir | `tmp/nr3d_eval_v10_guard_tight_probe_20260519_2f9afe4/` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_guard_tight_probe_20260519_2f9afe4/leaderboard_metrics.json` |
| SQLite run id | `v10_guard_tight_probe_20260519` |
| Workers | 2 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |

Commands:

```bash
printf '%s\n' \
  '["scannet/scene0629_00::2::19124","scannet/scene0389_00::2::41296"]' \
  > /tmp/nr3d_guard_tight_probe_ids_2f9afe4.json

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids /tmp/nr3d_guard_tight_probe_ids_2f9afe4.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v10_guard_tight_probe_20260519_2f9afe4 \
  --workers 2 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v10_guard_tight_probe_20260519_2f9afe4/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids /tmp/nr3d_guard_tight_probe_ids_2f9afe4.json \
  --output tmp/nr3d_eval_v10_guard_tight_probe_20260519_2f9afe4/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_guard_tight_probe_20260519_2f9afe4 \
  --run-id v10_guard_tight_probe_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 2f9afe4 \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_guard_tight_probe_20260519_2f9afe4/leaderboard_metrics.json \
  --notes "Two-case probe after guard target-semantics fix; sample ids omit target metadata; checks recovered closest-to case and generic object-door case."
```

Probe metrics:

| Metric | Value |
|---|---:|
| n | 2 |
| classification_acc_filtered | 50.00 |
| Easy | 50.00 |
| Hard | 0.00 |
| V-Dep | 0.00 |
| V-Indep | 50.00 |

Case outcomes:

| Sample | a6f6077 outcome | 2f9afe4 outcome | Reading |
|---|---|---|---|
| `scannet/scene0629_00::2::19124` | final failed; `TARGET_CATEGORY_GUARD` expected `object` and blocked door proposal `#2` | completed with `selected_object_id=0`, IoU 0.0; guard now expected/submitted category `door` | Guard bug is fixed, but visual state discrimination between two door proposals remains wrong. |
| `scannet/scene0389_00::2::41296` | completed wrong with `selected_object_id=3`; EFG/TADG interaction overrode `compare_proposals_spatial` rank-1 | completed with `selected_object_id=2`, IoU 1.0; compare rank `[2, 3]`, EFG/TADG did not block | Recovered. Query relation `closest_to` evidence is no longer displaced by rationale-only left/right wording. |

Reading: this is a diagnostic probe, not a leaderboard row. It confirms the
guard-semantics change recovers the closest-to regression and removes the
generic-object category deadlock, but the closed-door sample still needs a
visual/tool improvement for door-open/closed attributes.

### Cabinet-anchor target probe: 6cb6844

This probe follows the next 15-case subagent audit, focused on unsupported
spatial relations, visual attributes, and target/anchor separation. The code
change remains no-GT: `target_category_guard` aliases `kitchen cabinet(s)`,
`cabinet(s)`, and `cupboard(s)` into the same target category, recognizes
positional heads such as `top left of the cabinets`, and prevents a fridge
anchor from becoming the final answer when the query target is a cabinet. A
post-probe unit-only follow-up at `c94f5e1` also lets generic
`object you are looking for is X` fall through to the complement extractor when
the legacy `looking for` explicit-target regex finds no valid category.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `6cb6844` |
| Run-time code commit | `6cb6844` - no worktree drift |
| Probe IDs | `/tmp/nr3d_cabinet_anchor_probe_ids_6cb6844.json` |
| Output dir | `tmp/nr3d_eval_v10_cabinet_anchor_probe_20260519_6cb6844/` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_cabinet_anchor_probe_20260519_6cb6844/leaderboard_metrics.json` |
| SQLite run id | `v10_cabinet_anchor_probe_20260519` |
| Workers | 2 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |

Commands:

```bash
printf '%s\n' \
  '["scannet/scene0164_00::14::23053","scannet/scene0149_00::8::28069"]' \
  > /tmp/nr3d_cabinet_anchor_probe_ids_6cb6844.json

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids /tmp/nr3d_cabinet_anchor_probe_ids_6cb6844.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_v10_cabinet_anchor_probe_20260519_6cb6844 \
  --workers 2 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v10_cabinet_anchor_probe_20260519_6cb6844/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids /tmp/nr3d_cabinet_anchor_probe_ids_6cb6844.json \
  --output tmp/nr3d_eval_v10_cabinet_anchor_probe_20260519_6cb6844/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_cabinet_anchor_probe_20260519_6cb6844 \
  --run-id v10_cabinet_anchor_probe_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 6cb6844 \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_cabinet_anchor_probe_20260519_6cb6844/leaderboard_metrics.json \
  --notes "Two-case probe after cabinet/anchor target-category fix; sample ids omit target metadata; checks scene0164 cabinet-vs-fridge guard path and scene0149 kitchen-cabinet relation case."
```

Probe metrics:

| Metric | Value |
|---|---:|
| n | 2 |
| classification_acc_filtered | 50.00 |
| Easy | 0.00 |
| Hard | 50.00 |
| V-Dep | 50.00 |
| V-Indep | 0.00 |

Case outcomes:

| Sample | a6f6077 outcome | 6cb6844 outcome | Reading |
|---|---|---|---|
| `scannet/scene0149_00::8::28069` | completed wrong with `selected_object_id=4`; the agent treated lower/side cabinet co-occurrence as enough for `directly over/contains microwave` | completed with `selected_object_id=8`, IoU 1.0 | Recovered. The alias pool and revised target-category behavior kept the agent on cabinet candidates, but the deeper fix still needs explicit `over/contains` relation verification. |
| `scannet/scene0164_00::14::23053` | completed wrong with `selected_object_id=16` refrigerator after EFG pushed the agent from cabinet `#15` to the fridge anchor | completed with `selected_object_id=15`, IoU 0.0072; TCG expected/submitted `kitchen cabinet`, EFG did not recommend the refrigerator | Partially fixed. Target/anchor separation works, but the agent still selects the wrong cabinet within the cabinet group; group-internal `top-left` ranking remains unsolved. |

Reading: this is a diagnostic probe, not a leaderboard row. It shows the
cabinet/fridge target-category guard path is fixed enough to prevent anchor
substitution and recover one audited kitchen-cabinet relation case. Remaining
work from this 15-case audit is broader: relation verifiers for
`behind/in_front_of/opposite/across_from`, composite `over/contains/on`
relations, open/closed door crops, and positive-final candidate coverage for
view-dependent left/right cases.

### View-dependent side-evidence probe: 23f68de

This probe follows the next 15-case subagent audit. The audited failures split
into three buckets: view-dependent left/right evidence, composite target-anchor
relations, and group-internal ordinal / row-level reasoning. The code change is
prompt/skill-only and remains no-GT: the shared scene exploration playbook and
VG grounding playbooks now require a single viewer/anchor frame for
view-dependent left/right queries, recommend `select_by_proposal(...,
require_all=True)` for small co-visible candidate sets, and explicitly forbid
combining screen-left/right evidence from different camera viewpoints or
different candidate pairs.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `23f68de` |
| Run-time code commit | `23f68de` - no worktree drift |
| Probe IDs | `tmp/nr3d_artifacts/v10_viewdep15_probe_ids_23f68de.json` |
| Output dir | `tmp/nr3d_eval_v10_viewdep15_probe_20260519_23f68de/` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_viewdep15_probe_20260519_23f68de/leaderboard_metrics.json` |
| SQLite run id | `v10_viewdep15_probe_20260519` |
| Workers | 5 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |

Commands:

```bash
tmux new-session -d -s nr3d_v10_viewdep15_probe_23f68de \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
     --sample-ids tmp/nr3d_artifacts/v10_viewdep15_probe_ids_23f68de.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_viewdep15_probe_20260519_23f68de \
     --workers 5 \
     --sample-retries 2 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_v10_viewdep15_probe_23f68de.log"

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v10_viewdep15_probe_20260519_23f68de/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v10_viewdep15_probe_ids_23f68de.json \
  --output tmp/nr3d_eval_v10_viewdep15_probe_20260519_23f68de/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_viewdep15_probe_20260519_23f68de \
  --run-id v10_viewdep15_probe_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 23f68de \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_viewdep15_probe_20260519_23f68de/leaderboard_metrics.json \
  --notes "15-case probe after view-dependent side-evidence playbook tightening from a new 15-case subagent audit; sample ids omit target metadata; checks view-dependent side, composite relation, and ordinal/row failure buckets." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe metrics:

| Metric | Value |
|---|---:|
| n | 15 |
| classification_acc_filtered | 33.33 |
| Easy | 0.00 |
| Hard | 33.33 |
| V-Dep | 33.33 |
| V-Indep | 0.00 |

Case outcomes vs `a6f6077`:

| Sample | a6f6077 selected | 23f68de selected | Reading |
|---|---:|---:|---|
| `scannet/scene0095_00::24::19560` | 20 | 24 | Recovered. Prompt caused a co-visible all-mouse frame via `require_all=True`, avoiding cross-view pairwise leftness. |
| `scannet/scene0187_00::13::24038` | 0 | 13 | Recovered. The agent kept the table-facing frame as the ordering frame. |
| `scannet/scene0423_00::3::23715` | 1 | 3 | Recovered. The "two closest" subset was preserved before applying right-from-behind. |
| `scannet/scene0025_00::1::6057` | 0 | 1 | Recovered. The prompt reduced EFG over-steering from a non-left/right query and kept the keyboard-monitor relation in focus. |
| `scannet/scene0552_00::10::35833` | 9 | 10 | Recovered. The nested table relation was handled in the intended role order. |
| `scannet/scene0196_00::3::16361` | 1 | 1 | Still wrong. Door-view candidate coverage remains incomplete. |
| `scannet/scene0221_00::39::34411` | 40 | 40 | Still wrong. The direction word modifies `left bed`, not target `pillow`; EFG still treats it as target-left. |
| `scannet/scene0149_00::21::1743` | 4 | 9 | Still wrong. Cupboard/cabinet synonym coverage and under-counter / pantry relation binding remain missing. |
| `scannet/scene0231_00::16::3011` | 20 | 20 | Still wrong. Needs role-bound armchair-facing-kitchen then picture-above-anchor verification. |
| `scannet/scene0249_00::23::30460` | 27 | 27 | Still wrong. Superlative / visual attribute coverage across all trash cans remains missing. |
| `scannet/scene0030_00::18::30641` | 19 | 19 | Still wrong. Needs group-ordinal closure for "these 3". |
| `scannet/scene0030_00::23::28411` | 81 | 73 | Still wrong. Singular/plural book category and contained-in-selected-bookcase binding remain missing. |
| `scannet/scene0208_00::106::14558` | 6 | failed | Still blocked by target-category guard parsing `wall` as target instead of `bookshelf`. |
| `scannet/scene0307_00::15::33619` | 24 | 24 | Still wrong. Wall-row shelf candidate coverage remains incomplete. |
| `scannet/scene0500_00::25::13471` | 23 | 23 | Still wrong. Same-wall middle-window grouping remains incomplete. |

Reading: this is a diagnostic probe, not a leaderboard row. The playbook-only
change is useful: it recovers 5 / 15 audited failures without any GT input and
without code/tool changes. The remaining failures point to harder work:
role-bound composite relation evidence, noun-scoped EFG direction parsing,
target-category guard operative-clause parsing, and candidate-closure gates for
ordinal / superlative / wall-row queries.

### Bookshelf target-category guard probe: c5a75ae

This one-case probe follows the `23f68de` 15-case audit: the remaining
`scene0208_00::106::14558` failure was not a visual miss after the prompt
change, but a `TARGET_CATEGORY_GUARD` parse bug. The query contains context
phrases like "that wall" and "angled wall", followed by the operative clause
"Find the bookshelf...". Before this fix, demonstrative context heads were
evaluated before explicit action targets, so the guard expected `wall` and
blocked bookshelf submissions. Commit `c5a75ae` makes explicit target actions
(`find`, `select`, `choose`, `want`, etc.) take precedence over demonstrative
context heads.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `c5a75ae` |
| Run-time code commit | `c5a75ae` - no worktree drift |
| Probe IDs | `tmp/nr3d_artifacts/v10_bookshelf_guard_probe_ids_c5a75ae.json` |
| Output dir | `tmp/nr3d_eval_v10_bookshelf_guard_probe_20260519_c5a75ae/` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_bookshelf_guard_probe_20260519_c5a75ae/leaderboard_metrics.json` |
| SQLite run id | `v10_bookshelf_guard_probe_20260519` |
| Workers | 1 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |

Commands:

```bash
tmux new-session -d -s nr3d_bookshelf_guard_probe_c5a75ae \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
     --sample-ids tmp/nr3d_artifacts/v10_bookshelf_guard_probe_ids_c5a75ae.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_bookshelf_guard_probe_20260519_c5a75ae \
     --workers 1 \
     --sample-retries 2 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_bookshelf_guard_probe_c5a75ae.log"

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v10_bookshelf_guard_probe_20260519_c5a75ae/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v10_bookshelf_guard_probe_ids_c5a75ae.json \
  --output tmp/nr3d_eval_v10_bookshelf_guard_probe_20260519_c5a75ae/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_bookshelf_guard_probe_20260519_c5a75ae \
  --run-id v10_bookshelf_guard_probe_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit c5a75ae \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_bookshelf_guard_probe_20260519_c5a75ae/leaderboard_metrics.json \
  --notes "One-case probe after target-category guard explicit-target priority fix; checks scene0208 bookshelf query that was previously parsed as wall and failed." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe metrics:

| Metric | Value |
|---|---:|
| n | 1 |
| classification_acc_filtered | 100.00 |
| Hard | 100.00 |
| V-Dep | 100.00 |

Case outcome:

| Sample | 23f68de outcome | c5a75ae outcome | Reading |
|---|---|---|---|
| `scannet/scene0208_00::106::14558` | `failed`; guard expected `wall`, blocked bookshelf submissions, and the run fell into no-match / wall fallback | completed with `selected_object_id=106`, IoU 1.0; submit trace shows expected/submitted category `bookshelf` | Recovered. The target-category guard now follows the operative `Find the bookshelf` clause instead of the context `that wall` demonstratives. |

Reading: diagnostic only. This closes one of the 15-case audit's hardest guard
failures without GT inputs. It does not address the larger candidate-closure
and role-bound relation buckets still open from the same audit.

### Guard-scope 15-case probe: 6355781

This probe follows a fresh 15-case subagent audit over failures from
`v10_no_gt_fixes_strat600_20260519`. The audit found two no-GT guard errors
that could be fixed locally:

- Target-category guard parsed an earlier "want the bed" context as the target
  even when a later option head said "The pillow is...".
- Evidence-frame guard treated anchor / room-side phrases such as "on the
  larger desk to the left" and "left side of the room" as image-left
  constraints on the submitted target.

Commit `6355781` fixes those two guard scopes and keeps the evidence contract
unchanged: no initial keyframes, no pending-image side channel, and first-person
tool images only via `tool_trace.image_metadata`.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `6355781` |
| Run-time code commit | `6355781` - no worktree drift |
| Probe IDs used at launch | `/tmp/nr3d_guard_scope_probe15_sample_ids_20260519.json` |
| Durable probe IDs | `docs/benchmark/nr3d/assets/v10_guard_scope_probe15_sample_ids_20260519.json` |
| Durable probe IDs MD5 | `6663813191523d007d71457346fbe505` |
| Output dir | `tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781/` |
| Run log | `tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781/run.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781/side_by_side.json` |
| Side-by-side MD5 | `ae597f8851e88cf796284c3c8a769d71` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `13295426da7bd3de9cdbeea8752671a7` |
| SQLite run id | `v10_guard_scope_probe15_20260519` |
| Workers | 15 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |

The launch-time `/tmp` sample-id file included audit helper fields for human
review, but `run_nr3d_vg_side_by_side.load_sample_ids()` consumes only the
`sample_id` field. The durable copy above is therefore stored as a string-only
sample-id list for future reruns.

Commands:

```bash
tmux new-session -d -s nr3d_guard_probe15_6355781 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   mkdir -p tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781 && \
   PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids /tmp/nr3d_guard_scope_probe15_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781 \
     --workers 15 \
     --sample-retries 2 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781/run.log"

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids /tmp/nr3d_guard_scope_probe15_sample_ids_20260519.json \
  --output tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781 \
  --run-id v10_guard_scope_probe15_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 6355781 \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_guard_scope_probe15_20260519_6355781/leaderboard_metrics.json \
  --notes "Focused 15-case probe after TCG later-option target-head and EFG side-scope fixes; no GT inputs, standard guards."
```

Probe stability:

- 15 / 15 samples completed.
- 0 missing `selected_object_id`.
- 0 Tracebacks / logged Exceptions.
- Guard blocks in final-submit traces: TADG 2, TCG 1, EFG 4, no-match 0.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 15 |
| classification_acc_filtered | 53.33 |
| Easy | 0.00 |
| Hard | 53.33 |
| V-Dep | 53.33 |
| V-Indep | 0.00 |

Case outcomes vs `v10_no_gt_fixes_strat600_20260519`:

| Sample | a6f6077 selected | 6355781 selected | Reading |
|---|---:|---:|---|
| `scannet/scene0025_00::26::38922` | 2 | 26 | Recovered. EFG no longer treats "desk to the left" as a target image-left constraint. |
| `scannet/scene0095_00::29::20881` | 23 | 29 | Recovered. The monitor/keyboard relation stays scoped to the target candidate instead of drifting to the monitor anchor. |
| `scannet/scene0221_00::18::33317` | 19 | 18 | Recovered. Side evidence remains bound to the bag candidate instead of being overruled by broad room-side wording. |
| `scannet/scene0221_00::46::36120` | 8 | 46 | Recovered. TCG now chooses the later `pillow` option head rather than the earlier `bed` context. |
| `scannet/scene0231_00::55::11667` | 56 | 55 | Recovered. Same-category window comparison improved after the guard stopped over-constraining context-side language. |
| `scannet/scene0338_00::21::35740` | 19 | 21 | Recovered. The agent kept the co-visible box frame and selected the back-left box. |
| `scannet/scene0351_00::22::23771` | 23 | 22 | Recovered. "Left side of room" is no longer interpreted as image-left for the final monitor id. |
| `scannet/scene0565_00::6::7388` | 7 | 6 | Recovered. The side relation is resolved within the relevant marked box cluster. |
| `scannet/scene0081_00::0::38576` | 7 | 7 | Still wrong. Needs viewer-frame candidate closure for couch-side language. |
| `scannet/scene0231_00::55::41048` | 25 | 25 | Still wrong. Window superlative / room-side grouping remains unresolved. |
| `scannet/scene0246_00::39::37886` | 38 | 42 | Still wrong. Pillow/headboard side language still needs noun-scoped anchor handling beyond this guard fix. |
| `scannet/scene0329_00::31::21581` | 32 | 30 | Still wrong. Monitor/keyboard group ranking needs relation-bound candidate closure. |
| `scannet/scene0329_00::33::13927` | 32 | 32 | Still wrong. Same monitor cluster remains ambiguous without a stronger row/side verifier. |
| `scannet/scene0494_00::3::5674` | 1 | 1 | Still wrong. Chair "facing/behind" relation remains a visual/geometry gap. |
| `scannet/scene0565_00::7::7026` | 8 | 8 | Still wrong. Box ordinal / side comparison still picks the neighboring candidate. |

Reading: this is a diagnostic probe, not a leaderboard row. It recovers 8 / 15
freshly audited failures without introducing any GT evidence path. The strongest
next buckets are candidate closure for ordinal/superlative same-category sets,
role-bound relation evidence for monitor/keyboard and chair-facing queries, and
noun-scoped EFG for anchor-side phrases like pillow/bed/headboard.

### Candidate-closure 15-case probe: 06b8d58

This probe follows the next 15-case subagent audit over failures from
`v10_no_gt_fixes_strat600_20260519`. The audit groups were:

- view-dependent side/front/back anchors;
- ordinal / superlative / candidate-closure failures;
- composite target-anchor relations.

The common no-GT failure was premature finalization after seeing only part of a
small same-category candidate set. Commit `06b8d58` adds an EFG sub-check: when
the query has left/right, ordinal/superlative, or target-anchor relation cues
and the same-category candidate set is small, `submit_final` is soft-blocked
until every same-category candidate has appeared in marked evidence at least
once. The guard uses only the proposal pool, query text, and the agent's own
`mark_frame_with_bbox` trace; it does not use target ids or GT boxes.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `06b8d58` |
| Run-time code commit | `06b8d58` - no worktree drift |
| Probe IDs | `docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json` |
| Probe IDs MD5 | `8a9f5489e331631a2e10d13010effae5` |
| Output dir | `tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58/` |
| Run log | `tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58/run.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58/side_by_side.json` |
| Side-by-side MD5 | `6377587d5c4145126c18ef03ef4f4549` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `c0f824f3b1bebae8aff9156f112dd99a` |
| SQLite run id | `v10_candidate_closure_probe15_20260519` |
| Workers | 15 |
| Sample retries | 2 |
| Guards | TADG + no-match + evidence-frame |

Commands:

```bash
tmux new-session -d -s nr3d_candidate_closure_probe15_06b8d58 \
  "cd /Users/bytedance/project/3DVLMReasoning && \
   mkdir -p tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58 && \
   PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58 \
     --workers 15 \
     --sample-retries 2 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58/run.log"

PYTHONPATH=src .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
  --side-by-side tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json \
  --output tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58 \
  --run-id v10_candidate_closure_probe15_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 06b8d58 \
  --backend pack_v1 \
  --judge-model none \
  --leaderboard-metrics tmp/nr3d_eval_v10_candidate_closure_probe15_20260519_06b8d58/leaderboard_metrics.json \
  --notes "15-case probe after EFG same-category candidate-closure guard; no GT inputs, standard guards."
```

Probe stability:

- 15 / 15 samples completed.
- 0 missing `selected_object_id`.
- 0 Tracebacks / logged Exceptions.
- Candidate-closure EFG messages fired in 7 submit traces.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 15 |
| classification_acc_filtered | 33.33 |
| Easy | 28.57 |
| Hard | 37.50 |
| V-Dep | 28.57 |
| V-Indep | 37.50 |

Case outcomes vs `v10_no_gt_fixes_strat600_20260519`:

| Sample | a6f6077 selected | 06b8d58 selected | Reading |
|---|---:|---:|---|
| `scannet/scene0011_00::23::15020` | 24 | 23 | Recovered. Candidate-closure blocked the one-window answer until the other window candidate was marked. |
| `scannet/scene0025_00::18::35283` | 19 | 18 | Recovered. The run kept the complete trash-can candidate comparison instead of side-frame drift. |
| `scannet/scene0063_00::5::11741` | 4 | 5 | Recovered. Additional chair evidence prevented finalizing from a partial chair cluster under the missing-TV anchor. |
| `scannet/scene0077_00::1::22194` | 2 | 1 | Recovered. The agent could override the weak `below` geometry after complete printer/window marked evidence. |
| `scannet/scene0095_00::20::26800` | 24 | 20 | Recovered. Mouse candidate closure corrected the cross-view "second from top" ordering. |
| `scannet/scene0011_00::20::28662` | 28 | 1 | Still wrong. Needs facing-normalized right/left over cabinet candidates; closure alone can still drift to a kitchen-cabinet alias. |
| `scannet/scene0025_00::17::37165` | 38 | 38 | Still wrong. The file-cabinet query needs `end of desk` / `whiteboard behind it` relation support, not just closure. |
| `scannet/scene0025_00::30::15602` | 34 | 34 | Still wrong. Keyboard attribute / in-front-of-chair relation remains unresolved; closure did not force the black keyboard candidate. |
| `scannet/scene0030_00::0::23137` | 1 | 1 | Still wrong. Needs center/facing-away orientation support around chalkboard/table. |
| `scannet/scene0030_00::25::23351` | 80 | 80 | Still wrong. Large book/book-row set exceeds the small-set closure threshold; needs row/shelf comparator. |
| `scannet/scene0046_00::50::29027` | 18 | 18 | Still wrong. Closure fired, but anchor subtype "square green chair" still picked the wrong anchor relation. |
| `scannet/scene0050_00::32::30122` | 9 | 9 | Still wrong. Closure fired but door/doors group relation still needs closest-to-group semantics. |
| `scannet/scene0063_00::10::1029` | 9 | 9 | Still wrong. Closure fired but singular cabinet vs plural cabinet-run ranking remains unresolved. |
| `scannet/scene0081_00::0::10619` | 7 | 7 | Still wrong. Closure fired, but couch-left from a facing frame still needs a better viewpoint/side tool. |
| `scannet/scene0084_00::43::31587` | 39 | 34 | Still wrong. Closure fired but opposite-side-of-two-rails topology is unsupported. |

Reading: diagnostic only. The candidate-closure guard is useful and no-GT: it
recovers 5 / 15 failures from a fresh audited slice, with all samples completed.
It is not enough for topology, anchor-subtype grounding, or large row/ordinal
sets. The next practical tool-level targets are a small-set ordinal comparator
(`top/bottom/second`) and a relation helper for group/anchor semantics such as
`closest_to_group`, `opposite_side_of_group`, and facing-normalized side.

### Rationale/payload consistency probe: be3c4fc

This probe follows the 25-case topology / relation audit. One audited failure
(`scannet/scene0678_00::21::1134`) had enough no-GT evidence and the final
rationale explicitly selected `#21`, but the structured payload still submitted
`proposal_id=31`. Commit `be3c4fc` adds a no-GT submit-time consistency guard:
if the final rationale rules out the submitted proposal or names a different
final proposal id, `submit_final` soft-blocks instead of accepting the stale
payload. The guard reads only `payload.proposal_id` and the agent-written
`rationale`; it does not inspect target ids, GT bboxes, target visibility, or
metrics.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `be3c4fc` |
| Run-time code commit | `be3c4fc` - no worktree drift |
| Probe IDs used at run time | `tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json` |
| Durable probe IDs | `docs/benchmark/nr3d/assets/v10_rationale_payload_probe3_sample_ids_20260519.json` |
| Durable probe IDs MD5 | `4ef53f60e189403c1d66d718534d8e60` |
| Output dir | `tmp/nr3d_eval_v10_rationale_payload_probe3_20260519_be3c4fc/` |
| Run log | `/tmp/nr3d_rationale_payload_probe3_be3c4fc.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_rationale_payload_probe3_20260519_be3c4fc/side_by_side.json` |
| Side-by-side MD5 | `1e821cb1f03df277ac6c12b4290c818d` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_rationale_payload_probe3_20260519_be3c4fc/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `0f54b85ed95ed079316a81e0abc90851` |
| SQLite run id | `v10_rationale_payload_probe3_20260519` |
| Workers | 3 |
| Sample retries | 0 |
| Guards | TADG + no-match + evidence-frame + rationale/payload |

Commands:

```bash
tmux new-session -d -s nr3d-rpg-probe3-be3c4fc \
  "cd /Users/bytedance/project/3DVLMReasoning && bash -lc 'set -euo pipefail; \
   export PYTHONPATH=src PYTHONUNBUFFERED=1; \
   .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_rationale_payload_probe3_20260519_be3c4fc \
     --workers 3 \
     --sample-retries 0 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_rationale_payload_probe3_be3c4fc.log; \
   .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
     --side-by-side tmp/nr3d_eval_v10_rationale_payload_probe3_20260519_be3c4fc/side_by_side.json \
     --nr3d-data-root data/nr3d \
     --phase8-data-root data/nr3d/scannet \
     --sample-ids tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json \
     --output tmp/nr3d_eval_v10_rationale_payload_probe3_20260519_be3c4fc/leaderboard_metrics.json \
     --canonical-filter true \
     2>&1 | tee -a /tmp/nr3d_rationale_payload_probe3_be3c4fc.log'"

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_rationale_payload_probe3_20260519_be3c4fc \
  --run-id v10_rationale_payload_probe3_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit be3c4fc \
  --backend pack_v1 \
  --leaderboard-metrics tmp/nr3d_eval_v10_rationale_payload_probe3_20260519_be3c4fc/leaderboard_metrics.json \
  --notes "Diagnostic 3-case probe for rationale/payload consistency guard; sample ids from prior strat600 stale-payload offline scan." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe stability:

- 3 / 3 samples completed.
- 0 Tracebacks / logged Exceptions in the run log.
- Offline scan over the previous `a6f6077` strat600 `submit_final` traces found
  3 / 600 payload/rationale contradictions after false-positive tightening.
- In this live probe the guard did not need to fire: the model submitted
  self-consistent payloads on all three cases. This run therefore validates
  integration and outcome, while the unit tests and offline scan validate the
  stale-payload block path directly.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 3 |
| classification_acc_filtered | 66.67 |
| Easy | 50.00 |
| Hard | 100.00 |
| V-Dep | 0.00 |
| V-Indep | 100.00 |

Case outcomes vs `v10_no_gt_fixes_strat600_20260519`:

| Sample | a6f6077 selected | be3c4fc selected | Reading |
|---|---:|---:|---|
| `scannet/scene0678_00::21::1134` | 31 | 21 | Recovered. The old trace would now be blocked because the rationale ruled out `#31` and selected `#21`; the live run directly submitted `#21`. |
| `scannet/scene0222_00::20::35738` | 19 | 20 | Recovered. The old trace would now be blocked because the rationale named final `#20` while payload stayed `#19`; the live run directly submitted `#20`. |
| `scannet/scene0663_00::6::33306` | 33 | 33 | Still wrong. The old trace would now be blocked when it argues for absent / no-match while payload stays `#33`, but the live run is self-consistent and still prefers the plain chair over office chair `#6`; this needs category-alias / anchor-relation work, not a stale-payload guard. |

Reading: diagnostic only. The guard is a low-blast-radius no-GT correctness
check for stale structured payloads. It can recover cases only when the agent
has already reasoned to the right id in prose but failed to update the payload.
It does not solve topology, category aliasing, or anchor-relation ambiguity.

### Generic subtype category probe: dbd9adb

This probe follows the remaining miss in the rationale/payload consistency
subset. The failed sample says "chair", while the correct proposal is labeled
`office chair`. Commit `dbd9adb` makes generic target-category checks asymmetric:
a generic query category such as `chair` may accept subtype labels such as
`office chair`, but a specific query category such as `office chair` does not
accept a generic `chair` answer. The VG playbooks were also updated to tell the
agent not to reject a valid subtype when the query uses a generic category word.
The implementation reads only query text, proposal labels, and agent rationale;
it does not inspect target ids, GT bboxes, target visibility, or metrics.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `dbd9adb` |
| Run-time code commit | `dbd9adb` - no worktree drift |
| Probe IDs used at run time | `tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json` |
| Durable probe IDs | `docs/benchmark/nr3d/assets/v10_rationale_payload_probe3_sample_ids_20260519.json` |
| Durable probe IDs MD5 | `4ef53f60e189403c1d66d718534d8e60` |
| Output dir | `tmp/nr3d_eval_v10_subtype_probe3_20260519_dbd9adb/` |
| Run log | `/tmp/nr3d_subtype_probe3_dbd9adb.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_subtype_probe3_20260519_dbd9adb/side_by_side.json` |
| Side-by-side MD5 | `0733fdc85f2351e61b70167a1dbdc32b` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_subtype_probe3_20260519_dbd9adb/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `0f54b85ed95ed079316a81e0abc90851` |
| SQLite run id | `v10_subtype_probe3_20260519` |
| Workers | 3 |
| Sample retries | 0 |
| Guards | TADG + no-match + evidence-frame + rationale/payload |

Commands:

```bash
tmux new-session -d -s nr3d-subtype-probe3-dbd9adb \
  "cd /Users/bytedance/project/3DVLMReasoning && bash -lc 'set -euo pipefail; \
   export PYTHONPATH=src PYTHONUNBUFFERED=1; \
   .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_subtype_probe3_20260519_dbd9adb \
     --workers 3 \
     --sample-retries 0 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_subtype_probe3_dbd9adb.log; \
   .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
     --side-by-side tmp/nr3d_eval_v10_subtype_probe3_20260519_dbd9adb/side_by_side.json \
     --nr3d-data-root data/nr3d \
     --phase8-data-root data/nr3d/scannet \
     --sample-ids tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json \
     --output tmp/nr3d_eval_v10_subtype_probe3_20260519_dbd9adb/leaderboard_metrics.json \
     --canonical-filter true \
     2>&1 | tee -a /tmp/nr3d_subtype_probe3_dbd9adb.log'"

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_subtype_probe3_20260519_dbd9adb \
  --run-id v10_subtype_probe3_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit dbd9adb \
  --backend pack_v1 \
  --leaderboard-metrics tmp/nr3d_eval_v10_subtype_probe3_20260519_dbd9adb/leaderboard_metrics.json \
  --notes "Diagnostic 3-case probe after generic chair subtype category guard/playbook update; negative on scene0663 because nested anchor desk closest to window remains unresolved." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe stability:

- 3 / 3 samples completed.
- 0 Tracebacks / logged Exceptions in the run log.
- Pre-run verification covered the target-category guard, chassis tools, TADG,
  evidence-frame guard, and VG playbook loadability: 148 tests passed.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 3 |
| classification_acc_filtered | 66.67 |
| Easy | 50.00 |
| Hard | 100.00 |
| V-Dep | 0.00 |
| V-Indep | 100.00 |

Case outcomes vs `v10_no_gt_fixes_strat600_20260519`:

| Sample | a6f6077 selected | be3c4fc selected | dbd9adb selected | Reading |
|---|---:|---:|---:|---|
| `scannet/scene0678_00::21::1134` | 31 | 21 | 21 | Still recovered; no regression. |
| `scannet/scene0222_00::20::35738` | 19 | 20 | 20 | Still recovered; no regression. |
| `scannet/scene0663_00::6::33306` | 33 | 33 | 33 | Still wrong. The subtype check no longer rejects `office chair` for a generic `chair` query, but the live trace resolves the nested anchor incorrectly: it treats desk `#4` as the desk closest to window `#5`, then selects chair `#33` behind that desk. The correct no-GT workflow is to first resolve the anchor desk candidates against the window with `compare_proposals_spatial(..., relation='closest_to')`, then rank/verify chair candidates behind the resolved desk. |

Reading: diagnostic only. The subtype change removes one false rejection path,
but the target case is now clearly a nested-anchor failure rather than a
category-alias failure. The next no-GT improvement target is a playbook/tool-flow
rule for queries shaped like "target behind/next-to [anchor] closest/farthest to
[second anchor]": resolve the anchor superlative first, then compare targets
against that resolved anchor.

### Nested-anchor flow probe: 3b675f9

This probe tests the next no-GT prompt/skill change after the subtype probe.
Commit `3b675f9` adds a nested-anchor contract to the VG grounding and spatial
disambiguation skills: for expressions like "the chair behind the desk closest
to the window", resolve the anchor candidate set first with
`compare_proposals_spatial(candidate_ids=[anchor ids], anchor_id=#window, relation='closest_to')`,
then rank / visually verify the target candidates against the resolved anchor.
The change is prompt/skill-only; it does not add GT inputs or inspect target
ids during inference.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `3b675f9` |
| Run-time code commit | `3b675f9` - no worktree drift |
| Probe IDs used at run time | `tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json` |
| Durable probe IDs | `docs/benchmark/nr3d/assets/v10_rationale_payload_probe3_sample_ids_20260519.json` |
| Durable probe IDs MD5 | `4ef53f60e189403c1d66d718534d8e60` |
| Output dir | `tmp/nr3d_eval_v10_nested_anchor_probe3_20260519_3b675f9/` |
| Run log | `/tmp/nr3d_nested_anchor_probe3_3b675f9.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_nested_anchor_probe3_20260519_3b675f9/side_by_side.json` |
| Side-by-side MD5 | `78764bc7f1c2d427b8bc96dd04ed3502` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_nested_anchor_probe3_20260519_3b675f9/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `8d054cf50fd3df9bacb064316039a23d` |
| SQLite run id | `v10_nested_anchor_probe3_20260519` |
| Workers | 3 |
| Sample retries | 0 |
| Guards | TADG + no-match + evidence-frame + rationale/payload |

Commands:

```bash
tmux new-session -d -s nr3d-nested-probe3-3b675f9 \
  "cd /Users/bytedance/project/3DVLMReasoning && bash -lc 'set -euo pipefail; \
   export PYTHONPATH=src PYTHONUNBUFFERED=1; \
   .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_nested_anchor_probe3_20260519_3b675f9 \
     --workers 3 \
     --sample-retries 0 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_nested_anchor_probe3_3b675f9.log; \
   .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
     --side-by-side tmp/nr3d_eval_v10_nested_anchor_probe3_20260519_3b675f9/side_by_side.json \
     --nr3d-data-root data/nr3d \
     --phase8-data-root data/nr3d/scannet \
     --sample-ids tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json \
     --output tmp/nr3d_eval_v10_nested_anchor_probe3_20260519_3b675f9/leaderboard_metrics.json \
     --canonical-filter true \
     2>&1 | tee -a /tmp/nr3d_nested_anchor_probe3_3b675f9.log'"

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_nested_anchor_probe3_20260519_3b675f9 \
  --run-id v10_nested_anchor_probe3_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 3b675f9 \
  --backend pack_v1 \
  --leaderboard-metrics tmp/nr3d_eval_v10_nested_anchor_probe3_20260519_3b675f9/leaderboard_metrics.json \
  --notes "Diagnostic 3-case probe after nested-anchor playbook update; recovers scene0663 chair behind desk closest to window, but regresses scene0222 pillow negated-window case." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe stability:

- 3 / 3 samples completed.
- 0 Tracebacks / logged Exceptions in the run log.
- Pre-run TDD verification:
  - RED: nested-anchor playbook contract test failed on all four VG variants.
  - GREEN: `test_playbook_v9_consistency.py` + shared playbook tests passed
    51 / 51.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 3 |
| classification_acc_filtered | 66.67 |
| Easy | 50.00 |
| Hard | 100.00 |
| V-Dep | 100.00 |
| V-Indep | 50.00 |

Case outcomes vs previous probes:

| Sample | be3c4fc selected | dbd9adb selected | 3b675f9 selected | Reading |
|---|---:|---:|---:|---|
| `scannet/scene0678_00::21::1134` | 21 | 21 | 21 | Still recovered; no regression. |
| `scannet/scene0663_00::6::33306` | 33 | 33 | 6 | Recovered. The trace now calls `compare_proposals_spatial(candidate_ids=[3,4], anchor_id=5, relation='closest_to')`, which ranks desk `#3` before desk `#4`, then marks desk `#3`, window `#5`, and office chair `#6` in frame 0 before finalizing `#6`. |
| `scannet/scene0222_00::20::35738` | 20 | 20 | 19 | Regressed. The query is a negated anchor case ("pillow on the bed not next to the windows"), not a nested-anchor superlative. The agent over-relies on BEV/window proximity and selects pillow `#19`; the next guard/prompt target should treat negated anchor relations separately from nested-anchor dependency resolution. |

Reading: partial positive. The nested-anchor contract fixes the intended
failure and produces the desired no-GT tool sequence, but the 3-case slice is
not net-positive because a negated-window case regressed. Do not promote this
as a general improvement without either a larger failed-case probe or a
follow-up fix that separates negated anchor relations from nested superlative
anchors.

### Negated-anchor prompt probe: fd1a628

This probe tests a narrow follow-up to the `scene0222` regression from
`3b675f9`. Commit `fd1a628` adds prompt/skill text requiring a marked positive
counterexample for negated anchor relations, e.g. mark the pillow/bed/window
that is next to the forbidden window before selecting the remaining pillow.
The intent was to recover `scene0222_00::20::35738` without changing tools or
adding GT inputs.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `fd1a628` |
| Run-time code commit | `fd1a628` - no worktree drift |
| Probe IDs used at run time | `tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json` |
| Durable probe IDs | `docs/benchmark/nr3d/assets/v10_rationale_payload_probe3_sample_ids_20260519.json` |
| Durable probe IDs MD5 | `4ef53f60e189403c1d66d718534d8e60` |
| Output dir | `tmp/nr3d_eval_v10_negated_anchor_probe3_20260519_fd1a628/` |
| Run log | `/tmp/nr3d_negated_anchor_probe3_fd1a628.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_negated_anchor_probe3_20260519_fd1a628/side_by_side.json` |
| Side-by-side MD5 | `dd880a11f282f382db1ae1e3c946c808` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_negated_anchor_probe3_20260519_fd1a628/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `c0a83e5efab56364886fc3ad37be49c9` |
| SQLite run id | `v10_negated_anchor_probe3_20260519` |
| Workers | 3 |
| Sample retries | 0 |
| Guards | TADG + no-match + evidence-frame + rationale/payload |

Commands:

```bash
tmux new-session -d -s nr3d-negated-probe3-fd1a628 \
  "cd /Users/bytedance/project/3DVLMReasoning && bash -lc 'set -euo pipefail; \
   export PYTHONPATH=src PYTHONUNBUFFERED=1; \
   .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_negated_anchor_probe3_20260519_fd1a628 \
     --workers 3 \
     --sample-retries 0 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_negated_anchor_probe3_fd1a628.log; \
   .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
     --side-by-side tmp/nr3d_eval_v10_negated_anchor_probe3_20260519_fd1a628/side_by_side.json \
     --nr3d-data-root data/nr3d \
     --phase8-data-root data/nr3d/scannet \
     --sample-ids tmp/nr3d_artifacts/v10_rationale_payload_probe3_sample_ids_20260519.json \
     --output tmp/nr3d_eval_v10_negated_anchor_probe3_20260519_fd1a628/leaderboard_metrics.json \
     --canonical-filter true \
     2>&1 | tee -a /tmp/nr3d_negated_anchor_probe3_fd1a628.log'"

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_negated_anchor_probe3_20260519_fd1a628 \
  --run-id v10_negated_anchor_probe3_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit fd1a628 \
  --backend pack_v1 \
  --leaderboard-metrics tmp/nr3d_eval_v10_negated_anchor_probe3_20260519_fd1a628/leaderboard_metrics.json \
  --notes "Diagnostic 3-case probe after marked-positive negated-anchor prompt; recovers scene0222 but regresses scene0663 and scene0678, so the prompt change is negative." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe stability:

- 3 / 3 samples completed.
- 0 Tracebacks / logged Exceptions in the run log.
- Pre-run TDD verification:
  - RED: negated-anchor playbook contract test failed on all four VG variants.
  - GREEN: `test_playbook_v9_consistency.py` + shared playbook tests passed
    55 / 55.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 3 |
| classification_acc_filtered | 33.33 |
| Easy | 50.00 |
| Hard | 0.00 |
| V-Dep | 0.00 |
| V-Indep | 50.00 |

Case outcomes vs `3b675f9`:

| Sample | 3b675f9 selected | fd1a628 selected | Reading |
|---|---:|---:|---|
| `scannet/scene0222_00::20::35738` | 19 | 20 | Recovered. The agent now selects the pillow on the non-window bed. It still does not mark a window frame as cleanly as the earlier `dbd9adb` run, so this is a fragile prompt recovery. |
| `scannet/scene0663_00::6::33306` | 6 | 33 | Regressed. The agent still resolves desk `#3` as closest to the window, but then accepts chair `#33` in a later frame and discounts `office chair` subtype candidates. |
| `scannet/scene0678_00::21::1134` | 21 | 8 | Regressed. The outside-door case no longer preserves the previous recovered door `#21`. |

Reading: negative. The marked-positive negated-anchor prompt recovered the
intended negated-window sample but caused two regressions on the same small
diagnostic slice. Do not keep this prompt wording as an active improvement; use
the recorded run as evidence that negated-anchor handling needs either a
narrower guard or a tool-level relation helper rather than more broad playbook
prose.

### Above/below alignment tool probe: 54198ef

Commit `54198ef` changes `compare_proposals_spatial` for vertical relations:
`above` / `below` now rank candidates on the correct vertical side first, then
by horizontal center alignment, then by vertical magnitude. This addresses the
audited failure pattern behind cases like "the printer right under the window",
where a lower but laterally offset object can beat the object directly under the
anchor if the tool sorts mostly by z distance.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `54198ef` |
| Run-time code commit | `54198ef` - no worktree drift |
| Probe IDs | `docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json` |
| Probe IDs MD5 | `8a9f5489e331631a2e10d13010effae5` |
| Output dir | `tmp/nr3d_eval_v10_below_align_probe15_20260519_54198ef/` |
| Run log | `/tmp/nr3d_below_align_probe15_54198ef.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_below_align_probe15_20260519_54198ef/side_by_side.json` |
| Side-by-side MD5 | `b65964a4c506b846ef688248a65f7839` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_below_align_probe15_20260519_54198ef/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `a5d0716dd1ccd241f927e4365719ee83` |
| SQLite run id | `v10_below_align_probe15_20260519` |
| Workers | 8 |
| Sample retries | 0 |
| Guards | TADG + no-match + evidence-frame + rationale/payload |

Commands:

```bash
tmux new-session -d -s nr3d-below-align-probe15-54198ef \
  "cd /Users/bytedance/project/3DVLMReasoning && bash -lc 'set -euo pipefail; \
   export PYTHONPATH=src PYTHONUNBUFFERED=1; \
   .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_below_align_probe15_20260519_54198ef \
     --workers 8 \
     --sample-retries 0 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_below_align_probe15_54198ef.log; \
   .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
     --side-by-side tmp/nr3d_eval_v10_below_align_probe15_20260519_54198ef/side_by_side.json \
     --nr3d-data-root data/nr3d \
     --phase8-data-root data/nr3d/scannet \
     --sample-ids docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json \
     --output tmp/nr3d_eval_v10_below_align_probe15_20260519_54198ef/leaderboard_metrics.json \
     --canonical-filter true \
     2>&1 | tee -a /tmp/nr3d_below_align_probe15_54198ef.log'"

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_below_align_probe15_20260519_54198ef \
  --run-id v10_below_align_probe15_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 54198ef \
  --backend pack_v1 \
  --leaderboard-metrics tmp/nr3d_eval_v10_below_align_probe15_20260519_54198ef/leaderboard_metrics.json \
  --notes "Diagnostic 15-case probe after compare_proposals_spatial above/below horizontal-alignment ranking. Probe completes 15/15 and improves 5/15 -> 8/15 vs candidate-closure run, but traces do not exercise above/below compare_proposals_spatial, so live gain is not direct attribution." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe stability:

- 15 / 15 samples completed.
- 0 Tracebacks / logged Exceptions in the run log.
- Pre-run deterministic verification:
  - `test_compare_proposals_spatial_below_prefers_horizontal_alignment`
    fails before the code change and passes after it.
  - `test_compare_proposals_spatial_above_prefers_horizontal_alignment`
    fails before the code change and passes after it.
  - `PYTHONPATH=src .venv/bin/python -m pytest
    src/agents/packs/vg_embodiedscan/tests/test_tools.py
    src/agents/tests/test_tadg.py -q` passed 59 / 59.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 15 |
| classification_acc_filtered | 53.33 |
| Easy | 57.14 |
| Hard | 50.00 |
| V-Dep | 71.43 |
| V-Indep | 37.50 |

Case outcomes:

| Sample | a6f6077 selected | 06b8d58 selected | 54198ef selected | Reading |
|---|---:|---:|---:|---|
| `scannet/scene0011_00::20::28662` | 28 | 1 | 20 | Recovered vs both prior runs. Trace uses marked first-person cabinet views, not the changed vertical comparator. |
| `scannet/scene0030_00::0::23137` | 1 | 1 | 0 | Recovered vs both prior runs. Trace resolves the longer chalkboard and chair alignment visually; no vertical comparator call. |
| `scannet/scene0025_00::17::37165` | 38 | 38 | 17 | Recovered vs both prior runs. Trace compares file cabinets at desk ends visually; no vertical comparator call. |
| `scannet/scene0077_00::1::22194` | 2 | 1 | 1 | Still recovered. This is the audited "right under window" case, but the live trace resolves it through marked RGB evidence and does not call `compare_proposals_spatial`. |

Reading: the deterministic tool fix is valid and unit-covered, but this live
probe should not be used as direct causal evidence for the tool change because
none of the 15 traces called `compare_proposals_spatial` with `above` or
`below`. Treat the 8 / 15 result as a no-regression diagnostic on the same
failed-case slice. The next improvement target is prompt/skill routing: make
the agent explicitly use `compare_proposals_spatial` for above / below / under
candidate ranking before visual verification.

### Vertical relation routing prompt probe: 61042c3

Commit `61042c3` adds broad VG playbook / spatial-disambiguation prose telling
the agent to map "under", "right under", and "beneath" to
`relation='below'`, and "above", "over", and "on top of" to
`relation='above'`, then use `compare_proposals_spatial` before marked-frame
verification. This directly targets the caveat from the `54198ef` probe: the
tool was fixed, but the agent did not call it for vertical relations.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `61042c3` |
| Run-time code commit | `61042c3` - no worktree drift |
| Probe IDs | `docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json` |
| Output dir | `tmp/nr3d_eval_v10_vertical_route_probe15_20260519_61042c3/` |
| Run log | `/tmp/nr3d_vertical_route_probe15_61042c3.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_vertical_route_probe15_20260519_61042c3/side_by_side.json` |
| Side-by-side MD5 | `b3a0080d632251bf302443bf404abada` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_vertical_route_probe15_20260519_61042c3/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `ca7ce4384396ef0a96632884ad5c2311` |
| SQLite run id | `v10_vertical_route_probe15_20260519` |
| Workers | 8 |
| Sample retries | 0 |
| Guards | TADG + no-match + evidence-frame + rationale/payload |

Commands:

```bash
tmux new-session -d -s nr3d-vertical-route-probe15-61042c3 \
  "cd /Users/bytedance/project/3DVLMReasoning && bash -lc 'set -euo pipefail; \
   export PYTHONPATH=src PYTHONUNBUFFERED=1; \
   .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_vertical_route_probe15_20260519_61042c3 \
     --workers 8 \
     --sample-retries 0 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_vertical_route_probe15_61042c3.log; \
   .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
     --side-by-side tmp/nr3d_eval_v10_vertical_route_probe15_20260519_61042c3/side_by_side.json \
     --nr3d-data-root data/nr3d \
     --phase8-data-root data/nr3d/scannet \
     --sample-ids docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json \
     --output tmp/nr3d_eval_v10_vertical_route_probe15_20260519_61042c3/leaderboard_metrics.json \
     --canonical-filter true \
     2>&1 | tee -a /tmp/nr3d_vertical_route_probe15_61042c3.log'"

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_vertical_route_probe15_20260519_61042c3 \
  --run-id v10_vertical_route_probe15_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 61042c3 \
  --backend pack_v1 \
  --leaderboard-metrics tmp/nr3d_eval_v10_vertical_route_probe15_20260519_61042c3/leaderboard_metrics.json \
  --notes "Diagnostic 15-case probe after vertical relation playbook routing; creates one below compare_proposals_spatial call and keeps scene0077 correct, but overall drops to 5/15 vs 8/15 previous probe, so the broad prompt wording is negative." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe stability:

- 15 / 15 samples completed.
- 0 Tracebacks / logged Exceptions in the run log.
- Pre-run TDD verification:
  - RED: `test_playbooks_route_vertical_relations_to_spatial_compare`
    failed on all four VG skill/playbook variants.
  - GREEN: after the prompt change, the new contract test passed on all four
    variants; full playbook tests passed 55 / 55.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 15 |
| classification_acc_filtered | 33.33 |
| Easy | 28.57 |
| Hard | 37.50 |
| V-Dep | 28.57 |
| V-Indep | 37.50 |

Tool-routing evidence:

- `scene0077_00::1::22194` now calls
  `compare_proposals_spatial(candidate_ids=[1, 2], anchor_id=3, relation='below')`.
- That tool ranks `[1, 2]` with horizontal distances `[0.376, 0.687]`, and the
  agent submits `#1` with `relation_evidence={"evidence_id": "compare_proposals_spatial:0"}`.
- Across the 15 traces, relation calls were: `closest_to` x3, `next_to` x1,
  `below` x1, `above` x0.

Case outcomes vs `54198ef`:

| Sample | 54198ef selected | 61042c3 selected | Reading |
|---|---:|---:|---|
| `scannet/scene0077_00::1::22194` | 1 | 1 | Still correct, now with the desired `below` comparison and relation evidence. |
| `scannet/scene0011_00::20::28662` | 20 | 1 | Regressed. Non-vertical cabinet right-side query; broad prompt noise likely changed the tool/visual path. |
| `scannet/scene0011_00::23::15020` | 23 | 24 | Regressed. Non-vertical window-facing right-side query. |
| `scannet/scene0030_00::0::23137` | 0 | 1 | Regressed. Multi-clause chalkboard/office-chair alignment query. |

Reading: negative as a broad prompt change. It proves the agent can use the
fixed vertical comparator when instructed, but the prose is too broad for the
main VG playbooks and regresses unrelated cases on the same 15-case slice. Do
not keep this wording active; preserve the run as evidence that the next route
should be narrower, likely a guard/tool-level intervention for explicit
above/below submissions rather than more general playbook prose.

### Vertical relation TADG guard probe: 66c6218

Commit `66c6218` moves the vertical-routing attempt from broad playbook prose
into TADG: when the query explicitly asks for `under` / `below` / `beneath` /
`above`, `submit_final` must bind matching `compare_proposals_spatial`
relation evidence, or provide an override reason. This is still no-GT: the
guard reads only the query text, submitted payload, and tool trace.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Head commit at launch | `66c6218` |
| Run-time code commit | `66c6218` - no worktree drift |
| Probe IDs | `docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json` |
| Probe IDs MD5 | `8a9f5489e331631a2e10d13010effae5` |
| Output dir | `tmp/nr3d_eval_v10_vertical_tadg_probe15_20260519_66c6218/` |
| Run log | `/tmp/nr3d_vertical_tadg_probe15_66c6218.log` |
| Side-by-side JSON | `tmp/nr3d_eval_v10_vertical_tadg_probe15_20260519_66c6218/side_by_side.json` |
| Side-by-side MD5 | `38ed9aa52d13bab057e6542ddd357e2c` |
| Leaderboard metrics | `tmp/nr3d_eval_v10_vertical_tadg_probe15_20260519_66c6218/leaderboard_metrics.json` |
| Leaderboard metrics MD5 | `c7bdcd4467e049babe67d632650fa87d` |
| SQLite run id | `v10_vertical_tadg_probe15_20260519` |
| Workers | 8 |
| Sample retries | 0 |
| Guards | TADG + no-match + evidence-frame |

Commands:

```bash
tmux new-session -d -s nr3d-vertical-tadg-probe15-66c6218 \
  "cd /Users/bytedance/project/3DVLMReasoning && bash -lc 'set -euo pipefail; \
   export PYTHONPATH=src PYTHONUNBUFFERED=1; \
   .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_vertical_tadg_probe15_20260519_66c6218 \
     --workers 8 \
     --sample-retries 0 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_vertical_tadg_probe15_66c6218.log; \
   .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
     --side-by-side tmp/nr3d_eval_v10_vertical_tadg_probe15_20260519_66c6218/side_by_side.json \
     --nr3d-data-root data/nr3d \
     --phase8-data-root data/nr3d/scannet \
     --sample-ids docs/benchmark/nr3d/assets/v10_candidate_closure_probe15_sample_ids_20260519.json \
     --output tmp/nr3d_eval_v10_vertical_tadg_probe15_20260519_66c6218/leaderboard_metrics.json \
     --canonical-filter true \
     2>&1 | tee -a /tmp/nr3d_vertical_tadg_probe15_66c6218.log'"

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_vertical_tadg_probe15_20260519_66c6218 \
  --run-id v10_vertical_tadg_probe15_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 66c6218 \
  --backend pack_v1 \
  --leaderboard-metrics tmp/nr3d_eval_v10_vertical_tadg_probe15_20260519_66c6218/leaderboard_metrics.json \
  --notes "Diagnostic 15-case probe after TADG missing vertical relation evidence guard; no GT inputs; negative vs 54198ef." \
  --db docs/benchmark/nr3d/runs.sqlite
```

Probe stability:

- 15 / 15 samples completed.
- 0 Tracebacks / logged Exceptions in the run log.
- Pre-run TDD verification: targeted TADG tests passed 3 / 3; full TADG +
  chassis tool tests passed 67 / 67; `ruff` and `git diff --check` passed.

Probe metrics:

| Metric | Value |
|---|---:|
| n | 15 |
| classification_acc_filtered | 33.33 |
| Easy | 28.57 |
| Hard | 37.50 |
| V-Dep | 42.86 |
| V-Indep | 25.00 |

Tool-routing evidence:

- `TADG_MISSING_RELATION_EVIDENCE` fired in 2 submit traces:
  `scene0063_00::5::11741` and `scene0077_00::1::22194`.
- Across the 15 traces, relation calls were: `closest_to` x2, `next_to` x1,
  `below` x2, `above` x0.
- `scene0077_00::1::22194` was routed into
  `compare_proposals_spatial(candidate_ids=[1, 2], anchor_id=3,
  relation='below')`, which ranked `[1, 2]` and preserved the correct answer.
- `scene0063_00::5::11741` was also routed into a `below` comparison but still
  selected wrong proposal `4` instead of target `5`, showing that mandatory
  relation evidence can lock in a wrong anchor/candidate interpretation.

Case outcomes vs `54198ef`:

| Sample | 54198ef selected | 66c6218 selected | Reading |
|---|---:|---:|---|
| `scannet/scene0077_00::1::22194` | 1 | 1 | Still correct, now with a TADG block followed by `below` relation evidence. |
| `scannet/scene0063_00::5::11741` | 5 | 4 | Regressed. This is a vertical chair-under-TV query; the guard forces a compare call, but the resulting relation path picks the wrong chair. |
| `scannet/scene0011_00::20::28662` | 20 | 1 | Regressed. Non-vertical cabinet right-side query; no missing-relation block, so the stricter guard gives no monotonic benefit on the slice. |
| `scannet/scene0025_00::17::37165` | 17 | 38 | Regressed. File-cabinet/whiteboard relation regresses without being the targeted vertical case. |

Reading: negative as a guard-level intervention. It demonstrates that a TADG
block can force the desired tool call, but the current rule is not safe: the
same-slice accuracy stays at 5 / 15, matching the negative broad-prompt probe
and losing 3 cases vs the stable `54198ef` probe. Do not keep this guard active.
Future work should make the relation tool itself easier to call correctly
without forcing every explicit vertical final answer through a brittle block.

### Unavailable crop tool-surface probe: cbfb9ed -> 1d3b02d

The 5-worker audit found failures where `request_crops` returned the old
success-like stub text even though no concrete crop image was generated. That
violates the no-silent-fallback rule: an agent can cite crop evidence that does
not exist. Commit `d588b34` first made the generic callback fail loud with an
`ERROR`. The direct probe below showed that exposing an ERROR-only tool hurt
accuracy on this diagnostic slice, so commit `1d3b02d` instead hides
`request_crops` from tools, system prompt, and evidence nudges unless a real
crop callback is configured.

Run metadata:

| Item | Value |
|---|---|
| Branch | `feat/remove-initial-keyframes` |
| Parent run-time code commit | `5193cff` |
| Fail-loud run-time code commit | `cbfb9ed` |
| Hidden-tool run-time code commit | `1d3b02d` |
| Probe IDs | `docs/benchmark/nr3d/assets/v10_crop_fail_loud_probe26_sample_ids_20260519.json` |
| Probe IDs MD5 | `1546be2afa96bfe012b22a45d7211032` |
| Parent output dir | `tmp/nr3d_eval_v10_crop_parent_probe26_20260519_5193cff_maincwd/` |
| Fail-loud output dir | `tmp/nr3d_eval_v10_crop_fail_loud_probe26_20260519_cbfb9ed/` |
| Hidden-tool output dir | `tmp/nr3d_eval_v10_crop_hidden_probe26_20260519_1d3b02d/` |
| SQLite run ids | `v10_crop_parent_probe26_20260519`, `v10_crop_fail_loud_probe26_20260519`, `v10_crop_hidden_probe26_20260519` |
| Workers | 8 |
| Sample retries | 0 |
| Guards | TADG + no-match + evidence-frame |

Artifact checksums:

| Artifact | MD5 |
|---|---|
| parent `side_by_side.json` | `babec59a2a3e3ac9fb302d0f7bdcf1c7` |
| parent `leaderboard_metrics.json` | `6021ef9f9ed7109d3e6bc1e568adeeee` |
| fail-loud `side_by_side.json` | `f9d8b44980635b365a58ed4c96a0831b` |
| fail-loud `leaderboard_metrics.json` | `2dbfe92d22e8a6ede71a9a4830d61cfe` |
| hidden-tool `side_by_side.json` | `7f34041372e815e064580cf5163d7fdb` |
| hidden-tool `leaderboard_metrics.json` | `bc2ddb8f9cf0603df5c4a364f91ffa95` |

Commands:

```bash
tmux new-session -d -s nr3d-crop-hidden-probe26-1d3b02d \
  "cd /Users/bytedance/project/3DVLMReasoning && bash -lc 'set -euo pipefail; \
   export PYTHONPATH=src PYTHONUNBUFFERED=1; \
   .venv/bin/python -m evaluation.scripts.run_nr3d_vg_side_by_side \
     --sample-ids docs/benchmark/nr3d/assets/v10_crop_fail_loud_probe26_sample_ids_20260519.json \
     --data-root data/nr3d/scannet \
     --pack-name pack_nr3d_v9_catalog_first \
     --output-dir tmp/nr3d_eval_v10_crop_hidden_probe26_20260519_1d3b02d \
     --workers 8 \
     --sample-retries 0 \
     --use-tool-answer-disagreement-gate \
     --use-no-match-candidate-guard \
     --use-evidence-frame-guard \
     2>&1 | tee /tmp/nr3d_crop_hidden_probe26_1d3b02d.log; \
   .venv/bin/python -m evaluation.scripts.nr3d_leaderboard_metrics \
     --side-by-side tmp/nr3d_eval_v10_crop_hidden_probe26_20260519_1d3b02d/side_by_side.json \
     --nr3d-data-root data/nr3d \
     --phase8-data-root data/nr3d/scannet \
     --sample-ids docs/benchmark/nr3d/assets/v10_crop_fail_loud_probe26_sample_ids_20260519.json \
     --output tmp/nr3d_eval_v10_crop_hidden_probe26_20260519_1d3b02d/leaderboard_metrics.json \
     --canonical-filter true \
     2>&1 | tee -a /tmp/nr3d_crop_hidden_probe26_1d3b02d.log'"

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v10_crop_hidden_probe26_20260519_1d3b02d \
  --run-id v10_crop_hidden_probe26_20260519 \
  --branch feat/remove-initial-keyframes \
  --commit 1d3b02d \
  --backend pack_v1 \
  --leaderboard-metrics tmp/nr3d_eval_v10_crop_hidden_probe26_20260519_1d3b02d/leaderboard_metrics.json \
  --notes "Diagnostic 26-case probe after hiding unavailable request_crops tool; no crop tool calls; no GT runtime inputs." \
  --db docs/benchmark/nr3d/runs.sqlite
```

For the parent comparison, `5193cff` was run with
`PYTHONPATH=/tmp/3dvlm_crop_parent_5193cff/src` from the main repository cwd.
That cwd matters because the prepared sample artifacts contain relative
`scene_artifacts_dir` paths. Earlier parent attempts from the detached worktree
failed path preflight / visibility checks and were not used as the comparison.

Probe metrics:

| Variant | Commit | Overall | Easy | Hard | V-Dep | V-Ind | Statuses | Crop tool calls |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Old success-like crop stub | `5193cff` | 50.00 | 50.00 | 50.00 | 33.33 | 58.82 | 26 completed | 11 stub calls, 0 images |
| Fail-loud unavailable crop callback | `cbfb9ed` | 42.31 | 37.50 | 44.44 | 22.22 | 52.94 | 26 completed | 12 ERROR calls, 0 images |
| Hide unavailable crop tool | `1d3b02d` | 50.00 | 50.00 | 50.00 | 22.22 | 64.71 | 25 completed, 1 failed sentinel | 0 calls |

Case deltas:

| Comparison | Recoveries | Regressions |
|---|---|---|
| hidden-tool vs fail-loud | `scene0030_00::0::23137`, `scene0307_00::25::35532`, `scene0432_00::2::29554` | `scene0249_00::23::30460` |
| hidden-tool vs parent stub | `scene0030_00::0::23137`, `scene0307_00::25::35532`, `scene0549_00::7::17436`, `scene0565_00::23::30788` | `scene0025_00::1::6057`, `scene0149_00::25::10119`, `scene0249_00::23::30460`, `scene0574_00::5::13504` |

Reading: do not restore the old stub. It was not visual evidence and should not
be available under the no-fallback rule. The active fix is `1d3b02d`: when no
concrete crop renderer is configured, `request_crops` is absent from the tool
list and from runtime prompt/nudge text. The diagnostic slice returns to the
parent's 13 / 26 without allowing any false crop-evidence path. Future work
should either implement a real object/frame crop renderer or keep this tool
hidden for NR3D/ScanRefer pack-v1 runs.
