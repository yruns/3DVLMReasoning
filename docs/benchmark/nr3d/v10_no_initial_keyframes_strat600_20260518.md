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
