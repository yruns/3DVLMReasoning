# v8 Clean Initial Marked-On-Demand Random100 - 2026-05-14

NR3D random100 rerun on the same depth-aware v4/v6/v7 fold after changing the
VG evidence contract: initial keyframes are clean RGB images, while marked
`#id category` bbox images are only injected when the agent calls
`view_keyframe_marked(frame_id)`.

This is a partial ablation, not a public leaderboard row. The result is
negative relative to v7.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Tip commit at run time: `f86c161`
- Internal version: `v8_clean_initial_marked_on_demand_random100`
- Run ID: `v8_clean_initial_marked_on_demand_random100_20260514`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic target-instance classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v8_clean_initial_marked_on_demand_random100_20260514'`

## What Changed vs v7

- Initial `Stage2EvidenceBundle.keyframes[*].image_path` now points to raw
  clean RGB frames instead of annotated `bbox in frame` images.
- The initial VG prompt adds a text-only frame proposal inventory:
  `frame_id=N left_to_right: #id category, ...`.
- The initial prompt deliberately omits per-proposal 2D bbox coordinates and
  visibility scores.
- Marked images remain available through `view_keyframe_marked(frame_id)`;
  the runtime injects those marked images only after the tool call.
- NR3D pack prep emits `proposal.frame_views` so left-to-right ordering can be
  derived from actual per-frame 2D boxes.
- Stage 2 guard flags remain enabled:
  - `--use-tool-answer-disagreement-gate`
  - `--use-no-match-candidate-guard`
  - `--use-evidence-frame-guard`

## Fold

- Split: NR3D `test`
- Fold size: 100 samples
- Selection file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Selection mechanism: same deterministic v4/v6/v7 random100 fold.
- Canonical filter after restriction: `n_full=100`, `n_filtered=100`

## Raw Artifacts

- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_clean_initial_marked_on_demand/`
- Eval output:
  `tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/per_sample/pack_nr3d_v8_clean_initial_marked_on_demand/*.json`
- Side-by-side output:
  `tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/leaderboard_metrics.json`
- Log:
  `tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514_w100.log`
- Static 2T/2F trace viewer:
  `docs/benchmark/nr3d/v8_clean_initial_trace_2t2f_20260514.html`

## Commands

Prepare the clean-initial pack from the existing depth-aware v6 pack:

```bash
tmux new-session -d -s nr3d_v8_clean_prep_random100 "cd /Users/bytedance/project/3DVLMReasoning && PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python scripts/prepare_nr3d_rerender_pack_from_existing.py --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json --data-root data/nr3d/scannet --source-pack pack_nr3d_v6_inline_labels_depth_visible --out-pack pack_nr3d_v8_clean_initial_marked_on_demand 2>&1 | tee tmp/nr3d_v8_clean_prep_random100_20260514.log"
```

Run Stage 2 checkpoints:

```bash
tmux new-session -d -s nr3d_v8_clean_stage2_random100 'cd /Users/bytedance/project/3DVLMReasoning && source .venv/bin/activate && export PYTHONUNBUFFERED=1 PYTHONPATH=src && ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_clean_initial_marked_on_demand --output-dir tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514 --workers 100 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514_w100.log'
```

Assemble without re-running samples:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_clean_initial_marked_on_demand \
  --output-dir tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

Compute metrics and ingest:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514 \
  --run-id v8_clean_initial_marked_on_demand_random100_20260514 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit f86c161 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v8 random100 on same v4/v6/v7 fold; initial evidence uses clean RGB keyframes plus text-only left-to-right #id category inventory; marked images only via view_keyframe_marked; proposal frame_views emitted during pack prep; TADG + no-match + evidence-frame guards; workers=100 with 15GB RSS guard; 100/100 checkpoints; 0 failed sentinels; observed RSS about 4.1GB and LLM 429 backoff as throughput limiter" \
  --leaderboard-metrics tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v7 callbacks no-CLIP | v8 clean initial | Delta |
|---|---:|---:|---:|
| Overall | 73.00 | 67.00 | -6.00 pp |
| Easy | 82.93 | 85.37 | +2.44 pp |
| Hard | 66.10 | 54.24 | -11.86 pp |
| View-Dep | 70.59 | 58.82 | -11.77 pp |
| View-Indep | 74.24 | 71.21 | -3.03 pp |

### IoU Proxy on GT Pool

| Metric | v7 | v8 | Delta |
|---|---:|---:|---:|
| n | 100 | 100 | 0 |
| mean IoU | 0.7327 | 0.6724 | -0.0603 |
| Acc@0.25 | 0.7300 | 0.6700 | -0.0600 |
| Acc@0.50 | 0.7300 | 0.6700 | -0.0600 |
| failed sentinels | 0 | 0 | 0 |
| uncaught sample errors | 0 | 0 | 0 |

### Per-Sample Movement vs v7

| Check | Count |
|---|---:|
| Samples with changed selected proposal id | 25 |
| Wrong -> correct | 6 |
| Correct -> wrong | 12 |
| Net overall movement | -6 |

Wrong -> correct sample IDs:

- `scannet/scene0490_00::14::39189`
- `scannet/scene0565_00::0::3757`
- `scannet/scene0578_00::10::5147`
- `scannet/scene0629_00::6::19188`
- `scannet/scene0645_00::38::39739`
- `scannet/scene0663_00::3::36534`

Correct -> wrong sample IDs:

- `scannet/scene0249_00::36::39076`
- `scannet/scene0353_00::50::24580`
- `scannet/scene0353_00::52::27519`
- `scannet/scene0474_00::12::7632`
- `scannet/scene0568_00::15::20548`
- `scannet/scene0598_00::18::5365`
- `scannet/scene0618_00::26::28943`
- `scannet/scene0643_00::22::25584`
- `scannet/scene0648_00::15::24119`
- `scannet/scene0652_00::13::34162`
- `scannet/scene0653_00::16::3852`
- `scannet/scene0665_00::2::38846`

## Tool Trace / Runtime Notes

- Stage 2 completed 100/100 checkpoints with `workers=100`.
- Failed sentinels: 0.
- Wall-clock runtime for checkpoint generation was about 6 minutes
  (`19:40:20` to `19:46:25` local time).
- RSS guard limit was 15GB. Observed RSS stayed far below the guard, peaking
  around 4.1GB during the run.
- The log contains 192 retryable `429` lines and no retryable `503` lines.
  Throughput is therefore bounded by ModelHub qpm/tpm rather than local memory
  on this fold.
- The log contains no `callback is not configured` messages.
- The log contains 3 `Loading CLIP model` messages from text-query paths.
- Per-sample tool traces include:
  - `view_keyframe_marked`: 317 calls
  - `request_more_views`: 46 calls
  - `request_crops`: 19 calls
  - `switch_or_expand_hypothesis`: 5 calls
  - `inspect_stage1_metadata`: 12 calls
  - `retrieve_object_context`: 6 calls

## Interpretation

The clean-initial contract is visually cleaner and avoids overwhelming the
first model turn with many overlapping boxes. On this fold, however, it hurts
overall accuracy. The regression is concentrated in Hard and View-Dep examples,
which are exactly the cases that need stable proposal identity grounding early
in the reasoning chain.

This suggests the current agent does not reliably decide when to request marked
views before committing to a candidate. A safer next step is not to adopt v8 as
the default, but to keep v7 as the stronger random100 baseline and treat clean
initial evidence as a prompt/tool-policy problem: the agent needs a stronger
rule to call `view_keyframe_marked` before resolving same-class relational or
view-dependent references.

## Caveats

- This is a 100-sample internal ablation, not a full NR3D row.
- It preserves the v4/v6/v7 fold and depth-aware visibility source, but the
  pack was regenerated to point initial keyframes at clean raw images.
- The run uses the same guards as v7, so the main measured delta is the initial
  visual evidence contract and on-demand marked-image policy.
- `workers=100` already produced frequent ModelHub 429 backoff. Higher worker
  counts may improve queue fill for full runs, but this fold shows the practical
  bottleneck has shifted from local OOM risk to LLM qpm/tpm.
