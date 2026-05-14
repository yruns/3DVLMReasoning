# v10 GPT-4o Request Random100 - 2026-05-15

NR3D random100 rerun on the same depth-aware v9 selective-mark fold, changing
only the requested ModelHub model family from the default `gpt-5.4-2026-03-05`
to GPT-4o request strings.

This is a model-swap pilot, not a public leaderboard row. It should be read
with the ModelHub-routing caveat below: the request model names were GPT-4o,
but smoke responses reported different routed model metadata.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Tip commit at run time: `efb7bb4-dirty-modelhub-model-env`
  - The run used the committed v9 selective-mark code plus an uncommitted
    `MODELHUB_MODEL_NAME` env override in `Stage2DeepAgentConfig`.
- Internal version: `v10_gpt4o_request_random100`
- Run ID: `v10_gpt4o_request_random100_20260515`
- Stage 2 backend: internal ModelHub, requested GPT-4o model strings
- Judge: none; scoring is programmatic target-instance classification
- SQLite row:
  `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v10_gpt4o_request_random100_20260515'`

## ModelHub Routing Caveat

Before the benchmark, the user-provided AK/model combinations were smoke-tested
with the ModelHub chat endpoint.

- AK group A accepted `gpt-4o-2024-11-20`, but the response metadata reported
  model `gpt-5-mini-2025-08-07`.
- AK group A did not have permission for `gpt-4o-2024-08-06`.
- AK group B accepted `gpt-4o-2024-08-06`, but response metadata reported
  model `gpt-4.1-mini-2025-04-14`.
- One AK in group B did not have permission for `gpt-4o-2024-11-20`.

Therefore this run is recorded as a **ModelHub GPT-4o request** experiment, not
as a confirmed pure public GPT-4o run.

## What Changed vs v9

- Added a small config hook so `Stage2DeepAgentConfig.model_name` can come from
  `MODELHUB_MODEL_NAME`; default remains `gpt-5.4-2026-03-05`.
- Kept the v9 selective-mark evidence contract unchanged:
  - clean initial RGB keyframes with text-only inventories;
  - `list_frame_proposals(frame_id)`;
  - filtered `view_keyframe_marked(frame_id, categories=[...],
    proposal_ids=[...])`;
  - TADG, no-match, and evidence-frame guards enabled.
- Split the 100 samples by ModelHub permission:
  - shard A requested `gpt-4o-2024-11-20`;
  - shard B requested `gpt-4o-2024-08-06`.
- User-provided AKs are intentionally not written into this tracked document.

## Fold

- Split: NR3D `test`
- Fold size: 100 samples
- Selection file:
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- Selection mechanism: same deterministic v4/v6/v7/v8/v9 random100 fold.
- Canonical filter after restriction: `n_full=100`, `n_filtered=100`

## Raw Artifacts

- Prepared pack:
  `data/nr3d/scannet/<scene>/pack_nr3d_v8_clean_initial_marked_on_demand/`
- Eval output:
  `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/`
- Initial shard files:
  - `tmp/nr3d_artifacts/v9_gpt4o_random100_shard_a_20260514.json`
  - `tmp/nr3d_artifacts/v9_gpt4o_random100_shard_b_20260514.json`
- Rerun shard files:
  - `tmp/nr3d_artifacts/v9_gpt4o_random100_rerun_a_20260514.json`
  - `tmp/nr3d_artifacts/v9_gpt4o_random100_rerun_b_20260514.json`
  - `tmp/nr3d_artifacts/v9_gpt4o_random100_rerun_failed2_20260514.json`
- Failed checkpoint backups:
  - `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/failed_attempt1_20260514/`
  - `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/failed_attempt2_20260514/`
- Per-sample checkpoints:
  `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/per_sample/pack_nr3d_v8_clean_initial_marked_on_demand/*.json`
- Side-by-side output:
  `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/side_by_side.json`
- Leaderboard metrics:
  `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/leaderboard_metrics.json`
- Logs:
  - `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_shard_a.log`
  - `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_shard_b.log`
  - `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_rerun_a.log`
  - `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_rerun_b.log`
  - `tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_failed2.log`

## Commands

The actual AK values were supplied through `MODELHUB_AKS` at runtime and are
redacted here. The command shape is otherwise preserved.

Initial high-concurrency shards:

```bash
tmux new-session -d -s nr3d_v9_gpt4o_a 'cd /Users/bytedance/project/3DVLMReasoning && MODELHUB_MODEL_NAME=gpt-4o-2024-11-20 MODELHUB_AKS=<redacted-ak-group-a> MODELHUB_AK_WEIGHTS=1 PYTHONUNBUFFERED=1 PYTHONPATH=src ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v9_gpt4o_random100_shard_a_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_clean_initial_marked_on_demand --output-dir tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514 --workers 50 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_shard_a.log'

tmux new-session -d -s nr3d_v9_gpt4o_b 'cd /Users/bytedance/project/3DVLMReasoning && MODELHUB_MODEL_NAME=gpt-4o-2024-08-06 MODELHUB_AKS=<redacted-ak-group-b> MODELHUB_AK_WEIGHTS=1,1 PYTHONUNBUFFERED=1 PYTHONPATH=src ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v9_gpt4o_random100_shard_b_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_clean_initial_marked_on_demand --output-dir tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514 --workers 50 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_shard_b.log'
```

The initial 50+50 run produced 70 checkpoint files: 64 completed, 6 failed,
and 30 missing. The six failed sentinels were all
`StructuredOutputValidationError` from extra JSON after a structured response.

Controlled rerun under the user's 20GB memory cap:

```bash
tmux new-session -d -s nr3d_v9_gpt4o_rerun_a 'cd /Users/bytedance/project/3DVLMReasoning && MODELHUB_MODEL_NAME=gpt-4o-2024-11-20 MODELHUB_AKS=<redacted-ak-group-a> MODELHUB_AK_WEIGHTS=1 PYTHONUNBUFFERED=1 PYTHONPATH=src ./scripts/run_with_rss_guard.sh --rss-limit-mb 9000 --check-interval-sec 60 -- .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v9_gpt4o_random100_rerun_a_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_clean_initial_marked_on_demand --output-dir tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514 --workers 10 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_rerun_a.log'

tmux new-session -d -s nr3d_v9_gpt4o_rerun_b 'cd /Users/bytedance/project/3DVLMReasoning && MODELHUB_MODEL_NAME=gpt-4o-2024-08-06 MODELHUB_AKS=<redacted-ak-group-b> MODELHUB_AK_WEIGHTS=1,1 PYTHONUNBUFFERED=1 PYTHONPATH=src ./scripts/run_with_rss_guard.sh --rss-limit-mb 9000 --check-interval-sec 60 -- .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v9_gpt4o_random100_rerun_b_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_clean_initial_marked_on_demand --output-dir tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514 --workers 10 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_rerun_b.log'

tmux new-session -d -s nr3d_v9_gpt4o_failed2 'cd /Users/bytedance/project/3DVLMReasoning && MODELHUB_MODEL_NAME=gpt-4o-2024-08-06 MODELHUB_AKS=<redacted-ak-group-b> MODELHUB_AK_WEIGHTS=1,1 PYTHONUNBUFFERED=1 PYTHONPATH=src ./scripts/run_with_rss_guard.sh --rss-limit-mb 9000 --check-interval-sec 60 -- .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py --sample-ids tmp/nr3d_artifacts/v9_gpt4o_random100_rerun_failed2_20260514.json --data-root data/nr3d/scannet --pack-name pack_nr3d_v8_clean_initial_marked_on_demand --output-dir tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514 --workers 1 --checkpoint-only --sample-retries 2 --use-tool-answer-disagreement-gate --use-no-match-candidate-guard --use-evidence-frame-guard 2>&1 | tee tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514_failed2.log'
```

Assemble, compute metrics, and ingest:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v8_clean_initial_marked_on_demand \
  --output-dir tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514 \
  --workers 1 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
  --output tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514 \
  --run-id v10_gpt4o_request_random100_20260515 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit efb7bb4-dirty-modelhub-model-env \
  --backend pack_v1 \
  --judge-model none \
  --notes "v10 random100 same v9 selective-mark code/fold; requested ModelHub GPT-4o models via user-provided AKs split by permission; smoke responses reported routed model metadata not guaranteed pure GPT-4o; initial 50+50 workers left missing/failed samples, then reran A/B at 10+10 under 9GB per-shard guards and final failed sample at workers=1; 100/100 completed; observed rerun RSS stayed under 20GB" \
  --leaderboard-metrics tmp/nr3d_eval_v9_selective_mark_gpt4o_random100_20260514/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v9 gpt-5.4 selective mark | v10 GPT-4o request | Delta |
|---|---:|---:|---:|
| Overall | 74.00 | 65.00 | -9.00 pp |
| Easy | 90.24 | 75.61 | -14.63 pp |
| Hard | 62.71 | 57.63 | -5.08 pp |
| View-Dep | 64.71 | 55.88 | -8.82 pp |
| View-Indep | 78.79 | 69.70 | -9.09 pp |

Raw metrics:

```text
n_full=100 n_filtered=100
classification_acc_full     = 0.6500
classification_acc_filtered = 0.6500
Easy   (n=41): 0.7561
Hard   (n=59): 0.5763
V-Dep  (n=34): 0.5588
V-Ind  (n=66): 0.6970
```

### Per-Sample Movement vs v9

| Check | Count |
|---|---:|
| Samples with changed selected proposal id | 28 |
| Wrong -> correct | 7 |
| Correct -> wrong | 16 |
| Net overall movement | -9 |

Wrong -> correct sample IDs:

- `scannet/scene0164_00::11::8924`
- `scannet/scene0187_00::13::24713`
- `scannet/scene0338_00::0::33421`
- `scannet/scene0353_00::52::27519`
- `scannet/scene0578_00::10::5147`
- `scannet/scene0651_00::8::37347`
- `scannet/scene0653_00::16::3852`

Correct -> wrong sample IDs:

- `scannet/scene0025_00::32::23687`
- `scannet/scene0084_00::35::19742`
- `scannet/scene0095_00::24::39202`
- `scannet/scene0249_00::36::39076`
- `scannet/scene0300_00::8::22495`
- `scannet/scene0500_00::27::34282`
- `scannet/scene0535_00::9::18463`
- `scannet/scene0549_00::31::29364`
- `scannet/scene0568_00::15::20548`
- `scannet/scene0598_00::18::5365`
- `scannet/scene0618_00::26::28943`
- `scannet/scene0629_00::31::23523`
- `scannet/scene0663_00::3::36534`
- `scannet/scene0697_00::19::35451`
- `scannet/scene0704_00::0::16937`
- `scannet/scene0704_00::5::8434`

## Runtime / Tool Usage Notes

- Final checkpoint status: `completed=100`
- Missing checkpoints after all reruns: `0`
- Failed sentinels after all reruns: `0`
- Initial 50+50 run: 64 completed, 6 failed, 30 missing
- Controlled 10+10 rerun: recovered all missing and all but one failed sample
- Final `workers=1` rerun: recovered the remaining failed sample
- Observed controlled-rerun RSS stayed below 20GB:
  - early 10+10 rerun: about 8.7GB total RSS;
  - later single A shard: about 8.2GB total RSS;
  - final single-sample rerun: about 0.3GB RSS.
- RSS guard: no `RSS exceeded` event in any v10 log.
- Retry lines across v10 logs:
  - initial shard A: 0 retryable `429`, 113 transport failures;
  - initial shard B: 172 retryable `429`, 113 transport failures;
  - controlled rerun A: 0 retryable `429`, 0 transport failures;
  - controlled rerun B: 14 retryable `429`, 1 transport failure;
  - final failed-sample rerun: 0 retryable `429`, 0 transport failures.
- Tool calls across final 100 completed samples:
  - `view_keyframe_marked`: 313 calls across 94 samples
  - `inspect_proposal`: 231 calls across 90 samples
  - `submit_final`: 204 calls across 100 samples
  - `find_proposals_by_category`: 123 calls across 74 samples
  - `list_frame_proposals`: 74 calls across 42 samples
  - `compare_proposals_spatial`: 67 calls across 61 samples
  - `list_keyframes_with_proposals`: 51 calls across 48 samples
  - `request_crops`: 40 calls across 28 samples
  - `request_more_views`: 16 calls across 14 samples

## SQLite Reproduction Query

```sql
SELECT run_id, n, n_filtered,
       printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS view_dep,
       printf('%.4f', acc_view_indep) AS view_indep
FROM runs
WHERE run_id = 'v10_gpt4o_request_random100_20260515';
```

Verified output:

```text
v10_gpt4o_request_random100_20260515|100|100|0.6500|0.7561|0.5763|0.5588|0.6970
```

## Interpretation

On this fixed depth-aware random100 fold, the ModelHub GPT-4o request run is
materially worse than the default gpt-5.4 v9 run: -9.00 pp overall. The loss is
largest on Easy examples and view-dependent examples, suggesting the weaker or
routed model is less stable at mapping visual evidence back to the correct
proposal id even when selective marks are available.

This result does not support switching the NR3D/ScanRefer VG agent default away
from gpt-5.4. It is still useful as a cost/latency ablation and as evidence
that the selective-mark tool surface depends on model quality.

## Caveats

- This is a 100-sample internal model-swap pilot, not a public full NR3D row.
- ModelHub response metadata did not confirm a pure public GPT-4o backend.
- The run crossed midnight: artifacts are named `20260514`, while the metrics
  and SQLite row were harvested on `20260515`.
- The tracked docs redact AK values. Reproduction requires valid ModelHub
  credentials with the same model permissions.
