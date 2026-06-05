# NR3D v17 Codex SDK strat600

This run evaluates a new **Codex Agent SDK** Stage-2 entrypoint on the
canonical NR3D strat600 fold. It is intentionally separate from the production
DeepAgents runtime: Codex receives the BEV image plus compact proposal metadata
and returns a structured `proposal_id`; it does not use the DeepAgents
selector/mark-frame/tool-guard loop.

The result is **63.33 % Overall** on strat600. This is a successful SDK
integration run, not a new accuracy best.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `feat/intro-codex-agent-sdk` |
| Head commit at launch | `6037c0b` |
| Run-time code commit | `6037c0b` — no worktree drift |
| Implementation commit | `1c2d2aa` — `Add Codex SDK stage2 NR3D backend` |
| Runtime-state ignore commit | `6037c0b` — `Ignore Codex SDK runtime state` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold design | `docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md` |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_eval_codex_sdk_strat600_g54_20260605/` |
| Side-by-side JSON | `tmp/nr3d_eval_codex_sdk_strat600_g54_20260605/side_by_side.json` |
| Leaderboard metrics | `tmp/nr3d_eval_codex_sdk_strat600_g54_20260605/leaderboard_strat600.json` |
| SQLite run id | `v17_codex_sdk_strat600_g54_20260605` |
| Started | `2026-06-05T19:56:26+08:00` |
| Side-by-side completed | `2026-06-05T20:34:38+08:00` |
| Metrics completed | `2026-06-05T20:36:55+08:00` |
| SQLite ingested | `2026-06-05T20:37:22+08:00` |

## Exact Commands

Local Codex ModelHub adapter was run in tmux session `codex-adapter` from:

```bash
cd /Users/bytedance/aispace/codex_modelhub_adapter
PYTHONPATH=/Users/bytedance/aispace/codex_modelhub_adapter .venv/bin/python <adapter-launcher>
```

The launcher loaded the existing ModelHub AK pool from
`src/agents/core/agent_config.py`, set:

```bash
AIDP_CODEX_PROXY_UPSTREAM_ENV=office
AIDP_CODEX_PROXY_UPSTREAM_API=auto
AIDP_CODEX_PROXY_CHAT_COMPLETIONS_MODELS=gpt-5.5*,gpt-5.4*
AIDP_CODEX_PROXY_SESSION_ID=nr3d-codex-sdk
```

and started `uvicorn adapter.app:app --host 127.0.0.1 --port 8787`.

Initial smoke with the setup-guide default `gpt-5.5-2026-04-24` reached the
adapter but failed with:

```text
RuntimeError: unexpected status 401 Unauthorized: no model permission: gpt-5.5-2026-04-24
```

The benchmark run therefore used the model this repo already has permission
for, via `CODEX_AGENT_MODEL=gpt-5.4-2026-03-05`.

```bash
CODEX_HOME=/Users/bytedance/project/3DVLMReasoning/.codex-home \
CODEX_AGENT_MODEL=gpt-5.4-2026-03-05 \
PYTHONPATH=src \
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --output-dir tmp/nr3d_eval_codex_sdk_strat600_g54_20260605 \
  --backend codex_sdk \
  --workers 4 \
  --sample-retries 1
```

Leaderboard aggregation:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_codex_sdk_strat600_g54_20260605/side_by_side.json \
  --backend codex_sdk \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --output tmp/nr3d_eval_codex_sdk_strat600_g54_20260605/leaderboard_strat600.json
```

SQLite ingestion:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_codex_sdk_strat600_g54_20260605 \
  --run-id v17_codex_sdk_strat600_g54_20260605 \
  --branch feat/intro-codex-agent-sdk \
  --commit 6037c0b \
  --backend codex_sdk \
  --db docs/benchmark/nr3d/runs.sqlite \
  --leaderboard-metrics tmp/nr3d_eval_codex_sdk_strat600_g54_20260605/leaderboard_strat600.json \
  --notes "Codex Agent SDK catalog-plus-BEV NR3D strat600 run via local ModelHub adapter; gpt-5.4 runtime override after gpt-5.5 permission failure."
```

## Headline Metrics

| Metric | v17 Codex SDK | v11 enriched strat600 | v10 best observed | v9.3 text-first |
|---|---:|---:|---:|---:|
| Overall | 63.33 | 71.33 | 72.00 | 66.67 |
| Easy | 70.00 | 79.66 | 82.41 | 73.45 |
| Hard | 57.10 | 63.55 | 62.26 | 60.32 |
| V-Dep | 53.55 | 61.61 | 62.56 | 54.98 |
| V-Indep | 68.64 | 76.61 | 77.12 | 73.01 |

Counts:

| Slice | n | correct |
|---|---:|---:|
| Full / filtered | 600 | 380 |
| Easy | 290 | 203 |
| Hard | 310 | 177 |
| V-Dep | 211 | 113 |
| V-Indep | 389 | 267 |

Pack side-by-side metrics:

| Metric | Value |
|---|---:|
| `n` | 600 |
| `status=completed` | 600 |
| checkpoint `error` fields | 0 |
| mean IoU | 0.63869 |
| Acc@0.25 | 0.63333 |
| Acc@0.50 | 0.63333 |

## Runtime Notes

- The Codex SDK runtime is a separate backend selected with
  `--backend codex_sdk`; the existing DeepAgents path remains `--backend pack_v1`.
- The run used `openai-codex==0.1.0b2` and local adapter
  `/Users/bytedance/aispace/codex_modelhub_adapter`.
- Codex SDK input included the repo skill
  `.agents/skills/nr3d-codex-sdk/SKILL.md`, one text prompt, and the BEV image
  as `LocalImageInput`.
- Prompt construction uses `Stage2EvidenceBundle.extra_metadata.vg_proposal_pool`
  and `scene_catalog`; tests assert benchmark GT fields are not serialized into
  the Codex prompt.
- One adapter-side 429 was observed during monitoring, but the final run has
  600 completed checkpoints and 0 persisted errors.
- `workers=4` was stable with the shared project `.codex-home`; no SQLite lock
  or app-server concurrency errors were observed.

## SQLite Reproduction Query

```sql
SELECT
  run_id,
  n,
  ROUND(classification_acc_filtered * 100, 2) AS overall,
  ROUND(acc_easy * 100, 2) AS easy,
  ROUND(acc_hard * 100, 2) AS hard,
  ROUND(acc_view_dep * 100, 2) AS view_dep,
  ROUND(acc_view_indep * 100, 2) AS view_indep
FROM runs
WHERE run_id = 'v17_codex_sdk_strat600_g54_20260605';
-- v17_codex_sdk_strat600_g54_20260605 | 600 | 63.33 | 70.00 | 57.10 | 53.55 | 68.64
```

## Interpretation

This validates that the Codex Agent SDK can drive the NR3D pack format through
the local ModelHub adapter and produce benchmark-grade structured outputs over
the canonical 600-case fold. Accuracy is below the DeepAgents tool-using
runtime because this first SDK entrypoint is a one-turn catalog+BEV selector:
it does not call `select_by_text`, `mark_frame_with_bbox`,
`compare_proposals_spatial`, TADG, no-match guard, or evidence-frame guard.

The next integration step is to expose the existing VG tools to Codex through a
controlled MCP or command-tool surface, then rerun the same strat600 fold.
