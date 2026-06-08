# NR3D v22 Codex SDK CLI-only FULL

This run is the full-set confirmation of the v20 Codex SDK CLI-only strat600
pilot. It uses the Codex Agent SDK backend with MCP disabled and CLI evidence
tools enabled, routed through the local ModelHub adapter's weighted 3-AK TOML
pool.

## Pre-run Checklist

| Item | Value |
|---|---|
| Branch | `feat/intro-codex-agent-sdk` |
| Head commit at launch | `2837f2c` |
| Run-time code commit | `2837f2c` for the final continuation and assembly. The output directory is a resume artifact: 2622 valid initial checkpoints came from the prior `3f0236a` / `462b102` resume artifact, then the continuation ran from the current branch tip. |
| Worktree drift | No source-code worktree drift for the continuation. The temporary chunk runner under `tmp/` was patched operationally to quarantine retry-exhausted technical failures and continue. |
| Backend | `codex_sdk` |
| Model | `gpt-5.4-2026-03-05` |
| Fold | Full NR3D test fold: `tmp/nr3d_artifacts/full_test_sample_ids.json` |
| Fold size | 8584 full utterances / 7805 canonical filtered utterances |
| Pack | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_eval_codex_sdk_cli_isolatedhome_full_20260607_462b102_resume/` |
| Run id | `v22_codex_sdk_cli_isolatedhome_full_20260607_w70_3ak` |
| Workers / chunk size | `WORKERS=70`, `CHUNK_SIZE=100` |
| Tool mode | `CODEX_AGENT_ENABLE_MCP_TOOLS=0`, `CODEX_AGENT_ENABLE_CLI_TOOLS=1` |
| Prefix cache | Stable `CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_sdk_cli_isolatedhome_full_20260607_462b102`; per-turn `chat_run_id` used for weighted AK distribution |
| ModelHub AK routing | Private weighted TOML pool, 3 upstreams at 5:1:5. Real AK values stay in `/Users/bytedance/aispace/codex_modelhub_adapter/.modelhub_upstreams.toml`, not in tracked docs. |
| Stage-1 text retrieval | Serialized with `STAGE1_TEXT_RETRIEVAL_MAX_CONCURRENCY=1` |

## Exact Commands

The monitored run was launched and resumed through:

```bash
cd /Users/bytedance/project/3DVLMReasoning
WORKERS=70 CHUNK_SIZE=100 bash tmp/run_nr3d_full_chunked_cleanhome_3f0236a.sh
```

The runner assembled side-by-side output, computed leaderboard metrics, and
ingested SQLite with:

```bash
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_isolatedhome_full_20260607_462b102_resume \
  --workers 1 \
  --sample-retries 1

.venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_codex_sdk_cli_isolatedhome_full_20260607_462b102_resume/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --backend codex_sdk \
  --output tmp/nr3d_eval_codex_sdk_cli_isolatedhome_full_20260607_462b102_resume/leaderboard_full.json

.venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_isolatedhome_full_20260607_462b102_resume \
  --run-id v22_codex_sdk_cli_isolatedhome_full_20260607_w70_3ak \
  --branch feat/intro-codex-agent-sdk \
  --commit 2837f2c \
  --backend codex_sdk \
  --judge-model gpt-5.4-2026-03-05 \
  --leaderboard-metrics tmp/nr3d_eval_codex_sdk_cli_isolatedhome_full_20260607_462b102_resume/leaderboard_full.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Headline Metrics

| Metric | v22 Codex SDK CLI-only FULL | v20 Codex SDK CLI-only strat600 | v11 enrichment FULL | UniVLG |
|---|---:|---:|---:|---:|
| Overall filtered | **74.61** | 74.50 | 72.26 | 65.20 |
| Easy | **82.83** | 81.38 | 81.74 | 73.30 |
| Hard | **66.91** | 68.06 | 63.39 | 57.00 |
| View-dependent | **64.86** | 64.45 | 62.46 | 55.10 |
| View-independent | **79.91** | 79.95 | 77.60 | 69.90 |

Counts:

| Slice | n | correct | accuracy |
|---|---:|---:|---:|
| Full | 8584 | 6135 | 71.47 |
| Filtered | 7805 | 5823 | 74.61 |
| Easy | 3773 | 3125 | 82.83 |
| Hard | 4032 | 2698 | 66.91 |
| View-dependent | 2752 | 1785 | 64.86 |
| View-independent | 5053 | 4038 | 79.91 |

Pack side-by-side metrics:

| Metric | Value |
|---|---:|
| `n` | 8584 |
| `status=completed` | 8553 |
| `status=failed` | 31 |
| bad JSON checkpoints | 0 |
| final active technical failures | 0 |
| mean IoU | 0.71809 |
| Acc@0.25 | 0.71493 |
| Acc@0.50 | 0.71470 |

## Tool Distribution

SQLite contains 64256 tool-call rows for this run. Excluding the synthetic
`codex_sdk_turn` wrapper, the agent made 55672 evidence-tool calls, or 6.49
evidence-tool calls per sample.

| Tool | Calls |
|---|---:|
| `inspect_proposal` | 23722 |
| `select_by_proposal` | 14024 |
| `mark_frame_with_bbox` | 9303 |
| `codex_sdk_turn` | 8584 |
| `compare_proposals_spatial` | 4695 |
| `compare_candidates_to_anchors` | 2518 |
| `select_by_text` | 1084 |
| `list_frame_proposals` | 219 |
| `list_scene_proposals` | 53 |
| `view_bev` | 40 |
| `select_by_region` | 14 |

`select_by_text` had 1084 successful calls and no final active
`select_by_text` error signature in the health summary.

## Runtime Notes

- High concurrency did produce transient Codex/ModelHub technical failures,
  mainly retry-exhausted `429 Too Many Requests`, `stream disconnected`, and one
  transport-closed long tail. The chunk runner quarantined those checkpoint
  sentinels and continued from clean active checkpoints.
- Final `postprocess_health.json` reports 8584 active checkpoints, 0 bad JSON,
  and 0 active technical failures.
- MCP was intentionally disabled because prior SDK MCP attempts did not expose
  usable tool calls. This run validates the CLI-only tool surface at full scale.
- The run uses prefix-cache session stability and per-turn AK load distribution
  together: stable session id for cache affinity, per-turn `chat_run_id` for the
  weighted upstream pool.

## Artifact Retention

- The v22 raw artifacts are retained under
  `tmp/nr3d_eval_codex_sdk_cli_isolatedhome_full_20260607_462b102_resume/`.
- The ingested SQLite DB remains available locally at
  `docs/benchmark/nr3d/runs.sqlite`, but benchmark SQLite files are ignored by
  git after the 2026-06-08 repository cleanup to avoid committing large binary
  DB revisions.
- Historical temporary NR3D full-run outputs already summarized in docs and
  SQLite were pruned on 2026-06-08 to recover local disk space. The pruned set
  included old v4/v5 projection-invalidated full outputs, v9.1_fix/v9.2 full
  reproduction outputs, and failed/partial Codex SDK full-run probes from
  2026-06-06 / 2026-06-07.
- Project-local `.codex-home/sessions` and `.codex-home/shell_snapshots` were
  also pruned on 2026-06-08. These were ignored Codex session/shell histories,
  not benchmark source data or the v22 side-by-side/SQLite outputs.

## Spot Checks

Two trace-level samples were inspected after completion:

| Sample | Result | Evidence checked |
|---|---|---|
| `scannet/scene0164_00::11::35445` | Correct, selected `#11` | Trace includes `select_by_proposal`, `mark_frame_with_bbox`, and `codex_sdk_turn` images. Loaded RGB `000450-rgb.png` at 1296x968, marked image `frame_54_ids_2_11.png` at 1296x968, BEV `scene_bev_nr3d.png` at 1500x1500, and light point-cloud object pack with 56 objects. |
| `scannet/scene0565_00::25::34864` | Completed miss, selected `#1` for target `#25` | Trace includes selector image frames, `mark_frame_with_bbox`, and BEV. Loaded RGB `000000-rgb.png` at 1296x968, marked image `frame_0_ids_1_12.png` at 1296x968, BEV `scene_bev_nr3d.png` at 1500x1500, and light point-cloud object pack with 41 objects. |

## SQLite Reproduction Queries

```sql
SELECT
  run_id,
  n,
  ROUND(classification_acc_full * 100, 2) AS full_acc,
  ROUND(classification_acc_filtered * 100, 2) AS filtered_acc,
  ROUND(acc_easy * 100, 2) AS easy,
  ROUND(acc_hard * 100, 2) AS hard,
  ROUND(acc_view_dep * 100, 2) AS view_dep,
  ROUND(acc_view_indep * 100, 2) AS view_indep,
  n_filtered
FROM runs
WHERE run_id = 'v22_codex_sdk_cli_isolatedhome_full_20260607_w70_3ak';
```

Expected row:

```text
v22_codex_sdk_cli_isolatedhome_full_20260607_w70_3ak|8584|71.47|74.61|82.83|66.91|64.86|79.91|7805
```

```sql
SELECT tool_name, COUNT(*)
FROM tool_calls
WHERE run_id = 'v22_codex_sdk_cli_isolatedhome_full_20260607_w70_3ak'
GROUP BY tool_name
ORDER BY COUNT(*) DESC;
```

## Interpretation

v22 confirms the v20 strat600 signal on the full NR3D test fold. The filtered
full-set result is 74.61 %, which is +2.35 pp over v11 enrichment FULL and +9.41
pp over the public UniVLG classification row. The strongest remaining gap is
still view-dependent spatial grounding, where the run is 64.86 % despite the
denser CLI tool use.
