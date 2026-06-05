# NR3D v19 Codex SDK CLI fallback strat600

This run evaluates the Codex Agent SDK NR3D backend after enabling the CLI
fallback tool path on top of the local MCP server configuration, synced Codex
skills, and prefix-cache configuration. It uses the canonical NR3D strat600
fold, also called case600 in local iteration notes.

The result is **72.50 % Overall** on strat600. This is the best observed valid
strat600 pilot in this archive, but it is not a full-set replacement for the
v11 proposal-enrichment FULL row.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `feat/intro-codex-agent-sdk` |
| Head commit at launch | `c19a747` |
| Run-time code commit | `c19a747` - no worktree drift |
| Implementation commit | `c19a747` - `Add Codex SDK CLI fallback for NR3D tools` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold design | `docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md` |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747/` |
| Side-by-side JSON | `tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747/side_by_side.json` |
| Leaderboard metrics | `tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747/leaderboard_strat600.json` |
| SQLite run id | `v19_codex_sdk_cli_fallback_strat600_20260606` |
| Main checkpoints | `2026-06-06T01:51:27+08:00` to `2026-06-06T03:56:54+08:00` |
| Failed-checkpoint rerun | `2026-06-06T03:57+08:00` to `2026-06-06T04:00:12+08:00` |
| Full side-by-side rebuilt | `2026-06-06T04:02:48+08:00` |
| Metrics completed | `2026-06-06T04:04:59+08:00` |
| SQLite ingested | `2026-06-06T04:05:42+08:00`; re-ingested after tool-call ingestion fix |

## Exact Commands

Local Codex ModelHub adapter was started in tmux session
`codex-modelhub-adapter-8787-case600-cli`:

```bash
tmux new-session -d -s codex-modelhub-adapter-8787-case600-cli '
cd /Users/bytedance/aispace/codex_modelhub_adapter &&
export AIDP_GPT_AK=$(/Users/bytedance/project/3DVLMReasoning/.venv/bin/python - <<PY
import sys
sys.path.insert(0, "/Users/bytedance/project/3DVLMReasoning/src")
from agents.core.agent_config import Stage2DeepAgentConfig
print(Stage2DeepAgentConfig().api_keys[0])
PY
) &&
export AIDP_CODEX_PROXY_UPSTREAM_ENV=office &&
export AIDP_CODEX_PROXY_UPSTREAM_API=auto &&
export AIDP_CODEX_PROXY_CHAT_COMPLETIONS_MODELS="gpt-5.*" &&
uv run uvicorn adapter.app:app --host 127.0.0.1 --port 8787 \
  2>&1 | tee /tmp/codex_modelhub_adapter_8787_case600_cli.log'
```

Adapter health check returned `status=healthy`, `has_upstream_ak=true`, and
upstream base `https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online`.

Main strat600/case600 run:

```bash
tmux new-session -d -s nr3d-case600-codex-cli-20260606 '
cd /Users/bytedance/project/3DVLMReasoning
export PYTHONPATH=src
export CODEX_AGENT_ENABLE_MCP_TOOLS=1
export CODEX_AGENT_ENABLE_CLI_TOOLS=1
export CODEX_AGENT_ENABLE_PREFIX_CACHE=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_cli_case600_20260606_c19a747
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747 \
  --workers 20 \
  --sample-retries 2 \
  2>&1 | tee /tmp/nr3d_codex_sdk_cli_fallback_case600_20260606_c19a747.log'
```

Two samples initially failed with transient JSON decode errors:

```text
scannet/scene0643_00::26::21245
scannet/scene0153_00::18::27180
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

After deleting only those failed checkpoint files, they were rerun with low
concurrency and additional retries:

```bash
tmux new-session -d -s nr3d-case600-codex-cli-rerun2-20260606 '
cd /Users/bytedance/project/3DVLMReasoning
export PYTHONPATH=src
export CODEX_AGENT_ENABLE_MCP_TOOLS=1
export CODEX_AGENT_ENABLE_CLI_TOOLS=1
export CODEX_AGENT_ENABLE_PREFIX_CACHE=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_cli_case600_20260606_c19a747
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/codex_sdk_cli_fallback_case600_failed2_20260606.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747 \
  --workers 1 \
  --sample-retries 4 \
  2>&1 | tee /tmp/nr3d_codex_sdk_cli_fallback_case600_failed2_rerun_20260606.log'
```

Both failed samples completed on rerun:

| Sample | Status | Selected object | IoU |
|---|---|---:|---:|
| `scannet/scene0643_00::26::21245` | completed | empty normalized prediction | 0.0000 |
| `scannet/scene0153_00::18::27180` | completed | target object | 1.0000 |

The full `side_by_side.json` was rebuilt from the 600 completed checkpoints:

```bash
tmux new-session -d -s nr3d-case600-codex-cli-assemble-20260606 '
cd /Users/bytedance/project/3DVLMReasoning
export PYTHONPATH=src
export CODEX_AGENT_ENABLE_MCP_TOOLS=1
export CODEX_AGENT_ENABLE_CLI_TOOLS=1
export CODEX_AGENT_ENABLE_PREFIX_CACHE=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_cli_case600_20260606_c19a747
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747 \
  --workers 1 \
  --sample-retries 1 \
  2>&1 | tee /tmp/nr3d_codex_sdk_cli_fallback_case600_assemble_20260606.log'
```

Leaderboard aggregation:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --backend codex_sdk \
  --output tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747/leaderboard_strat600.json
```

SQLite ingestion:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747 \
  --run-id v19_codex_sdk_cli_fallback_strat600_20260606 \
  --branch feat/intro-codex-agent-sdk \
  --commit c19a747 \
  --backend codex_sdk \
  --judge-model nr3d-classifier \
  --notes "Codex SDK backend with MCP enabled and CLI fallback tool calls; prefix cache enabled; failed checkpoint retry repaired two JSONDecodeError samples." \
  --leaderboard-metrics tmp/nr3d_eval_codex_sdk_cli_fallback_case600_20260606_c19a747/leaderboard_strat600.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Headline Metrics

| Metric | v19 Codex SDK CLI fallback | v18 Codex SDK MCP/text | v17 Codex SDK | v11 enriched strat600 | v10 best observed |
|---|---:|---:|---:|---:|---:|
| Overall | 72.50 | 63.00 | 63.33 | 71.33 | 72.00 |
| Easy | 80.34 | 70.69 | 70.00 | 79.66 | 82.41 |
| Hard | 65.16 | 55.81 | 57.10 | 63.55 | 62.26 |
| V-Dep | 59.72 | 55.45 | 53.55 | 61.61 | 62.56 |
| V-Indep | 79.43 | 67.10 | 68.64 | 76.61 | 77.12 |

Counts:

| Slice | n | correct |
|---|---:|---:|
| Full / filtered | 600 | 435 |
| Easy | 290 | 233 |
| Hard | 310 | 202 |
| V-Dep | 211 | 126 |
| V-Indep | 389 | 309 |

Pack side-by-side metrics:

| Metric | Value |
|---|---:|
| `n` | 600 |
| `status=completed` | 600 |
| checkpoint `error` fields | 0 |
| mean IoU | 0.72919 |
| Acc@0.25 | 0.72500 |
| Acc@0.50 | 0.72500 |
| empty normalized predictions | 0 |
| IoU 0 samples | 107 |

## Tool and Runtime Notes

- `CODEX_AGENT_ENABLE_MCP_TOOLS=1` and `CODEX_AGENT_ENABLE_CLI_TOOLS=1` were
  both active.
- Prefix cache config was active with
  `CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_cli_case600_20260606_c19a747`.
- The CLI fallback tool path was exercised in the actual strat600 run. This is
  the main difference from v18, where only `codex_sdk_turn` appeared.
- Total tool trace entries: 2592, average 4.32 per sample, max 12.
- SQLite `tool_calls` was populated from `side_by_side.per_sample[].tool_trace`
  after fixing `scripts/ingest_nr3d_run.py`; it contains the same 2592 calls.

Observed tool counts:

| Tool | Count |
|---|---:|
| `inspect_proposal` | 1574 |
| `codex_sdk_turn` | 600 |
| `compare_proposals_spatial` | 149 |
| `select_by_proposal` | 139 |
| `compare_candidates_to_anchors` | 37 |
| `list_frame_proposals` | 28 |
| `select_by_text` | 24 |
| `mark_frame_with_bbox` | 23 |
| `list_scene_proposals` | 16 |
| `view_bev` | 1 |
| `select_by_region` | 1 |

## SQLite Reproduction Queries

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
WHERE run_id = 'v19_codex_sdk_cli_fallback_strat600_20260606';
-- v19_codex_sdk_cli_fallback_strat600_20260606 | 600 | 72.50 | 80.34 | 65.16 | 59.72 | 79.43
```

```sql
SELECT tool_name, COUNT(*)
FROM tool_calls
WHERE run_id = 'v19_codex_sdk_cli_fallback_strat600_20260606'
GROUP BY tool_name
ORDER BY COUNT(*) DESC;
-- inspect_proposal 1574; codex_sdk_turn 600; compare_proposals_spatial 149; ...
```

## Interpretation

This run validates that the Codex SDK backend can run the canonical case600
fold with real CLI fallback tool calls and the local MCP server configuration.
Accuracy jumps +9.50 pp over v18 and +9.17 pp over v17, mainly because the SDK
path now reaches the same style of proposal-inspection and spatial-comparison
tools instead of staying in a one-turn catalog selector.

The run is also +1.17 pp over the v11 enriched strat600 pilot and +0.50 pp over
the previous best observed valid strat600 row, v10 multi-anchor TADG. This
delta is smaller than the strat600 90 % Overall comparison band, so it should be
treated as a strong integration result and best observed pilot, not a stable
new full-set claim. A full 7805-query filtered run is needed before changing the
published/full-set headline.

The main runtime caveat is throughput: the Codex SDK CLI/MCP path starts
per-sample Codex app-server and MCP server processes, so wall time was much
slower than the pure DeepAgents runner. Two transient JSON decode failures were
repairable by deleting only the failed checkpoints and rerunning those sample
ids with low concurrency.
