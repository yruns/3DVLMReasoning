# NR3D v18 Codex SDK MCP/text strat600

This run evaluates the Codex Agent SDK NR3D backend after wiring the local MCP
tool server, syncing DeepAgents playbook/guideline content into Codex skills,
and enabling prefix-cache configuration. It uses the canonical NR3D strat600
fold, also called case600 in local iteration notes.

The result is **63.00 % Overall** on strat600. This is an integration
validation run, not a new accuracy best.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `feat/intro-codex-agent-sdk` |
| Head commit at launch | `348ad52` |
| Run-time code commit | `348ad52` - no worktree drift |
| Implementation commit | `348ad52` - `Add Codex SDK MCP NR3D tools` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold design | `docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md` |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter/` |
| Side-by-side JSON | `tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter/side_by_side.json` |
| Leaderboard metrics | `tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter/leaderboard_strat600.json` |
| SQLite run id | `v18_codex_sdk_mcp_text_strat600_20260605` |
| Main side-by-side launched | `2026-06-05T21:55+08:00` |
| Side-by-side rebuilt | `2026-06-05T23:28+08:00` |
| Metrics completed | `2026-06-05T23:30+08:00` |
| SQLite ingested | `2026-06-05T23:32:26+08:00` |

## Exact Commands

Local Codex ModelHub adapter was started in tmux session
`codex-modelhub-adapter-8787`:

```bash
tmux new-session -d -s codex-modelhub-adapter-8787 '
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
  2>&1 | tee /tmp/codex_modelhub_adapter_8787.log'
```

Adapter health check returned `status=healthy`, `has_upstream_ak=true`, and
upstream base `https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online`.

Main strat600/case600 run:

```bash
tmux new-session -d -s nr3d-case600-codex-adapter-20260605 '
cd /Users/bytedance/project/3DVLMReasoning
export PYTHONPATH=src
export CODEX_AGENT_ENABLE_MCP_TOOLS=1
export CODEX_AGENT_ENABLE_PREFIX_CACHE=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_case600_20260605_348ad52_adapter
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter \
  --workers 20 \
  2>&1 | tee /tmp/nr3d_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter.log'
```

One sample initially failed with a transient JSON decode error:

```text
scannet/scene0025_00::0::19704
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

After deleting only that failed checkpoint, it was rerun with retries:

```bash
PYTHONPATH=src CODEX_AGENT_ENABLE_MCP_TOOLS=1 \
CODEX_AGENT_ENABLE_PREFIX_CACHE=1 \
CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_case600_20260605_348ad52_adapter \
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids /tmp/nr3d_failed_one_348ad52.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter \
  --workers 1 \
  --sample-retries 4
```

The failed sample completed on rerun with `selected_object_id=0`,
`iou=1.0000000000000042`, and `status=completed`. The full
`side_by_side.json` was then rebuilt from the 600 completed checkpoints.

Leaderboard aggregation:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --backend codex_sdk \
  --output tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter/leaderboard_strat600.json
```

SQLite ingestion:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter \
  --run-id v18_codex_sdk_mcp_text_strat600_20260605 \
  --branch feat/intro-codex-agent-sdk \
  --commit 348ad52 \
  --backend codex_sdk \
  --judge-model nr3d-classifier \
  --notes "Codex SDK backend with MCP tools enabled, prefix cache enabled, and lazy select_by_text selector support; local ModelHub adapter gpt-5.4 path; model emitted no MCP tool calls in this strat600 run." \
  --leaderboard-metrics tmp/nr3d_eval_codex_sdk_mcp_select_by_text_case600_20260605_348ad52_adapter/leaderboard_strat600.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Headline Metrics

| Metric | v18 Codex SDK MCP/text | v17 Codex SDK | v11 enriched strat600 | v10 best observed | v9.3 text-first |
|---|---:|---:|---:|---:|---:|
| Overall | 63.00 | 63.33 | 71.33 | 72.00 | 66.67 |
| Easy | 70.69 | 70.00 | 79.66 | 82.41 | 73.45 |
| Hard | 55.81 | 57.10 | 63.55 | 62.26 | 60.32 |
| V-Dep | 55.45 | 53.55 | 61.61 | 62.56 | 54.98 |
| V-Indep | 67.10 | 68.64 | 76.61 | 77.12 | 73.01 |

Counts:

| Slice | n | correct |
|---|---:|---:|
| Full / filtered | 600 | 378 |
| Easy | 290 | 205 |
| Hard | 310 | 173 |
| V-Dep | 211 | 117 |
| V-Indep | 389 | 261 |

Pack side-by-side metrics:

| Metric | Value |
|---|---:|
| `n` | 600 |
| `status=completed` | 600 |
| checkpoint `error` fields | 0 |
| mean IoU | 0.63556 |
| Acc@0.25 | 0.63000 |
| Acc@0.50 | 0.63000 |

## Tool and Runtime Notes

- Codex SDK input loaded four skill files:
  `.agents/skills/nr3d-codex-sdk/SKILL.md`,
  `.agents/skills/scene-exploration-playbook/SKILL.md`,
  `.agents/skills/vg-grounding-playbook/SKILL.md`, and
  `.agents/skills/vg-spatial-disambiguation/SKILL.md`.
- `CODEX_AGENT_ENABLE_MCP_TOOLS=1` was active. Per-sample trace input records
  `mcp_tools_enabled=true`, `mcp_server_path=src/agents/mcp/nr3d_tools_server.py`,
  and a per-sample MCP state path.
- Actual observed tool trace was only `codex_sdk_turn` for all 600 samples:
  `{'codex_sdk_turn': 600}`. No `select_by_text`, `view_bev`,
  `inspect_proposal`, or MCP trace JSON call files were produced.
- Prefix cache config was active with
  `CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_case600_20260605_348ad52_adapter`.
- Model path in persisted tool trace is `gpt-5.4-2026-03-05` through
  `modelhub_adapter`.

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
WHERE run_id = 'v18_codex_sdk_mcp_text_strat600_20260605';
-- v18_codex_sdk_mcp_text_strat600_20260605 | 600 | 63.00 | 70.69 | 55.81 | 55.45 | 67.10
```

## Interpretation

This run validates that the Codex SDK backend still completes the canonical
600-case fold after MCP server wiring, playbook-to-skill sync, lazy
`select_by_text` selector initialization support, and prefix-cache
configuration. It is statistically flat with the earlier v17 one-turn SDK run
(-0.33 pp Overall), and remains below the tool-using DeepAgents rows.

The main caveat is that the model did not invoke MCP tools. Treat this as an
SDK/backend plus configuration validation, not evidence that MCP-augmented
reasoning improves NR3D accuracy.

## Preflight Failure

An earlier attempt before starting the local adapter failed 20 / 20 samples
with:

```text
RuntimeError: stream disconnected before completion: error sending request for url (http://127.0.0.1:8787/v1/responses)
```

Root cause was local `codex_modelhub_adapter` not listening on
`127.0.0.1:8787`. Starting the adapter fixed the connectivity issue; a
single-sample smoke then completed before the full case600 run was launched.
