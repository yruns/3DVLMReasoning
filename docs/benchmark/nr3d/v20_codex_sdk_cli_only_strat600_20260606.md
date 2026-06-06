# NR3D v20 Codex SDK CLI-only strat600

This run evaluates the Codex Agent SDK NR3D backend after disabling MCP tools
and keeping only the CLI evidence-tool surface. It uses the canonical NR3D
strat600 fold, also called case600 in local iteration notes.

The result is **74.50 % Overall** on strat600. This is the best observed valid
strat600 pilot in this archive, but it is not a full-set replacement for the
v11 proposal-enrichment FULL row until the full filtered 7805-query slice is
rerun.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `feat/intro-codex-agent-sdk` |
| Head commit at launch | `33376e2` |
| Run-time code commit | `33376e2` - no worktree drift |
| Implementation commit | `33376e2` - `Make Codex SDK NR3D tools CLI-only by default` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold design | `docs/benchmark/nr3d/v9_3_strat600_subset_design_20260517.md` |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Backend | `codex_sdk` |
| Output dir | `tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2/` |
| Side-by-side JSON | `tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2/side_by_side.json` |
| Leaderboard metrics | `tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2/leaderboard_strat600.json` |
| SQLite run id | `v20_codex_sdk_cli_only_strat600_20260606` |
| Main checkpoints | 600 checkpoints produced; 14 transient JSONDecodeError checkpoints rerun |
| Final side-by-side rebuilt | `2026-06-06T14:52:41+08:00` |
| SQLite ingested | `v20_codex_sdk_cli_only_strat600_20260606` in `docs/benchmark/nr3d/runs.sqlite` |

## Exact Commands

Local Codex ModelHub adapter was started in tmux session
`codex-modelhub-adapter-8787-20260606`. The first attempt used a degraded
fallback key; it was restarted with `AIDP_GPT_AK` from
`Stage2DeepAgentConfig().api_keys[0]`. The accepted health check returned
`status=healthy`, `has_upstream_ak=true`, and upstream base
`https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online`.

Main strat600/case600 run:

```bash
tmux new-session -d -s nr3d-case600-codex-cli-only-20260606 '
cd /Users/bytedance/project/3DVLMReasoning
export PYTHONPATH=src
export CODEX_HOME=/Users/bytedance/project/3DVLMReasoning/.codex-home
export CODEX_AGENT_ENABLE_MCP_TOOLS=0
export CODEX_AGENT_ENABLE_CLI_TOOLS=1
export CODEX_AGENT_ENABLE_PREFIX_CACHE=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_cli_only_case600_20260606_33376e2
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2 \
  --workers 20 \
  --sample-retries 2'
```

Fourteen samples initially failed with transient JSON decode checkpoint
sentinels. After deleting only those failed checkpoint files, they were rerun
with low concurrency and additional retries:

```bash
tmux new-session -d -s nr3d-case600-codex-cli-only-rerun14-20260606 '
cd /Users/bytedance/project/3DVLMReasoning
export PYTHONPATH=src
export CODEX_HOME=/Users/bytedance/project/3DVLMReasoning/.codex-home
export CODEX_AGENT_ENABLE_MCP_TOOLS=0
export CODEX_AGENT_ENABLE_CLI_TOOLS=1
export CODEX_AGENT_ENABLE_PREFIX_CACHE=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_cli_only_case600_20260606_33376e2
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/codex_sdk_cli_only_case600_failed14_20260606.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2 \
  --workers 1 \
  --sample-retries 4'
```

The full `side_by_side.json` was rebuilt from the 600 checkpoints:

```bash
tmux new-session -d -s nr3d-case600-codex-cli-only-assemble-20260606 '
cd /Users/bytedance/project/3DVLMReasoning
export PYTHONPATH=src
export CODEX_HOME=/Users/bytedance/project/3DVLMReasoning/.codex-home
export CODEX_AGENT_ENABLE_MCP_TOOLS=0
export CODEX_AGENT_ENABLE_CLI_TOOLS=1
export CODEX_AGENT_ENABLE_PREFIX_CACHE=1
export CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_cli_only_case600_20260606_33376e2
.venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --backend codex_sdk \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2 \
  --workers 1 \
  --sample-retries 1'
```

Leaderboard aggregation:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --backend codex_sdk \
  --output tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2/leaderboard_strat600.json
```

SQLite ingestion:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2 \
  --run-id v20_codex_sdk_cli_only_strat600_20260606 \
  --branch feat/intro-codex-agent-sdk \
  --commit 33376e2 \
  --backend codex_sdk \
  --judge-model nr3d-classifier \
  --notes "Codex SDK backend with MCP disabled and CLI evidence tools only by default; prefix cache enabled; 14 JSONDecodeError checkpoints rerun at workers=1/sample-retries=4 before final assembly." \
  --leaderboard-metrics tmp/nr3d_eval_codex_sdk_cli_only_case600_20260606_33376e2/leaderboard_strat600.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Headline Metrics

| Metric | v20 Codex SDK CLI-only | v19 CLI fallback | v18 MCP/text | v11 enriched strat600 | v10 best observed |
|---|---:|---:|---:|---:|---:|
| Overall | 74.50 | 72.50 | 63.00 | 71.33 | 72.00 |
| Easy | 81.38 | 80.34 | 70.69 | 79.66 | 82.41 |
| Hard | 68.06 | 65.16 | 55.81 | 63.55 | 62.26 |
| V-Dep | 64.45 | 59.72 | 55.45 | 61.61 | 62.56 |
| V-Indep | 79.95 | 79.43 | 67.10 | 76.61 | 77.12 |

Counts:

| Slice | n | correct |
|---|---:|---:|
| Full / filtered | 600 | 447 |
| Easy | 290 | 236 |
| Hard | 310 | 211 |
| V-Dep | 211 | 136 |
| V-Indep | 389 | 311 |

Pack side-by-side metrics:

| Metric | Value |
|---|---:|
| `n` | 600 |
| `status=completed` | 599 |
| `status=failed` | 1 |
| checkpoint JSONDecodeError sentinels after rerun | 0 |
| mean IoU | 0.74797 |
| Acc@0.25 | 0.74500 |
| Acc@0.50 | 0.74500 |
| IoU 0 samples | 109 |

The one final failed-status sample was
`scannet/scene0221_00::51::24914` (`The towel in the furthest corner of the
room`). It had a trace length of 6 and no checkpoint JSON decode error; it is a
model/business final-status miss rather than a runtime sentinel.

## Tool and Runtime Notes

- `CODEX_AGENT_ENABLE_MCP_TOOLS=0` and `CODEX_AGENT_ENABLE_CLI_TOOLS=1` were
  active.
- Prefix cache config was active with
  `CODEX_AGENT_PREFIX_CACHE_SESSION_ID=nr3d_codex_cli_only_case600_20260606_33376e2`.
- The Codex SDK prompt hid `submit_final`, `list_skills`, and `load_skill` from
  the runtime tool policy; playbook guidance was attached through the SDK skill
  prompt and the CLI evidence-tool examples.
- Actual trace contains 600 `codex_sdk_turn` records and no MCP tool-call
  records.
- Total SQLite `tool_calls` rows: 4621.
- Evidence-tool calls excluding `codex_sdk_turn`: 4021 total, average 6.70 per
  sample, median 6, min 3, max 18.

Observed tool counts:

| Tool | Count |
|---|---:|
| `inspect_proposal` | 1680 |
| `select_by_proposal` | 1006 |
| `mark_frame_with_bbox` | 667 |
| `codex_sdk_turn` | 600 |
| `compare_proposals_spatial` | 330 |
| `compare_candidates_to_anchors` | 163 |
| `select_by_text` | 156 |
| `list_frame_proposals` | 11 |
| `list_scene_proposals` | 4 |
| `select_by_region` | 2 |
| `view_bev` | 2 |

## Case Spot Checks

Three concrete samples were inspected after aggregate metrics and SQLite
ingestion. For each scene, the marked RGB evidence was opened and the lightweight
point-cloud/proposal cache was loaded from
`conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.light.pkl.gz`. These scene
packs do not include `conceptgraph/mesh.ply`, so the sanity check used the
available RGB marks, `scene_catalog.json`, `enriched_objects.json`, visibility
indices, and light point-cloud caches.

| Sample | Status | Evidence checked | Finding |
|---|---|---|---|
| `scannet/scene0256_00::4::40234` | correct, IoU 1.0 | `frame_0_ids_3_4_12.png`, light PCD cache with 14 objects | Marked frame shows #4 as the bottom box in the three-box stack, matching the query and final `proposal_id=4`. |
| `scannet/scene0081_00::8::292` | completed, wrong, IoU 0.07045 | `frame_5_ids_2_6_7_8.png`, light PCD cache with 10 objects | Tool path worked; #7 and #8 are both couch segments near the ottoman/table cluster. The miss is a semantic/part-boundary error, not a runtime failure. |
| `scannet/scene0221_00::51::24914` | final `failed`, IoU 0.0 | `frame_94_ids_50_51.png`, light PCD cache with 52 objects | The two towel-labeled proposals look like wall-mounted fixtures/hairdryer hardware; the model returned `-1`. This is a business/model failure, not a checkpoint JSON decode sentinel. |

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
WHERE run_id = 'v20_codex_sdk_cli_only_strat600_20260606';
-- v20_codex_sdk_cli_only_strat600_20260606 | 600 | 74.50 | 81.38 | 68.06 | 64.45 | 79.95
```

```sql
SELECT tool_name, COUNT(*)
FROM tool_calls
WHERE run_id = 'v20_codex_sdk_cli_only_strat600_20260606'
GROUP BY tool_name
ORDER BY COUNT(*) DESC;
-- inspect_proposal 1680; select_by_proposal 1006; mark_frame_with_bbox 667; ...
```

## Interpretation

This run validates that MCP can be removed from the Codex SDK NR3D path without
losing tool-augmented reasoning, as long as the CLI evidence tools are exposed
clearly in the prompt. Accuracy improves +2.00 pp over v19 and +11.50 pp over
v18, while tool usage becomes substantially denser: v20 averages 6.70
evidence-tool calls per sample versus v19's 3.32 evidence-tool calls per sample
(4.32 total trace entries per sample including `codex_sdk_turn`).

The largest behavioral difference from v19 is that the agent now regularly
uses proposal-targeted visual verification: `select_by_proposal` rises from 139
to 1006 calls and `mark_frame_with_bbox` rises from 23 to 667 calls. The
spatial tools also rise from 186 combined spatial comparisons in v19 to 493 in
v20. That matches the intended skill change: inspect candidates, initialize
from `select_by_text` when useful, verify candidate views, and bind spatial
relations before final selection.

Treat this as the new best observed valid strat600 pilot, not as a new full-set
headline. The strat600 variance budget is still about +/-2.3 pp Overall at
90 %, so the +2.00 pp delta over v19 is promising but should be confirmed on
the full filtered 7805-query slice before making a public benchmark claim.
