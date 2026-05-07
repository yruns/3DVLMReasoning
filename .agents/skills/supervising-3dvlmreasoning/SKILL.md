---
name: supervising-3dvlmreasoning
description: Use when a Codex-only supervisor is needed for 3DVLMReasoning benchmark fixes, multimodal case audits, long evaluations, worker coordination, reviews, or sign-off.
---

# Supervising 3DVLMReasoning

## Core Rule

Act as the strict Codex tech lead for `/Users/bytedance/project/3DVLMReasoning`: delegate bounded work to Codex workers in tmux sessions, but personally verify evidence, diffs, tests, benchmark artifacts, sampled cases, and project-rule compliance before sign-off.

## Non-Negotiables

- **tmux only for workers.** Do not delegate workers through cmux surfaces. Use tmux sessions for all Codex workers, audits, reviews, and long evals.
- **Codex only.** Never invoke, resume, monitor, or delegate through Claude Code, `claude`, `ttadk code`, `auto-claude.sh`, or Claude-backed sessions.
- **Best model only.** All delegated Codex workers use `gpt-5.5` with `xhigh`. If not explicit and confirmed, do not delegate. Cost is irrelevant.
- **Recoverability first.** Before delegation, rename every Codex thread and record active tmux sessions in a ledger.
- **Context guard.** Below 20% context, immediately use `handoff`, write the doc, `/clear`, read it, and continue.
- **No silent fallback.** If images, point clouds, checkpoints, GPU, or dependencies cannot load, fix the prerequisite or report blocked.
- **Verify.** Worker reports never replace reading diffs, logs, tests, images, point-cloud artifacts, and benchmark outputs yourself.

## Required Skills

Use and require workers to use relevant skills first: `superpowers:systematic-debugging` for failures; `superpowers:test-driven-development` before behavior changes; `superpowers:writing-plans` for multi-step work; `superpowers:requesting-code-review` after worker output and before sign-off; `superpowers:verification-before-completion` before any complete/fixed/passing claim; `handoff` before 20% context. Do not use native subagents as workers; tmux sessions are the worker mechanism.

## tmux Session Lifecycle

For implementation, long evals, independent reviews, and case audits, open one Codex CLI worker per non-overlapping task in its own tmux session. Session names must be stable and role-specific:

```bash
tmux new-session -d -s worker-audit-YYYYMMDD-HHMMSS \
  'cd /Users/bytedance/project/3DVLMReasoning && codex -m gpt-5.5 -c model_reasoning_effort=\"xhigh\" --cd /Users/bytedance/project/3DVLMReasoning'
tmux capture-pane -t worker-audit-YYYYMMDD-HHMMSS -p -S -80
tmux send-keys -t worker-audit-YYYYMMDD-HHMMSS '/rename worker-audit-YYYYMMDD-HHMMSS' Enter
```

Always verify the Codex screen shows `gpt-5.5 xhigh` and that `/rename` was accepted before sending work.

## Recovery Ledger

Maintain `.ccb/supervisor_sessions/<YYYYMMDD>-<task-slug>.md` from the start. Record:

- task summary, milestones, branch, baseline commit, and dirty-worktree note;
- `tmux list-sessions` snapshot;
- every active tmux session: role, tmux session name, Codex thread rename, cwd, assigned scope, status, blocker, next review time;
- retired sessions and reason;
- communication tags, e.g. `[SUPERVISOR]`, `[WORKER-AUDIT:worker-audit-20260507-224730]`;
- recovery instructions: rerun `tmux list-sessions`, `tmux capture-pane`, relaunch missing Codex workers with `gpt-5.5/xhigh`, reapply `/rename`, continue from this ledger.

Update it after spawning, renaming, delegation, milestone review, worker retirement, handoff, and final sign-off.

Helper scripts are in `scripts/`:

```bash
.agents/skills/supervising-3dvlmreasoning/scripts/init_ledger.sh <task-slug>
.agents/skills/supervising-3dvlmreasoning/scripts/launch_codex_session.sh <role> <task-slug>
.agents/skills/supervising-3dvlmreasoning/scripts/send_brief_to_session.sh <tmux-session> <brief.md>
.agents/skills/supervising-3dvlmreasoning/scripts/monitor_sessions.sh .ccb/supervisor_sessions/<file>.md
```

## Worker Brief Protocol

Every tmux worker brief must include:

- identity tag and response protocol: write a status line in the tmux session and/or artifact path;
- goal, files/artifacts, branch, baseline commit, write scope, non-goals, output path, verification commands;
- `gpt-5.5/xhigh` recording, context reporting, and the 20% handoff rule;
- peer worker names/scopes so workers do not duplicate or overwrite work;
- "You are not alone in the codebase; do not revert edits made by others."

Use separate output paths per worker, e.g. `tmp/review_worker_a.md`, `tmp/case_audit_worker_b.md`. Send long briefs from a file with `send_brief_to_session.sh`; verify with `tmux capture-pane -t <session> -p -S -60`.

## Monitoring Loop

Every cycle:

1. Run `tmux list-sessions`.
2. Read each worker with `tmux capture-pane -t <session> -p -S -120`.
3. Review `git diff <baseline>` and worker artifacts.
4. Update the ledger.
5. Send feedback with file paths, evidence, and exact verification expected.

Use a 20-minute cadence by default, 10 minutes after must-fix feedback, and immediate checks for blocker/status messages. Long commands inside workers still run inside their tmux session, with logs:

```bash
tmux send-keys -t <session> "cd /Users/bytedance/project/3DVLMReasoning && <command> 2>&1 | tee /tmp/<name>.log" Enter
tmux capture-pane -t <session> -p -S -120
```

## Multimodal Case Audit

For every suspect OpenEQA, ScanRefer, SQA3D, or embodied VG case used for a decision/sign-off, delegate an independent Codex case-audit worker unless the user forbids delegation.

Briefs must include `case_id`/`qid`, query/refexp, prediction/output, GT, image/crop paths, point-cloud/mesh/proposal paths, trace/log paths, and expected audit table. The audit must:

- re-read the query and task type;
- open images/crops and check visibility, distractors, occlusion, and visual support;
- load, render, or compute on actual PLY/mesh/proposal geometry; metadata alone is insufficient;
- trace Stage 1 parse, selected keyframes, proposal ids, tool calls, and Stage 2 answer;
- compare output to GT/evidence and classify retrieval, proposal coverage, ranking, reasoning, metric, or documentation failure.

Never accept "code agents cannot inspect images or point clouds." Codex can open images, load visualizations, write inspection scripts, summarize geometry, render point clouds/meshes, or request/generate screenshots.

## Benchmark Gate

Do not accept aggregate metrics alone. For every benchmark validation, sample multiple concrete cases, including successes and failures when available. Load the sampled images/crops and point-cloud/mesh artifacts before accepting the result. Benchmark runs also require `docs/benchmark/<name>/`, README/leaderboard updates, and SQLite ingestion.

## Project Gates

- Linux perception work uses conda `conceptgraph`; macOS uses `uv` / `.venv`.
- On Linux, never use `CUDA_VISIBLE_DEVICES=1`.
- Respect the dirty worktree; do not revert unrelated user changes.
- Final sign-off requires fresh tests, sampled case inspection with loaded images/point clouds, benchmark-doc checks when applicable, and residual risks.
