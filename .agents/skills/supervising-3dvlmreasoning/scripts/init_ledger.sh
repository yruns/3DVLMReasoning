#!/usr/bin/env bash
set -euo pipefail

task_slug="${1:-supervision}"
repo="${REPO_ROOT:-/Users/bytedance/project/3DVLMReasoning}"
date_sh="$(TZ=Asia/Shanghai date +%Y%m%d)"
time_sh="$(TZ=Asia/Shanghai date +%H%M%S)"
ledger_dir="$repo/.ccb/supervisor_sessions"
ledger="$ledger_dir/${date_sh}-${task_slug}.md"

mkdir -p "$ledger_dir"
if [[ -e "$ledger" ]]; then
  ledger="$ledger_dir/${date_sh}-${task_slug}-${time_sh}.md"
fi
cd "$repo"

branch="$(git branch --show-current 2>/dev/null || true)"
baseline="$(git rev-parse HEAD 2>/dev/null || true)"
status_short="$(git status --short 2>/dev/null || true)"
tmux_sessions="$(tmux list-sessions 2>&1 || true)"

cat > "$ledger" <<EOF
# 3DVLMReasoning Supervisor Session

- Created: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')
- Task slug: ${task_slug}
- Repo: ${repo}
- Branch: ${branch}
- Baseline commit: ${baseline}

## tmux Sessions

\`\`\`text
${tmux_sessions}
\`\`\`

## Dirty Worktree

\`\`\`text
${status_short}
\`\`\`

## Active tmux Sessions

| Role | tmux session | Codex thread name | CWD | Scope | Status | Next review |
| --- | --- | --- | --- | --- | --- | --- |
| supervisor | TBD | TBD | ${repo} | coordination | active | TBD |

## Retired Sessions

| tmux session | Role | Reason | Time |
| --- | --- | --- | --- |

## Milestones

- [ ] Define milestones.
- [ ] Delegate non-overlapping worker scopes.
- [ ] Review diffs, artifacts, and sampled benchmark cases.

## Recovery

1. Run \`tmux list-sessions\`.
2. Read likely sessions with \`tmux capture-pane -t <session> -p -S -120\`.
3. Relaunch missing Codex workers with \`gpt-5.5/xhigh\`.
4. Reapply recorded \`/rename\` names.
5. Continue from this ledger and the latest handoff if present.
EOF

echo "$ledger"
