#!/usr/bin/env bash
set -euo pipefail

role_raw="${1:?usage: launch_codex_session.sh <role> <task-slug>}"
task_slug_raw="${2:?usage: launch_codex_session.sh <role> <task-slug>}"
role="$(printf '%s' "$role_raw" | tr -c 'A-Za-z0-9_-' '-')"
task_slug="$(printf '%s' "$task_slug_raw" | tr -c 'A-Za-z0-9_-' '-')"
repo="${REPO_ROOT:-/Users/bytedance/project/3DVLMReasoning}"
date_sh="$(TZ=Asia/Shanghai date +%Y%m%d)"
time_sh="$(TZ=Asia/Shanghai date +%H%M%S)"
session_name="worker-${role}-${date_sh}-${time_sh}"
ledger="$repo/.ccb/supervisor_sessions/${date_sh}-${task_slug}.md"
codex_cmd="cd '$repo' && codex -m gpt-5.5 -c model_reasoning_effort=\\\"xhigh\\\" --cd '$repo'"

if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "ERROR: tmux session already exists: $session_name" >&2
  exit 2
fi

tmux new-session -d -s "$session_name" "$codex_cmd"
sleep 4
tmux send-keys -t "$session_name" "/rename $session_name" C-m
sleep 2
screen="$(tmux capture-pane -t "$session_name" -p -S -80 2>&1 || true)"
if ! grep -q "Thread renamed to $session_name" <<<"$screen"; then
  tmux send-keys -t "$session_name" C-m
  sleep 2
fi
screen="$(tmux capture-pane -t "$session_name" -p -S -80 2>&1 || true)"

mkdir -p "$(dirname "$ledger")"
{
  echo
  echo "## tmux Session Launch: ${session_name}"
  echo
  echo "- Time: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "- Role: ${role}"
  echo "- tmux session: ${session_name}"
  echo "- Command: codex -m gpt-5.5 -c model_reasoning_effort=\"xhigh\" --cd ${repo}"
  echo
  echo '```text'
  echo "$screen"
  echo '```'
} >> "$ledger"

echo "$session_name"
