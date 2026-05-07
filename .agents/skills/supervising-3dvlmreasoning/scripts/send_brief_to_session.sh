#!/usr/bin/env bash
set -euo pipefail

session="${1:?usage: send_brief_to_session.sh <tmux-session> <brief.md>}"
brief_file="${2:?usage: send_brief_to_session.sh <tmux-session> <brief.md>}"

if [[ ! -f "$brief_file" ]]; then
  echo "ERROR: brief file not found: $brief_file" >&2
  exit 2
fi

if ! tmux has-session -t "$session" 2>/dev/null; then
  echo "ERROR: tmux session not found: $session" >&2
  exit 3
fi

tmux load-buffer -b 3dvlm-supervisor-brief "$brief_file"
tmux paste-buffer -b 3dvlm-supervisor-brief -t "$session"
tmux send-keys -t "$session" Enter
tmux capture-pane -t "$session" -p -S -80
