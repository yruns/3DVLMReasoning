#!/usr/bin/env bash
set -euo pipefail

ledger="${1:?usage: monitor_sessions.sh <ledger.md> [lines]}"
lines="${2:-120}"

if [[ ! -f "$ledger" ]]; then
  echo "ERROR: ledger not found: $ledger" >&2
  exit 2
fi

sessions="$(grep -Eo 'worker-[A-Za-z0-9_-]+-[0-9]{8}-[0-9]{6}' "$ledger" | sort -u || true)"
if [[ -z "$sessions" ]]; then
  echo "No worker tmux sessions found in $ledger" >&2
  exit 0
fi

for session in $sessions; do
  echo "===== ${session} ====="
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux capture-pane -t "$session" -p -S "-$lines" 2>&1 || true
  else
    echo "tmux session not running"
  fi
  echo
done
