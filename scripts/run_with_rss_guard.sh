#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run_with_rss_guard.sh --rss-limit-mb <mb> [--check-interval-sec <sec>] -- <command> [args...]

Runs a command and exits with 137 if the command process tree exceeds the RSS
limit. This is intended for tmux-launched benchmark jobs where unbounded memory
growth would otherwise destabilize the workstation.
EOF
}

rss_limit_mb=""
check_interval_sec="60"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rss-limit-mb)
      rss_limit_mb="${2:-}"
      shift 2
      ;;
    --check-interval-sec)
      check_interval_sec="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$rss_limit_mb" || $# -eq 0 ]]; then
  usage
  exit 2
fi

if ! [[ "$rss_limit_mb" =~ ^[0-9]+$ ]] || [[ "$rss_limit_mb" -le 0 ]]; then
  echo "--rss-limit-mb must be a positive integer" >&2
  exit 2
fi

if ! [[ "$check_interval_sec" =~ ^[0-9]+$ ]] || [[ "$check_interval_sec" -le 0 ]]; then
  echo "--check-interval-sec must be a positive integer" >&2
  exit 2
fi

rss_limit_kb=$((rss_limit_mb * 1024))

rss_tree_kb() {
  local root_pid="$1"
  local total=0
  local queue=("$root_pid")
  local pid
  local rss
  local children

  while [[ "${#queue[@]}" -gt 0 ]]; do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    rss="$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print $1}' || true)"
    if [[ -n "$rss" ]]; then
      total=$((total + rss))
    fi
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    if [[ -n "$children" ]]; then
      while IFS= read -r child_pid; do
        [[ -n "$child_pid" ]] && queue+=("$child_pid")
      done <<< "$children"
    fi
  done

  echo "$total"
}

"$@" &
child_pid=$!
guard_done="$(mktemp -t rss_guard_done.XXXXXX)"
rm -f "$guard_done"

guard_status=0
(
  last_check=0
  while kill -0 "$child_pid" 2>/dev/null; do
    if [[ -e "$guard_done" ]]; then
      exit 0
    fi
    now="$(date +%s)"
    if [[ $((now - last_check)) -lt "$check_interval_sec" ]]; then
      sleep 1
      continue
    fi
    last_check="$now"
    rss_kb="$(rss_tree_kb "$child_pid")"
    if [[ "$rss_kb" -gt "$rss_limit_kb" ]]; then
      rss_mb=$((rss_kb / 1024))
      echo "[rss_guard] RSS ${rss_mb}MB exceeded limit ${rss_limit_mb}MB; terminating pid ${child_pid}" >&2
      kill -TERM "$child_pid" 2>/dev/null || true
      sleep 10
      kill -KILL "$child_pid" 2>/dev/null || true
      exit 137
    fi
  done
) &
guard_pid=$!

set +e
wait "$child_pid" 2>/dev/null
child_status=$?
touch "$guard_done"
wait "$guard_pid" 2>/dev/null
guard_status=$?
set -e
rm -f "$guard_done"

if [[ "$guard_status" -eq 137 ]]; then
  exit 137
fi

exit "$child_status"
