#!/bin/bash
# auto-claude.sh - 3DVLMReasoning Task Automation
#
# Executes actionable tasks from TASKS.md against the current repository state.
# The task ledger is defined in TASKS.md with 6 phases.
#
# Usage:
#   chmod +x auto-claude.sh
#   ./auto-claude.sh              # Show status
#   ./auto-claude.sh --status     # Show phase progress
#   ./auto-claude.sh --execute    # Execute next pending task
#   ./auto-claude.sh --execute-all # Execute all pending tasks
#   ./auto-claude.sh --phase N    # Execute tasks in phase N only
#   ./auto-claude.sh --dry-run    # Show what would happen
#
# Configuration:
#   TARGET_ROOT - Target repository (default: ../3DVLMReasoning)
#   MODEL       - Claude model to use (default: claude-opus-4-5)

set -euo pipefail

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$SCRIPT_DIR"
TARGET_ROOT="${TARGET_ROOT:-/Users/bytedance/project/3DVLMReasoning}"
MODEL="${MODEL:-claude-opus-4-5}"

DRY_RUN=false
EXECUTE_MODE=false
EXECUTE_ALL=false
SHOW_STATUS=false
TARGET_PHASE=""

# Files
TASKS_FILE="$SCRIPT_DIR/TASKS.md"
MIGRATION_PLAN="$SCRIPT_DIR/MIGRATION_PLAN.md"
LOG_DIR="$SCRIPT_DIR/migration_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_LOG="$LOG_DIR/migration_$TIMESTAMP.log"

# Phase ranges (bash 3.2 compatible - using functions instead of arrays)
get_phase_range() {
    case $1 in
        1) echo "100:102" ;;   # Phase 1: Cleanup
        2) echo "110:131" ;;   # Phase 2: Migration
        3) echo "200:222" ;;   # Phase 3: Architecture
        4) echo "300:302" ;;   # Phase 4: Multi-dataset
        5) echo "400:420" ;;   # Phase 5: Testing
        6) echo "500:502" ;;   # Phase 6: Validation
        *) echo "" ;;
    esac
}

get_phase_name() {
    case $1 in
        1) echo "Cleanup Unused Code" ;;
        2) echo "Migrate Missing Modules" ;;
        3) echo "Architectural Rewrite" ;;
        4) echo "Multi-Dataset Support" ;;
        5) echo "Equivalence Testing" ;;
        6) echo "Integration Validation" ;;
        *) echo "Unknown Phase" ;;
    esac
}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# ════════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ════════════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)
            EXECUTE_MODE=true
            shift
            ;;
        --execute-all)
            EXECUTE_MODE=true
            EXECUTE_ALL=true
            shift
            ;;
        --status)
            SHOW_STATUS=true
            shift
            ;;
        --phase)
            TARGET_PHASE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --target)
            TARGET_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "3DVLMReasoning Task Automation"
            echo ""
            echo "Options:"
            echo "  --status           Show migration progress by phase"
            echo "  --execute          Execute next actionable task"
            echo "  --execute-all      Execute all actionable tasks"
            echo "  --phase N          Only execute tasks in phase N (1-6)"
            echo "  --dry-run          Show what would happen"
            echo "  --target PATH      Override target repository path"
            echo "  --help, -h         Show this help"
            echo ""
            echo "Phases:"
            echo "  1: Cleanup Unused Code (TASK-100 to TASK-102)"
            echo "  2: Migrate Missing Modules (TASK-110 to TASK-131)"
            echo "  3: Architectural Rewrite (TASK-200 to TASK-222)"
            echo "  4: Multi-Dataset Support (TASK-300 to TASK-302)"
            echo "  5: Equivalence Testing (TASK-400 to TASK-420)"
            echo "  6: Integration Validation (TASK-500 to TASK-502)"
            echo ""
            echo "Examples:"
            echo "  $0 --status                # Show progress"
            echo "  $0 --execute               # Run next actionable task"
            echo "  $0 --execute --phase 1     # Run next actionable Phase 1 task"
            echo "  $0 --execute-all           # Run all actionable tasks"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# ════════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════════

mkdir -p "$LOG_DIR"

log() {
    local level="$1"
    local msg="$2"
    local color=""
    case $level in
        INFO)  color="$BLUE" ;;
        WARN)  color="$YELLOW" ;;
        ERROR) color="$RED" ;;
        OK)    color="$GREEN" ;;
        TASK)  color="$CYAN" ;;
        PHASE) color="$MAGENTA" ;;
    esac
    echo -e "${color}[$level]${NC} $(date '+%H:%M:%S') $msg"
    if [ -n "${SESSION_LOG:-}" ] && [ -d "$(dirname "$SESSION_LOG")" ]; then
        echo "[$level] $(date '+%H:%M:%S') $msg" >> "$SESSION_LOG" 2>/dev/null || true
    fi
}

check_prerequisites() {
    if [ ! -f "$TASKS_FILE" ]; then
        log ERROR "TASKS.md not found at $TASKS_FILE"
        exit 1
    fi
    if [ ! -d "$TARGET_ROOT" ]; then
        log ERROR "Target repository not found: $TARGET_ROOT"
        log INFO "Please ensure 3DVLMReasoning repository exists"
        exit 1
    fi
    if [ "$EXECUTE_MODE" = true ]; then
        if command -v ttadk &> /dev/null; then
            :
        elif command -v claude &> /dev/null; then
            log WARN "ttadk not found, will use 'claude' command instead"
        else
            log ERROR "Neither 'ttadk' nor 'claude' command is available"
            exit 1
        fi
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# TASK COUNTING
# ════════════════════════════════════════════════════════════════════════════════

count_phase_tasks() {
    local phase="$1"
    local range=$(get_phase_range "$phase")
    local start="${range%%:*}"
    local end="${range##*:}"

    local total=0
    local completed=0

    # Count tasks by reading lines and checking numeric range
    while IFS= read -r line; do
        # Extract task number
        local num=$(echo "$line" | sed -n 's/.*TASK-\([0-9]\{3\}\).*/\1/p')
        if [ -n "$num" ]; then
            # Remove leading zeros for comparison
            num=$((10#$num))
            if [ "$num" -ge "$start" ] && [ "$num" -le "$end" ]; then
                total=$((total + 1))
                if echo "$line" | grep -q '^\- \[x\]'; then
                    completed=$((completed + 1))
                fi
            fi
        fi
    done < <(grep -E '^\- \[.\] TASK-[0-9]+' "$TASKS_FILE")

    echo "$completed/$total"
}

count_tasks_by_marker() {
    local marker="$1"
    grep -E -c "^\- \[${marker}\] TASK-[0-9]+" "$TASKS_FILE" || true
}

count_phase_tasks_by_marker() {
    local phase="$1"
    local marker="$2"
    local range=$(get_phase_range "$phase")
    local start="${range%%:*}"
    local end="${range##*:}"
    local count=0

    while IFS= read -r line; do
        local num=$(echo "$line" | sed -n 's/.*TASK-\([0-9]\{3\}\).*/\1/p')
        if [ -n "$num" ]; then
            num=$((10#$num))
            if [ "$num" -ge "$start" ] && [ "$num" -le "$end" ]; then
                if echo "$line" | grep -q "^\- \[${marker}\]"; then
                    count=$((count + 1))
                fi
            fi
        fi
    done < <(grep -E '^\- \[.\] TASK-[0-9]+' "$TASKS_FILE")

    echo "$count"
}

show_status() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  3DVLMReasoning Task Status"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Source: $SOURCE_ROOT"
    echo "Target: $TARGET_ROOT"
    echo ""

    printf "%-40s %10s\n" "Phase" "Progress"
    echo "──────────────────────────────────────────────────────"

    local total_completed=0
    local total_tasks=0

    for phase in 1 2 3 4 5 6; do
        local name=$(get_phase_name "$phase")
        local progress=$(count_phase_tasks "$phase")
        local comp="${progress%%/*}"
        local tot="${progress##*/}"

        total_completed=$((total_completed + comp))
        total_tasks=$((total_tasks + tot))

        # Color based on progress
        if [ "$comp" -eq "$tot" ] && [ "$tot" -gt 0 ]; then
            printf "${GREEN}%-40s %10s${NC}\n" "Phase $phase: $name" "$progress"
        elif [ "$comp" -gt 0 ]; then
            printf "${YELLOW}%-40s %10s${NC}\n" "Phase $phase: $name" "$progress"
        else
            printf "%-40s %10s\n" "Phase $phase: $name" "$progress"
        fi
    done

    echo "──────────────────────────────────────────────────────"
    printf "%-40s %10s\n" "Total" "$total_completed/$total_tasks"
    echo ""

    # Show next actionable task
    local next_task=$(get_next_task)
    local blocked_count=$(count_tasks_by_marker "!")
    if [ -n "$next_task" ]; then
        log INFO "Next actionable task: $next_task"
        if [ "$blocked_count" -gt 0 ]; then
            log WARN "Blocked tasks remaining: $blocked_count"
        fi
        echo ""
        echo "Run: $0 --execute"
    elif [ "$blocked_count" -gt 0 ]; then
        log WARN "No actionable tasks remain, but $blocked_count blocked task(s) still need manual resolution."
    else
        log OK "All tasks completed!"
    fi
    echo ""
}

# ════════════════════════════════════════════════════════════════════════════════
# TASK MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════════

# Get next actionable task ID
get_next_task() {
    if [ -n "$TARGET_PHASE" ]; then
        local range=$(get_phase_range "$TARGET_PHASE")
        local start="${range%%:*}"
        local end="${range##*:}"

        # Find first actionable task in phase range ([ ] or [~])
        while IFS= read -r line; do
            local task_id=$(echo "$line" | sed -n 's/.*\(TASK-[0-9]\{3\}\).*/\1/p')
            local num="${task_id#TASK-}"
            if [ -n "$num" ]; then
                num=$((10#$num))
                if [ "$num" -ge "$start" ] && [ "$num" -le "$end" ]; then
                    echo "$task_id"
                    return
                fi
            fi
        done < <(grep -E '^\- \[[ ~]\] TASK-[0-9]+' "$TASKS_FILE")
    else
        # Find first actionable task overall
        grep -m1 -E '^\- \[[ ~]\] TASK-[0-9]+' "$TASKS_FILE" | \
            sed -E 's/.*(TASK-[0-9]{3}).*/\1/' || echo ""
    fi
}

# Get task details
get_task_details() {
    local task_id="$1"
    awk "
        /$task_id/ { found=1 }
        found && /^- \[/ && !/$task_id/ { exit }
        found && /^##/ { exit }
        found && /^---/ { exit }
        found { print }
    " "$TASKS_FILE"
}

# Mark task status
mark_task_status() {
    local task_id="$1"
    local status="$2"  # [ ] = pending, [~] = in_progress, [x] = done, [!] = blocked

    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/^\- \[.\] $task_id/- [$status] $task_id/" "$TASKS_FILE"
    else
        sed -i "s/^\- \[.\] $task_id/- [$status] $task_id/" "$TASKS_FILE"
    fi
}

# Get phase number for a task
get_task_phase() {
    local task_id="$1"
    local num="${task_id#TASK-}"
    num=$((10#$num))

    for phase in 1 2 3 4 5 6; do
        local range=$(get_phase_range "$phase")
        local start="${range%%:*}"
        local end="${range##*:}"
        if [ "$num" -ge "$start" ] && [ "$num" -le "$end" ]; then
            echo "$phase"
            return
        fi
    done
    echo "0"
}

# ════════════════════════════════════════════════════════════════════════════════
# TASK EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

build_task_prompt() {
    local task_id="$1"
    local task_details="$2"

    cat << EOF
# Repository Task: $task_id

You are executing a task from TASKS.md for the current 3DVLMReasoning repository.

## Context

- **Repository Root**: $TARGET_ROOT
- **Task Ledger**: $TASKS_FILE
- **Current Task**: $task_id

## Task Details

$task_details

## Working Rules

1. Treat TASKS.md as the current source of truth for task status.
2. Inspect the current repository state before changing code.
3. Implement only the work needed to satisfy this task in the current repo.
4. Verify with targeted tests or commands when possible.
5. Update TASKS.md only if the task status or acceptance notes are now materially different.

## Historical References

- MIGRATION_PLAN.md and older completion reports are background only.
- If they conflict with the current code or TASKS.md, follow the current repository state.

## Important

- Preserve unrelated user changes already present in the worktree.
- Do not claim completion unless the current repo state verifies it.
- Prefer minimal, targeted fixes with explicit verification.
- Keep imports and docs aligned with the current module layout.

Execute this task now.
EOF
}

execute_task() {
    local task_id="$1"

    log TASK "Executing: $task_id"

    # Get task details
    local task_details=$(get_task_details "$task_id")

    if [ -z "$task_details" ]; then
        log ERROR "Could not find task details for $task_id"
        return 1
    fi

    # Get phase info
    local phase=$(get_task_phase "$task_id")
    log PHASE "Phase $phase: $(get_phase_name "$phase")"

    # Build prompt
    local prompt=$(build_task_prompt "$task_id" "$task_details")

    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "DRY RUN: Would execute task $task_id"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "Task Details:"
        echo "$task_details"
        echo ""
        echo "Working Directory: $TARGET_ROOT"
        echo ""
        return 0
    fi

    # Mark as in progress
    mark_task_status "$task_id" "~"

    # Execute with Claude (in target directory)
    log INFO "Running Claude in $TARGET_ROOT..."

    local task_log="$LOG_DIR/${task_id}_$TIMESTAMP.log"

    # Execute task
    cd "$TARGET_ROOT"

    local task_log="$LOG_DIR/${task_id}_$TIMESTAMP.log"

    # Use ttadk if available, otherwise claude
    if command -v ttadk &> /dev/null; then
        # ttadk requires -a to pass args to claude
        echo "$prompt" | ttadk code -m "$MODEL" -a "--print --dangerously-skip-permissions" 2>&1 | tee "$task_log"
    else
        echo "$prompt" | claude code --model "$MODEL" --print --dangerously-skip-permissions 2>&1 | tee "$task_log"
    fi
    local exit_code=${PIPESTATUS[1]}

    cd "$SOURCE_ROOT"

    if [ $exit_code -eq 0 ]; then
        log OK "Task $task_id completed"
        mark_task_status "$task_id" "x"
        return 0
    else
        log ERROR "Task $task_id failed (exit code: $exit_code)"
        mark_task_status "$task_id" "!"
        return 1
    fi
}

run_execution() {
    check_prerequisites

    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  3DVLMReasoning Task Execution"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""

    if [ -n "$TARGET_PHASE" ]; then
        log INFO "Target phase: $TARGET_PHASE ($(get_phase_name "$TARGET_PHASE"))"
    fi

    local iteration=0
    local max_iterations=100

    while [ $iteration -lt $max_iterations ]; do
        local task_id=$(get_next_task)

        if [ -z "$task_id" ]; then
            if [ -n "$TARGET_PHASE" ]; then
                local phase_blocked=$(count_phase_tasks_by_marker "$TARGET_PHASE" "!")
                if [ "$phase_blocked" -gt 0 ]; then
                    log WARN "Phase $TARGET_PHASE has no actionable tasks, but $phase_blocked blocked task(s) remain."
                else
                    log OK "Phase $TARGET_PHASE complete!"
                fi
            else
                local blocked_count=$(count_tasks_by_marker "!")
                if [ "$blocked_count" -gt 0 ]; then
                    log WARN "No actionable tasks remain, but $blocked_count blocked task(s) still need manual resolution."
                else
                    log OK "All actionable tasks complete!"
                fi
            fi
            break
        fi

        execute_task "$task_id"
        local result=$?

        if [ $result -ne 0 ]; then
            log ERROR "Task failed, stopping execution"
            break
        fi

        if [ "$EXECUTE_ALL" = false ]; then
            break
        fi

        iteration=$((iteration + 1))
        sleep 2
    done

    echo ""
    show_status
}

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

main() {
    if [ "$EXECUTE_MODE" = true ]; then
        run_execution
    elif [ "$SHOW_STATUS" = true ]; then
        check_prerequisites
        show_status
    else
        # Default: show status
        check_prerequisites
        show_status
    fi
}

main
