#!/usr/bin/env bash
# v9 catalog-first NR3D random100 launcher.
#
# Per CLAUDE.md "Long-Running Tasks: tmux (MANDATORY)" the actual eval runs
# inside a detached tmux session so it survives SSH drops and Bash tool
# timeouts. Operator invokes this on the eval host (Linux GPU box).

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

TAG="${1:-v9_full}"
DATE="$(date +%Y%m%d_%H%M)"
OUT_DIR="tmp/nr3d_eval_${TAG}_${DATE}"
LOG="/tmp/nr3d_${TAG}_${DATE}.log"
SESS="nr3d-${TAG}-${DATE}"
FROZEN_FOLD="${FROZEN_FOLD:-tmp/nr3d_artifacts/random100_frozen.json}"
PACK_NAME="${PACK_NAME:-pack_nr3d_v9_catalog_first}"
DATA_ROOT="${DATA_ROOT:-data/nr3d/scannet}"

mkdir -p "$OUT_DIR"

# The actual side-by-side runner is run_nr3d_vg_side_by_side; we drive a
# random100-style frozen fold through it. If the project ships a dedicated
# `run_nr3d_random100` driver in the future, swap the python invocation
# below accordingly.
tmux new-session -d -s "$SESS" "
set -euo pipefail
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
PYTHONPATH=src python -m evaluation.scripts.run_nr3d_vg_side_by_side \
    --sample-ids '$FROZEN_FOLD' \
    --data-root '$DATA_ROOT' \
    --pack-name '$PACK_NAME' \
    --output-dir '$OUT_DIR' \
    --judge-model gemini-2.5-pro \
    2>&1 | tee '$LOG'
"
echo "Launched tmux session: $SESS"
echo "Tail logs: tmux capture-pane -t $SESS -p -S -100"
echo "Output dir: $OUT_DIR"
