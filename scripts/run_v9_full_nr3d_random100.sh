#!/usr/bin/env bash
# v9 catalog-first NR3D random100 launcher.
#
# Drives the canonical random100 fold end-to-end:
#   1) pack-prep (BEV + scene_catalog + camera trajectory + per-sample artifacts)
#   2) Stage-2 eval (per-sample agent checkpoints, workers=W, RSS-guarded)
#   3) side_by_side.json assembly
#   4) NR3D leaderboard metrics (classification accuracy, programmatic — no
#      LLM judge needed for VG)
#
# Per CLAUDE.md "Long-Running Tasks: tmux (MANDATORY)" the actual prep + eval
# run inside detached tmux sessions so they survive SSH drops and Bash tool
# timeouts. ModelHub AKs default to the fallbacks hardcoded in
# src/agents/core/agent_config.py:_default_modelhub_api_keys; override with
# MODELHUB_AKS if needed.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

TAG="${1:-v9_full}"
DATE="$(date +%Y%m%d_%H%M)"
OUT_DIR="tmp/nr3d_eval_${TAG}_${DATE}"
PREP_LOG="tmp/nr3d_v9_catalog_first_prep_${TAG}_${DATE}.log"
EVAL_LOG="tmp/nr3d_v9_catalog_first_eval_${TAG}_${DATE}.log"
PREP_SESS="nr3d-${TAG}-prep-${DATE}"
EVAL_SESS="nr3d-${TAG}-eval-${DATE}"
FROZEN_FOLD="${FROZEN_FOLD:-tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json}"
PACK_NAME="${PACK_NAME:-pack_nr3d_v9_catalog_first}"
DATA_ROOT="${DATA_ROOT:-data/nr3d/scannet}"
NR3D_ROOT="${NR3D_ROOT:-data/nr3d}"
WORKERS="${WORKERS:-50}"
RSS_LIMIT_MB="${RSS_LIMIT_MB:-15000}"
KEYFRAME_MODE="${KEYFRAME_MODE:-gt_target}"

mkdir -p "$OUT_DIR" tmp/nr3d_artifacts

if [[ ! -f "$FROZEN_FOLD" ]]; then
  echo "ERROR: fold not found: $FROZEN_FOLD" >&2
  exit 2
fi

# --- 1) pack-prep ------------------------------------------------------------
# v9 catalog-first writes scene_catalog.json + bev.png + camera_trajectory.json
# per scene, plus per-sample artifacts. gt_target keyframe mode is the fastest
# (no Gemini calls) and keyframes themselves are not consumed by the v9 agent.
PREP_CMD=$(cat <<EOF
set -euo pipefail
cd $ROOT
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH=src PYTHONUNBUFFERED=1
python -m evaluation.scripts.prepare_pack_v1_inputs_nr3d \\
    --sample-ids '$FROZEN_FOLD' \\
    --data-root '$DATA_ROOT' \\
    --nr3d-root '$NR3D_ROOT' \\
    --pack-name '$PACK_NAME' \\
    --split test \\
    --keyframe-mode '$KEYFRAME_MODE' \\
    --ensure-lightweight-cache \\
    --max-selector-cache-size 2 \\
    --max-scene-artifact-cache-size 1 \\
    2>&1 | tee '$PREP_LOG'
EOF
)

tmux new-session -d -s "$PREP_SESS" "$PREP_CMD"
echo "Launched pack-prep tmux session: $PREP_SESS"
echo "  log: $PREP_LOG"
echo "  tail: tmux capture-pane -t $PREP_SESS -p -S -100"

# --- 2) eval -----------------------------------------------------------------
# checkpoint-only mode writes per-sample JSON checkpoints (resumable). After
# pack-prep finishes, the operator should manually start this session, or use
# `tmux wait-for -S nr3d-${TAG}-prep-done` if the prep cmd signals it.
EVAL_CMD=$(cat <<EOF
set -euo pipefail
cd $ROOT
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
export PYTHONPATH=src PYTHONUNBUFFERED=1
./scripts/run_with_rss_guard.sh --rss-limit-mb $RSS_LIMIT_MB --check-interval-sec 60 -- \\
python -m evaluation.scripts.run_nr3d_vg_side_by_side \\
    --sample-ids '$FROZEN_FOLD' \\
    --data-root '$DATA_ROOT' \\
    --pack-name '$PACK_NAME' \\
    --output-dir '$OUT_DIR' \\
    --workers $WORKERS \\
    --sample-retries 2 \\
    --use-tool-answer-disagreement-gate \\
    --use-no-match-candidate-guard \\
    --use-evidence-frame-guard \\
    2>&1 | tee '$EVAL_LOG'
python -m evaluation.scripts.nr3d_leaderboard_metrics \\
    --side-by-side '$OUT_DIR/side_by_side.json' \\
    --nr3d-data-root '$NR3D_ROOT' \\
    --phase8-data-root '$DATA_ROOT' \\
    --sample-ids '$FROZEN_FOLD' \\
    --output '$OUT_DIR/leaderboard_metrics.json' \\
    --canonical-filter true \\
    2>&1 | tee -a '$EVAL_LOG'
EOF
)

# Print the eval cmd as a follow-up so the operator can run it after prep
# completes. We do not auto-chain via tmux wait-for because the prep cmd
# in this script does not signal back; explicit handoff is clearer.
echo
echo "After pack-prep finishes, kick off the eval with:"
echo "tmux new-session -d -s $EVAL_SESS \"$EVAL_CMD\""
echo
echo "Output dir: $OUT_DIR"
echo "Eval log:   $EVAL_LOG"
