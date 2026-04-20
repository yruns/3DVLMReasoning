#!/usr/bin/env bash
# v15 OpenEQA 1050Q eval matrix launcher
# Assumes:
#   - branch feat/v15-trajectory-aware checked out
#   - .venv activated
#   - frozen 1050Q set at /tmp/v15_frozen_questions.json
#   - T5b CLI plumbing committed (--pose-aware / --frustum-method / --enable-temporal-fan / --force-selection)

set -euo pipefail

REPO="/Users/bytedance/project/3DVLMReasoning"
cd "$REPO"
source .venv/bin/activate

FROZEN="$REPO/tmp/v15_artifacts/v15_frozen_questions.json"
if [ ! -s "$FROZEN" ]; then
    echo "ERROR: frozen questions file missing: $FROZEN" >&2
    exit 1
fi

# Common args (identical to v14 production invocation except --force-selection pins the set)
COMMON_ARGS="--max-samples 2000 --workers 6 \
    --llm-rewrite --confidence-guard 0.6 \
    --max-reasoning-turns 10 --max-additional-views 2 \
    --force-selection $FROZEN \
    --resume \
    --evaluate --eval-model gemini-2.5-pro"

MODE="${1:-smoke}"  # smoke | batch1 | batch2 | all | single-<col>

launch() {
    local session_name="$1"
    local output_root="$2"
    shift 2
    local extra_args="$*"

    local log="/tmp/v15_eval_${session_name}.log"
    echo "[launch] session=$session_name log=$log"

    mkdir -p "$output_root"

    tmux new-session -d -s "$session_name" \
        "cd $REPO && source .venv/bin/activate && \
         PYTHONPATH=src python -m agents.examples.openeqa_official_question_pilot \
         $COMMON_ARGS $extra_args \
         --output-root $output_root 2>&1 | tee $log"
}

case "$MODE" in
    smoke)
        # Tiny sanity run — 3 questions, no eval scoring, verify CLI + overlay + selector init
        echo "[smoke] 3-question sanity run"
        mkdir -p tmp/v15_smoke
        PYTHONPATH=src python -m agents.examples.openeqa_official_question_pilot \
            --max-samples 3 --workers 2 \
            --llm-rewrite --confidence-guard 0.6 \
            --max-reasoning-turns 5 --max-additional-views 2 \
            --pose-aware --frustum-method l1 --enable-temporal-fan \
            --output-root tmp/v15_smoke 2>&1 | tee /tmp/v15_smoke.log | tail -30
        ;;

    batch1)
        # Parallel 3: control + L1-S1 + L2-S1
        echo "[batch1] launching v14-rerun + v15-S1-L1 + v15-S1-L2"
        launch "eval-v14-rerun"   "tmp/openeqa_eval_v14_rerun"
        launch "eval-v15-s1-l1"   "tmp/openeqa_eval_v15_s1_l1"   --pose-aware --frustum-method l1
        launch "eval-v15-s1-l2"   "tmp/openeqa_eval_v15_s1_l2"   --pose-aware --frustum-method l2
        echo "[batch1] launched. Monitor: tmux list-sessions"
        ;;

    batch2)
        # Parallel 2: S1+S2 on L1 and L2
        echo "[batch2] launching v15-S1+S2-L1 + v15-S1+S2-L2"
        launch "eval-v15-s1s2-l1" "tmp/openeqa_eval_v15_s1s2_l1" --pose-aware --frustum-method l1 --enable-temporal-fan
        launch "eval-v15-s1s2-l2" "tmp/openeqa_eval_v15_s1s2_l2" --pose-aware --frustum-method l2 --enable-temporal-fan
        echo "[batch2] launched. Monitor: tmux list-sessions"
        ;;

    all)
        # Fire all 5 at once — only if Azure rate limits allow and workers scaled down
        echo "[all] launching 5 parallel runs (reduce workers if Azure complains)"
        launch "eval-v14-rerun"   "tmp/openeqa_eval_v14_rerun"
        launch "eval-v15-s1-l1"   "tmp/openeqa_eval_v15_s1_l1"   --pose-aware --frustum-method l1
        launch "eval-v15-s1-l2"   "tmp/openeqa_eval_v15_s1_l2"   --pose-aware --frustum-method l2
        launch "eval-v15-s1s2-l1" "tmp/openeqa_eval_v15_s1s2_l1" --pose-aware --frustum-method l1 --enable-temporal-fan
        launch "eval-v15-s1s2-l2" "tmp/openeqa_eval_v15_s1s2_l2" --pose-aware --frustum-method l2 --enable-temporal-fan
        ;;

    single-*)
        col="${MODE#single-}"
        case "$col" in
            v14-rerun)  launch "eval-v14-rerun"   "tmp/openeqa_eval_v14_rerun" ;;
            s1-l1)      launch "eval-v15-s1-l1"   "tmp/openeqa_eval_v15_s1_l1" --pose-aware --frustum-method l1 ;;
            s1-l2)      launch "eval-v15-s1-l2"   "tmp/openeqa_eval_v15_s1_l2" --pose-aware --frustum-method l2 ;;
            s1s2-l1)    launch "eval-v15-s1s2-l1" "tmp/openeqa_eval_v15_s1s2_l1" --pose-aware --frustum-method l1 --enable-temporal-fan ;;
            s1s2-l2)    launch "eval-v15-s1s2-l2" "tmp/openeqa_eval_v15_s1s2_l2" --pose-aware --frustum-method l2 --enable-temporal-fan ;;
            *) echo "unknown column: $col"; exit 1 ;;
        esac
        ;;

    *)
        echo "usage: $0 {smoke|batch1|batch2|all|single-<col>}"
        echo "columns: v14-rerun | s1-l1 | s1-l2 | s1s2-l1 | s1s2-l2"
        exit 1
        ;;
esac
