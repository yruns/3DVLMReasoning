#!/usr/bin/env bash
# scripts/run_scanrefer_v3_query_driven.sh
#
# End-to-end driver for ScanRefer v3 query-driven track. Runs the four
# stages in sequence, each writing a tee'd log so failures can be
# diagnosed without re-running.
#
# Pre-flight:
#   - feat/scanrefer-v3-query-driven branch checked out
#   - all 141 NR3D-shared scenes have data/nr3d/scannet/<s>/conceptgraph/enriched_objects.json
#   - .venv with open_clip_torch installed
#   - tmp/scanrefer_artifacts/full_val_sample_ids.json exists (9508 utts)
#
# Wall estimate: ~12-20h (pack-prep ~6-10h + agent ~5-7h + ingest <30 min)
# $ estimate: ~$50 Stage 1 LLM + ~$200-400 agent gpt-5.4

set -euo pipefail

PROJECT_ROOT="/Users/bytedance/project/3DVLMReasoning"
cd "$PROJECT_ROOT"

if [[ ! -f .venv/bin/activate ]]; then
    echo "FATAL: .venv missing — run 'uv venv' first" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

DATE="$(date +%Y%m%d)"
RUN_ID="v3_query_driven_track_${DATE}"
SAMPLE_IDS="tmp/scanrefer_artifacts/full_val_sample_ids.json"
PACK_NAME="pack_scanrefer_v3_query_driven"
EVAL_DIR="tmp/scanrefer_eval_${RUN_ID}"
AGG_EVAL_DIR="${EVAL_DIR}_aggregation_gt"
LOG_DIR="${EVAL_DIR}/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$SAMPLE_IDS" ]]; then
    echo "FATAL: $SAMPLE_IDS missing" >&2
    exit 1
fi

# Pre-flight: every scene must have enriched_objects.json
MISSING_ENRICHED=$(
    python -c "
import json, sys
from pathlib import Path
ids = json.load(open('$SAMPLE_IDS'))
scenes = sorted({r['scene_id'] for r in ids})
missing = [s for s in scenes if not (Path('data/nr3d/scannet') / s / 'conceptgraph/enriched_objects.json').exists()]
print('|'.join(missing))
"
)
if [[ -n "$MISSING_ENRICHED" ]]; then
    echo "FATAL: enriched_objects.json missing for: $MISSING_ENRICHED" >&2
    echo "Run: python -m src.scripts.enrich_objects --scannet_root data/nr3d/scannet" >&2
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"
echo "[v3] starting run $RUN_ID (branch=$BRANCH commit=$COMMIT)"
echo "[v3] eval dir: $EVAL_DIR"

#-----------------------------------------------------------------------
# Stage 1: pack-prep with --keyframe-mode query_driven (~6-10h)
#-----------------------------------------------------------------------
echo
echo "[v3 stage 1] pack-prep query_driven on 9508 utts → $PACK_NAME"
PYTHONPATH=src python -m evaluation.scripts.prepare_pack_v1_inputs_scanrefer \
    --sample-ids "$SAMPLE_IDS" \
    --data-root data/scanrefer/scannet \
    --pack-name "$PACK_NAME" \
    --keyframe-mode query_driven \
    --keyframe-llm-model gemini-2.5-pro \
    2>&1 | tee "${LOG_DIR}/01_pack_prep.log"

#-----------------------------------------------------------------------
# Stage 2: agent run (gpt-5.4-2026-03-05, workers=32, ~5-7h)
#-----------------------------------------------------------------------
echo
echo "[v3 stage 2] agent run on 9508 utts → $EVAL_DIR"
PYTHONPATH=src python -m evaluation.scripts.run_scanrefer_vg_side_by_side \
    --sample-ids "$SAMPLE_IDS" \
    --data-root data/scanrefer/scannet \
    --output-dir "$EVAL_DIR" \
    --pack-name "$PACK_NAME" \
    --workers 32 \
    --sample-retries 1 \
    2>&1 | tee "${LOG_DIR}/02_agent_run.log"

#-----------------------------------------------------------------------
# Stage 3: paper-comparable GT rescan
#-----------------------------------------------------------------------
echo
echo "[v3 stage 3] rescan IoU against ScanNet aggregation GT → $AGG_EVAL_DIR"
PYTHONPATH=src python -m evaluation.scripts.rescan_with_aggregation_gt \
    --side-by-side "${EVAL_DIR}/side_by_side.json" \
    --output-dir "$AGG_EVAL_DIR" \
    --scannet-aux-root data/nr3d/scannet_aux \
    --mesh-root data/nr3d/scannet_aux_meshes \
    2>&1 | tee "${LOG_DIR}/03_rescan_aggregation_gt.log"

#-----------------------------------------------------------------------
# Stage 4: aggregator (Unique/Multiple slicing × Acc@0.25/0.50)
#-----------------------------------------------------------------------
echo
echo "[v3 stage 4] leaderboard metrics aggregator"
PYTHONPATH=src python -m evaluation.scripts.scanrefer_leaderboard_metrics \
    --side-by-side "${AGG_EVAL_DIR}/side_by_side.json" \
    --scanrefer-data-root data/scanrefer \
    --phase8-data-root data/nr3d/scannet \
    --output "${AGG_EVAL_DIR}/leaderboard_metrics.json" \
    2>&1 | tee "${LOG_DIR}/04_aggregator.log"

#-----------------------------------------------------------------------
# Stage 5: SQLite ingest with keyframe_mode=query_driven
#-----------------------------------------------------------------------
echo
echo "[v3 stage 5] SQLite ingest"
python scripts/ingest_scanrefer_run.py \
    --output-dir "$AGG_EVAL_DIR" \
    --run-id "$RUN_ID" \
    --branch "$BRANCH" \
    --commit "$COMMIT" \
    --backend pack_v1 \
    --keyframe-mode query_driven \
    --leaderboard-metrics "${AGG_EVAL_DIR}/leaderboard_metrics.json" \
    --notes "v3 query-driven keyframe selection (KeyframeSelector.select_keyframes_v2 with use_visual_context=False); Mask3D pool unchanged; IoU rescanned against ScanNet aggregation GT. Compare to v2_aggregation_gt_track_$(date -v-1d +%Y%m%d) which used GT view oracle keyframes." \
    2>&1 | tee "${LOG_DIR}/05_ingest.log"

#-----------------------------------------------------------------------
# Quick metrics dump
#-----------------------------------------------------------------------
echo
echo "[v3 done] headline metrics:"
sqlite3 docs/benchmark/scanrefer/runs.sqlite \
    "SELECT printf('U25=%.2f U50=%.2f M25=%.2f M50=%.2f O25=%.2f O50=%.2f',
                   100*acc25_unique, 100*acc50_unique,
                   100*acc25_multiple, 100*acc50_multiple,
                   100*acc25_overall, 100*acc50_overall)
     FROM runs WHERE run_id='$RUN_ID';"

echo
echo "[v3 done] next steps:"
echo "  - write docs/benchmark/scanrefer/v3_query_driven_track_${DATE}.md (use v2 doc as template)"
echo "  - update docs/benchmark/scanrefer/README.md headline → v3"
echo "  - update docs/benchmark/scanrefer/leaderboard.md → v3 row, demote v2 with [‡]"
