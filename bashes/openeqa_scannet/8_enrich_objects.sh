#!/usr/bin/env bash
# Enrich ConceptGraph objects with LLM-generated metadata.
#
# Sends object images to Gemini and generates category, description,
# location, nearby objects, color, and usability for each object.
#
# Usage:
#   bash bashes/openeqa_scannet/8_enrich_objects.sh <clip_id>
#   bash bashes/openeqa_scannet/8_enrich_objects.sh --all
#   FORCE=1 bash bashes/openeqa_scannet/8_enrich_objects.sh <clip_id>
#
# Environment variables:
#   MAX_WORKERS  - parallel LLM requests per scene (default: 7)
#   MAX_RETRIES  - max retries per object (default: 10)
#   FORCE        - set to 1 to re-process existing results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

OPENEQA_SCANNET_ROOT="${ROOT_DIR}/data/OpenEQA/scannet"
MAX_WORKERS="${MAX_WORKERS:-7}"
MAX_RETRIES="${MAX_RETRIES:-10}"
FORCE="${FORCE:-0}"

# Platform-dependent environment activation
if [[ "$(uname -s)" == "Linux" ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph
else
    # macOS — uv / .venv
    source .venv/bin/activate 2>/dev/null || { uv venv && source .venv/bin/activate; }
fi

FORCE_FLAG=""
if [[ "${FORCE}" == "1" ]]; then
    FORCE_FLAG="--force"
fi

MODE="${1:?Usage: $0 <clip_id> | --all}"

if [[ "${MODE}" == "--all" ]]; then
    echo "=== Enriching ALL scenes in ${OPENEQA_SCANNET_ROOT} ==="
    python -m src.scripts.enrich_objects \
        --scannet_root "${OPENEQA_SCANNET_ROOT}" \
        --max_workers "${MAX_WORKERS}" \
        --max_retries "${MAX_RETRIES}" \
        ${FORCE_FLAG}
else
    CLIP_ID="${MODE}"
    SCENE_PATH="${OPENEQA_SCANNET_ROOT}/${CLIP_ID}"
    if [[ ! -d "${SCENE_PATH}/conceptgraph" ]]; then
        echo "ERROR: Scene not found: ${SCENE_PATH}/conceptgraph" >&2
        exit 1
    fi
    echo "=== Enriching scene: ${CLIP_ID} ==="
    python -m src.scripts.enrich_objects \
        --scene_path "${SCENE_PATH}" \
        --max_workers "${MAX_WORKERS}" \
        --max_retries "${MAX_RETRIES}" \
        ${FORCE_FLAG}
fi

echo "=== Done ==="
