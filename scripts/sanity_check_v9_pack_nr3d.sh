#!/usr/bin/env bash
# Sanity check: build v9 catalog-first pack artifacts for a single NR3D scene
# and verify scene_catalog.json + camera_trajectory.json + BEV png are produced.
#
# Usage:
#   DATA_ROOT=/path/to/scannet bash scripts/sanity_check_v9_pack_nr3d.sh scene0011_00

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SCENE_ID="${1:-scene0011_00}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/scannet}"
PACK_NAME="pack_nr3d_v9_catalog_first"

if [[ ! -d "$DATA_ROOT/$SCENE_ID" ]]; then
    echo "Scene dir missing: $DATA_ROOT/$SCENE_ID" >&2
    exit 2
fi

# Activate venv if available; otherwise rely on the caller's environment.
if [[ -f .venv/bin/activate ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

PYTHONPATH=src python -m evaluation.scripts.prepare_pack_v1_inputs_nr3d \
    --sample-ids "${SAMPLE_IDS:-tmp/nr3d_artifacts/random100_frozen.json}" \
    --data-root "$DATA_ROOT" \
    --pack-name "$PACK_NAME" \
    --max-samples 1

pack_dir="$DATA_ROOT/$SCENE_ID/$PACK_NAME"
echo "Checking artifacts in $pack_dir"
test -f "$pack_dir/scene_catalog.json" || { echo "FAIL: no scene_catalog.json"; exit 1; }
test -f "$pack_dir/camera_trajectory.json" || { echo "FAIL: no camera_trajectory.json"; exit 1; }
test -f "$pack_dir/bev/scene_bev_nr3d.png" || { echo "FAIL: no BEV png"; exit 1; }

PYTHONPATH=src python <<PY
import json
from agents.catalog import SceneCatalog

data = json.load(open("${pack_dir}/scene_catalog.json"))
cat = SceneCatalog(**data)
assert cat.scene_id == "${SCENE_ID}", f"scene_id mismatch: {cat.scene_id!r}"
assert cat.proposals, "empty proposals"
assert cat.bev_image_path.endswith(".png")
print(f"OK: catalog parses, scene_id={cat.scene_id}, proposals={len(cat.proposals)}")
PY
echo "PASS: NR3D v9 pack sanity check"
