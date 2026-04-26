#!/usr/bin/env bash
# Stage all data needed for the pack-v1 side-by-side run under
# data/embodiedscan/pack_v1/. Run from worktree root.
set -euo pipefail

WORKTREE=/home/ysh/codecase/3DVLMReasoning/.worktrees/plan-a
cd "$WORKTREE"

source .venv-agents/bin/activate

OUT_ROOT="data/embodiedscan/pack_v1"
STAGING="$OUT_ROOT/staging"
INPUTS_DIR="$STAGING/vdetr_inputs"
OUTPUTS_DIR="$STAGING/vdetr_outputs"
PROPOSALS_DIR="$OUT_ROOT/vdetr_proposals"

mkdir -p "$STAGING" "$PROPOSALS_DIR"

echo "================================================================"
echo "[1/5] Materialize scene-level point clouds for the 10 batch30 scenes"
echo "================================================================"
PYTHONPATH=src python -m benchmarks.embodiedscan_bbox_feasibility.cli prepare-inputs \
    --data-root data/embodiedscan \
    --scene-data-root data/embodiedscan/scannet \
    --scannet-root data/scannetv2 \
    --conditions scannet_full \
    --target-categories cabinet chair curtain desk door picture sink table window \
    --require-visible-frames \
    --max-scenes 10 \
    --max-targets-per-scene 3 \
    --scene-level-full \
    --output-dir "$INPUTS_DIR" \
    2>&1

echo
echo "Materialized scene PLYs:"
ls "$INPUTS_DIR/scannet_full/" 2>&1 | head -20
echo

echo "================================================================"
echo "[2/5] Run V-DETR on each scene PLY (GPU 7, vdetr conda env)"
echo "================================================================"
# Switch to vdetr env for the actual GPU run
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vdetr
export CUDA_VISIBLE_DEVICES=7

PYTHONPATH=src python -m benchmarks.embodiedscan_bbox_feasibility.cli run-detector \
    --inputs-jsonl "$INPUTS_DIR/detector_inputs.jsonl" \
    --output-dir "$OUTPUTS_DIR" \
    --method vdetr-scannet \
    --detector-profile vdetr \
    --vdetr-repo-dir "$WORKTREE/external/V-DETR" \
    --vdetr-checkpoint "$WORKTREE/external/V-DETR/checkpoints/scannet_540ep.pth" \
    --vdetr-python /home/ysh/miniconda3/envs/vdetr/bin/python \
    --vdetr-num-points 40000 \
    --vdetr-conf-thresh 0.05 \
    --vdetr-top-k 256 \
    --cuda-device 0 \
    2>&1

echo
echo "V-DETR predictions written:"
ls "$OUTPUTS_DIR/predictions/" 2>&1 | head -20
echo

echo "================================================================"
echo "[3/5] Reorganize V-DETR outputs into <scene>/predictions.json layout"
echo "================================================================"
# Switch back to .venv-agents (Python 3.11) for the rest
conda deactivate
source "$WORKTREE/.venv-agents/bin/activate"

PYTHONPATH=src python <<'PY'
import json
import shutil
from pathlib import Path

src = Path("data/embodiedscan/pack_v1/staging/vdetr_outputs/predictions")
dst_root = Path("data/embodiedscan/pack_v1/vdetr_proposals")

# scene-level full predictions are named <scene>_scene_scannet_full.json
files = sorted(src.glob("*_scene_scannet_full.json"))
if not files:
    raise SystemExit(f"no scene-level predictions found in {src}")

for fp in files:
    scene_id = fp.name.split("_scene_scannet_full.json")[0]
    dst_dir = dst_root / scene_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "predictions.json"
    # Copy + verify schema
    payload = json.loads(fp.read_text())
    n = len(payload.get("proposals", []))
    if n == 0:
        raise SystemExit(f"empty proposals in {fp}")
    dst.write_text(json.dumps(payload, indent=2))
    print(f"  {scene_id}: {n} proposals -> {dst}")

print(f"\nReorganized {len(files)} scenes into {dst_root}")
PY

echo
echo "================================================================"
echo "[4/5] Run prepare_pack_v1_inputs to produce per-sample artifacts"
echo "================================================================"
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs.py \
    --sample-ids "$OUT_ROOT/batch30_sample_ids.json" \
    --vdetr-proposals-dir "$PROPOSALS_DIR" \
    --embodiedscan-data-root data/embodiedscan \
    --output-dir "$OUT_ROOT" \
    --source vdetr \
    2>&1

echo
echo "================================================================"
echo "[5/5] Verify final layout"
echo "================================================================"
echo "Scene artifacts:"
find "$OUT_ROOT/scenes" -type f 2>&1 | head -30
echo
echo "Sample artifacts:"
ls "$OUT_ROOT/samples/" 2>&1 | head -10
echo "  (total: $(ls "$OUT_ROOT/samples/" 2>&1 | wc -l) sample json files)"
echo
echo "Done."
