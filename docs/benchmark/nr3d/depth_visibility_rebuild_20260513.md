# NR3D Depth-Aware Visibility Rebuild - 2026-05-13

## Status

This is a data-quality fix, not a new benchmark result. No NR3D metric should
be claimed from this document alone. The next valid metric requires:

1. regenerate NR3D packs from the rebuilt visibility indices;
2. rerun Stage 2 on the full fold;
3. ingest and document the new run.

## Root Cause

The NR3D `view_to_objects` / `object_to_views` indices used by v1 through v5.1
were not depth-aware. A local audit found:

- `data/nr3d/scannet/*/conceptgraph/indices/visibility_index.pkl`: 141 files
- `metadata.use_depth=false`: 141 / 141
- projection fallback mappings: 1 scene before rebuild

That means the mappings represented point projection / frustum candidates, not
occlusion-aware object visibility. They were therefore invalid as
agent-visible evidence.

## Code Changes

- `scripts.build_visibility_index.build_visibility_index(...)` now fails
  loudly when `use_depth=True` lacks depth maps, has too few depth maps, or
  cannot read a depth image.
- Depth-aware scoring now computes coverage from depth-visible points, not all
  in-bounds projected points.
- `src/scripts/nr3d_gt_conceptgraph.py` defaults to depth-aware visibility and
  refuses foreground projection fallbacks when `use_depth=True`.
- `prepare_pack_v1_inputs_nr3d.py` rejects NR3D visibility indices unless
  `metadata.use_depth=true` and `num_projection_fallback_objects=0`.
- `prepare_pack_v1_inputs_scanrefer.py` rejects explicit
  `metadata.use_depth=false`; future ScanRefer builds now save
  `metadata.use_depth=true`.
- Added `scripts/rebuild_nr3d_depth_visibility.py` for visibility-only rebuilds
  without rerunning CLIP feature extraction or Stage 2.

## Rebuild Command

Executed on branch `feat/nr3d-v4-agent-guards-fair-views` from the local
working tree after commit `c4afadb`.

```bash
tmux new-session -d -s nr3d_depth_visibility_rebuild 'cd /Users/bytedance/project/3DVLMReasoning && PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python scripts/rebuild_nr3d_depth_visibility.py --data-root data/nr3d/scannet --workers 1 --report tmp/nr3d_depth_visibility_rebuild_20260513.json 2>&1 | tee tmp/nr3d_depth_visibility_rebuild_20260513.log'
```

`--workers 1` was intentional: the full object pkl files can expand
substantially in memory, so the rebuild avoids concurrent pkl loads.

## Rebuild Result

Raw artifacts:

- Report: `tmp/nr3d_depth_visibility_rebuild_20260513.json`
- Log: `tmp/nr3d_depth_visibility_rebuild_20260513.log`

Summary:

| Check | Value |
|---|---:|
| Scenes rebuilt | 141 |
| Old mappings | 224,664 |
| New mappings | 192,090 |
| Removed mappings | 32,574 |
| Relative reduction | 14.50% |
| `metadata.use_depth=true` after rebuild | 141 / 141 |
| `visibility_kind=depth_occlusion_point_visibility` | 141 / 141 |
| Projection fallback mappings after rebuild | 0 |

Single-scene sanity check before the full rebuild:

| Scene | Old mappings | New mappings | Delta |
|---|---:|---:|---:|
| `scene0474_00` | 1,700 | 1,427 | -273 |

## Spotcheck Visualization

Generated from the rebuilt indices:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_nr3d_depth_visibility_spotcheck.py \
  --data-root data/nr3d/scannet \
  --scenes scene0474_00 scene0030_00 scene0153_00 scene0435_00 \
  --frames-per-scene 3 \
  --max-marks 18 \
  --out-html docs/benchmark/nr3d/depth_visibility_spotcheck_20260513.html \
  --asset-dir docs/benchmark/nr3d/assets/depth_visibility_spotcheck_20260513 \
  --rebuild-report tmp/nr3d_depth_visibility_rebuild_20260513.json
```

Output:

- HTML: `docs/benchmark/nr3d/depth_visibility_spotcheck_20260513.html`
- Assets: `docs/benchmark/nr3d/assets/depth_visibility_spotcheck_20260513/`
- Rendered frames: 12
- Verification: 12 image references, 0 missing files, 0 blank images

The rendered frames draw only objects present in the rebuilt depth-aware
`view_to_objects[view_id]` for that frame.

## Benchmark Impact

All previous NR3D rows that relied on the old Phase 8 visibility source are
invalidated as benchmark claims, including v3, v4, v5, and v5.1. Their raw
numbers remain in the archive as audit records only.

No corrected metric exists yet. The corrected sequence must be:

1. regenerate packs from the rebuilt visibility indices;
2. rerun full NR3D with the same Stage 2 guard configuration;
3. compare the corrected full run against the invalidated v5.1 record and
   public rows.
