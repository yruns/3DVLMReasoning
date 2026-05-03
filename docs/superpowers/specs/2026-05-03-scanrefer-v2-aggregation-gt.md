# ScanRefer v2 — Aggregation-based GT Re-aggregation

**Date:** 2026-05-03
**Branch:** `feat/scanrefer-vg-benchmark` (continuation; new commits on top of v1)
**Tip commit at design time:** `0c52f43` (v1 ship)
**Authors:** Shuhao Yue (designer / executor)

## Goal

Replace v1's Phase 8 GT-CG bbox with the **ScanNet aggregation-based GT
bbox** in the IoU computation, **without re-running the agent**. This
makes our Acc@0.25 / Acc@0.50 numbers paper-comparable to ZSVG3D /
SeeGround / CSVG / Z3D (all of which compute IoU against
aggregation-derived GT, since Mask3D ScanNet200 itself was trained
against that GT).

The headline outcome is a v2 row in `runs.sqlite` and a new
`v2_aggregation_gt_track_20260503.md` doc that reports the same 9508
agent decisions against the standard GT, eligible to be quoted alongside
published baselines.

Estimated impact (from oracle analysis, now folded into v2 doc §
"Audit trail": `docs/benchmark/scanrefer/v2_aggregation_gt_track_20260503.md`):
Acc@0.25 moves from 51.65 → ~60-65, Acc@0.50 moves from 16.22 → ~35-45.
The agent run is reused as-is; only the GT side of the IoU changes.

## Why this is the right fix (not a re-run)

v1's per-sample checkpoints already record the agent's predicted bbox
(= the chosen Mask3D candidate's bbox in 9-DoF axis-aligned form). The
predicted side is invariant to GT derivation. Therefore the agent run
(8 hours, gpt-5.4 API, real money) does NOT need to repeat. The
aggregation step (~30 minutes) is what changes. v2 is a re-aggregation,
not a re-eval.

## Background — what's already in place

- **v1 outputs** at `tmp/scanrefer_eval_v1_full/` — `side_by_side.json`
  has 9508 per-sample records `{sample_id, status, iou, predicted_bbox_3d_9dof, gt_bbox_3d_9dof, ...}`.
  Note `gt_bbox_3d_9dof` is the v1 (Phase 8) GT — we'll write it ASIDE
  and substitute the aggregation-derived GT for v2's IoU computation.
- **ScanNet aggregation files** at `data/nr3d/scannet_aux/<scene>/`:
  - `<scene>.aggregation.json` — segGroups: `[{id, objectId, label, segments: [seg_ids...]}, ...]`
  - `<scene>_vh_clean_2.0.010000.segs.json` — `segIndices: [seg_id_per_vertex...]`
  - `<scene>.txt` — axis alignment matrix (4×4)
- **What's missing locally**: the ScanNet `.ply` mesh per scene. Without
  the mesh we cannot resolve `seg_id → vertex_xyz` to compute the AABB.
  See "Open question 1" below.

## Brainstorming-locked decisions

These decisions are LOCKED for v2 unless explicitly reopened:

1. **GT bbox derivation** = AABB of mesh vertices belonging to the target
   `objectId`, in axis-aligned scene coordinates (apply
   `scene.txt` axisAlignment matrix to vertices first, then min/max).
2. **No agent re-run.** v2 reuses v1's per-sample predicted bbox and
   the agent's `selected_object_id`.
3. **GT lookup keyed by ScanNet `objectId`** = ScanRefer JSON's
   `object_id` field (cast to int). Same convention as v1.
4. **Background filter:** none on the GT side (ScanRefer references
   real objects, never wall/floor/ceiling, so this is a no-op).
5. **IoU function** = the same `compute_oriented_iou_3d` with Euler=0.
6. **Run ID** = `v2_aggregation_gt_track_20260503` — clearly different
   from v1 so SQLite has both rows.
7. **Test fold** = identical to v1: full ScanRefer val 9508 utts,
   141 scenes.
8. **Output destination** = `tmp/scanrefer_eval_v2_aggregation_gt/`
   for the new `side_by_side.json` (re-aggregated) and
   `leaderboard_metrics.json`.
9. **Phase 8 fall-back** behind `--gt-source phase8` for diagnostics
   only; `aggregation` is the default and the published number.
10. **The v1 doc + leaderboard caveats stay in place.** v2 is an
    additional row in the leaderboard table, not a v1 retraction.

## Architecture

```
v1 outputs (already on disk)              ScanNet aux (already on disk)
  side_by_side.json                          <scene>.aggregation.json
  per_sample/*.json                          <scene>_vh_clean_2.0.010000.segs.json
                                             <scene>.txt   (axis alignment)
                                             <scene>_vh_clean_2.ply   (NEW download — see open Q)
       │                                              │
       ▼                                              ▼
       └────────────── new module ───────────────────┘
                       aggregation_gt_loader.py
                       (per-scene: load mesh + segs + agg → {objectId: bbox_9dof})
                                  │
                                  ▼
                       v2 aggregator wrapper
                       (reuse src/evaluation/scripts/scanrefer_leaderboard_metrics.py:
                        only swap the sample_meta source so gt_bbox_9dof comes from
                        aggregation_gt_loader instead of ScanRefVGDataset)
                                  │
                                  ▼
                       new side_by_side_v2.json
                       (per-sample IoU recomputed; predicted bbox unchanged;
                        same status; new iou; new acc25/acc50)
                                  │
                                  ▼
                       leaderboard_metrics_v2.json + runs.sqlite row
                       v2_aggregation_gt_track_20260503
```

## Scope

### In scope (this design)

- **New loader**: `src/benchmarks/scanrefer_aggregation_gt.py` with
  `load_aggregation_gt_bboxes(scene_id, scannet_root) -> dict[objectId, bbox_9dof]`.
- **New re-aggregator**: `src/evaluation/scripts/rescan_with_aggregation_gt.py`
  CLI that reads v1 `side_by_side.json` + `aggregation_gt_loader`, recomputes
  IoU per sample, writes new `side_by_side.json` + `leaderboard_metrics.json`.
- **Tests** for the loader (synthetic mesh + agg + segs → known AABB).
- **v2 doc** at `docs/benchmark/scanrefer/v2_aggregation_gt_track_20260503.md`.
- **Leaderboard update**: add v2 row alongside (not replacing) v1.
- **README + Active Benchmarks update** to highlight v2 as the
  paper-comparable headline.

### Out of scope

- Modifying the v1 doc's numbers (v1 stays as the historical record;
  caveat banners already added 2026-05-03).
- Modifying the agent runner (no re-run).
- Switching GT pool source (still Phase 8 / NR3D for non-IoU lookups
  like `is_unique` class-counts; only the **bbox** geometry changes).
- New backbone or detector ablation (v3+).
- Touching `src/agents/**`.

## Components

### Component 1 — `src/benchmarks/scanrefer_aggregation_gt.py` (new)

```python
def load_aggregation_gt_bboxes(
    scene_id: str,
    *,
    scannet_aux_root: Path = Path("data/nr3d/scannet_aux"),
    mesh_root: Path,                 # location of <scene>_vh_clean_2.ply
    apply_axis_alignment: bool = True,
) -> dict[int, list[float]]:
    """For one scene, return {objectId: bbox_9dof} where bbox is the
    axis-aligned AABB of the mesh vertices belonging to that objectId.

    Pipeline:
      1. Read <scene>.aggregation.json → segGroups: [{objectId, segments:[seg_id_list], ...}]
      2. Read <scene>_vh_clean_2.0.010000.segs.json → segIndices: list[int] (per-vertex seg_id)
      3. Read <scene>_vh_clean_2.ply → vertex_xyz: (N, 3) float
      4. (optional) Read <scene>.txt → axisAlignment 4×4 matrix; apply to vertex_xyz.
      5. For each segGroup, collect vertex indices where segIndices ∈ segments,
         compute AABB on those vertices' (axis-aligned) xyz.
      6. Convert AABB to 9-DoF [cx,cy,cz,dx,dy,dz,0,0,0] (Euler=0).
      7. Return dict keyed by objectId.

    Raises FileNotFoundError on any missing aux file.
    """
```

Tests (synthetic, no real mesh):
- 1 segGroup with 3 segments covering 4 vertices at known coords → AABB matches min/max of those 4 vertices.
- 2 segGroups; vertex set per group is disjoint; both AABBs returned.
- Axis-alignment matrix is applied before AABB.
- Missing seg id in `segIndices` is silently skipped (defensive).
- Empty segGroup raises ValueError.

### Component 2 — `src/evaluation/scripts/rescan_with_aggregation_gt.py` (new)

CLI:
```
--side-by-side tmp/scanrefer_eval_v1_full/side_by_side.json
--scannet-aux-root data/nr3d/scannet_aux
--mesh-root data/scanrefer/scannet_meshes      # see open Q on mesh source
--scanrefer-data-root data/scanrefer
--phase8-data-root data/nr3d/scannet            # for is_unique class counts
--output-dir tmp/scanrefer_eval_v2_aggregation_gt
--backend pack_v1
```

For each per-sample record in the input `side_by_side.json`:
1. Parse sample_id → (scene_id, target_id, ann_id).
2. From cached aggregation GT (one dict per scene, lazy-loaded), look up `gt_9dof = gt_bboxes[target_id]`.
3. Skip the sample if `target_id` is missing in the aggregation GT (record stat); this is rare and indicates an annotation mismatch.
4. Recompute `iou = compute_oriented_iou_3d(predicted_bbox_3d_9dof, gt_9dof)` if `predicted_bbox_3d_9dof` is non-null; else `iou = 0`.
5. Write a new per-sample record with the recomputed `iou` and the new `gt_bbox_3d_9dof`.

Output = a new `side_by_side.json` and the v1-style summary.

### Component 3 — Re-use existing `scanrefer_leaderboard_metrics.py`

Run the existing aggregator on the v2 `side_by_side.json`. The
aggregator loads sample_meta from `ScanRefVGDataset` (Phase 8) for
`is_unique`; **this is the right behavior** because `is_unique` =
"only one same-class GT instance in scene" is unaffected by GT bbox
geometry. We keep using Phase 8's class counts.

(Apply the OOM workaround documented in v1: scene-by-scene meta
construction. v2 should fix this in the aggregator itself with a
`sample_ids` filter — small, in-scope improvement.)

### Component 4 — New ingester invocation

Reuse `scripts/ingest_scanrefer_run.py` unchanged with `--run-id v2_aggregation_gt_track_20260503`.

### Component 5 — `docs/benchmark/scanrefer/v2_aggregation_gt_track_20260503.md` (new)

Mirror of v1 doc with:
- Headline numbers under aggregation GT.
- Explicit comparison vs v1 numbers (same agent, same picks, only GT changes).
- SOTA comparison table — v2 row replaces v1 row in the leaderboard's
  zero-shot reference table.

## Implementation order

1. Acquire ScanNet `.ply` meshes for 141 ScanRefer val scenes
   (see open Q below — possibly already on local disk under another path,
   or downloadable from ScanNet via license-required download script).
2. Write Component 1 (loader) + tests, RED → GREEN.
3. Write Component 2 (re-aggregator CLI) — smoke on 5-utt subset, then
   full 9508-utt run.
4. Run Component 3 (aggregator) on v2 `side_by_side.json`.
5. Run Component 4 (ingester) → new SQLite row.
6. Write Component 5 (v2 doc) — substitute measured numbers.
7. Update `scanrefer/README.md` + `leaderboard.md` + `docs/benchmark/README.md`
   to feature v2 as the paper-comparable headline (v1 stays as historical record
   with caveat banner).
8. Final regression sweep + push.

Estimated effort: 3-4 hours of dev wall + 30 min compute. No new agent
runs; no new test infrastructure; no `src/agents/**` changes.

## Acceptance criteria

- [ ] Component 1 loader passes ≥5 unit tests with synthetic mesh+agg+segs.
- [ ] Re-aggregator produces a new `side_by_side.json` covering the same
      9508 sample_ids as v1, with re-computed IoUs.
- [ ] Aggregator runs cleanly on the v2 output and reports
      Acc@0.25/0.50 × {Unique, Multiple, Overall}.
- [ ] SQLite has a `v2_aggregation_gt_track_20260503` row with
      non-null Acc@0.25/0.50.
- [ ] v2 numbers are within the predicted band (Acc@0.25 ∈ [55, 70],
      Acc@0.50 ∈ [30, 50]). If not, oracle-analysis was wrong; surface
      to user before continuing.
- [ ] No regression on v1 SQLite row (still queryable; numbers unchanged).
- [ ] `docs/benchmark/scanrefer/v2_aggregation_gt_track_20260503.md`
      exists with measured numbers + side-by-side table vs v1.
- [ ] Branch pushed; clean working tree (other than session-only files).

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| ScanNet `.ply` mesh not available locally | High | Open Q below — check the ScanNet download path; if missing, defer v2 until we have it. The aggregation analysis is doable without `.ply` only via segindex-only AABB reconstruction, which is wrong. |
| `objectId` indexing differs between aggregation file and ScanRefer JSON | Low | NR3D v3 verified target_id ↔ objectId 100 % match across 8584 utts; ScanRefer val uses the same convention per the loader audit. Verify on first 5-utt smoke. |
| AxisAlignment matrix already applied to mesh in some scans | Low | Check both modes in the test fixture; default `apply_axis_alignment=True` and let user override per scene if needed. |
| v2 numbers come out worse than v1 | Very low | If oracle ceiling on aggregation GT is ≤ Phase 8 ceiling, the bbox theory is wrong. Run a 5-utt sanity smoke first; abort if oracle ceiling drops. |
| Mask3D bbox doesn't actually align with aggregation GT either | Medium | Mask3D was trained on ScanNet200 instance segmentation labels which derive from the same aggregation files; this is the most-aligned GT available short of human re-annotation. If v2 still has a low ceiling, that's a Mask3D-segmentation-quality cap, not a GT issue. |

## Open questions / explicit deferrals

1. **Mesh source.** Where do we get `<scene>_vh_clean_2.ply` for the
   141 ScanRefer val scenes? Options:
   - ScanNet official download (license-required).
   - Re-derive vertex coordinates from depth+pose (lossy; not recommended).
   - Linux server may already have them under a different path (worth grepping
     `/data/`, `/home/ysh/`).

   **Action before v2 starts:** confirm mesh availability. If absent, the
   v2 work is blocked until they're rsync'd locally.

2. **Texture / color preservation.** Not needed for AABB; mesh vertex
   coordinates only.

3. **Per-LLM-call durability gap.** Carried forward from v1 / NR3D v3 —
   `tool_calls` / `llm_calls` SQLite tables stay empty until a separate
   instrumentation pass lands.

## Appendix — predicted impact (from oracle analysis)

If we assume:
- Mask3D bbox ≈ aggregation GT bbox (Mask3D was trained on this label
  source; deviation is empirical Mask3D segmentation noise, not GT
  derivation).
- Agent picking quality is invariant to GT (it's about which Mask3D
  candidate gets chosen, not about how we measure it).

Then:
- v2 oracle ceiling ≈ Mask3D-self-vs-aggregation-GT ceiling (~80-90 %
  for Acc@0.25, ~50-65 % for Acc@0.50, based on Mask3D-pool method
  ceilings in the ZSVG3D / SeeGround / CSVG papers).
- v2 agent Acc@0.25 ≈ 73.97 % × 80-90 % ≈ 60-65 %.
- v2 agent Acc@0.50 ≈ 79.00 % × 50-65 % ≈ 40-50 %.

These bands are in the same range as Camp-A SOTA. We should land
between SeeGround (44.1 / 39.4) and Z3D (58.9 / 52.7), with v3+
ablations targeting the gap to Z3D.
