# ScanRefer v3 Query-Driven Keyframe Track — Design

## Goal

Remove the GT view oracle from v1/v2's keyframe selection. Replace
`select_keyframes_from_phase8_target(target_id)` with the same
query-driven Stage 1 entry that OpenEQA uses
(`KeyframeSelector.select_keyframes_v2(query, ...)`). Everything
downstream — Mask3D pool, aggregation GT, annotated PNG rendering,
agent, VG tools, playbook, runner, aggregator — stays unchanged. v3
becomes the apples-to-apples zero-shot Camp-A headline; v2 demotes to
"GT-view-assisted upper bound" for ablation.

## Branch

`feat/scanrefer-v3-query-driven` (off `feat/scanrefer-vg-benchmark`
@ `609b0be`). v2 caveat docs already on this branch (commit `7e6eb40`).

## Locked decisions

1. **Stage 1 entry** = `KeyframeSelector.from_scene_path(<scene>/conceptgraph)`
   then `select_keyframes_v2(query, k=5, use_visual_context=False)`.
   Same call OpenEQA pilot uses.
2. **`use_visual_context=False`** disables BEV-image generation in
   hypothesis parsing. ScanNet `_vh_clean.ply` mesh under
   `data/scannetv2/` is not present locally; we rely on text-only
   hypothesis parsing. Quality-side: for short ScanRefer utterances
   (mean ~15 tokens) the text-only parse is empirically sufficient
   per smoke.
3. **`stride=10`** matches the raw RGB sampling we use elsewhere.
   Visibility index was built at stride=1 — selector logs a stride-
   mismatch warning but still returns valid `frame_id` integers usable
   by `_resolve_raw_rgb_path(raw_frames_root/scene, frame_id)`.
4. **`llm_model="gemini-2.5-pro"`** for hypothesis parsing — cheap
   ($0.005/call × 9508 ≈ $50) and matches what OpenEQA uses.
5. **Empty-keyframes fallback**: when `select_keyframes_v2` returns
   `keyframe_indices=[]` (no executable hypothesis or empty result),
   fall back to **scene-frame-with-most-Mask3D-candidates top-5**.
   This is query-blind but proposal-aware; preserves a non-trivial
   visual prior without leaking GT.
6. **Pack name** = `pack_scanrefer_v3_query_driven`. v1/v2
   `pack_scanrefer_v1` artifacts left untouched for ablation
   comparison.
7. **CLI flag** on `prepare_pack_v1_inputs_scanrefer.py`:
   `--keyframe-mode {gt_target,query_driven}` (default `gt_target` to
   preserve back-compat). v3 invokes with `query_driven`.
8. **Pre-flight**: run `enrich_objects.py` on all 141 NR3D-shared
   scenes (selector hard-requires `enriched_objects.json`). One-time
   cost ~$50, ~2-3h wall.
9. **NR3D parallel** = **out of scope for v3**. NR3D's headline metric
   is top-1 id-acc; GT pool already gives the label, and the same
   keyframe-oracle effect is smaller. Re-evaluate after v3 lands.
10. **Backend** = same `gpt-5.4-2026-03-05` agent as v1/v2; runner
    `run_scanrefer_vg_side_by_side.py` unchanged. workers = 32, same
    sample-retries = 1.

## Architecture (delta vs v2)

```
ScanRefer JSON utt  ──→  scanrefer_loader.ScanRefVGSample
                                 │
                                 │ pack-prep  (--keyframe-mode query_driven)
                                 ▼
        ┌──────────── select_keyframes (NEW) ─────────────┐
        │                                                 │
        │  if mode == gt_target:                          │
        │     select_keyframes_from_phase8_target(target_id, k=5)   ← v1/v2 path
        │  else:  # query_driven                          │
        │     selector = KeyframeSelector.from_scene_path(<scene>/conceptgraph)
        │     res = selector.select_keyframes_v2(query=utt, k=5,
        │                                       use_visual_context=False)
        │     if res.keyframe_indices is empty:           │
        │        return _fallback_top5_by_mask3d_density(scene)
        │     return [{"keyframe_idx": i, "frame_id": fid,
        │              "image_path": _resolve_raw_rgb_path(scene, fid)}
        │             for i, fid in enumerate(res.keyframe_indices)]
        │                                                 │
        └─────────────────────────────────────────────────┘
                                 │
                                 ▼
        normalize_prepared_keyframes  (same as v1/v2; swaps in annotated PNG)
                                 │
                                 ▼
        write_sample_artifact → samples/<safe_id>.json
        proposal pool / annotated dir / visibility.json:  reuse pack_scanrefer_v1 artifacts
        (the per-scene pre-rendered images don't depend on which 5 frames are picked
         per query; we just point keyframes at the same annotated_dir)
```

## Components

### Component 1 — `prepare_pack_v1_inputs_scanrefer.py` (modify)

- Add `--keyframe-mode {gt_target, query_driven}` arg (default
  `gt_target`).
- Lazy-import `KeyframeSelector` only when mode == `query_driven`.
- Cache `KeyframeSelector` per scene (constructor is expensive: loads
  pcd + enriched + visibility + builds CLIP model). Process samples
  scene-by-scene to amortize.
- Write the same `samples/<safe_id>.json` schema; only `keyframes`
  list differs by mode.
- Add `pack_scanrefer_v3_query_driven` writer path so v2 dirs aren't
  touched.

Implementation note: `select_keyframes_v2` may take 1-3 s per query
(LLM call + executor). Process scenes serially per worker; 9508 utts
across ~141 scenes → process scene-by-scene with intra-scene
parallelism only if needed.

### Component 2 — `_fallback_top5_by_mask3d_density` (new helper)

When Stage 1 returns no keyframes:

```python
def _fallback_top5_by_mask3d_density(
    scene_id: str, scene_artifacts: SceneArtifacts, k: int = 5
) -> list[dict]:
    """Top-k frames by len(visible Mask3D proposals). Query-blind but
    proposal-aware — preserves a visual prior without GT lookup."""
    visibility = json.loads(scene_artifacts.visibility_json.read_text())  # frame_id -> [proposal_ids]
    ranked = sorted(visibility.items(), key=lambda kv: -len(kv[1]))[:k]
    return [{"keyframe_idx": i, "frame_id": int(fid),
             "image_path": str(_resolve_raw_rgb_path(...))} 
            for i, (fid, _) in enumerate(ranked)]
```

Fallback usage telemetry: count + log per pack-prep run. Expected
fallback rate < 5 % based on Stage 1 hypothesis-parser robustness on
short utterances.

### Component 3 — `run_scanrefer_vg_side_by_side.py` (no change)

Consumer of `samples/*.json`. The `pack_name` arg accepts
`pack_scanrefer_v3_query_driven`. Already supports `--pack-name`.

### Component 4 — `scanrefer_leaderboard_metrics.py` (no change)

Consumes `side_by_side.json`. Pack-agnostic.

### Component 5 — `scripts/ingest_scanrefer_run.py` (modify)

Add column `keyframe_mode TEXT` to `runs` table (defaults to NULL on
existing rows, "gt_target" / "query_driven" on new). New `--keyframe-
mode` CLI arg, written to that column. SQL migration: `ALTER TABLE
runs ADD COLUMN keyframe_mode TEXT;`.

### Component 6 — `docs/benchmark/scanrefer/v3_query_driven_track_<date>.md` (new, post-run)

Mirrors v2 doc structure: Headline → Run Identity → Methodology →
SOTA Comparison (now showing v3 as headline + v2 demoted to "GT-view-
assisted upper bound") → Sanity check (oracle ceiling unchanged from
v2; picking quality the new headline) → Caveats → Cross-version
table.

### Component 7 — README + leaderboard (post-run)

Update `docs/benchmark/scanrefer/README.md` headline section + version
timeline; update `docs/benchmark/scanrefer/leaderboard.md` to make v3
the unmarked row and v2 carry the [‡] GT-view-oracle marker.

## Pre-flight checklist

- [x] Branch created: `feat/scanrefer-v3-query-driven` (commit `7e6eb40`)
- [x] v2 caveat docs committed
- [x] `open_clip_torch` installed in `.venv`
- [ ] **In progress**: `enrich_objects.py` on all 141 scenes
  (~2-3h, ~$50). Background task.
- [ ] Smoke `KeyframeSelector` on 1 enriched scene + 1 ScanRefer
  utterance, confirm `keyframe_indices` non-empty + frame_ids resolve
  to existing PNG files.
- [ ] Implement Component 1 + 2 + 5.
- [ ] Smoke pack-prep query_driven on 1 scene (10-30 utts).
- [ ] Smoke runner + metrics on smoke pack.
- [ ] Full pack-prep regen (9508 utts, ~6-13h).
- [ ] Full agent run (9508 utts, ~5-7h).
- [ ] Aggregator + ingest.
- [ ] Component 6 + 7.

## Risks

| Risk | Mitigation |
|---|---|
| Stage 1 LLM rate limits / cost overrun | gemini-2.5-pro pool already used by enrich; same pool for hypothesis parse. $50 budget. |
| Empty-keyframes rate > 10 % | Fallback to top-5 by Mask3D density (proposal-aware, query-blind). Acceptable but degrades; monitor. |
| Stride mismatch in visibility index | Smoke confirmed selector still returns valid frame_ids despite warning. If frame_id resolution fails, build a fresh stride-10 visibility index per scene (~30 min once). |
| `enriched_objects.json` quality differences vs OpenEQA | Same script, same prompt. NR3D objects are richer (Phase 8 GT-CG, not detector noise) so enrichment should be cleaner. |
| Mask3D pool quality bound caps v3 too | Carries over from v2 (Acc@0.50 ceiling 84.77 %). Independent of keyframe-mode. |
| Numbers drop sharply (e.g. v3 ≈ Z3D) | Useful negative result — proves v2 gain was view oracle, points us at detector / backbone upgrades next. |

## Definition of Done

- v3 SQLite row in `docs/benchmark/scanrefer/runs.sqlite` with
  `keyframe_mode = "query_driven"` and 6 non-null Acc columns.
- v3 doc exists with full methodology, SOTA table, Caveats including
  fallback rate.
- README + leaderboard show v3 as headline; v2 demoted with [‡].
- v1 caveat re-text the same way (the GT-view-oracle [‡] still
  applies to v1).
- Branch ready to PR.
