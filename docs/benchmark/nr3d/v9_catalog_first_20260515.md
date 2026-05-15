# NR3D — v9 catalog-first (random100 fold)

Two runs, same fold, same code path *except for the Stage-1 keyframe leak fix*:

| Run | Tip commit | Overall | Easy | Hard | V-Dep | V-Indep | What changed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v9_full (leaky) | `e0ab061` | 81.00 | 85.37 | 77.97 | 73.53 | 84.85 | First random100; `bundle.keyframes` from pack-prep were auto-injected by `build_evidence_update_message`, leaking 5 GT-target-visible Stage-1 frames into context. |
| **v9_full_clean (leak-fixed)** | this commit | **86.00** | **92.68** | **81.36** | **82.35** | **87.88** | `Stage2RuntimeState.initial_keyframe_paths` snapshots Stage-1 seeds at runtime construction; `build_evidence_update_message` skips them so the agent only ever sees images it *explicitly* fetched via selector / view_keyframe / view_bev / request_crops. |

Per-split deltas from leaky → clean: Overall **+5.00**, Easy **+7.31**, Hard
**+3.39**, V-Dep **+8.82**, V-Indep **+3.03**. *Every split improved.* The
hypothesis that "the leak was propping the number up" was wrong; the leak was
actually *biasing the agent toward the GT-target-visible seeds and starving it
of evidence about same-category distractors*.

## Provenance

- **Branch**: `feat/v9-catalog-first-scene-exploration`
- **Pack**: `pack_nr3d_v9_catalog_first`
- **Fold**: `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
  (same fold as v9_selective_mark for apples-to-apples comparison)
- **Stage-2 backend**: `gpt-5.4-2026-03-05` via internal ModelHub (3-key rotator)
- **Judge**: none — NR3D leaderboard is *programmatic*
  (`selected_object_id == target_id`).
- **Launcher**: `scripts/run_v9_full_nr3d_random100.sh`
- **Raw artifacts (clean)**: `tmp/nr3d_eval_v9_full_clean_20260515_1442/`
- **Raw artifacts (leaky)**: `tmp/nr3d_eval_v9_full_20260515_1401/`
- **Date harvested**: 2026-05-15

## The Stage-1 leak (root cause)

`build_user_message` was correctly catalog-first: it only attached the BEV
image and the SceneCatalog text. But `build_evidence_update_message` —
which runs after every agent turn that queues new images — also drained
**every keyframe** present in `bundle.keyframes`, including the 5
GT-target-visible Stage-1 seeds that pack-prep had pre-selected. So on the
agent's *first* premature `submit_final` attempt (which `evidence_frame_guard`
defers), all 5 Stage-1 seed frames quietly entered the context.

```diff
# src/agents/runtime/deepagents_agent.py — build_evidence_update_message
        new_images: list[str] = []
+       # v9 catalog-first: never auto-inject pack-prep "seed" keyframes
+       # (Stage-1 GT-target-visible RGBs). The agent must explicitly fetch
+       # first-person frames via view_keyframe / select_* / view_bev /
+       # request_crops. Tool-produced keyframes (e.g. request_crops crops)
+       # are not in initial_keyframe_paths and are still drained below.
+       initial_seeds = runtime.initial_keyframe_paths
        for keyframe in runtime.bundle.keyframes:
+           if keyframe.image_path in initial_seeds:
+               continue
            if Path(keyframe.image_path).exists():
                if keyframe.image_path not in runtime.seen_image_paths:
                    new_images.append(keyframe.image_path)
```

`Stage2RuntimeState.initial_keyframe_paths` is populated when
`DeepAgentsStage2Runtime.build_agent` constructs the runtime, by
snapshotting the seed paths from `bundle.keyframes`. `request_crops`
appends *new* keyframes whose paths are not in the snapshot, so its
output is still surfaced when the legacy crop callback fires.

## What changed vs v9_selective_mark

| Layer | v9_selective_mark (a5f3625) | v9_full_clean (this run) |
| --- | --- | --- |
| Initial HumanMessage | first-person seed + Cat-B text inventory | **BEV image + Cat-B text inventory only (no first-person seed, leak now fixed)** |
| Tool set | Stage-1 callbacks + `view_keyframe_marked` | 6 catalog-first selectors + unified `view_keyframe(mode='auto')` + `view_bev` |
| Playbook | `vg_grounding_playbook` (v9 part 1) | new shared `scene_exploration_playbook` + rewritten `vg_grounding_playbook` |
| Guards | TADG / no_match_guard / evidence_frame_guard against legacy names | same guards, rewired to `view_keyframe` (mode='marked'/'auto') and `list_scene_proposals` |
| Pack prep | writes proposals.jsonl + annotated keyframes | also writes BEV + `scene_catalog.json` + `camera_trajectory.json`. Legacy `keyframes` list still emitted for runner backward-compat but is **not** auto-injected. |

## Headline

| Metric | v9_selective_mark | v9_full leaky | **v9_full_clean** | Δ vs v9_sel | Δ vs v9_full_leaky |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall (n=100) | 74.00 | 81.00 | **86.00** | **+12.00** | +5.00 |
| Easy (n=41) | 90.24 | 85.37 | **92.68** | **+2.44** | +7.31 |
| Hard (n=59) | 62.71 | 77.97 | **81.36** | **+18.65** | +3.39 |
| View-Dep (n=34) | 64.71 | 73.53 | **82.35** | **+17.64** | +8.82 |
| View-Indep (n=66) | 78.79 | 84.85 | **87.88** | **+9.09** | +3.03 |

(`n_full == n_filtered == 100` — no samples dropped by the canonical
`mentions_target_class` filter.)

## Tool-usage profile (clean run)

| Family | Calls (across 100 samples) | Notes |
| --- | ---: | --- |
| `view_keyframe` | 882 | Dominant evidence acquisition path. Most calls in `mode='auto'` (marked for VG). |
| `inspect_proposal` | 429 | Catalog dive — agent reads proposal metadata before grounding. |
| `submit_final` | 228 | Median 2.3 calls per sample (most include a deferred retry that gets re-injected). |
| `load_skill` | 203 | scene-exploration-playbook + vg-grounding-playbook, twice per sample. |
| `select_by_proposal` | 161 | Cheap filter: which frames contain candidates X, Y, Z? |
| `view_bev` | 111 | Re-render BEV with `highlight=[...]` to declutter. |
| `compare_proposals_spatial` | 58 | Used for left/right/near/next_to relations. |
| `list_scene_proposals` | 46 | Catalog dump on demand. |
| `select_by_frame_neighbor` | 38 | Temporal expansion around an anchor frame. |
| `request_crops` | 21 | **No-op in v9** — callback returns text-only placeholder, so the agent's frame_indices don't actually produce crops. Cycles wasted; safe to drop in v10. |
| `list_frame_proposals` | 6 | Used when an agent wants ids in a specific frame. |
| `retrieve_object_context` | 6 | Legacy `pack_v1` context retriever. |
| `select_by_text` | 3 | CLIP text query — rarely used. |
| `select_by_region` | 2 | BEV bbox query — rarely used. |
| `select_by_coverage` | 2 | Frame-count-based query — rarely used. |

Cleanups suggested for v10:

- Remove `request_crops` entirely — 21 wasted calls per 100 samples and
  the callback is already a no-op.
- Promote `view_keyframe(mode='auto')` to the default for VG, since
  `mode='auto' == 'marked'` everywhere in this run.

## Caveats

- Pack-prep still writes a non-empty `keyframes` list into each sample
  artifact (5 GT-target-visible frames per sample). The list is now
  purely a *backward-compat anchor* — required by the side-by-side
  runner's `if not keyframes: raise ValueError` assertion and used by
  `request_crops` for `frame_idx → image_path` lookup. The agent never
  sees those frames unless it explicitly calls `request_crops`. A clean
  v10 should drop the assertion and the list.
- The leaky and clean runs share the *same* fold and the *same* pack
  prep output (BEV / scene_catalog / camera_trajectory are
  deterministic). The only delta between them is the
  `initial_keyframe_paths` filter.
- `request_crops` in v9 is a no-op (returns placeholder text). The agent's
  21 calls had no effect on context; treat the count as protocol overhead.

## SQLite ingestion (TODO before paper / leaderboard update)

```bash
python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v9_full_clean_20260515_1442/ \
    --run-id v9_full_clean_20260515_1442 \
    --branch feat/v9-catalog-first-scene-exploration \
    --commit $(git rev-parse --short HEAD) \
    --judge-model none \
    --notes "v9 catalog-first leak-fixed; +5.0 overall vs leaky v9_full; Easy +7.3 Hard +3.4 VDep +8.8" \
    --db docs/benchmark/nr3d/runs.sqlite
```

Reproduce headline from DB:

```sql
SELECT
  COUNT(*) AS n,
  AVG(CAST(selected_object_id = target_id AS REAL)) AS classification_acc
FROM samples
WHERE run_id = 'v9_full_clean_20260515_1442';
```

## Trace viewer

A LangSmith-style static viewer for 4 representative cases (2 correct, 2
failed; mix of Easy / Hard / VDep / VInd splits) is at:

- HTML: [v9_catalog_first_langsmith_trace_20260515.html](v9_catalog_first_langsmith_trace_20260515.html)
- Assets: `assets/v9_catalog_first_langsmith_trace_20260515/`

The viewer was rewritten for v9 catalog-first (see
`scripts/generate_nr3d_langsmith_trace_html.py`):

- Turn-0 panel shows the **BEV image + SceneCatalog table** (exactly
  what `build_user_message` injects). No Stage-1 seed-keyframe cluster.
- Tool cards are color-coded by family: `setup` / `catalog` / `selector`
  / `view` / `reason` / `final`.
- Legacy tools (`view_keyframe_marked`, `request_more_views`,
  `find_proposals_by_category`, etc.) render with a dashed amber
  outline + a per-case warning bar, so a historical run is still
  readable but visually flagged as "needs migration".
