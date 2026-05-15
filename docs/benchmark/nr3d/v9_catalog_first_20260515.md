# NR3D — v9 catalog-first (random100 fold)

- **Branch**: feat/v9-catalog-first-scene-exploration
- **Tip commit**: `e0ab061` (feat(bev): per-scene render cache + perspective-aware label projection)
- **Pack**: `pack_nr3d_v9_catalog_first`
- **Fold**: random100 frozen at
  `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
  (same fold as v9_selective_mark for apples-to-apples comparison)
- **Stage-2 backend**: `gpt-5.4-2026-03-05` via internal ModelHub (3-key rotator)
- **Judge**: none — NR3D leaderboard is *programmatic* (`selected_object_id == target_id`)
  via `evaluation.scripts.nr3d_leaderboard_metrics`. We keep the `--judge-model`
  field in run docs for cross-benchmark consistency but no LLM judge is invoked.
- **Launcher**: `scripts/run_v9_full_nr3d_random100.sh`
- **Raw artifacts**: `tmp/nr3d_eval_v9_full_20260515_1401/`
- **Pack-prep run**: `20260515_1335` (~24 min, 58 scenes, fully BEV-cached)
- **Eval run**: `20260515_1401` (~5 min wall-clock with workers=50, 100/100 completed)
- **Date harvested**: 2026-05-15

## What changed vs v9_selective_mark

| Layer | v9_selective_mark (a5f3625) | v9_full (this run) |
| --- | --- | --- |
| Initial HumanMessage | first-person seed + Cat-B text inventory | **BEV image + Cat-B text inventory (no first-person seed)** |
| Tool set | Stage-1 callbacks + `view_keyframe_marked` | 6 catalog-first selectors + unified `view_keyframe(mode='auto')` + `view_bev` |
| Playbook | `vg_grounding_playbook` (v9 part 1) | new shared `scene_exploration_playbook` + rewritten `vg_grounding_playbook` |
| Guards | TADG / no_match_guard / evidence_frame_guard against legacy names | same guards, rewired to `view_keyframe` (mode='marked'/'auto') and `list_scene_proposals` |
| Pack prep | `prepare_pack_v1_inputs_nr3d` writes proposals.jsonl + annotated keyframes | **same script** but also writes BEV + `scene_catalog.json` + `camera_trajectory.json` |

## Headline

| Metric | v9_selective_mark (a5f3625) | **v9_full (this run)** | Δ |
| --- | ---: | ---: | ---: |
| Overall (n=100) | 74.00 | **81.00** | **+7.00** |
| Easy (n=41) | 90.24 | **85.37** | -4.87 |
| Hard (n=59) | 62.71 | **77.97** | **+15.26** |
| View-Dep (n=34) | 64.71 | **73.53** | **+8.82** |
| View-Indep (n=66) | 78.79 | **84.85** | **+6.06** |

(`n_full == n_filtered == 100` — no samples dropped by the canonical
`mentions_target_class` filter; this matches the v9_selective_mark fold.)

## Per-question failure pattern (quick scan)

- 19 incorrect predictions overall (`classification_acc = 81/100`).
- Disproportionately concentrated in Hard / view-dep with same-category clutter:
  - Same-category adjacency: e.g. "the lamp between the two beds" → off-by-one.
  - "this" / first-person anchor without enough first-person frames: BEV-only
    initial prompt loses pronoun grounding — agent has to invoke selectors to
    recover; sometimes it picks a near-duplicate proposal instead.
- All 100 samples reached `completed` status (no timeouts, no judge
  unreachable, no proposal-pool fallout).

Detailed per-sample regression vs v9_selective_mark requires SQLite
ingestion (next step below).

## Wall-clock & infra

- Mac (Apple Silicon) `.venv-agents`, `workers=50`, RSS guard 15GB, default
  3-key ModelHub AK rotator. No GPU used (Stage-2 is API-bound).
- Pack-prep one-time cost: ~24 min for 58 scenes (BEV rendered + cached).
  Subsequent reruns hit the per-scene `render_hash` cache and skip Open3D
  entirely.
- Eval cost: ~5 min wall-clock from launch to side_by_side.json (100 samples,
  workers=50, agents typically 2–3 turns with 25–30 tool calls each).
- ModelHub: ~few `429` retries handled transparently by `ModelHubKeyRotator`
  (20s demotion, automatic recovery).

## Caveats

- Same frozen 100 sample fold as v9_selective_mark; the numbers are directly
  comparable on a per-question basis (use the `samples` table in
  `docs/benchmark/nr3d/runs.sqlite`).
- BEV builder rendering is deterministic given mesh + intrinsics; the per-scene
  cache key (`render_hash`) is folded into the BEV filename so different
  meshes / proposals would yield distinct cache entries. The eval host and the
  prep host are the same Mac, so cache hits are guaranteed within this run.
- The `--judge-model gemini-2.5-pro` flag in the side-by-side runner is
  cosmetic; NR3D classification accuracy is a pure id-equality check.
- Per the design spec's risk #1: if Hard / View-Dep had dropped below v7
  levels we planned to add an auxiliary trajectory-overlay image to the
  initial HumanMessage. With Hard +15.26 and View-Dep +8.82 vs
  v9_selective_mark, **risk #1 did not materialize** — the BEV-only initial
  view is sufficient.

## SQLite ingestion (TODO before paper / leaderboard update)

```bash
python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v9_full_20260515_1401/ \
    --run-id v9_full_20260515_1401 \
    --branch feat/v9-catalog-first-scene-exploration \
    --commit e0ab061 \
    --judge-model none \
    --notes "v9 catalog-first first random100 run; +7.0 overall vs v9_selective_mark, Hard +15.26" \
    --db docs/benchmark/nr3d/runs.sqlite
```

Reproduce headline from DB:

```sql
SELECT
  COUNT(*) AS n,
  AVG(CAST(selected_object_id = target_id AS REAL)) AS classification_acc
FROM samples
WHERE run_id = 'v9_full_20260515_1401';
```

Per-question regression vs v9_selective_mark (after both runs ingested):

```sql
SELECT
  s_new.sample_id,
  s_old.selected_object_id AS old_pred,
  s_new.selected_object_id AS new_pred,
  s_old.target_id          AS gt,
  (s_old.selected_object_id = s_old.target_id) AS old_correct,
  (s_new.selected_object_id = s_new.target_id) AS new_correct
FROM samples s_new
JOIN samples s_old USING (sample_id)
WHERE s_new.run_id = 'v9_full_20260515_1401'
  AND s_old.run_id = 'v9_selective_mark_<date>'
  AND old_correct != new_correct;
```
