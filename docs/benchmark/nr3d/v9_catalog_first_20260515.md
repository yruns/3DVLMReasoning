# NR3D — v9 catalog-first (random100 fold) — skeleton

- **Branch**: feat/v9-catalog-first-scene-exploration
- **Tip commit**: TBD (fill in when run starts)
- **Pack**: `pack_nr3d_v9_catalog_first`
- **Fold**: random100 frozen at `tmp/nr3d_artifacts/random100_frozen.json`
  (same fold as v9_selective_mark for apples-to-apples comparison)
- **Judge**: `gemini-2.5-pro`
- **Launcher**: `scripts/run_v9_full_nr3d_random100.sh`
- **Raw artifacts**: `tmp/nr3d_eval_v9_full_<DATE>/`

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
| Overall | 74.00 | TBD | TBD |
| Easy | 90.24 | TBD | TBD |
| Hard | 62.71 | TBD | TBD |
| View-Dep | 64.71 | TBD | TBD |
| View-Indep | 78.79 | TBD | TBD |

## Caveats

- Same frozen 100 sample fold as v9_selective_mark; the numbers are directly
  comparable on a per-question basis (use the `samples` table in
  `docs/benchmark/nr3d/runs.sqlite`).
- BEV builder rendering is deterministic given mesh + intrinsics; if the eval
  host's mesh assets differ from the prep host's, the BEV image bytes will
  differ even when the catalog is identical.
- Per the design spec's risk #1: if Hard / View-Dep drop below v7 levels we
  plan to add an auxiliary trajectory-overlay image to the initial
  HumanMessage. This is *not* shipped in v9_full to keep the prompt cache
  warm.

## SQLite ingestion

```bash
python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v9_full_<DATE>/ \
    --run-id v9_full_<DATE> \
    --branch feat/v9-catalog-first-scene-exploration \
    --commit $(git rev-parse --short HEAD) \
    --judge-model gemini-2.5-pro \
    --notes "v9 catalog-first first random100 run" \
    --db docs/benchmark/nr3d/runs.sqlite
```

Reproduce headline:

```sql
SELECT category, AVG(stage2_score) AS mean_score, COUNT(*) AS n
FROM samples
WHERE run_id = 'v9_full_<DATE>'
GROUP BY category;
```
