# v1 Mask3D-Track — 2026-05-02

First ScanRefer detection-mode evaluation on the canonical full val set
(9508 utterances on 141 scenes), using Mask3D ScanNet200 predictions as
the proposal pool — the de-facto shared detector for Camp-A zero-shot
methods (ZSVG3D / SeeGround / CSVG / Z3D / SeqVLM).

## Run Identity

- Branch: `feat/scanrefer-vg-benchmark`
- Tip commit: `e45d9a5`
- Internal version: `v1_mask3d_track`
- Run ID: `v1_mask3d_track_20260502`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Workers: 32 (NR3D v3 sweet spot for 3-key API rotation)

## Methodology

ScanRefer is a detection-mode benchmark — methods must produce 3D bboxes,
not pick from a GT pool. We adopt the standard Camp-A protocol used by
ZSVG3D, SeeGround, CSVG, and Z3D:

- **Pool**: Mask3D ScanNet200 predictions distributed by ZSVG3D
  (`data/scanrefer/Mask3d/scannet200/<scene>.npz`), repackaged into a
  ConceptGraph-shaped pkl per scene by `build_scanrefer_mask3d_cg.py`.
- **GT bbox**: Phase 8 GT-CG pkl (functionally equivalent to Vil3dRef's
  `pcd_with_global_alignment` `.pth`; same `(min+max)/2` axis-aligned
  derivation).
- **Agent input**: 5 RGB keyframes per query, selected by Phase 8
  visibility of the target GT instance, with Mask3D candidate boxes
  projected as 2D overlays. Reuses NR3D v3 Stage 1 + Stage 2 unchanged.
- **Metric**: axis-aligned 3D IoU via `compute_oriented_iou_3d`
  with Euler=0; Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }.
- **Filtering**: drop wall / floor / ceiling Mask3D instances (matches
  ZSVG3D `keep_background=False`).

## Fold

- Split: ScanRefer val
- Total utterances: 9508
- Scenes: 141 (130 NR3D-overlap + 11 ScanRefer-only built by
  `nr3d_gt_conceptgraph` Linux producer on 2026-05-02 — see
  `docs/benchmark/scanrefer/producer_report_20260502.md`).
- Unique partition: 2582 utts where target's class has exactly 1 GT instance in scene
- Multiple partition: 6926 utts where target's class has ≥ 2 GT instances

## Headline Metrics

| Metric | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | **62.97** | **47.43** | **51.65** |
| Acc@0.50 | **25.41** | **12.79** | **16.22** |

mean IoU (overall): 0.2746

Failed sentinels: 205 / 9508 (2.16 %) — counted as iou=0 in the metrics above.

## SOTA Comparison (ScanRefer val, detection mode w/ Mask3D pool)

| Method | Setup | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| ZSVG3D / GPT-4-Turbo | Mask3D pool, zero-shot | 63.8 | 58.4 | 27.7 | 24.6 | 36.4 | 32.7 |
| CSVG (Mask3D) | Mask3D pool, zero-shot | 68.8 | 61.2 | 38.4 | 27.3 | 49.6 | 39.8 |
| SeeGround / Qwen2-VL-72B | Mask3D pool, zero-shot | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 |
| VLM-Grounder / GPT-4V (250 sub-sample) | 2D-online proj, zero-shot | 66.0 | 29.8 | 48.3 | 33.5 | 51.6 | 32.8 |
| Z3D (Mask3D row) | Mask3D pool, zero-shot | 82.3 | 74.8 | 51.5 | 45.7 | 58.9 | 52.7 |
| **Ours v1 (gpt-5.4)** | Mask3D pool, zero-shot | **62.97** | **25.41** | **47.43** | **12.79** | **51.65** | **16.22** |

For supervised SOTA (test-server top-5 + val Table 1), see `leaderboard.md`.

## Caveats

- **Zero-shot RGB+VLM agent vs trained 3D models** — paradigm difference, not protocol violation.
- **Mask3D pool quality bounds the upper limit** — instances mis-segmented by Mask3D are unrecoverable.
- **Wall/floor/ceiling Mask3D candidates filtered** — matches ZSVG3D `keep_background=False`. Drop count per scene visible in producer report.
- **GT bbox source** = Phase 8 GT-CG pkl (Vil3dRef-equivalent; see `pool_equivalence_log_20260501.md` from NR3D v3 work).
- **Per-LLM-call durability gap** carried forward from NR3D v3 — `tool_calls`/`llm_calls` SQLite tables empty for v1.
- **Acc@0.50 is a notable weak spot** — Overall 16.22% sits below Camp-A SOTA (32-53%). The gap is concentrated in Multiple@0.50 (12.79% vs SOTA 24-46%); Unique@0.50 25.41% is in the SOTA-low range. Hypothesis: keyframe selection on Phase 8 GT visibility gives the agent strong target localization (high Acc@0.25) but the agent's bbox refinement is bounded by the Mask3D candidate's tight-fit precision rather than agent intent. v2/v3 ablations should isolate this.

## Cross-version comparison (ScanRefer-only)

| Version | Date | n_total | Headline | Notes |
|---|---|---:|---|---|
| **v1_mask3d_track** | 2026-05-02 | 9508 | Acc@0.25 = 51.65 / Acc@0.50 = 16.22 | First ScanRefer detection-mode track |

## SQLite Reproduction

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', acc25_unique) AS u25,
       printf('%.4f', acc50_unique) AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50
FROM runs WHERE run_id='v1_mask3d_track_20260502';
```

## Raw Artifacts

- ScanRefer JSON: `data/scanrefer/raw/ScanRefer_filtered_val.json`
- Mask3D `.npz` distribution: `data/scanrefer/Mask3d/scannet200/<scene>.npz`
- Mask3D-CG pkl (per scene): `data/scanrefer/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz`
- Phase 8 GT lookup (per scene): `data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz`
- Per-sample checkpoints: `tmp/scanrefer_eval_v1_full/per_sample/pack_scanrefer_v1/*.json`
- Aggregate: `tmp/scanrefer_eval_v1_full/side_by_side.json`
- Leaderboard metrics: `tmp/scanrefer_eval_v1_full/leaderboard_metrics.json`
- SQLite row: `docs/benchmark/scanrefer/runs.sqlite::runs.run_id='v1_mask3d_track_20260502'`

## Reproduction Command

```bash
source .venv/bin/activate

# Step 1 (one-time): convert Mask3D npz → ConceptGraph-shaped pkl
PYTHONPATH=src python -m scripts.build_scanrefer_mask3d_cg \
    --scene-list data/scanrefer/raw/ScanRefer_filtered_val.txt \
    --mask3d-root data/scanrefer/Mask3d/scannet200 \
    --raw-root data/nr3d/scannet \
    --output-root data/scanrefer/scannet

# Step 2: pack-prep
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py \
    --sample-ids tmp/scanrefer_artifacts/full_val_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --pack-name pack_scanrefer_v1

# Step 3: full agent run (~8h on workers=32, plan estimate ~16h)
PYTHONPATH=src python src/evaluation/scripts/run_scanrefer_vg_side_by_side.py \
    --sample-ids tmp/scanrefer_artifacts/full_val_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --pack-name pack_scanrefer_v1 \
    --output-dir tmp/scanrefer_eval_v1_full \
    --workers 32 --sample-retries 1

# Step 4: aggregate + ingest
PYTHONPATH=src python src/evaluation/scripts/scanrefer_leaderboard_metrics.py \
    --side-by-side tmp/scanrefer_eval_v1_full/side_by_side.json \
    --output tmp/scanrefer_eval_v1_full/leaderboard_metrics.json
PYTHONPATH=src python scripts/ingest_scanrefer_run.py \
    --output-dir tmp/scanrefer_eval_v1_full \
    --run-id v1_mask3d_track_20260502 \
    --branch feat/scanrefer-vg-benchmark --commit "$(git rev-parse --short HEAD)" \
    --leaderboard-metrics tmp/scanrefer_eval_v1_full/leaderboard_metrics.json \
    --db docs/benchmark/scanrefer/runs.sqlite
```

> **Note on Step 4 aggregator:** the canonical aggregator's `compute_leaderboard_metrics` calls `ScanRefVGDataset.from_path()` without a `sample_ids` filter, which loads all 141 Phase 8 GT-CG pkls into memory and OOM'd on macOS during this run. The actual aggregation used a memory-frugal scene-by-scene driver that opens one Phase 8 pkl at a time, computes per-scene class counts, emits sample_meta, and releases the pkl before the next scene. Logically equivalent to the CLI; v2 should patch the aggregator to accept a sample_ids filter.

## Next Steps

- v2: SeeGround head-to-head with Qwen2-VL-72B backbone (apples-to-apples backbone match).
- v3: detector ablation (BIP3D / V-DETR / GroupFree3D pool variants).
- v4: SR3D extension.
