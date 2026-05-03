# v2 Aggregation-GT-Track — 2026-05-03

Re-aggregation of the v1 9508-utt agent run against ScanNet
aggregation-based GT bbox (the canonical Camp-A reference). **Same agent
decisions, different GT measurement** — v2 swaps Phase 8 GT-CG bbox for
the mesh-aggregation-derived bbox that Mask3D was trained against and
that ZSVG3D / SeeGround / CSVG / Z3D all evaluate against. No agent
re-run was needed; only the IoU-target side of every per-sample IoU was
recomputed (~30 s wall on a Mac).

## Run Identity

- Branch: `feat/scanrefer-vg-benchmark`
- Tip commit: `3ffac81` (v1 ship + post-v1 caveats; v2 commit on top)
- Internal version: `v2_aggregation_gt_track`
- Run ID: `v2_aggregation_gt_track_20260503`
- Source agent run: `v1_mask3d_track_20260502` (gpt-5.4-2026-03-05,
  workers=32, sample-retries=1, ~8h on 2026-05-02 → 2026-05-03)
- Re-aggregation script: `src/evaluation/scripts/rescan_with_aggregation_gt.py`
- Spec: `docs/superpowers/specs/2026-05-03-scanrefer-v2-aggregation-gt.md`

## Why v2 exists

v1 reported Acc@0.50 = 16.22 % using Phase 8 GT-CG bbox as the IoU
reference. Oracle analysis (`v1_oracle_analysis_20260503.md`) showed
Phase 8 GT-CG bbox is systematically ~2× larger by volume than Mask3D
bbox / ScanNet aggregation GT, so v1's @0.50 number was bottlenecked by
GT geometry rather than agent quality (oracle ceiling on v1 was Acc@0.50
= 20.53 % — i.e. Z3D's published 52.7 % was unreachable with that GT).

v2 fixes this by deriving GT bbox from `_vh_clean_2.ply` +
`_vh_clean_2.0.010000.segs.json` + `<scene>.aggregation.json` +
axis-alignment matrix from `<scene>.txt` — the canonical pipeline that
Mask3D itself, ZSVG3D, SeeGround, CSVG, and Z3D all use.

## Methodology

Identical to v1 except for the GT bbox source:

- **Pool**: same Mask3D ScanNet200 predictions (ZSVG3D distribution).
- **GT bbox**: derived from ScanNet aggregation files + mesh per scene
  (axis-aligned 8-corner AABB of mesh vertices belonging to each
  `objectId`).
- **Agent input**: same 5 RGB keyframes per query, same Stage 1 + Stage 2
  pipeline, same gpt-5.4-2026-03-05 backend.
- **Metric**: same `compute_oriented_iou_3d` with Euler = 0.
- **Filtering**: drop wall / floor / ceiling Mask3D instances (matches
  ZSVG3D `keep_background=False`).

## Fold

- Identical to v1: ScanRefer val 9508 utts, 141 scenes.
- Unique partition: 2582 utts.
- Multiple partition: 6926 utts.

## Headline Metrics

| Metric | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | **83.11** | **65.00** | **69.92** |
| Acc@0.50 | **76.49** | **57.68** | **62.79** |

mean IoU (overall): **0.6022** (v1: 0.2746).
Failed sentinels: 205 / 9508 (2.16 %) — same as v1 (carried over).

## v1 → v2 delta

Same 9508 agent decisions. Only the GT side of IoU changed.

| Metric | v1 (Phase 8 GT-CG) | v2 (aggregation GT) | Δ |
|---|---:|---:|---:|
| mean_iou_overall | 0.2746 | 0.6022 | +0.328 (+119 %) |
| Acc@0.25 Overall | 51.65 | 69.92 | **+18.27 pp** |
| Acc@0.50 Overall | 16.22 | 62.79 | **+46.57 pp** |
| Acc@0.25 Unique | 62.97 | 83.11 | +20.14 pp |
| Acc@0.50 Unique | 25.41 | 76.49 | +51.08 pp |
| Acc@0.25 Multiple | 47.43 | 65.00 | +17.57 pp |
| Acc@0.50 Multiple | 12.79 | 57.68 | +44.89 pp |

The Acc@0.50 jumps are dramatic because v1's GT was ~2× the volume of
the Mask3D candidate bbox, so even perfect picks were stuck near
IoU = 0.5; v2's GT is mesh-tight and Mask3D-aligned, so correct picks
score ~0.7-1.0.

## SOTA Comparison (ScanRefer val, detection mode w/ Mask3D pool)

| Method | Setup | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| ZSVG3D / GPT-4-Turbo | Mask3D pool, zero-shot | 63.8 | 58.4 | 27.7 | 24.6 | 36.4 | 32.7 |
| CSVG (Mask3D) | Mask3D pool, zero-shot | 68.8 | 61.2 | 38.4 | 27.3 | 49.6 | 39.8 |
| SeeGround / Qwen2-VL-72B | Mask3D pool, zero-shot | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 |
| VLM-Grounder / GPT-4V (250 sub-sample) | 2D-online proj, zero-shot | 66.0 | 29.8 | 48.3 | 33.5 | 51.6 | 32.8 |
| Z3D (Mask3D row) | Mask3D pool, zero-shot | 82.3 | 74.8 | 51.5 | 45.7 | 58.9 | 52.7 |
| **Ours v2 (gpt-5.4)** | Mask3D pool, zero-shot | **83.11** | **76.49** | **65.00** | **57.68** | **69.92** | **62.79** |

We outperform Z3D (the previous Camp-A SOTA) on every column:
Unique@0.25 +0.81 / Unique@0.50 +1.69 / Multiple@0.25 +13.5 /
Multiple@0.50 +11.98 / Overall@0.25 +11.02 / Overall@0.50 +10.09. The
Multiple gap is the largest — language-disambiguation is where the
RGB+VLM agent advantage shows most.

For supervised SOTA reference, see `leaderboard.md`.

## Oracle / picking-quality sanity check

To confirm the v2 numbers aren't a measurement bug, we recomputed the
oracle ceiling (best-Mask3D-vs-aggregation-GT IoU) and the
oracle-normalized agent picking quality:

| | Oracle Acc | Agent Acc | Picking quality |
|---|---:|---:|---:|
| @0.25 | 93.31 % | 69.92 % | 74.93 % |
| @0.50 | 84.77 % | 62.79 % | 74.07 % |
| mean IoU | 0.8008 | 0.6022 | — |

Picking quality is consistent with v1 (73.97 % @0.25 / 79.00 % @0.50)
and with NR3D v3 classification_acc 80.79 %. Same agent, same skill;
the headline metric improvement comes entirely from the GT change.

## Caveats

- **Same-agent property**: v2 reuses v1's per-sample agent decisions
  byte-for-byte. The improvement is fully attributable to switching
  the GT-bbox source. v3+ work targeting the agent itself would
  improve the picking-quality factor (75 → 80 %, etc.).
- **Mask3D pool quality bound**: the oracle ceiling at Acc@0.50 = 84.77 %
  represents Mask3D's segmentation accuracy, not a GT issue. Higher
  numbers would require a better detector (V-DETR, BIP3D, etc.) — see
  v3 next-step.
- **Wall/floor/ceiling Mask3D candidates filtered** — matches Camp-A.
- **Per-LLM-call durability gap** carried forward from NR3D v3 —
  `tool_calls`/`llm_calls` SQLite tables empty for v2.

## Cross-version comparison (ScanRefer-only)

| Version | Date | n_total | Headline | Notes |
|---|---|---:|---|---|
| v1_mask3d_track | 2026-05-02 | 9508 | Acc@0.25 = 51.65 / Acc@0.50 = 16.22 | First eval; **Phase 8 GT-CG bbox** — not paper-comparable. |
| **v2_aggregation_gt_track** | 2026-05-03 | 9508 | **Acc@0.25 = 69.92 / Acc@0.50 = 62.79** | Same v1 agent run; aggregation-GT re-aggregation. Camp-A SOTA across all 6 columns. |

## SQLite Reproduction

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', acc25_unique) AS u25,
       printf('%.4f', acc50_unique) AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50
FROM runs WHERE run_id='v2_aggregation_gt_track_20260503';
```

## Raw Artifacts

- ScanNet meshes (rsync'd 2026-05-03): `data/nr3d/scannet_aux_meshes/<scene>/<scene>_vh_clean_2.ply` (141 / 141, 863 MB total).
- ScanNet aux (already on disk): `data/nr3d/scannet_aux/<scene>/{<scene>.aggregation.json, <scene>_vh_clean_2.0.010000.segs.json, <scene>.txt}`.
- v1 source side_by_side: `tmp/scanrefer_eval_v1_full/side_by_side.json`.
- v2 side_by_side (re-aggregated): `tmp/scanrefer_eval_v2_aggregation_gt/side_by_side.json`.
- v2 leaderboard metrics: `tmp/scanrefer_eval_v2_aggregation_gt/leaderboard_metrics.json`.
- SQLite row: `docs/benchmark/scanrefer/runs.sqlite::runs.run_id='v2_aggregation_gt_track_20260503'`.

## Reproduction Command

Assuming v1 has already run and produced `tmp/scanrefer_eval_v1_full/`:

```bash
source .venv/bin/activate

# 1. Pull ScanNet meshes (one-time, 863 MB)
rsync -av --mkpath --files-from=<(awk '{print $1"/"$1"_vh_clean_2.ply"}' \
        data/scanrefer/raw/ScanRefer_filtered_val.txt) \
    bupt:/home/ysh/Datasets/ScanNet/scans/ \
    data/nr3d/scannet_aux_meshes/

# 2. Re-aggregate v1 outputs against aggregation GT
PYTHONPATH=src python src/evaluation/scripts/rescan_with_aggregation_gt.py \
    --side-by-side tmp/scanrefer_eval_v1_full/side_by_side.json \
    --output-dir tmp/scanrefer_eval_v2_aggregation_gt \
    --scannet-aux-root data/nr3d/scannet_aux \
    --mesh-root data/nr3d/scannet_aux_meshes

# 3. Aggregator (Unique/Multiple slicing) — uses memory-frugal scene-by-scene meta
#    (the canonical CLI's compute_leaderboard_metrics OOMs on macOS at full-val
#    Phase 8 load; this is documented in v1 doc § Reproduction.)
PYTHONPATH=src python -c "
import gzip, json, pickle
from collections import defaultdict, Counter
from pathlib import Path
from evaluation.scripts.scanrefer_leaderboard_metrics import aggregate
side = json.load(open('tmp/scanrefer_eval_v2_aggregation_gt/side_by_side.json'))
val = json.load(open('data/scanrefer/raw/ScanRefer_filtered_val.json'))
val_by_sid = {f\"scannet/{r['scene_id']}::{r['object_id']}::{r['ann_id']}\": r for r in val}
by_scene = defaultdict(list)
for rec in side['pack_v1']['per_sample']:
    by_scene[rec['sample_id'].split('::')[0].split('/')[-1]].append(rec['sample_id'])
sample_meta = []
for scene, sids in sorted(by_scene.items()):
    p = Path(f'data/nr3d/scannet/{scene}/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz')
    with gzip.open(p, 'rb') as f: objs = pickle.load(f).get('objects') or []
    counts = Counter()
    for o in objs:
        cn = o.get('class_name')
        if isinstance(cn, list) and cn: counts[cn[0].lower()] += 1
    for sid in sids:
        v = val_by_sid.get(sid)
        if v is None: continue
        sample_meta.append({'sample_id': sid, 'is_unique': counts.get(v['object_name'].lower(), 0) == 1, 'target_id': int(v['object_id']), 'target': v['object_name']})
m = aggregate(side['pack_v1']['per_sample'], sample_meta)
json.dump(m, open('tmp/scanrefer_eval_v2_aggregation_gt/leaderboard_metrics.json', 'w'), indent=2)
print(f'overall acc25={m[\"acc25_overall\"]:.4f} acc50={m[\"acc50_overall\"]:.4f}')
"

# 4. Ingest into SQLite
PYTHONPATH=src python scripts/ingest_scanrefer_run.py \
    --output-dir tmp/scanrefer_eval_v2_aggregation_gt \
    --run-id v2_aggregation_gt_track_20260503 \
    --branch feat/scanrefer-vg-benchmark --commit "$(git rev-parse --short HEAD)" \
    --leaderboard-metrics tmp/scanrefer_eval_v2_aggregation_gt/leaderboard_metrics.json \
    --db docs/benchmark/scanrefer/runs.sqlite
```

Total wall time end-to-end (assuming v1 outputs on disk + Linux SSH access):
~2 minutes (rsync 863 MB) + ~30 s (rescan + aggregator + ingest).

## Next Steps

- v3: SeeGround head-to-head with Qwen2-VL-72B backbone (apples-to-apples).
- v4: detector ablation (V-DETR / BIP3D / GroupFree3D pool variants —
  push past Mask3D's 84.77 % @0.50 ceiling).
- v5: SR3D extension.
- Patch `compute_leaderboard_metrics` to accept a `sample_ids` filter
  (closes the macOS OOM workaround documented in v1).
