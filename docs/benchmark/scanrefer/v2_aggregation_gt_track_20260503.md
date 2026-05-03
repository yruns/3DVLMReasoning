# v2 Aggregation-GT-Track — 2026-05-03

ScanRefer detection-mode evaluation on the canonical full val set (9508
utterances on 141 scenes), Mask3D ScanNet200 pool from ZSVG3D's
distribution, gpt-5.4-2026-03-05 agent. Beats every Camp-A baseline on
all 6 columns (Z3D +10 pp on Acc@0.50), **but the keyframe selector
uses a GT view oracle — see § Caveats below; numbers are not directly
comparable to zero-shot Camp-A methods until v3 query-driven**. v2
fixed a GT-bbox-source bug carried by v1 (see "Audit trail" section
below); same agent decisions, only the GT side of every IoU was
recomputed against ScanNet aggregation-derived bboxes.

## Headline

| Metric | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | **83.11** | **65.00** | **69.92** |
| Acc@0.50 | **76.49** | **57.68** | **62.79** |

mean IoU (overall) = **0.6022** · n_total = 9508 (Unique = 2582,
Multiple = 6926) · failed sentinels = 205 / 9508 (2.16 %, counted as
iou = 0).

> ⚠ **GT view oracle in keyframe selection.** The 5 initial RGB
> keyframes per query are selected by Phase 8 visibility of the GT
> `target_id`, not by a query-driven Stage 1 retrieval. The target
> object is therefore guaranteed to appear in the initial visual
> evidence. This is GT-assisted evidence selection (a view oracle),
> not full label leakage — agent still picks the proposal id from the
> Mask3D pool. The numbers above are best read as a **controlled
> upper bound on agent picking ability under GT-visible RGB**. The
> planned v3 query-driven track removes the oracle. See § Caveats.

## Run Identity

- Branch: `feat/scanrefer-vg-benchmark` (tip `40c57b8`)
- Run ID: `v2_aggregation_gt_track_20260503`
- Source agent run: `v1_mask3d_track_20260502` (~8 h on 2026-05-02 →
  2026-05-03; gpt-5.4-2026-03-05; workers = 32; sample-retries = 1)
- Re-aggregation script: `src/evaluation/scripts/rescan_with_aggregation_gt.py`
- Spec: `docs/superpowers/specs/2026-05-03-scanrefer-v2-aggregation-gt.md`

## Methodology

ScanRefer is a detection-mode benchmark; methods produce 3D bboxes
rather than picking from a GT pool. We adopt the standard Camp-A
protocol used by ZSVG3D / SeeGround / CSVG / Z3D:

- **Pool**: Mask3D ScanNet200 predictions (ZSVG3D distribution),
  repackaged into Phase-8-shaped ConceptGraph pkls per scene by
  `build_scanrefer_mask3d_cg.py`.
- **GT bbox**: AABB of ScanNet mesh vertices belonging to each `objectId`,
  derived from `_vh_clean_2.ply` + `_vh_clean_2.0.010000.segs.json` +
  `<scene>.aggregation.json` + axis-alignment matrix from `<scene>.txt`.
  This is the reference Mask3D itself was trained against and that
  every Camp-A paper evaluates against.
- **Agent input**: 5 RGB keyframes per query, selected by Phase 8
  visibility of the target GT instance, with Mask3D candidate boxes
  projected as 2D overlays. Reuses the NR3D v3 Stage 1 + Stage 2
  pipeline unchanged.
- **Metric**: `compute_oriented_iou_3d` with Euler = 0;
  Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }.
- **Filtering**: drop wall / floor / ceiling Mask3D instances at the
  converter stage (matches ZSVG3D `keep_background=False`).
- **Mask3D scores hidden** from the agent — uniform `score = 1.0` in
  proposals; `ins_scores` retained only as diagnostics.

## SOTA Comparison (Mask3D-pool zero-shot)

| Method | Setup | Unique@0.25 | Unique@0.50 | Multiple@0.25 | Multiple@0.50 | Overall@0.25 | Overall@0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| ZSVG3D / GPT-4-Turbo | Mask3D pool, zero-shot | 63.8 | 58.4 | 27.7 | 24.6 | 36.4 | 32.7 |
| CSVG (Mask3D) | Mask3D pool, zero-shot | 68.8 | 61.2 | 38.4 | 27.3 | 49.6 | 39.8 |
| SeeGround / Qwen2-VL-72B | Mask3D pool, zero-shot | 75.7 | 68.9 | 34.0 | 30.0 | 44.1 | 39.4 |
| VLM-Grounder / GPT-4V (250 sub) | 2D-online proj, zero-shot | 66.0 | 29.8 | 48.3 | 33.5 | 51.6 | 32.8 |
| Z3D (Mask3D row) | Mask3D pool, zero-shot | 82.3 | 74.8 | 51.5 | 45.7 | 58.9 | 52.7 |
| **Ours v2 (gpt-5.4)** | Mask3D pool, zero-shot | **83.11** | **76.49** | **65.00** | **57.68** | **69.92** | **62.79** |

Δ vs Z3D: U@0.25 +0.81 / U@0.50 +1.69 / **M@0.25 +13.5 / M@0.50 +11.98** /
**O@0.25 +11.02 / O@0.50 +10.09**. The Multiple split shows the largest
gain — language-disambiguation is where the RGB+VLM agent advantage
shows most. Supervised reference numbers in `leaderboard.md`.

## Sanity check (oracle ceiling + picking quality)

| | Oracle Acc | Agent Acc | Picking quality |
|---|---:|---:|---:|
| @0.25 | 93.31 % | 69.92 % | 74.93 % |
| @0.50 | 84.77 % | 62.79 % | 74.07 % |
| mean IoU | 0.8008 | 0.6022 | — |

*Oracle Acc* = best-Mask3D-vs-aggregation-GT IoU per sample.
*Picking quality* = `agent_acc / oracle_acc` = how often the agent picks
the right Mask3D candidate when one with the required IoU exists.
74-75 % matches v1's picking quality (73.97 % @0.25 / 79.00 % @0.50)
and NR3D v3 classification_acc (80.79 %) — same agent skill, the
metric improvement is fully attributable to the GT-bbox source change.

## Caveats

- **GT view oracle in keyframe selection (paper-comparability)** —
  pack-prep picks the 5 initial RGB keyframes via Phase 8 visibility
  of the GT `target_id` (`select_keyframes_from_phase8_target` in
  `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`,
  decision #2 in the v1 spec). This bypasses query-driven Stage 1
  retrieval and guarantees the target object appears in the initial
  visual evidence. The agent still picks `proposal_id` from the
  Mask3D pool — `proposal_id` is not leaked. But the ranking of
  RGB views is GT-driven, which dramatically reduces retrieval
  difficulty (especially in the Multiple split, where same-category
  distractors otherwise compete for view-budget).
  - Z3D / SeeGround / ZSVG3D / CSVG do **not** use a GT view oracle:
    Z3D and ZSVG3D do proposal-driven multi-view rendering; SeeGround
    uses query-aligned synthetic rendering; CSVG is pure 3D+text. So
    v2's per-column wins over those baselines are **not apples-to-
    apples** as a zero-shot Camp-A comparison.
  - OpenEQA's pack-prep correctly uses
    `query_scene.keyframe_selector.select_keyframes_v2(query, ...)`
    — hypothesis-driven, no GT lookup. The Stage 1 → Stage 2
    architecture stated in `CLAUDE.md` (`Hypothesis as soft prior`,
    `Evidence-seeking`) is what v3 will restore.
  - Treat v2 as a **controlled upper bound** on agent picking ability
    under GT-visible RGB, useful as an ablation against v3 (planned)
    to quantify how much of the Multiple-class gain is the agent vs
    the view oracle.
- **Mask3D pool quality bound**: oracle ceiling at Acc@0.50 = 84.77 %
  represents Mask3D's segmentation accuracy, not a GT issue. Higher
  numbers would require a better detector (V-DETR / BIP3D / ...).
- **Wall/floor/ceiling Mask3D candidates filtered** — Camp-A standard.
- **Per-LLM-call durability gap** carried over from NR3D v3 —
  `tool_calls` / `llm_calls` SQLite tables are empty for v2 until a
  callback-based instrumentation pass lands.

---

## Audit trail — why v2 exists

v1 reported Acc@0.50 = 16.22 % using the wrong GT bbox source. This
section is the post-mortem that motivated v2 (folded into the version
doc; the standalone analysis file has been retired).

### v1 used the wrong GT bbox source

v1 used **Phase 8 GT-CG bbox** as the IoU reference (Vil3dRef-equivalent
metadata, but the bbox itself was the AABB of a ConceptGraph
reconstructed point cloud per object, not the AABB of mesh-aggregation
vertices). This was correct for NR3D v3's classification metric — which
only checks `object_id` match — but wrong for ScanRefer's
detection-mode IoU metric.

Phase 8 GT-CG bbox is **systematically ~2× larger by volume** than the
ScanNet aggregation-derived bbox (which is also what Mask3D was trained
to predict). Measured on the 5444 samples with Mask3D oracle IoU ≥ 0.30
in v1:

| | mean | median | p25 | p75 |
|---|---:|---:|---:|---:|
| volume_ratio (Mask3D / Phase8_GT) | 0.594 | 0.502 | 0.389 | 0.677 |

Per-axis size ratio (median): x = 0.796, y = 0.821, z = 0.815. 49.6 %
of samples have Mask3D bbox volume below 50 % of Phase 8 volume; 85.3 %
below 80 %. Center offset is small (median ~5 cm; p95 ~39 cm) — the
issue is uniform inflation across all three axes from CG point-cloud
over-extension (adjacent surfaces, depth noise, merged fragments).

### Consequence for v1 numbers

A perfectly-segmented Mask3D instance contained in a 2× larger Phase 8
box gives IoU = 0.5 by construction, plus center offset and shape
mismatch drops the IoU into the 0.3-0.4 band — exactly the modal band
of v1's oracle distribution:

```
oracle distribution (Mask3D best-match vs Phase 8 GT, n=9508):
[0.00, 0.10)  n=  252   2.65 %
[0.10, 0.25)  n= 2617  27.53 %
[0.25, 0.30)  n= 1195  12.57 %
[0.30, 0.40)  n= 1996  20.99 %  ← mode here
[0.40, 0.50)  n= 1496  15.73 %
[0.50, 1.00)  n= 1952  20.53 %
```

v1 oracle ceiling: **Acc@0.25 = 69.83 %** (any picker), **Acc@0.50 =
20.53 %**. Z3D's published 52.7 % was mathematically unreachable on this
fold — not because Z3D's number is wrong, but because v1 was measuring
against the wrong reference.

### v1 numbers (kept for audit)

| Metric | v1 (Phase 8 GT-CG) | v2 (aggregation GT) | Δ |
|---|---:|---:|---:|
| Acc@0.25 Overall | 51.65 | 69.92 | +18.27 pp |
| Acc@0.50 Overall | 16.22 | 62.79 | **+46.57 pp** |
| Acc@0.25 Unique | 62.97 | 83.11 | +20.14 pp |
| Acc@0.50 Unique | 25.41 | 76.49 | +51.08 pp |
| Acc@0.25 Multiple | 47.43 | 65.00 | +17.57 pp |
| Acc@0.50 Multiple | 12.79 | 57.68 | +44.89 pp |
| mean_iou | 0.2746 | 0.6022 | +119 % |

v1 picking quality was **73.97 % @0.25 / 79.00 % @0.50**; v2 picking
quality is **74.93 % / 74.07 %**. Same agent, same skill — the entire
delta is GT-side measurement.

### Why NR3D v3 (Acc = 80.79) didn't surface this bug

NR3D v3's headline metric is `classification_acc` — pick the right
`object_id` from a GT pool. **No bbox geometry enters the metric.**
Phase 8 GT-CG correctly indexes objects by `object_id` (1:1 with
ScanNet aggregation `objectId`), so the lookup is right; bbox dimensions
only matter for IoU-based eval. NR3D shielded us from the bug; ScanRefer
exposed it.

### v1 retained as audit trail

The v1 SQLite row (`v1_mask3d_track_20260502`) is preserved in
`runs.sqlite` alongside v2. v1 is **not paper-comparable** to ZSVG3D /
SeeGround / CSVG / Z3D (different GT scale); v2 is.

---

## Cross-version

| Version | Date | n_total | Acc@0.25 | Acc@0.50 | Notes |
|---|---|---:|---:|---:|---|
| v1_mask3d_track | 2026-05-02 | 9508 | 51.65 | 16.22 | Phase 8 GT-CG bbox — not paper-comparable. Audit trail only. |
| **v2_aggregation_gt_track** | 2026-05-03 | 9508 | **69.92** | **62.79** | Same v1 agent, aggregation-GT re-aggregation. Camp-A SOTA on all 6 columns. |

## SQLite

```sql
SELECT run_id, n_total, n_unique, n_multiple,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', acc25_unique)   AS u25,
       printf('%.4f', acc50_unique)   AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50
FROM runs;
```

## Raw artifacts (gitignored)

- ScanNet meshes: `data/nr3d/scannet_aux_meshes/<scene>/<scene>_vh_clean_2.ply` (141 / 141, 863 MB).
- ScanNet aux: `data/nr3d/scannet_aux/<scene>/{<scene>.aggregation.json, <scene>_vh_clean_2.0.010000.segs.json, <scene>.txt}`.
- Mask3D-CG pkls: `data/scanrefer/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz`.
- v1 source: `tmp/scanrefer_eval_v1_full/{side_by_side.json, per_sample/...}`.
- v2 outputs: `tmp/scanrefer_eval_v2_aggregation_gt/{side_by_side.json, leaderboard_metrics.json}`.
- SQLite row: `docs/benchmark/scanrefer/runs.sqlite::runs.run_id='v2_aggregation_gt_track_20260503'`.

## Reproduction

End-to-end (assuming v1 outputs on disk + Linux SSH access):

```bash
source .venv/bin/activate

# 1. Pull ScanNet meshes (one-time, 863 MB, ~2 min)
python -c "
from pathlib import Path
val = Path('data/scanrefer/raw/ScanRefer_filtered_val.txt').read_text().split()
with open('/tmp/mesh_include.txt', 'w') as f:
    for s in val:
        f.write(f'{s}/{s}_vh_clean_2.ply\n')
"
rsync -av --mkpath --files-from=/tmp/mesh_include.txt \
    bupt:/home/ysh/Datasets/ScanNet/scans/ \
    data/nr3d/scannet_aux_meshes/

# 2. Re-aggregate v1 outputs against aggregation GT (~30 s)
PYTHONPATH=src python src/evaluation/scripts/rescan_with_aggregation_gt.py \
    --side-by-side tmp/scanrefer_eval_v1_full/side_by_side.json \
    --output-dir tmp/scanrefer_eval_v2_aggregation_gt \
    --scannet-aux-root data/nr3d/scannet_aux \
    --mesh-root data/nr3d/scannet_aux_meshes

# 3. Aggregate Unique/Multiple slicing (memory-frugal scene-by-scene driver
#    — the canonical CLI's compute_leaderboard_metrics OOMs on macOS at
#    full-val Phase 8 load; v3 should patch with a sample_ids filter).
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

To reproduce v1 from scratch (8 h API run + ~30 min pack-prep), see the
spec at `docs/superpowers/specs/2026-05-02-scanrefer-design.md`.

To audit oracle / picking-quality numbers, see the analysis script at
the end of this file.

## Next steps

- v3: SeeGround head-to-head with Qwen2-VL-72B backbone (apples-to-apples).
- v4: detector ablation (V-DETR / BIP3D / GroupFree3D pool variants —
  push past Mask3D's 84.77 % @0.50 ceiling).
- v5: SR3D extension.
- Patch `compute_leaderboard_metrics` to accept a `sample_ids` filter
  (closes the macOS OOM workaround documented above).

## Appendix — oracle / picking-quality reproduction script

```python
"""Recompute oracle ceiling + agent picking-quality from v1 or v2 outputs."""
import gzip, json, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from benchmarks.embodiedscan_eval import compute_oriented_iou_3d
from benchmarks.scanrefer_aggregation_gt import load_aggregation_gt_bboxes

def aabb_to_9dof(corners):
    a = np.asarray(corners, dtype=np.float64); mn, mx = a.min(0), a.max(0)
    return [*((mn+mx)/2.0).tolist(), *(mx-mn).tolist(), 0.0, 0.0, 0.0]

# Switch to v1's Phase 8 GT here to reproduce the v1 oracle numbers
USE_AGGREGATION_GT = True
SIDE = ('tmp/scanrefer_eval_v2_aggregation_gt/side_by_side.json'
        if USE_AGGREGATION_GT else
        'tmp/scanrefer_eval_v1_full/side_by_side.json')

side = json.load(open(SIDE))
per_sample = side['pack_v1']['per_sample']
by_scene = defaultdict(list)
for rec in per_sample:
    by_scene[rec['sample_id'].split('::')[0].split('/')[-1]].append(rec)

results = []
for scene, recs in sorted(by_scene.items()):
    if USE_AGGREGATION_GT:
        gt_bboxes = load_aggregation_gt_bboxes(scene)
    else:
        with gzip.open(f'data/nr3d/scannet/{scene}/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz', 'rb') as f:
            gt_objs = pickle.load(f).get('objects') or []
        gt_bboxes = {i: aabb_to_9dof(o['bbox_np']) for i, o in enumerate(gt_objs)}
    m3 = Path(f'data/scanrefer/scannet/{scene}/conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz')
    with gzip.open(m3, 'rb') as f:
        m3_objs = pickle.load(f).get('objects') or []
    m3_bb = [aabb_to_9dof(o['bbox_np']) for o in m3_objs]
    for rec in recs:
        tid = int(rec['sample_id'].split('::')[1])
        gt = gt_bboxes.get(tid)
        if gt is None or not m3_bb: continue
        results.append({
            'sample_id': rec['sample_id'],
            'agent_iou': float(rec.get('iou') or 0.0),
            'oracle_max_iou': float(max(compute_oriented_iou_3d(gt, b) for b in m3_bb)),
        })

a = np.array([r['agent_iou'] for r in results])
o = np.array([r['oracle_max_iou'] for r in results])
print(f"agent  acc25={(a>=0.25).mean()*100:.2f}%  acc50={(a>=0.50).mean()*100:.2f}%")
print(f"oracle acc25={(o>=0.25).mean()*100:.2f}%  acc50={(o>=0.50).mean()*100:.2f}%")
print(f"picking quality: @0.25 {(a>=0.25).sum()/(o>=0.25).sum()*100:.2f}%  @0.50 {(a>=0.50).sum()/(o>=0.50).sum()*100:.2f}%")
```
