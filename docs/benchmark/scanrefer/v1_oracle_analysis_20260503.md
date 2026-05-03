# v1 Oracle-Picker Analysis — Why Acc@0.50 = 16.22 is misleading

**Date:** 2026-05-03
**Source run:** `v1_mask3d_track_20260502` (9508 utts, 141 scenes, gpt-5.4, Mask3D-pool)
**Raw analysis JSON:** `tmp/scanrefer_eval_v1_full/oracle_analysis.json` (one record per sample with `agent_iou`, `oracle_max_iou`, `n_m3_candidates`)
**Reproduction script:** see appendix at the bottom of this doc.

## TL;DR

The Acc@0.50 = 16.22 number in v1 is **NOT directly comparable** to published Camp-A
zero-shot baselines (ZSVG3D 32.7 / SeeGround 39.4 / CSVG 39.8 / Z3D 52.7).
The cause is GT bbox derivation, not agent quality:

- **Phase 8 GT-CG bbox is systematically ~2× larger than Mask3D bbox**
  (median Mask3D-volume / Phase8-volume = **0.502**; per-axis median ratio
  ≈ 0.80; 49.6 % of samples have Mask3D bbox < 50 % the Phase 8 volume).
- **Even an oracle picker would only reach Acc@0.50 = 20.53 %** on this fold
  with this GT — i.e. Z3D's published 52.7 % is mathematically unreachable
  here regardless of agent quality.
- The agent's **picking quality is high** when normalized against the
  ceiling: 73.97 % at @0.25, **79.00 % at @0.50** — comparable to or
  better than NR3D v3 classification_acc = 80.79 %, which is what we'd
  expect from the same agent on the same backbone.

The fix is a v2 track that uses ScanNet aggregation-based GT bbox
(from `data/nr3d/scannet_aux/<scene>/<scene>.aggregation.json` +
`<scene>_vh_clean_2.0.010000.segs.json` + ScanNet mesh `.ply`),
matching what every other Camp-A method uses. Spec at
`docs/superpowers/specs/2026-05-03-scanrefer-v2-aggregation-gt.md`.

## Why the gap matters

The headline metric on a detection-mode benchmark is IoU between the
agent's predicted bbox and the ground-truth bbox. In v1 the predicted
bbox is the agent's chosen Mask3D candidate's bbox; the GT bbox is
derived by `_phase8_corners_to_9dof` from the Phase 8 GT-CG pkl. Both
sides go into `compute_oriented_iou_3d` with Euler = 0.

If the GT bbox is geometrically different from what other methods use,
the IoU is computed against a different reference, and the resulting
Acc@0.25 / Acc@0.50 numbers are not on the same scale. The bbox
**source** of comparison must match for cross-method comparison.

## Evidence 1 — Oracle ceiling

For each of the 9508 samples we computed `oracle_max_iou =
max(IoU(GT_phase8, Mask3D_candidate_i) for i in scene)`. This is the
upper bound any agent could achieve given this fold + this GT
derivation + this Mask3D pool.

| Threshold | Oracle Acc | Agent Acc | Headroom |
|---|---:|---:|---:|
| @0.10 | 97.35 % | — | — |
| @0.25 | **69.83 %** | 51.65 % | 18.18 pp |
| @0.50 | **20.53 %** | 16.22 % | 4.31 pp |
| @0.75 | 2.65 % | — | — |

**Critical observation:** the oracle ceiling at Acc@0.50 is 20.53 %.
Z3D's published 52.7 % is **2.6× our oracle ceiling**, which is
arithmetically impossible given the same Mask3D pool. The only
explanation is a different GT bbox source — Z3D and the other Camp-A
papers compute IoU against ScanRefer's official mesh-aggregation GT
bbox, not Phase 8 GT-CG bbox.

### Oracle distribution

```
Phase 8 GT-CG vs best-matching Mask3D candidate IoU (n = 9508):
  [0.00, 0.10)  n=  252   2.65 %  #
  [0.10, 0.20)  n= 1369  14.40 %  #######
  [0.20, 0.25)  n= 1248  13.13 %  ######
  [0.25, 0.30)  n= 1195  12.57 %  ######
  [0.30, 0.40)  n= 1996  20.99 %  ##########
  [0.40, 0.50)  n= 1496  15.73 %  #######
  [0.50, 0.60)  n=  904   9.51 %  ####
  [0.60, 0.70)  n=  610   6.42 %  ###
  [0.70, 0.80)  n=  340   3.58 %  #
  [0.80, 0.90)  n=   79   0.83 %
  [0.90, 1.00)  n=   19   0.20 %
```

The mode of the distribution sits in [0.30, 0.40) — exactly the band
that fails Acc@0.50 but passes Acc@0.25. This is the geometric
fingerprint of a systematic GT-bbox-too-large bias: a perfectly-segmented
Mask3D instance contained inside a 2× larger Phase 8 box gives
IoU = 0.5 by construction, plus center offset and shape mismatch drops
it into the 0.3-0.4 band.

## Evidence 2 — Bbox geometry

For the 5444 samples with `oracle_max_iou >= 0.30` (high confidence
that Mask3D segmented the right instance), we measured the geometry of
the best-matching Mask3D candidate vs the Phase 8 GT bbox:

```
volume_ratio (Mask3D / Phase8_GT):
    mean   = 0.594
    median = 0.502
    p25    = 0.389
    p75    = 0.677

per-axis size ratio (median):  x = 0.796   y = 0.821   z = 0.815

center offset (m):
    mean   = 0.104  (~10 cm)
    median = 0.053  (~5 cm)
    p75    = 0.121
    p95    = 0.386
```

Volume-ratio histogram:

```
[0.00, 0.50)  n=2700  49.60 %   <-- Mask3D < half of Phase 8 volume
[0.50, 0.80)  n=1942  35.67 %
[0.80, 0.90)  n= 187   3.43 %
[0.90, 1.00)  n= 153   2.81 %
[1.00, 1.10)  n= 124   2.28 %
[1.10, 1.25)  n=  92   1.69 %
[1.25, 1.50)  n=  74   1.36 %
[1.50, 2.00)  n= 124   2.28 %
[2.00, 3.00)  n=  48   0.88 %
```

**85.3 % of samples have Mask3D volume < 80 % of Phase 8 GT volume.**
Per-axis sizes are ~80 % of Phase 8's. This means Phase 8 GT-CG bbox
expands every dimension by ~25 % vs Mask3D's segmentation, growing the
volume by `(1.25)³ ≈ 1.95×`.

The cause is in Phase 8's reconstruction: ConceptGraph builds per-object
point clouds from SAM-Florence detections back-projected through depth,
which over-extends to adjacent surfaces (table edge → table + chair),
incorporates depth noise, and merges spatially-close fragments. Mask3D
ScanNet200 was trained against the official `.aggregation.json` mesh
segments and thus produces a tighter, mesh-aligned bbox.

## Evidence 3 — Picking quality (agent vs oracle)

The right way to measure agent skill on this fold is "given that an
oracle ≥0.25 (or ≥0.50) candidate exists, did the agent pick it?":

| Threshold | Oracle has good candidate | Agent picked good | Picking quality |
|---|---:|---:|---:|
| @0.25 | 6639 / 9508 (69.83 %) | 4911 / 6639 | **73.97 %** |
| @0.50 | 1952 / 9508 (20.53 %) | 1542 / 1952 | **79.00 %** |

For comparison, the same Stage 1 + Stage 2 agent on NR3D v3 reaches
classification_acc = 80.79 %. That's a metric of "given a GT pool, did
you pick the right object_id?" which is conceptually identical to "did
you pick the right candidate from the Mask3D pool?" — and the numbers
agree (~74-80 %). The agent is not the bottleneck; the GT bbox is.

## Stratified by Unique vs Multiple

| Group | n | Oracle Acc@0.25 | Oracle Acc@0.50 | Agent Acc@0.25 | Agent Acc@0.50 |
|---|---:|---:|---:|---:|---:|
| Unique | 2582 | 72.08 % | **28.51 %** | 62.97 % | 25.41 % |
| Multiple | 6926 | 68.99 % | **17.56 %** | 47.43 % | 12.79 % |

Both subgroups have the same GT-derivation problem; Multiple is harder
because of the language-disambiguation requirement, which is reflected
in the picking-quality drop (Unique: 62.97 / 72.08 = 87.4 %; Multiple:
47.43 / 68.99 = 68.7 %). This split is **agent-driven** and is meaningful.

## Where each Acc@0.50 failure comes from

```
agent failed @0.50 = 7966 / 9508 (83.78 %)
  oracle had no >=0.50 candidate (Mask3D-vs-Phase8 ceiling): 7556 (94.85 % of failures)
  oracle had >=0.50 but agent picked worse (agent mistake):   410 ( 5.15 % of failures)
```

**95 % of @0.50 failures are GT-derivation losses, not agent errors.**

By contrast, at the @0.25 threshold, agent contribution is non-trivial
(37.6 % of failures are agent-pickable losses) — that's where
prompt / picker improvements would actually move the number.

## Why NR3D v3 (Acc=80.79) doesn't have this problem

NR3D v3's headline metric is `classification_acc`: did the agent pick
the right `object_id` from a GT pool? **No bbox geometry enters this
metric.** Phase 8 GT-CG correctly indexes objects by `object_id` (1:1 to
the source ScanNet `.aggregation.json` `objectId`), so the lookup is
right; the bbox dimensions only matter for IoU-based eval.

For NR3D the same agent achieves 80.79 % classification_acc; for
ScanRefer the same agent achieves 79.00 % oracle-normalized picking
quality at @0.50 and 73.97 % at @0.25. These are all consistent
"pick the right candidate" rates. NR3D was not magically easier — it
just shielded us from the GT-bbox-derivation flaw.

## Implications

1. **v1's Acc@0.25 = 51.65 % is approximately comparable to published
   Camp-A baselines.** The threshold is loose enough that the GT
   over-expansion mostly doesn't push valid picks below 0.25. We sit in
   the SOTA mid-pack (between SeeGround 44.1 and Z3D 58.9; close to
   VLM-Grounder 51.6 and CSVG 49.6).
2. **v1's Acc@0.50 = 16.22 % is NOT comparable** to published Camp-A
   baselines. The threshold is tight enough that the systematic ~2×
   GT volume difference dominates. The published baselines compute IoU
   against a tighter GT (mesh-aggregation), which we should adopt.
3. **The same v1 agent run, re-aggregated with aggregation-based GT,
   would likely produce Acc@0.50 in the 35-45 range** (estimate: agent
   picking quality 79 % × oracle ceiling under aggregation GT, expected
   around 50-60 % for Mask3D-pool against its training-source GT).
   Acc@0.25 should rise to 60-65 %.
4. **No v1 agent re-run is needed for the v2 fix.** Predictions are
   the agent's chosen Mask3D candidate's bbox; only GT changes. v2 = a
   new aggregation-based GT loader + a re-aggregation pass over the
   existing v1 `side_by_side.json`. ~30 minutes of compute, not 8 hours.

## Reproduction

The analysis above was produced by a script that:

1. Loaded `tmp/scanrefer_eval_v1_full/side_by_side.json` (per-sample agent IoU + status).
2. For each sample, opened the scene's Phase 8 GT-CG pkl and Mask3D-CG pkl, computed `IoU(gt_phase8, m3_i)` for every Mask3D candidate, took the max → `oracle_max_iou`.
3. Stratified by Unique/Multiple using per-scene Phase 8 GT class-name counts and ScanRefer val JSON's `object_name`.
4. For samples with `oracle_max_iou >= 0.30`, took the best-matching Mask3D candidate and recorded center offset + per-axis size ratio + volume ratio vs Phase 8 GT.

Full script (Python, runs in ~1 min on macOS):

```python
import gzip, json, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from benchmarks.embodiedscan_eval import compute_oriented_iou_3d

def aabb_to_9dof(corners):
    a = np.asarray(corners, dtype=np.float64)
    mn, mx = a.min(0), a.max(0)
    return [*((mn+mx)/2.0).tolist(), *(mx-mn).tolist(), 0.0, 0.0, 0.0]

side = json.load(open('tmp/scanrefer_eval_v1_full/side_by_side.json'))
per_sample = side['pack_v1']['per_sample']
by_scene = defaultdict(list)
for r in per_sample:
    by_scene[r['sample_id'].split('::')[0].split('/')[-1]].append(r)

results = []
for scene, recs in sorted(by_scene.items()):
    p8 = Path(f'data/nr3d/scannet/{scene}/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz')
    m3 = Path(f'data/scanrefer/scannet/{scene}/conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz')
    with gzip.open(p8, 'rb') as f: gt_objs = pickle.load(f).get('objects') or []
    with gzip.open(m3, 'rb') as f: m3_objs = pickle.load(f).get('objects') or []
    m3_b = [aabb_to_9dof(o['bbox_np']) for o in m3_objs]
    for r in recs:
        tid = int(r['sample_id'].split('::')[1])
        if tid < 0 or tid >= len(gt_objs): continue
        gt9 = aabb_to_9dof(gt_objs[tid]['bbox_np'])
        ious = [compute_oriented_iou_3d(gt9, b) for b in m3_b]
        results.append({
            'sample_id': r['sample_id'],
            'agent_iou': float(r.get('iou') or 0.0),
            'oracle_max_iou': float(max(ious) if ious else 0.0),
        })

# Then aggregate: oracle Acc@thresh, picking-quality, etc.
```
