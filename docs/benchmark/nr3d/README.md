# NR3D VG Evaluation Results

This directory tracks NR3D visual-grounding evaluations for the Stage-2
task-pack pipeline.

**Benchmark:** NR3D ScanNet visual grounding (test split now has Phase 8 GT-CG bboxes for local scoring)
**Metric (v3, leaderboard track):** classification accuracy = `selected_object_id == target_id`, with Easy/Hard + View-Dep/View-Indep slicing — matches the public ReferIt3D leaderboard
**Metric (v1/v2, GT-pool IoU track):** 9-DoF oriented 3D IoU, reported as Acc@0.25 / Acc@0.50 / mean IoU
**Judge:** none; scoring is fully programmatic

## Version Timeline

| Version | Date | Headline | Eval Scale | Key Change |
|---------|------|----------|------------|------------|
| [v1_phase8_smoke20_mac](v1_phase8_smoke20_mac_20260430.md) | 2026-04-30 | Acc@0.25=0.7000, Acc@0.50=0.6500, mean IoU=0.6701 | 20Q smoke | First green pack_v1 numbers on Mac (PCA-aligned 9-DoF, endpoint reachable). 13/20 IoU=1.0 (GT-pool inflation). Linux side of same fold had hit SSL EOF on every retry — failure mode summarized in the `What Changed vs Prior Smoke` section of the Mac doc. |
| [v2_phase8_full](v2_phase8_full_20260501.md) | 2026-05-01 | Acc@0.25=**0.7767**, Acc@0.50=**0.7762**, mean IoU=**0.7800** | 8584Q full test | First full NR3D test sweep — 130/130 scenes, gpt-5.4 backend, 32 workers. 5150/8584 IoU=1.0 (60%, GT-pool). 287 failed (3.3%, mostly upstream image-500). Bug fixes: bg-skip in pack prep + failed-sentinel in runner. |
| [v3_referit3d_track](v3_referit3d_track_20260501.md) | 2026-05-01 | Overall=**0.8079**, Easy=**0.8606**, Hard=**0.7587**, V-Dep=**0.7246**, V-Indep=**0.8534** | 8584Q (n_filtered=7805) | **First leaderboard-comparable NR3D run** — canonical filter chain (`mentions_target_class_only=True`), classification accuracy as headline, 5-column SOTA-aligned table. Post-aggregated from v2 outputs (no agent re-run). Pool / fold equivalence empirically verified (see `pool_equivalence_log_20260501.md`). |
| [v4_agent_guards_fair_views](v4_agent_guards_fair_views_20260512.md) | 2026-05-12 | Overall=**0.7100**, Easy=**0.8293**, Hard=**0.6271**, V-Dep=**0.6765**, V-Indep=**0.7273** | 100Q partial pilot | Query-driven fair keyframes plus TADG / no-match / evidence-frame guards. Same-fold baseline was Overall=0.8000, so v4 is -9.00 pp overall; no LLM/service failures while ramping workers 30 -> 60 -> 100. **Partial diagnostic, not a public leaderboard replacement.** |

## Current Interpretation

**Latest full-test leaderboard row (v3_referit3d_track, 2026-05-01)**:
post-aggregation of v2
predictions under the canonical ReferIt3D leaderboard protocol — the
**first apples-to-apples NR3D number** for our pipeline. Headline:
**classification_acc = 80.79 %** (Easy=86.06, Hard=75.87, V-Dep=72.46,
V-Indep=85.34) on n_filtered=7805 (after canonical
`mentions_target_class_only=True` filter), with full Easy/Hard +
View-Dep/View-Indep breakdown. Pool / fold equivalence to canonical is
verified empirically in `pool_equivalence_log_20260501.md`. See
`v3_referit3d_track_20260501.md` for the full SOTA comparison table
against ReferIt3DNet, BUTD-DETR, MVT, 3D-VisTA, MiKASA, and UniVLG GT-track.

**Latest diagnostic pilot (v4_agent_guards_fair_views, 2026-05-12)**:
100-sample fixed fold with fair query-driven keyframes and the ScanRefer guard
stack. Overall classification accuracy is **71.00 %** on the 100Q fold, while
the same-fold v1/v3 baseline is **80.00 %**. This is intentionally not a public
leaderboard replacement: it measures the cost of removing GT-target-visible
keyframe selection and running the guarded agent on NR3D. All 100 samples
completed with zero LLM/service failures while ramping workers from 30 to 100.

The v2 numbers (Acc@0.25/0.50 = 77.67/77.62) remain valid as the IoU-on-GT-pool
proxy of classification accuracy, but they are **superseded by v3** as the
leaderboard-comparable headline going forward. Empirically,
`classification_acc_full = 0.7762` matches `v2 Acc@0.50 = 0.7762` to four
decimals — confirming that under GT-pool, IoU ≥ 0.50 ⟺ correct ID, and the
v2 number was already a classification accuracy in disguise.

**Earlier (v2_phase8_full, 2026-05-01)**: full NR3D test split (8584
utterances, 130/130 scenes) scored end-to-end on Stage 1 + Stage 2 with
`gpt-5.4-2026-03-05` and the Phase 8 GT-CG candidate pool. Acc@0.25 =
77.67%, Acc@0.50 = 77.62%, mean IoU = 0.7800 (8297 completed + 287 failed,
3.34% failure rate, mostly upstream image-500). The IoU distribution is
bi-modal — 60% at 1.0, 21% at 0.0 — which is the GT-pool signature. v3 makes
this explicit by reporting classification accuracy directly.

An earlier Linux-side attempt on the 20-sample fold (commit `3c491a9`)
reached the first Stage 2 chat-completion call but the internal ModelHub
endpoint returned `[SSL: UNEXPECTED_EOF_WHILE_READING]` on every retry, so
no metric was produced and no separate doc was kept; the failure context is
embedded in `v1_phase8_smoke20_mac_20260430.md` instead.

The Mac re-run (v1_phase8_smoke20_mac, 2026-04-30) on tip `9115fd7` (which
adds the loader's PCA-aligned 9-DoF OBB recovery for Phase 8 corners)
produced the first green pack_v1 numbers: **Acc@0.25 = 70.0 %**, **Acc@0.50 =
65.0 %**, **mean IoU = 0.6701** on 19 completed + 1 failed samples. Of the
14 IoU ≥ 0.25 hits, 13 are exactly IoU = 1.0 because the GT-pool setup
includes every aggregation instance (including the GT itself) as a
candidate; a detector-pool variant is tracked separately.

The Phase-2 plumbing path loads the canonical NR3D CSV, derives train/test
membership from upstream scene lists, filters bad contexts and clothing rows by
default, and can still score train split predictions against EmbodiedScan PKL
boxes. The Phase-8 path adds local test split GT boxes from
`data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz`.

## Reproduction Pattern

Download the raw NR3D annotation files:

```bash
bash scripts/download_nr3d.sh data/nr3d
```

Load test samples with Phase 8 GT-CG boxes:

```bash
PYTHONPATH=src python -c "
from benchmarks.nr3d_loader import Nr3dDataset
ds = Nr3dDataset.from_path(
    data_root='data/nr3d',
    split='test',
    bbox_source='phase8_gt_cg',
    phase8_data_root='data/nr3d/scannet',
    max_samples=200,
)
print(f'loaded={len(ds)} stats={ds.stats}')
"
```

Ingest a future side-by-side output:

```bash
PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_<run> \
    --run-id <run> \
    --branch feat/nr3d-vg-benchmark \
    --commit <short_sha> \
    --backend pack_v1 \
    --judge-model none \
    --notes "NR3D pack_v1 run"
```

Prepare a Phase 8 pack:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --split test
```

Run the pack-v1 Stage 2 runner:

```bash
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --output-dir tmp/nr3d_eval_v1_smoke20 \
    --workers 1
```

## SQLite

Canonical DB: `docs/benchmark/nr3d/runs.sqlite`

```sql
SELECT run_id, n,
       printf('%.4f', classification_acc_filtered) AS acc_filtered,
       n_filtered,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50
FROM runs;
```

## Caveats

- EmbodiedScan PKL boxes remain available through
  `bbox_source="embodiedscan_pkl"` for the train split.
- NR3D test split local scoring now uses Phase 8 GT-CG boxes through
  `bbox_source="phase8_gt_cg"`.
- Pool / fold equivalence to canonical referit3d is empirically verified
  in `pool_equivalence_log_20260501.md` (8584 utterances, 130/130 scenes,
  100% target_id ↔ ScanNet objectId class match).
- View-dep / view-indep breakdown — implemented in v3 via the canonical
  10-token literal set from `referit3d/analysis/utterances.py:103-105`.
- Easy / Hard breakdown — implemented in v3 via `n_objects ≤ 2`, matching
  `referit3d/analysis/deepnet_predictions.py:34-36`.
