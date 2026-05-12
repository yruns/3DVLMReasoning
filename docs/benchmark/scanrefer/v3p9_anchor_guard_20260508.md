# ScanRefer v3.9 - anchor/evidence guards

**Branch**: `feat/scanrefer-v3p7-final-selection`
**Tip commit at harvest time**: `8acd3da` (working tree dirty; v3.9 code
changes not yet committed)
**Run ID** (SQLite): `v3p9_anchor_guard_v18_20260508`
**Date harvested**: 2026-05-08 GMT+8

## Headline

Random100 fold, frozen seed=20260503:
`tmp/scanrefer_artifacts/random100_sample_ids.json`.

| Variant | Acc@0.25 | Acc@0.50 | mean IoU | no-prediction / failed |
|---|---:|---:|---:|---:|
| v3.3 vertical spatial | 48.0% | 42.0% | 0.4130 | 5 |
| v3.7 final-selection guards | 46.0% | 42.0% | 0.4024 | 0 |
| v3.8 relation-ranking fresh | 49.0% | 43.0% | 0.4120 | 0 |
| **v3.9 anchor/evidence guards** | **53.0%** | **46.0%** | **0.4424** | **0** |

Per-column leaderboard slices after aggregation-GT rescoring:

| | Unique | Multiple | Overall |
|---|---:|---:|---:|
| Acc@0.25 | 78.57 | 43.06 | **53.00** |
| Acc@0.50 | 78.57 | 33.33 | **46.00** |

Interpretation: v3.9 is the best honest query-driven random100 result so far:
+4pp/+3pp over the v3.8 fresh gate, +5pp/+4pp over the v3.3 development
baseline, and `100/100` completed with no failed/no-prediction samples. It is
still below Z3D's zero-shot Mask3D-pool reference of 58.9 / 52.7.

## Hypothesis Tested

v3.8 improved relation ranking, but still showed final-selection variance:
correct proposals could be found and then bypassed by direct structured final
answers, weak direct no-match answers, or frame citations that did not actually
show the spatial anchor needed by the description.

v3.9 tightens the same trajectory without adding GT at inference:

- direct VG structured responses are no longer accepted as final answers before
  the chassis `submit_final` path validates them;
- newly queued visual evidence is injected before accepting a same-turn final
  answer;
- TADG uses a wider lookback and stricter anchor checks;
- EFG filters target candidates and adds a spatial-anchor frame-citation guard;
- VG spatial prompt guidance is stricter for pronouns, secondary clues,
  same-category anchors, and ambiguous spatial anchors;
- `chassis_tools_version` is bumped to 18 to avoid prompt-cache reuse of older
  tool contracts;
- `invalid_prompt` / `-4321` sample failures are treated as transient retryable
  infra errors;
- the ScanRefer runner bounds the per-scene `KeyframeSelector` cache to avoid
  retaining one CLIP model per scene during long random100 resumes.

## Commands

The main random100 gate used tmux and durable per-sample checkpoints:

```bash
tmux new-session -d -s v3p9_fresh_random100_v18 \
  -c /Users/bytedance/project/3DVLMReasoning \
  "CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p9_fresh_random100_v18_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 2 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard \
  2>&1 | tee -a tmp/v3p9_fresh_random100_v18_eval.log"
```

One sample hit a transient `invalid_prompt` / `-4321` policy-side failure in the
main checkpoint set. It was rerun with the same raw ScanRefer query and no GT at
inference, then copied over the failed sentinel:

```bash
CVRA_DISABLE=1 PYTHONPATH=src .venv/bin/python \
  -m evaluation.scripts.run_scanrefer_vg_side_by_side \
  --sample-ids tmp/v3p9_v18_policy_sample_ids.json \
  --data-root data/scanrefer/scannet \
  --output-dir tmp/v3p9_v18_policy_retry_eval \
  --pack-name pack_scanrefer_v3p_iterative \
  --workers 1 \
  --sample-retries 4 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard
```

The first main run exited abruptly at 56/100 with no traceback. The root cause
was an unbounded per-scene selector cache retaining separate CLIP-loaded
selectors across many scenes. After the bounded-cache fix, the same output dir
was resumed and completed 100/100.

Aggregation-GT rescore:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.rescan_with_aggregation_gt \
  --side-by-side tmp/v3p9_fresh_random100_v18_eval/side_by_side.json \
  --output-dir tmp/v3p9_fresh_random100_v18_eval_agg_gt
```

Leaderboard slicing:

```bash
PYTHONPATH=src .venv/bin/python -m evaluation.scripts.scanrefer_leaderboard_metrics \
  --side-by-side tmp/v3p9_fresh_random100_v18_eval_agg_gt/side_by_side.json \
  --sample-ids tmp/scanrefer_artifacts/random100_sample_ids.json \
  --output tmp/v3p9_fresh_random100_v18_eval_agg_gt/leaderboard_metrics.json
```

SQLite ingest:

```bash
PYTHONPATH=src .venv/bin/python scripts/ingest_scanrefer_run.py \
  --output-dir tmp/v3p9_fresh_random100_v18_eval_agg_gt \
  --run-id v3p9_anchor_guard_v18_20260508 \
  --branch feat/scanrefer-v3p7-final-selection \
  --commit 8acd3da \
  --leaderboard-metrics tmp/v3p9_fresh_random100_v18_eval_agg_gt/leaderboard_metrics.json \
  --db docs/benchmark/scanrefer/runs.sqlite \
  --keyframe-mode query_driven \
  --notes "v3.9 v18 anchor/evidence guards fresh random100: no GT at inference; direct-final deferral, TADG lookback and strict anchor checks, EFG target/filter plus spatial-anchor citation guard, transient invalid_prompt/-4321 retry classification, and bounded ScanRefer selector cache to prevent per-scene CLIP model retention; fresh result 53/46 on frozen random100."
```

## Raw Artifacts

- Agent output: `tmp/v3p9_fresh_random100_v18_eval/`
- Aggregation-GT rescore: `tmp/v3p9_fresh_random100_v18_eval_agg_gt/`
- Frozen fold: `tmp/scanrefer_artifacts/random100_sample_ids.json`
- Policy retry sample ids: `tmp/v3p9_v18_policy_sample_ids.json`
- Policy retry output: `tmp/v3p9_v18_policy_retry_eval/`
- SQLite DB: `docs/benchmark/scanrefer/runs.sqlite`

## Evidence

Main runner raw side-by-side output, before aggregation-GT rescore:

```text
n=100 mean_iou=0.2212 acc25=0.4700 acc50=0.1400
failed=[]
```

Aggregation-GT rescore:

```text
n=100 mean_iou=0.4424 acc25=0.5300 acc50=0.4600
stats: missing_scene=0 missing_target=0 no_prediction=0 completed_new_gt=100
```

Leaderboard slicing:

```text
n_total=100 (Unique=28, Multiple=72)
acc25  overall=0.5300 unique=0.7857 multiple=0.4306
acc50  overall=0.4600 unique=0.7857 multiple=0.3333
```

SQLite reproduction:

```sql
SELECT run_id, n_total,
       printf('%.4f', acc25_overall) AS acc25,
       printf('%.4f', acc50_overall) AS acc50,
       printf('%.4f', mean_iou_overall) AS miou,
       printf('%.4f', acc25_unique) AS u25,
       printf('%.4f', acc50_unique) AS u50,
       printf('%.4f', acc25_multiple) AS m25,
       printf('%.4f', acc50_multiple) AS m50,
       keyframe_mode
FROM runs
WHERE run_id IN (
  'v3p3_vertical_spatial_random100_20260504',
  'v3p7_final_selection_random100_20260508',
  'v3p8_relation_ranking_fresh_random100_20260508',
  'v3p9_anchor_guard_v18_20260508'
)
ORDER BY run_id;
```

Expected:

```text
v3p3_vertical_spatial_random100_20260504|100|0.4800|0.4200|0.4130|0.7500|0.7500|0.3750|0.2917|mask3d_query_driven
v3p7_final_selection_random100_20260508|100|0.4600|0.4200|0.4024|0.7500|0.7500|0.3472|0.2917|query_driven
v3p8_relation_ranking_fresh_random100_20260508|100|0.4900|0.4300|0.4120|0.7500|0.7500|0.3889|0.3056|query_driven
v3p9_anchor_guard_v18_20260508|100|0.5300|0.4600|0.4424|0.7857|0.7857|0.4306|0.3333|query_driven
```

Tool trace durability: the v3.9 run ingested 2639 rows into `tool_calls`, and
all 100 samples are `status='completed'` in SQLite.

## Delta vs v3.8 Fresh

| Slice | v3.8 | v3.9 | Delta |
|---|---:|---:|---:|
| mean IoU | 0.4120 | 0.4424 | +0.0304 |
| Overall Acc@0.25 | 49.00 | 53.00 | +4.00 |
| Overall Acc@0.50 | 43.00 | 46.00 | +3.00 |
| Unique Acc@0.25 | 75.00 | 78.57 | +3.57 |
| Unique Acc@0.50 | 75.00 | 78.57 | +3.57 |
| Multiple Acc@0.25 | 38.89 | 43.06 | +4.17 |
| Multiple Acc@0.50 | 30.56 | 33.33 | +2.78 |

Acc@0.25 flips: 12 up, 8 down. Acc@0.50 flips: 10 up, 7 down.

Notable recoveries include:

| Sample | Target | v3.8 IoU | v3.9 IoU |
|---|---|---:|---:|
| `scannet/scene0063_00::13::2` | table | 0.0090 | 0.9818 |
| `scannet/scene0328_00::15::0` | kitchen_cabinet | 0.1123 | 1.0000 |
| `scannet/scene0351_00::2::3` | chair | 0.0000 | 0.9674 |
| `scannet/scene0353_00::49::4` | shelf | 0.0000 | 0.7293 |
| `scannet/scene0377_00::3::1` | shelf | 0.0000 | 0.8658 |
| `scannet/scene0535_00::1::0` | window | 0.0015 | 0.5618 |
| `scannet/scene0552_00::14::0` | cabinet | 0.0000 | 0.9990 |
| `scannet/scene0648_00::23::1` | shelf | 0.0000 | 0.7745 |
| `scannet/scene0660_00::4::4` | chair | 0.0000 | 1.0000 |
| `scannet/scene0663_00::6::0` | office_chair | 0.0000 | 1.0000 |

Notable regressions include:

| Sample | Target | v3.8 IoU | v3.9 IoU |
|---|---|---:|---:|
| `scannet/scene0100_00::26::0` | shower_door | 0.7175 | 0.0858 |
| `scannet/scene0203_00::14::3` | pillow | 0.9759 | 0.0000 |
| `scannet/scene0354_00::5::1` | chair | 1.0000 | 0.0445 |
| `scannet/scene0474_00::15::3` | chair | 0.8727 | 0.0000 |
| `scannet/scene0690_00::18::2` | armchair | 0.8865 | 0.0000 |
| `scannet/scene0697_00::11::2` | dresser | 0.9726 | 0.0000 |
| `scannet/scene0699_00::11::3` | cabinet | 0.8333 | 0.0000 |

The net lift comes from recoveries on both Unique and Multiple slices, but the
remaining regressions show the final-selection path is still stochastic and
not yet robust enough for a full-val claim.

## Caveats

- This is the frozen random100 development fold, not full 9508 val.
- The v3.9 output is a checkpoint-resumed fresh gate, not a single uninterrupted
  process. Resume artifacts are durable per-sample JSON files, and the final
  side-by-side file contains all 100 samples.
- No GT is used during inference. GT appears only in post-run aggregation-GT IoU
  scoring and in this diagnosis.
- One transient policy-side `invalid_prompt` / `-4321` sample was rerun with the
  same query and no GT at inference, then copied over the failed checkpoint.
- The result remains below Z3D's Mask3D-pool zero-shot reference
  (58.9 / 52.7 overall), so the task is not solved.
- The next step should audit the seven Acc@0.50 regressions above before scaling
  to full 9508 validation.
