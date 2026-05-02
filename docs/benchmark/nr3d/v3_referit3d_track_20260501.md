# v3 ReferIt3D-Track — 2026-05-01

First NR3D evaluation aligned with the canonical ReferIt3D leaderboard
protocol (classification accuracy on the GT-pool, with the canonical
filter chain applied). Built by post-aggregating the v2 full-test
`selected_object_id` predictions — no Stage 2 agent re-run.

## Run Identity

- Branch: `feat/nr3d-vg-benchmark`
- Tip commit at run time: `b6f211a`
- Working tree: `/Users/bytedance/project/3DVLMReasoning` (Mac, uv `.venv`, Python 3.12)
- Internal version: `v3_referit3d_track`
- Run ID: `v3_referit3d_track_20260501`
- Stage 2 backend (inherited from v2): `gpt-5.4-2026-03-05` via internal ModelHub
- Aggregator: `src/evaluation/scripts/nr3d_leaderboard_metrics.py`
- Wall time: under 1 minute (no agent calls)

## Methodology

This version reuses every per-sample agent prediction from `v2_phase8_full`
(8584 utterances, 130/130 NR3D test scenes, GT-pool) and runs a pure
post-aggregator over them under the canonical ReferIt3D protocol:

- Filter: `mentions_target_class_only=True` (canonical default per
  `referit3d/in_out/arguments.py:50`).
- Metric: classification accuracy = mean over filtered fold of
  `selected_object_id == target_id`.
- Slicing: easy/hard via `n_objects ≤ 2`; view-dep via the 10-token literal
  set from `referit3d/analysis/utterances.py:103-105`.

The mathematical equivalence to a re-run is justified in
`docs/superpowers/specs/2026-05-01-nr3d-fairness-design.md`: the agent's
prediction for any given `sample_id` depends only on
`(bundle, query, agent_config)`, none of which change when the canonical
filter is applied — the filter only drops rows.

## Fold

- Split: NR3D `test`
- After baseline filters (blacklist + clothes drop): 8584 utterances on 130 scenes (== v2 fold)
- After canonical `mentions_target_class_only=True`: **7805** utterances on 130 scenes
  - retain rate = 7805 / 8584 = 90.92% (consistent with ReferIt3D paper p.9: 91.6%)
- Easy / Hard partition (`n_objects ≤ 2`): n_easy=3773 / n_hard=4032
- View-Dep / View-Indep (token ∩ {front,behind,…} ≠ ∅): n_view_dep=2752 / n_view_indep=5053

## Headline Metrics (Leaderboard Track)

| Metric | Value |
|---|---:|
| Overall (classification accuracy on filtered fold) | **80.79** |
| Easy | **86.06** |
| Hard | **75.87** |
| View-Dep | **72.46** |
| View-Indep | **85.34** |

## SOTA Comparison (NR3D test, classification, GT-track)

| Method | Setup | Overall | Easy | Hard | View-Dep | View-Indep |
|---|---|---:|---:|---:|---:|---:|
| ReferIt3DNet (NR3D-only) | classification | 35.6 | 43.6 | 27.9 | 32.5 | 37.1 |
| BUTD-DETR | classification | 54.6 | 60.7 | 48.4 | 46.0 | 58.0 |
| MVT | classification | 55.1 | 61.3 | 49.1 | 54.3 | 55.4 |
| 3D-VisTA | classification | 64.2 | 72.1 | 56.7 | 61.5 | 65.1 |
| MiKASA | classification | 64.4 | 69.7 | 59.4 | 65.4 | 64.0 |
| UniVLG (GT-track) | classification | 65.2 | 73.3 | 57.0 | 55.1 | 69.9 |
| **Ours (v3, zero-shot RGB+VLM)** | classification | **80.79** | **86.06** | **75.87** | **72.46** | **85.34** |

The "Ours" row is now apples-to-apples with the public leaderboard column
(canonical filter chain, GT-pool, classification metric, identical 5-column
slicing). What remains a paradigm difference — not a protocol violation — is
the input modality (RGB keyframes + 9-DoF box overlays vs trained
3D-point-cloud methods) and the model class (zero-shot LLM vs trained 3D
models).

## Internal Cross-Check Table

| Metric | Value | Notes |
|---|---:|---|
| `classification_acc_full` (n=8584 denominator) | 0.7762 | unfiltered baseline; differs from filtered by 3.17 pp |
| `classification_acc_filtered` (n=7805 denominator) | **0.8079** | **headline** — leaderboard-comparable |
| v2 `Acc@0.50` (9-DoF IoU on GT-pool, n=8584) | 0.7762 | same predictions; under GT-pool, IoU ≥ 0.50 ⟺ correct ID, so this is the IoU-based proxy of `classification_acc_full` |
| v2 `mean_iou` | 0.7800 | unchanged; not a leaderboard metric |

The v2 `Acc@0.50` (77.62 %) and `classification_acc_full` (77.62 %) match
to 4 decimals — empirical confirmation of the GT-pool collapse: under
GT-pool, the agent's bbox is exactly the GT bbox when the right ID is
picked, so IoU = 1.0; otherwise IoU is near 0. The v2 Acc@0.50 number was
already classification accuracy in disguise; v3 just makes it explicit and
adds the canonical filter + slicing.

## Failure Sentinels

The 287 v2 sentinel failures (upstream `InternalServerError 500`,
`BadRequestError 400`, agent malformed payload) carry
`selected_object_id = null`. Under the leaderboard metric, these score
`is_correct = False`, so they pull the headline down. They remain in the
denominator — i.e., we DO NOT exclude them, matching how leaderboard
methods report on their full test set.

## Caveats

- **Input modality asymmetry**: zero-shot RGB+VLM agent vs trained 3D-only
  models. Documented as paradigm difference, not protocol violation. The
  comparison answers "how does a zero-shot multimodal LLM agent score on
  NR3D under the canonical protocol", not "is our LLM stronger than 3D-VisTA
  at the same task."
- **Classification metric on GT-pool collapses to "right ID picked"**: in
  the GT-pool setting the bbox-IoU and ID-match metrics are functionally
  equivalent (because pool members ARE the GT bboxes). This is true for our
  v3 numbers and for every published GT-track leaderboard number above.
- **Per-LLM-call durability gap**: `tool_calls` / `llm_calls` SQLite tables
  remain empty for v3 (carried forward from v2). Tracked in
  `CLAUDE.md::Per-LLM-call durability`.
- **`correct_guess` filter NOT applied**: per audit (Q4), `correct_guess`
  is not part of the canonical referit3d filter chain — only UniVLG-side
  add-on. Applying it would inflate our number by dropping human-failed
  utterances; leaderboard methods don't drop those.
- **130 / 130 NR3D test scene coverage**: see
  `docs/benchmark/nr3d/pool_equivalence_log_20260501.md` for the empirical
  proof that our pool, fold, and indexing are bit-equal to canonical.

## Cross-Version Comparison (NR3D-only)

| Version | Date | n_full | n_filtered | Headline (Overall) | Notes |
|---|---|---:|---:|---:|---|
| v1_phase8_smoke20_mac | 2026-04-30 | 20 | — | Acc@0.50 = 0.65 | first green pack-v1 numbers |
| v2_phase8_full | 2026-05-01 | 8584 | — | Acc@0.50 = 0.7762 | first full sweep on GT-pool, IoU metric |
| **v3_referit3d_track** | 2026-05-01 | 8584 | **7805** | **classification_acc_filtered = 0.8079** | **first leaderboard-comparable run; 5-column SOTA-aligned table** |

## SQLite Reproduction Query

```sql
SELECT run_id,
       printf('%.4f', classification_acc_full) AS acc_full,
       printf('%.4f', classification_acc_filtered) AS acc_filtered,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind,
       n_filtered
FROM runs WHERE run_id='v3_referit3d_track_20260501';
-- → ('v3_referit3d_track_20260501', '0.7762', '0.8079',
--    '0.8606', '0.7587', '0.7246', '0.8534', 7805)
```

## Raw Artifacts

- v2 per-sample checkpoints (input): `tmp/nr3d_eval_v1_full/per_sample/pack_nr3d_v1/*.json`
- v2 aggregate: `tmp/nr3d_eval_v1_full/side_by_side.json`
- v3 leaderboard metrics (intermediate): `tmp/nr3d_eval_v1_full/leaderboard_metrics.json`
- SQLite row: `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v3_referit3d_track_20260501'`
- Aggregator entry point: `src/evaluation/scripts/nr3d_leaderboard_metrics.py`

## Reproduction Command

```bash
source .venv/bin/activate
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
    --side-by-side tmp/nr3d_eval_v1_full/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --output tmp/nr3d_eval_v1_full/leaderboard_metrics.json \
    --canonical-filter true

PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v1_full \
    --run-id v3_referit3d_track_20260501 \
    --branch feat/nr3d-vg-benchmark \
    --commit "$(git rev-parse --short HEAD)" \
    --backend pack_v1 --judge-model none \
    --notes "v3 leaderboard-track" \
    --leaderboard-metrics tmp/nr3d_eval_v1_full/leaderboard_metrics.json \
    --db docs/benchmark/nr3d/runs.sqlite
```

## Next Steps (post-v3)

- Detection-mode track using V-DETR / BIP3D / ConceptGraph proposals (separate P3 work)
- SR3D integration
- Tokenization alignment (referit3d's `pre_process_text`) if v4 wants strict input parity
