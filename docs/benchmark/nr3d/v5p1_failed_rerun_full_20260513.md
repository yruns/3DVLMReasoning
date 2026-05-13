# v5.1 Failed-Case Rerun Full - 2026-05-13

Full NR3D result after rerunning every failed sentinel from
`v5_agent_guards_fair_views_full_20260513` and merging the recovered
checkpoints back into the same full-test fold.

**Status update on 2026-05-13:** this row is invalidated pending rerun. The
NR3D `visibility_index.pkl` files used by the packs have
`metadata.use_depth=false`, so their `view_to_objects` / `object_to_views`
mappings are projection/frustum candidates rather than depth-occlusion
visibility. The numbers below are kept as an audit record only and must not be
quoted as a valid fair-view or SOTA comparison.

## Run Identity

- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Tip commit at failed-rerun time: `c404536`
- Working tree: `/Users/bytedance/project/3DVLMReasoning` (Mac, `.venv`, Python 3.12)
- Internal version: `v5p1_failed_rerun_full`
- Run ID: `v5p1_failed_rerun_full_20260513`
- Base run: `v5_agent_guards_fair_views_full_20260513`
- Stage 1 keyframe parser model: `gemini-2.5-pro`
- Stage 2 backend: `gpt-5.4-2026-03-05` via internal ModelHub
- Judge: none; scoring is programmatic
- SQLite row: `docs/benchmark/nr3d/runs.sqlite::runs.run_id='v5p1_failed_rerun_full_20260513'`

## What Changed vs v5

No pipeline behavior changed. v5.1 only reruns the 241 failed per-sample
checkpoints from v5 and overlays those outputs onto the original full v5
checkpoint set.

- Rerun subset: `tmp/nr3d_artifacts/v5_failed241_sample_ids_20260513.json`
- Rerun output: `tmp/nr3d_eval_v5_failed_rerun_20260513/`
- Merged output: `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- Original v5 output inherited for non-failed samples:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/`
- Guard stack remains:
  - `--use-tool-answer-disagreement-gate`
  - `--use-no-match-candidate-guard`
  - `--use-evidence-frame-guard`
- Keyframes remain fair query-driven evidence. The target id and GT bbox are
  used for scoring only, not for keyframe selection.
- Later audit found that the object-frame visibility source was not fair
  enough: it did not use depth occlusion. This invalidates the run even though
  the target id / GT bbox were not used for keyframe selection.

## Fold

- Split: NR3D `test`
- Full fold size: 8584 samples
- Canonical filtered size: 7805 samples
- Selection file: `tmp/nr3d_artifacts/full_test_sample_ids.json`
- Failed-rerun subset size: 241 samples
- Candidate scenes: 130

## Raw Artifacts

- Failed rerun sample ids:
  `tmp/nr3d_artifacts/v5_failed241_sample_ids_20260513.json`
- Persistent no-match sample id:
  `tmp/nr3d_artifacts/v5_failed_rerun_remaining1_sample_ids_20260513.json`
- Failed rerun output:
  `tmp/nr3d_eval_v5_failed_rerun_20260513/`
- Failed rerun logs:
  - `tmp/nr3d_eval_v5_failed_rerun_20260513_w100.log`
  - `tmp/nr3d_eval_v5_failed_rerun_one_20260513.log`
- Merged output:
  `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- Merged side-by-side output:
  `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/side_by_side.json`
- Merged leaderboard metrics:
  `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/leaderboard_metrics.json`
- Visual stage1+stage2 case-study HTML:
  `docs/benchmark/nr3d/v5p1_case_studies_20260513.html`
  - The HTML was regenerated after the post-run bbox-rendering fix that
    requires in-image bbox surface samples before drawing a 2D mark. Raw tool
    responses still show the original annotated-frame paths; the displayed
    images are the corrected visualization of the same trace, not a new model
    run.
  - The corrected HTML still uses the same recorded trace. It does not repair
    the underlying v5.1 run, because the agent-visible frame/object mappings
    came from projection-only visibility.
- Merge / metrics logs:
  - `tmp/nr3d_eval_v5_failed_rerun_merged_20260513_assemble.log`
  - `tmp/nr3d_eval_v5_failed_rerun_merged_20260513_metrics.log`

## Commands

Extract the v5 failed checkpoints:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path
from evaluation.scripts.run_nr3d_vg_side_by_side import load_sample_ids, sample_result_path

sample_ids = load_sample_ids(Path("tmp/nr3d_artifacts/full_test_sample_ids.json"))
out = Path("tmp/nr3d_eval_v4_agent_guards_fair_views_full")
pack = "pack_nr3d_v4_agent_guards_fair_views"
failed = []
for sid in sample_ids:
    p = sample_result_path(out, "pack_v1", sid, pack_name=pack)
    if not p.exists():
        failed.append(sid)
        continue
    d = json.loads(p.read_text())
    if d.get("status") != "completed" or d.get("selected_object_id") is None:
        failed.append(sid)
Path("tmp/nr3d_artifacts/v5_failed241_sample_ids_20260513.json").write_text(
    json.dumps(failed, ensure_ascii=False, indent=2) + "\n"
)
print(len(failed))
PY
```

Rerun all 241 failed samples:

```bash
tmux new-session -d -s nr3d_v5_failed_rerun 'cd /Users/bytedance/project/3DVLMReasoning && \
  source .venv/bin/activate && \
  export PYTHONUNBUFFERED=1 PYTHONPATH=src && \
  ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
    python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
      --sample-ids tmp/nr3d_artifacts/v5_failed241_sample_ids_20260513.json \
      --data-root data/nr3d/scannet \
      --pack-name pack_nr3d_v4_agent_guards_fair_views \
      --output-dir tmp/nr3d_eval_v5_failed_rerun_20260513 \
      --workers 100 \
      --checkpoint-only \
      --sample-retries 3 \
      --use-tool-answer-disagreement-gate \
      --use-no-match-candidate-guard \
      --use-evidence-frame-guard \
    2>&1 | tee tmp/nr3d_eval_v5_failed_rerun_20260513_w100.log'
```

One sample still submitted no-match after the high-concurrency rerun. It was
retried once more with a single worker and five retries:

```bash
tmux new-session -d -s nr3d_v5_failed_rerun_one 'cd /Users/bytedance/project/3DVLMReasoning && \
  source .venv/bin/activate && \
  export PYTHONUNBUFFERED=1 PYTHONPATH=src && \
  ./scripts/run_with_rss_guard.sh --rss-limit-mb 15000 --check-interval-sec 60 -- \
    python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
      --sample-ids tmp/nr3d_artifacts/v5_failed_rerun_remaining1_sample_ids_20260513.json \
      --data-root data/nr3d/scannet \
      --pack-name pack_nr3d_v4_agent_guards_fair_views \
      --output-dir tmp/nr3d_eval_v5_failed_rerun_20260513 \
      --workers 1 \
      --checkpoint-only \
      --sample-retries 5 \
      --use-tool-answer-disagreement-gate \
      --use-no-match-candidate-guard \
      --use-evidence-frame-guard \
    2>&1 | tee tmp/nr3d_eval_v5_failed_rerun_one_20260513.log'
```

Merge recovered checkpoints over the base full run:

```bash
rm -rf tmp/nr3d_eval_v5_failed_rerun_merged_20260513
mkdir -p tmp/nr3d_eval_v5_failed_rerun_merged_20260513/per_sample
cp -R \
  tmp/nr3d_eval_v4_agent_guards_fair_views_full/per_sample/pack_nr3d_v4_agent_guards_fair_views \
  tmp/nr3d_eval_v5_failed_rerun_merged_20260513/per_sample/
cp \
  tmp/nr3d_eval_v5_failed_rerun_20260513/per_sample/pack_nr3d_v4_agent_guards_fair_views/*.json \
  tmp/nr3d_eval_v5_failed_rerun_merged_20260513/per_sample/pack_nr3d_v4_agent_guards_fair_views/
```

Assemble, score, and ingest:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v4_agent_guards_fair_views \
  --output-dir tmp/nr3d_eval_v5_failed_rerun_merged_20260513 \
  --workers 100 \
  --max-new-samples 0 \
  --sample-retries 2 \
  --use-tool-answer-disagreement-gate \
  --use-no-match-candidate-guard \
  --use-evidence-frame-guard

PYTHONPATH=src .venv/bin/python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
  --side-by-side tmp/nr3d_eval_v5_failed_rerun_merged_20260513/side_by_side.json \
  --nr3d-data-root data/nr3d \
  --phase8-data-root data/nr3d/scannet \
  --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
  --output tmp/nr3d_eval_v5_failed_rerun_merged_20260513/leaderboard_metrics.json \
  --canonical-filter true

PYTHONPATH=src .venv/bin/python scripts/ingest_nr3d_run.py \
  --output-dir tmp/nr3d_eval_v5_failed_rerun_merged_20260513 \
  --run-id v5p1_failed_rerun_full_20260513 \
  --branch feat/nr3d-v4-agent-guards-fair-views \
  --commit c404536 \
  --backend pack_v1 \
  --judge-model none \
  --notes "v5.1 merged full NR3D after rerunning all 241 v5 failed sentinels; 240 completed, 1 persistent no-match; fair-view query-driven keyframes; TADG + no-match + evidence-frame guards" \
  --leaderboard-metrics tmp/nr3d_eval_v5_failed_rerun_merged_20260513/leaderboard_metrics.json \
  --db docs/benchmark/nr3d/runs.sqlite
```

## Metrics

### Leaderboard-Track Classification

| Metric | v5 before failed rerun | v5.1 after failed rerun | Delta |
|---|---:|---:|---:|
| Overall | 66.53 | 68.48 | +1.95 pp |
| Easy | 76.09 | 78.43 | +2.33 pp |
| Hard | 57.59 | 59.18 | +1.59 pp |
| View-Dep | 55.74 | 57.38 | +1.64 pp |
| View-Indep | 72.41 | 74.53 | +2.12 pp |

The failed-case rerun recovered 159 additional correct predictions on the full
8584Q fold and 152 additional correct predictions inside the canonical
filtered 7805Q fold.

### Full-Fold IoU Proxy on GT Pool

| Metric | Value |
|---|---:|
| n_full | 8584 |
| n_filtered | 7805 |
| classification_acc_full | 65.39 |
| classification_acc_filtered | 68.48 |
| mean IoU | 0.6577 |
| Acc@0.25 | 65.41 |
| Acc@0.50 | 65.39 |

### Rerun Completion

| Check | Value |
|---|---:|
| v5 failed checkpoints selected for rerun | 241 |
| Rerun checkpoints written | 241 |
| Rerun completed outputs | 240 |
| Persistent no-match outputs | 1 |
| Merged completed outputs | 8583 |
| Merged failed outputs | 1 |

The only persistent failed output is
`scannet/scene0655_00::19::8270`, query `Box in corner of the room`. The
agent explicitly submitted no-match after a second single-worker retry. This
row is filtered out by the canonical `mentions_target_class` filter, so it
does not enter the 7805Q headline number.

### Rerun Subset Audit

Among the 241 rerun samples, 159 are correct on the full fold. After the
canonical target-class filter, 152 of 216 rerun samples are correct.

| Sample | Query | Result | Audit note |
|---|---|---|---|
| `scannet/scene0474_00::9::27195` | `Find the backpack laying on the sectional sofa.` | selected 9 == target 9 | Recovered correct filtered sample. Trace listed backpack proposals 8/9 and inspected marked keyframes including frames 149/150/59. Annotated images exist and are non-blank (`1296x968`). |
| `scannet/scene0203_00::12::37998` | `the window that the couch is directly facing` | selected 8 != target 12 | Recovered completed but still incorrect filtered sample. Trace found window proposals 8/12/13 and couch proposals 2/3/26, then chose the wrong window. Annotated images exist and are non-blank (`1296x968`). |
| `scannet/scene0655_00::19::8270` | `Box in corner of the room` | selected `null`, target 19 | Persistent no-match. Trace found no `box` / `carton` / `bin` category in the proposal pool and `request_more_views` callback was unavailable. This row is filtered out of the 7805Q headline because `mentions_target_class=false`. |

A richer visual walkthrough for these and two additional cases lives in
`v5p1_case_studies_20260513.html`. It includes stage1 keyframes, marked
proposal images, proposal tables, and every stage2 tool call in chronological
order.

## Comparison to v3 Unfair / GT-Visible Row

| Metric | v3 GT-visible full | v5.1 fair-view full | Delta |
|---|---:|---:|---:|
| Overall | 80.79 | 68.48 | -12.31 pp |
| Easy | 86.06 | 78.43 | -7.63 pp |
| Hard | 75.87 | 59.18 | -16.69 pp |
| View-Dep | 72.46 | 57.38 | -15.08 pp |
| View-Indep | 85.34 | 74.53 | -10.81 pp |

v3 remains the historical recorded high row, but it inherited the older
GT-target-visible keyframe shortcut and the same projection-only visibility
source. The v5.1 numbers are no longer treated as a valid fair-view
measurement because their object-frame visibility source was projection-only.

## Public Nr3D SOTA Context

Source: https://referit3d.github.io/benchmarks.html, checked on 2026-05-13.
The current public Nr3D top row there is UniVLG at 65.2 overall.

The table below is retained only to show what was originally reported before
the visibility audit. It must not be used as a valid SOTA comparison.

| Metric | UniVLG public SOTA | v5.1 recorded, invalidated | Delta |
|---|---:|---:|---:|
| Overall | 65.20 | 68.48 | +3.28 pp |
| Easy | 73.30 | 78.43 | +5.13 pp |
| Hard | 57.00 | 59.18 | +2.18 pp |
| View-Dep | 55.10 | 57.38 | +2.28 pp |
| View-Indep | 69.90 | 74.53 | +4.63 pp |

Metric slicing matched the canonical NR3D target-instance classification
headline, but the evidence source did not: v5.1 used projection-only
object-frame mappings. A new depth-aware full run is required before comparing
against UniVLG or other public rows.

## SQLite Reproduction Query

```sql
SELECT run_id, n, n_filtered,
       printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind
FROM runs
WHERE run_id='v5p1_failed_rerun_full_20260513';
-- v5p1_failed_rerun_full_20260513|8584|7805|0.6848|0.7843|0.5918|0.5738|0.7453
```

## Interpretation

The original v5 aggregate understated the recorded pipeline because 241 service
/ checkpoint sentinels were still counted as failures. Rerunning all of them
recovers most of the lost metric mass and moves the recorded headline from
66.53 to 68.48 overall.

The result is still substantially below the historical v3 GT-visible row, but
it is not currently comparable to public Nr3D SOTA rows because the
object-frame visibility data was not depth-aware.

Post-run visualization audit found two related issues. First, the v5.1
annotated frames drew full clamped 3D bboxes for every proposal listed in
`visibility.json`. Second, and more importantly, the underlying
`object_to_views` / `view_to_objects` index was built with `use_depth=false`,
so it never encoded occlusion-aware visibility. Current code rejects such NR3D
indices and requires depth-aware visibility before pack generation. A full
rerun is needed before claiming any corrected agent-visible metric.
