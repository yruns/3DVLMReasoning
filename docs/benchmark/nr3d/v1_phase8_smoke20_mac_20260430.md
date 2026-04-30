# v1 Phase8 Smoke (Mac) - 2026-04-30

## Run Identity

- Branch: `feat/nr3d-vg-benchmark`
- Tip commit at run start: `e08bb10` (i.e. effectively `9115fd7` since `e08bb10` only adds the Linux→Mac handoff doc)
- Working tree: `/Users/bytedance/project/3DVLMReasoning` (Mac, uv `.venv`, Python 3.12)
- Internal version: `v1_phase8_smoke20_mac`
- Run ID: `v1_phase8_smoke20_20260430_mac`
- Outcome: `green — first NR3D pack-v1 numbers`

## What Changed vs Prior Smoke

This is the same fold as `v1_phase8_smoke_20260430.md` (20 utterances × 5 scenes), but two differences make the numbers meaningful for the first time:

1. **Code**: tip is `9115fd7` (loader's PCA-aligned 9-DoF OBB recovery from 8-corner Phase 8 boxes). The earlier Linux smoke ran on `3c491a9` which still emitted axis-aligned `[..., 0, 0, 0]` boxes; numerically it never reached metric output anyway because of the endpoint failure, but had it reached metrics they would have been biased by the AABB shortcut.
2. **Endpoint**: Mac's network reaches `https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl` cleanly (TLS handshakes, returns 404 on probe, completes POST chat-completions). The Linux smoke hit `[SSL: UNEXPECTED_EOF_WHILE_READING]` on every transport attempt and produced no `side_by_side.json`.

## Exact Commands

Loader sanity (post-rsync, 10 samples):

```bash
. .venv/bin/activate && PYTHONPATH=src python -c "
from benchmarks.nr3d_loader import Nr3dDataset
ds = Nr3dDataset.from_path(
    data_root='data/nr3d',
    split='test',
    bbox_source='phase8_gt_cg',
    phase8_data_root='data/nr3d/scannet',
    max_samples=10,
)
print(f'loaded={len(ds)}, sample[0]={ds[0].scan_id}::{ds[0].target_id}')
print(f'bbox={[round(x,3) for x in ds[0].gt_bbox_3d]}')
"
```

Pack prep (smoke fold):

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --split test
```

Runner (in tmux per CLAUDE.md long-running rule):

```bash
tmux new-session -d -s nr3d-smoke "source .venv/bin/activate && PYTHONPATH=src python \
    src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --output-dir tmp/nr3d_eval_v1_smoke20 \
    --workers 1 --sample-retries 0 2>&1 | tee /tmp/nr3d_smoke_run.log"
```

Ingest:

```bash
PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_v1_smoke20/ \
    --run-id v1_phase8_smoke20_20260430_mac \
    --branch feat/nr3d-vg-benchmark \
    --commit "$(git rev-parse --short HEAD)" \
    --backend pack_v1 \
    --judge-model none \
    --notes "first NR3D test smoke on Phase 8 GT-CG, Mac endpoint reachable (Linux had SSL EOF)" \
    --db docs/benchmark/nr3d/runs.sqlite
```

## Raw Artifacts

- Sample IDs: `tmp/nr3d_artifacts/v1_smoke20_sample_ids.json` (rsynced from Linux worktree, identical fold)
- Prepared packs:
  - `data/nr3d/scannet/scene0011_00/pack_nr3d_v1`
  - `data/nr3d/scannet/scene0015_00/pack_nr3d_v1`
  - `data/nr3d/scannet/scene0019_00/pack_nr3d_v1`
  - `data/nr3d/scannet/scene0025_00/pack_nr3d_v1`
  - `data/nr3d/scannet/scene0030_00/pack_nr3d_v1`
- Runner output dir: `tmp/nr3d_eval_v1_smoke20/`
  - `side_by_side.json` (18.6 KB, 20 per-sample records)
  - `per_sample/` (per-sample agent traces)
- Logs: `/tmp/nr3d_smoke_prep.log`, `/tmp/nr3d_smoke_run.log`
- SQLite: `docs/benchmark/nr3d/runs.sqlite` row `run_id='v1_phase8_smoke20_20260430_mac'`

## Fold

- Split: NR3D `test`
- Size: 20 utterances
- Scenes: `scene0011_00`, `scene0015_00`, `scene0019_00`, `scene0025_00`, `scene0030_00` (4 per scene, alphabetically by `target_id::assignment_id`)
- Selection: first 4 non-blacklisted, non-clothing test utterances per selected scene with an existing Phase 8 GT-CG package
- Judge model: none; scoring is programmatic 9-DoF oriented IoU

## Results

```
== pack_v1 ==
  n            = 20
  mean_iou     = 0.6701
  Acc@0.25     = 0.7000  (14/20)
  Acc@0.50     = 0.6500  (13/20)
  status_dist  = {'completed': 19, 'failed': 1}
  iou min/median/max = 0.000 / 1.000 / 1.000
```

IoU bucket distribution:

| Bucket | Count |
|---|---:|
| 0.00 (full miss) | 3 |
| 0.01 | 1 |
| 0.04 | 1 |
| 0.06 | 1 |
| 0.30 | 1 (partial — "middle of three windows") |
| 1.00 (perfect — same instance ID as GT) | 13 |

Per-sample turn / tool-call totals: every completed sample terminated at **turn 1 via chassis `submit_final`** with 10–25 inner tool calls. Total wall time ~25 minutes for 20 samples.

## Failure breakdown

- 1 status=failed: `scannet/scene0011_00::20::29043` query="the cabinet next to a blackboard" — agent emitted `failed` rather than picking a candidate.
- 5 status=completed but IoU < 0.25 (off-by-one candidate selection):
  - `scene0015_00::2::5693` — "When standing in the middle of the room facing the windows ..." (view-dependent, hard)
  - `scene0015_00::6::39122` — "select the window near the middle row of tables" (mentions distractor relation)
  - `scene0025_00::10::37171` — "The larger desk with a curve" (relative size)
  - `scene0030_00::2::22274` — "The chair that's tucked right into the table" (relational)
  - `scene0030_00::49::13215` — "the table that is along the same wall as the two windo[ws]" (relational)

The pattern: relational / view-dependent queries are where the GT-pool agent picks a wrong candidate. 13 perfect hits at IoU=1.0 reflect the GT-pool setup (every aggregation instance is a candidate including the GT itself, so a correct candidate ID yields exact IoU).

## SQLite query reproducing the headline

```sql
SELECT run_id, n,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50,
       branch, commit_hash, notes
FROM runs
WHERE run_id = 'v1_phase8_smoke20_20260430_mac';
-- → ('v1_phase8_smoke20_20260430_mac', 20, '0.6701', '0.7000', '0.6500',
--    'feat/nr3d-vg-benchmark', 'e08bb10', 'first NR3D test smoke ...')
```

## Cross-version comparison (NR3D only)

| Version | Date | n | Acc@0.25 | Acc@0.50 | mean_iou | Notes |
|---|---|---:|---:|---:|---:|---|
| v1_phase8_smoke (Linux) | 2026-04-30 | 0 | — | — | — | Endpoint SSL EOF on every attempt |
| v1_phase8_smoke20_mac | 2026-04-30 | 20 | 0.7000 | 0.6500 | 0.6701 | First green pack_v1 numbers |

Sample size is tiny (n=20). Wilson 95 % CI on Acc@0.25 = 14/20 is roughly
[47 %, 87 %]. These numbers should be read as a plumbing-green signal, not a
headline result; scale-up is tracked as `[P1]`.

NR3D and EmbodiedScan VG are independent benchmarks with different language
distributions, annotation conventions, and proposal pools — their numbers are
not cross-comparable, so this doc only stacks NR3D versions.

## Caveats

- **Per-LLM-call durability gap**: `tool_calls` and `llm_calls` tables in `runs.sqlite` are empty for this run. The NR3D ingester does not yet read the runner's per-sample agent trace into per-tool / per-call rows. CLAUDE.md's "Per-LLM-call durability (going forward)" applies — populating these tables is a follow-up that needs the runner to emit a per-question trace file the ingester can pick up.
- **GT-pool inflation**: with `bbox_source="phase8_gt_cg"` every aggregation instance (including the GT) is a candidate. IoU=1.0 for 13/20 samples reflects the agent picking the right instance ID, not bbox regression quality. A detector-pool variant (mirroring EmbodiedScan v4–v6 BIP3D / V-DETR / ConceptGraph pools) is tracked as `[P3]` in the handoff and should produce stricter numbers.
- **Mac vs Linux endpoint reachability**: this run confirms Mac can hit the internal ModelHub endpoint that Linux could not on the same day. Root cause unknown — likely network egress / IP allowlist. If a future Linux run fails the same way, this is environmental and not a code regression.
- **Easy/Hard and View-dep/View-indep splits not reported**: NR3D's per-utterance metadata (`is_view_dep`, `is_hard`) is on the loader sample but the side-by-side runner does not aggregate by it. Worth adding when the fold is large enough.
- **9-DoF angles use loader-level PCA recovery**: extents are exact for OBB inputs; a small Euler is emitted only when the 8-corner cloud is actually rotated, otherwise R=I. The reported `mean_iou` is 9-DoF oriented IoU (not axis-aligned).

## Next Steps (NR3D only)

- [P1] Scale to ~500–1000 NR3D test utterances; needs rsync of additional scenes' `raw/` (or run on Linux once endpoint is fixed). 130/130 scenes have CG output, so no scene-side gating.
- Add NR3D-native breakdowns once the fold is large enough: easy/hard, view-dep/view-indep (both flags exist on the per-utterance metadata).
- Detector-pool NR3D variant to remove the GT-pool IoU=1.0 inflation (mirrors the existing detector-pool work on a different benchmark, but tracked as a separate NR3D-side branch — no cross-benchmark numerical comparison implied).
