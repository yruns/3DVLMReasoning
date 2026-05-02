# ScanRefer Detection-Track Design

**Date:** 2026-05-02
**Branch:** `feat/scanrefer-vg-benchmark`
**Tip commit at design time:** `e4bb62d`
**Authors:** Shuhao Yue (designer / executor), Codex worker (paper survey + cross-validation), Claude Code worker (code-side audit)

## Goal

Ship a v1 ScanRefer **detection-mode** evaluation on the canonical full
val (9508 utterances on 141 scenes) using our Stage 1 + Stage 2 RGB+VLM
agent. Reuse all NR3D v3 infrastructure where it fits; add the minimum
ScanRefer-specific code to bridge the protocol gap.

The headline metrics are Acc@0.25 / Acc@0.50 with Unique / Multiple /
Overall slicing — the canonical ScanRefer leaderboard columns.

## Background — what's already in place

Two prior efforts converged into the inputs for this design:

- **NR3D leaderboard track (v3)**, shipped in
  `docs/superpowers/specs/2026-05-01-nr3d-fairness-design.md`. The
  ConceptGraph-shaped per-scene packages, the keyframe selection,
  pack-prep / runner / aggregator template, and the SQLite ingester
  schema all came out of that work. We will mirror the same shape.
- **ScanRefer protocol audits**, written by two parallel workers:
  - Code-side: `tmp/scanrefer_zsl_code_audit.md` (audited ZSVG3D + VLM-Grounder)
  - Paper-side: `tmp/scanrefer_zsl_paper_survey.md` (surveyed 6 zero-shot methods)
  - SeeGround + VoG deep-audit: `tmp/seeground_vog_audit_{claude,codex}.md`
  These reports identified Mask3D as the de-facto shared 3D detector
  among proposal-based zero-shot ScanRefer methods (ZSVG3D, SeeGround,
  CSVG, Z3D Mask3D row), with no canonical proposal-file URL —
  ZSVG3D's CUHK SharePoint `.npz` is the cleanest drop-in artifact.

After today's data prep, every ScanRefer val asset is local:

| Resource | Path | Coverage |
|---|---|---|
| ScanRefer annotations | `data/scanrefer/raw/ScanRefer_filtered_*.json` | 9508 val utts |
| Mask3D ScanNet200 `.npz` | `data/scanrefer/Mask3d/scannet200/<scene>.npz` | 312 / 312 ScanNet val |
| GT lookup (Phase 8 pkl) | `data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz` | 141 / 141 ScanRefer val scenes |
| Posed RGB frames | `data/nr3d/scannet/<scene>/raw/` | 141 / 141, stride=10 |
| ScanNet aux (.txt + .aggregation + .segs) | `data/nr3d/scannet_aux/<scene>/` | 141 / 141 |

## Brainstorming-locked decisions

The following 10 decisions were locked during brainstorming on 2026-05-02:

1. **Agent visual input** = **NR3D-style real RGB keyframes** (5 frames per query) with Mask3D candidates projected as 2D bbox overlays. Reuses Stage 1 keyframe selector and Stage 2 agent unchanged.
2. **Keyframe selection** = top-5 frames where the **target GT instance** is visible (using existing Phase 8 visibility index, which is keyed by ScanNet `objectId`).
3. **Per-instance text schema** = NR3D `proposals.jsonl` shape `{id, bbox_3d, score, label, label_idx}`. `label` = ScanNet200 class name from Mask3D `ins_labels`.
4. **Mask3D scores hidden from agent** — uniform `score=1.0` in proposals; `ins_scores` recorded only for diagnostics.
5. **Wall / floor / ceiling filtered** — drop any Mask3D instance whose `ins_labels[i].lower()` is in `{wall, floor, ceiling}`. Mirrors ZSVG3D `keep_background=False`.
6. **IoU function** = axis-aligned 6-DoF IoU. Implementation reuses `compute_oriented_iou_3d` with Euler=0 (functionally equivalent and bit-stable on AABB inputs).
7. **Metrics + slicing** = Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall }. **Unique** ⟺ scene contains exactly one GT instance whose class equals the target's `object_name`. Computed from the Phase 8 GT-CG pkl per scene.
8. **Test fold** = full ScanRefer val (9508 utts on 141 scenes). No `mentions_target_class` or `correct_guess` filter (those are NR3D conventions, not ScanRefer's).
9. **Stage 2 backend** = `gpt-5.4-2026-03-05` via internal ModelHub, identical to NR3D v3.
10. **Runner concurrency** = `workers=32`, identical to NR3D v3 (sweet spot for 3-key API rotation).

## Architecture and data flow

```
ScanRefer JSON (9508 utts)
        │
        │  scanrefer_loader.ScanRefVGDataset.from_path()
        ▼
ScanRefVGSample(scene_id, target_id, target, query, gt_bbox_3d, is_unique, ann_id)
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────┐
        │                                                                   │
        │ GT bbox lookup                                                    │ proposal pool
        ▼                                                                   ▼
data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/                  data/scanrefer/scannet/<scene>/conceptgraph/pcd_saves/
   full_pcd_gt_axisaligned_post.pkl.gz                                full_pcd_mask3d_axisaligned.pkl.gz
   (Phase 8 GT-CG, NR3D-shared, indexed by ScanNet objectId)          (NEW: Mask3D-CG, ScanRefer-only)
                                                                     ▲
                                                                     │ produced once per scene
                                                                     │ build_scanrefer_mask3d_cg.py
                                                                     │
                                                                     data/scanrefer/Mask3d/scannet200/<scene>.npz
                                                                     (ZSVG3D-distributed Mask3D output)
        │                                                                   │
        │                                                                   │
        └────────────────┬──────────────────────────────────────────────────┘
                         │
                         │ prepare_pack_v1_inputs_scanrefer.py
                         │  (mirrors NR3D pack-prep, with two pkl sources)
                         ▼
data/scanrefer/scannet/<scene>/pack_scanrefer_v1/
   ├── proposals.jsonl                   ← Mask3D pool (filtered)
   ├── visibility.json                   ← view-to-objects for Mask3D pool
   ├── annotated/frame_<id>.png          ← Mask3D 2D-bbox overlays on raw RGB
   └── samples/<safe_sample_id>.json     ← per-utterance bundle
                         │
                         │ run_scanrefer_vg_side_by_side.py
                         │   (same Stage2DeepResearchAgent as NR3D)
                         ▼
tmp/scanrefer_eval_<run>/per_sample/pack_scanrefer_v1/*.json
                         ▼
tmp/scanrefer_eval_<run>/side_by_side.json
                         │
                         │ scanrefer_leaderboard_metrics.py
                         ▼
tmp/scanrefer_eval_<run>/leaderboard_metrics.json
   (Unique/Multiple slicing × Acc@0.25/0.50)
                         │
                         │ ingest_scanrefer_run.py
                         ▼
docs/benchmark/scanrefer/runs.sqlite
                         │
                         ▼
docs/benchmark/scanrefer/v1_mask3d_track_<date>.md
```

The Stage 2 agent itself is **untouched**. The keyframe selector (Stage 1) is **untouched**. We add five new modules, one new producer step, and one new SQLite ingester.

## Scope

### In scope (this design)

- Mask3D `.npz` → ConceptGraph-shaped pkl converter, plus per-scene visibility index for the Mask3D pool.
- ScanRefer loader emitting `ScanRefVGSample` (analogous to `Nr3dVGSample`).
- Pack-prep `prepare_pack_v1_inputs_scanrefer.py` mirroring NR3D's pack-prep with two pkl sources.
- Runner `run_scanrefer_vg_side_by_side.py` mirroring NR3D's runner; sample_id format `scannet/<scene>::<object_id>::<ann_id>`.
- Aggregator `scanrefer_leaderboard_metrics.py` with Unique/Multiple/Overall × {0.25, 0.50}.
- Ingester `scripts/ingest_scanrefer_run.py` with ScanRefer-specific schema.
- v1 evaluation run on 9508 utts.
- v1 version doc + index files (README, leaderboard).
- Tests for the converter, loader, and aggregator.

### Out of scope

- Detection-mode ablations using non-Mask3D detectors (BIP3D / GroupFree3D / V-DETR). Future P2 work.
- ScanRefer test-set submission (server-only, not on val). Out of scope.
- SeeGround-style synthetic PyTorch3D rendering (architectural change). Future P3 ablation.
- Qwen2-VL-72B head-to-head with SeeGround. Future P3 backbone ablation; v1 stays on `gpt-5.4-2026-03-05`.
- Mask3D fine-tuning or re-running the detector. We consume ZSVG3D's pre-distributed `.npz` verbatim.
- SR3D dataset.

## Components

### Component 1 — `src/scripts/build_scanrefer_mask3d_cg.py` (new)

A producer that takes ZSVG3D's Mask3D `.npz` and emits a ConceptGraph-shaped pkl + visibility index per scene, filtering wall/floor/ceiling.

Public function:

```python
def build_mask3d_cg_for_scene(
    scene_id: str,
    mask3d_npz_path: Path,
    raw_dir: Path,           # data/nr3d/scannet/<scene>/raw/
    output_pkl: Path,        # data/scanrefer/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz
    output_visibility: Path, # data/scanrefer/scannet/<scene>/conceptgraph/indices/visibility_index.pkl
    output_scene_info: Path, # data/scanrefer/scannet/<scene>/conceptgraph/scene_info.json
    drop_background: bool = True,
) -> dict:
    """Convert Mask3D .npz to ConceptGraph-shaped pkl + visibility index.

    Returns summary dict with n_kept, n_dropped, n_visibility_mappings.
    """
```

Per Mask3D instance i passing the wall/floor/ceiling filter, emit object dict:

```python
{
    "bbox_np": axis_aligned_corners,      # (8, 3), built from min/max of pcds[:, :3]
    "class_name": [str(ins_labels[i])],   # list-of-1 string
    "class_id": [scannet200_idx_of(label)],  # list-of-1 int; lookup via conceptgraph/scannet200_classes.txt (existing in repo)
    "pcd_np": pcds[:, :3].astype(np.float64),
    "pcd_color_np": pcds[:, 3:6] if pcds.shape[1] >= 6 else None,
    "is_background": 0,
    "num_detections": 1,
    "n_points": [len(pcds)],
    "conf": [float(ins_scores[i])],       # diagnostic only; pack-prep ignores
    # all other Phase 8 fields default to None / empty list
}
```

Visibility index: project each kept Mask3D bbox into each kept frame using `intrinsic_color.txt` + per-frame `XXXXXX.txt` poses (already in `raw_dir`). Reuse `src/scripts/build_visibility_index.py::build_visibility_index` if its API accepts a generic objects list; otherwise factor a thin wrapper.

CLI driver:

```bash
python -m src.scripts.build_scanrefer_mask3d_cg \
    --scenes <list> | --scene-list data/scanrefer/raw/ScanRefer_filtered_val.txt \
    --mask3d-root data/scanrefer/Mask3d/scannet200 \
    --raw-root data/nr3d/scannet \
    --output-root data/scanrefer/scannet \
    --report tmp/scanrefer_handoff/converter_report.md
```

Expected wall: ~5-15 seconds per scene (no GPU; pure numpy + bbox projection). Total ~141 × 10 s = ~25 min wall.

### Component 2 — `src/benchmarks/scanrefer_loader.py` (new)

Mirror of `nr3d_loader.py`, parameterized on the ScanRefer JSON shape (no `mentions_target_class`, no `tokens` field, no `correct_guess`).

Sample dataclass:

```python
@dataclass
class ScanRefVGSample(BenchmarkSample):
    scan_id: str           # "scannet/<scene>"
    target_id: int         # int(ScanRefer JSON's object_id)
    target: str            # ScanRefer JSON's object_name
    ann_id: str            # ScanRefer JSON's ann_id
    description: str       # ScanRefer JSON's description (alias of query)
    gt_bbox_3d: list[float] | None  # 9-DoF, Euler=0; from Phase 8 pkl
    is_unique: bool        # computed: (count of same-class GT in scene) == 1
```

Loader signature:

```python
ScanRefVGDataset.from_path(
    data_root: Path,                # data/scanrefer
    phase8_data_root: Path,         # data/nr3d/scannet
    split: str = "val",
    sample_ids: set[str] | None = None,
) -> ScanRefVGDataset
```

Sample ID format: `scannet/<scene>::<object_id>::<ann_id>` (mirrors NR3D's `scannet/<scene>::<target_id>::<assignment_id>`).

For each utterance:
1. Look up GT bbox via Phase 8 pkl indexed by `int(object_id)`.
2. Compute `is_unique` by counting Phase 8 instances in the same scene whose `class_name[0] == object_name`.
3. Emit `ScanRefVGSample`.

If a scene's Phase 8 pkl is missing OR target_id is out of range, skip the sample with a stat counter (mirrors NR3D loader's `skipped_missing_or_ambiguous_bbox`).

### Component 3 — `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py` (new)

Mirror of `prepare_pack_v1_inputs_nr3d.py` with two changes:

1. Proposal source is the Mask3D-CG pkl from Component 1, NOT the Phase 8 GT-CG pkl.
2. Sample ID format and `safe_sample_id` adapt to the new format.

```python
def prepare_pack_v1_inputs_scanrefer(
    sample_ids_path: Path,
    data_root: Path = Path("data/scanrefer/scannet"),
    pack_name: str = "pack_scanrefer_v1",
    split: str = "val",
    nr3d_root: Path = Path("data/scanrefer/raw"),
    phase8_data_root: Path = Path("data/nr3d/scannet"),  # for GT lookup only
    max_samples: int | None = None,
) -> list[Path]
```

Pack-prep output per scene:

```
data/scanrefer/scannet/<scene>/pack_scanrefer_v1/
├── proposals.jsonl              # Mask3D pool (filtered, axis-aligned)
├── visibility.json              # view-to-Mask3D-objects mapping
├── annotated/frame_<id>.png     # rendered Mask3D bbox overlays
└── samples/<safe_id>.json       # per-utterance bundle (gt_bbox_3d_9dof from Phase 8)
```

The `samples/*.json` bundle's `gt_bbox_3d_9dof` is sourced from the Phase 8 GT-CG pkl (i.e., the loader's `get_gt_bbox(scan_id, target_id)`). This is the GT used by the runner for IoU scoring.

### Component 4 — `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py` (new)

Mirror of `run_nr3d_vg_side_by_side.py`. Differences:

- `parse_scanrefer_sample_id`: parses `scannet/<scene>::<object_id>::<ann_id>`.
- Same pack-v1 backend, same `Stage2DeepResearchAgent`.
- Same failed-sentinel wrapper to prevent single-sample crashes from killing the run.
- Same per-sample checkpointing under `tmp/scanrefer_eval_<run>/per_sample/pack_scanrefer_v1/`.

The IoU computation reuses `compute_oriented_iou_3d`. Both the predicted bbox and GT bbox are 9-DoF with Euler=0 (functionally axis-aligned). Documented in component-3 invariant.

CLI mirror:

```bash
python -m src.evaluation.scripts.run_scanrefer_vg_side_by_side \
    --sample-ids tmp/scanrefer_artifacts/full_val_sample_ids.json \
    --data-root data/scanrefer/scannet \
    --pack-name pack_scanrefer_v1 \
    --output-dir tmp/scanrefer_eval_v1_full \
    --workers 32 --sample-retries 1
```

Expected wall: 9508 utts × similar throughput as NR3D v2 (~5-23 finishes/min) ≈ 11-32 hours; budget **~16 hours** as the rough estimate, matching v2's 16h50m on 8584 utts.

### Component 5 — `src/evaluation/scripts/scanrefer_leaderboard_metrics.py` (new)

Mirror of `nr3d_leaderboard_metrics.py` with new slicing. Pure aggregator function:

```python
def aggregate(
    predictions: dict[str, dict],   # sample_id -> {selected_object_id, predicted_bbox_3d, iou}
    sample_meta: list[dict],        # [{sample_id, target_id, target, is_unique}, ...]
) -> dict
```

Returned metrics dict keys (all required):

| key | type | meaning |
|---|---|---|
| `n_total` | int | utterances in side_by_side (9508 expected) |
| `n_unique` | int | utterances with `is_unique=True` |
| `n_multiple` | int | utterances with `is_unique=False` |
| `acc25_overall` | float | mean(iou >= 0.25) over n_total |
| `acc50_overall` | float | mean(iou >= 0.50) over n_total |
| `acc25_unique` / `acc50_unique` | float | over `is_unique=True` subset |
| `acc25_multiple` / `acc50_multiple` | float | over `is_unique=False` subset |
| `mean_iou_overall` | float | mean iou |
| `per_sample` | list[dict] | per-utterance booleans (is_correct_25, is_correct_50, is_unique) |

Invariants enforced as runtime asserts:

1. Every sample_id in `sample_meta` appears in `predictions` (no silent partial denominators).
2. `n_unique + n_multiple == n_total`.
3. `len(per_sample) == n_total`.

### Component 6 — `src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py` (new)

Pytest module covering:

- `test_aggregate_all_correct` — 4 hits → all acc=1.0
- `test_aggregate_all_wrong` — 4 misses → all acc=0.0
- `test_aggregate_iou_thresholds` — IoU exactly at 0.25 / 0.50 boundaries
- `test_aggregate_unique_multiple_partition` — verify slicing denominators
- `test_aggregate_failed_sentinel` — `iou=None` → acc=0
- `test_aggregate_raises_on_missing_sample_id` — sanity for inner-join
- `test_aggregate_invariants` — n_unique + n_multiple == n_total

### Component 7 — `scripts/ingest_scanrefer_run.py` (new)

Mirror of `ingest_nr3d_run.py` with ScanRefer-specific schema. New SQLite DB at `docs/benchmark/scanrefer/runs.sqlite`.

`runs` table columns:

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, branch TEXT, commit_hash TEXT, output_dir TEXT,
    backend TEXT, n_total INTEGER, n_unique INTEGER, n_multiple INTEGER,
    acc25_overall REAL, acc50_overall REAL,
    acc25_unique REAL, acc50_unique REAL,
    acc25_multiple REAL, acc50_multiple REAL,
    mean_iou_overall REAL,
    judge_model TEXT, started_at REAL, ingested_at REAL NOT NULL,
    notes TEXT
);
```

`samples` table columns:

```sql
CREATE TABLE samples (
    run_id TEXT, sample_id TEXT, scene_id TEXT, target_id INTEGER, ann_id TEXT,
    description TEXT, status TEXT,
    selected_object_id INTEGER, confidence REAL,
    iou REAL, acc25 INTEGER, acc50 INTEGER, is_unique INTEGER,
    predicted_bbox_3d_9dof TEXT, gt_bbox_3d_9dof TEXT,
    PRIMARY KEY (run_id, sample_id)
);
```

CLI:

```bash
python scripts/ingest_scanrefer_run.py \
    --output-dir tmp/scanrefer_eval_v1_full \
    --leaderboard-metrics tmp/scanrefer_eval_v1_full/leaderboard_metrics.json \
    --run-id v1_mask3d_track_<date> \
    --branch feat/scanrefer-vg-benchmark \
    --commit "$(git rev-parse --short HEAD)" \
    --backend pack_v1 --judge-model none \
    --notes "v1 ScanRefer detection-mode track on Mask3D pool" \
    --db docs/benchmark/scanrefer/runs.sqlite
```

### Component 8 — `docs/benchmark/scanrefer/v1_mask3d_track_<date>.md` (new)

Sections:

1. Run identity (branch, commit, backend, fold size, wall time)
2. Methodology (Mask3D-pool detection-mode track, 5-keyframe RGB+VLM agent, gpt-5.4-2026-03-05)
3. Fold (9508 utts on 141 scenes; 100% canonical val coverage; verified via ScanRefer_filtered_val.json)
4. Headline metrics (Acc@0.25 / Acc@0.50 × Unique / Multiple / Overall)
5. SOTA comparison (ZSVG3D / SeeGround / CSVG / Z3D / VLM-Grounder / VoG; supervised top-5 from SeeGround Table 1)
6. Caveats:
   - Input modality asymmetry — zero-shot RGB+VLM agent vs trained 3D model class
   - Mask3D pool quality bounds the upper limit (instances mis-segmented by Mask3D are unrecoverable)
   - 287-class wall/floor/ceiling filter (matches ZSVG3D `keep_background=False`)
   - GT bbox source = Phase 8 GT-CG pkl (functionally Vil3dRef-equivalent; pool_equivalence_log_20260501.md proves NR3D coverage)
7. Cross-version comparison (NR3D v3 vs ScanRefer v1 — different benchmarks, listed for narrative continuity)
8. SQLite reproduction query
9. Raw artifact paths
10. Reproduction command

### Component 9 — index files

- `docs/benchmark/scanrefer/README.md` (new) — version timeline + Current Interpretation
- `docs/benchmark/scanrefer/leaderboard.md` (new) — public leaderboard reference table
- `docs/benchmark/README.md` — add ScanRefer row to Active Benchmarks

## Implementation order

| step | deliverable | dependency | est. wall time |
|---|---|---|---|
| 1 | failing tests for converter (Component 1) | — | 30 min |
| 2 | converter implementation; run on 141 scenes | step 1 | 1.5 hours |
| 3 | ScanRefer loader (Component 2) + tests | step 2 | 1 hour |
| 4 | pack-prep (Component 3) + smoke test | step 3 | 1 hour |
| 5 | runner (Component 4) + 20-utt smoke run | step 4 | 1 hour |
| 6 | aggregator (Component 5) + tests | step 5 | 1 hour |
| 7 | ingester (Component 7) + tests | step 6 | 30 min |
| 8 | full v1 run on 9508 utts | step 5 | **~16 hours** (mostly agent waits) |
| 9 | aggregate + ingest v1 outputs | step 8 | 30 min |
| 10 | v1 doc (Component 8) | step 9 | 1 hour |
| 11 | index updates (Component 9) | step 10 | 30 min |

Active dev wall: ~7 hours. Agent run wall: ~16 hours (background). Total elapsed: ~23 hours, achievable over a single weekend.

## Acceptance criteria

The design is complete when ALL of these are true:

1. `python -m src.scripts.build_scanrefer_mask3d_cg --scene-list data/scanrefer/raw/ScanRefer_filtered_val.txt --output-root data/scanrefer/scannet/` runs to completion; all 141 scenes produce both pkl and visibility files.
2. `pytest src/benchmarks/tests/test_scanrefer_loader.py src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py src/evaluation/scripts/tests/test_ingest_scanrefer_run.py -v` all pass.
3. `python -m src.evaluation.scripts.prepare_pack_v1_inputs_scanrefer` smoke run on 20 utts produces complete `pack_scanrefer_v1/` directories.
4. Full v1 run completes with `n_total=9508` and `n_failed < 5%`.
5. SQLite query `SELECT acc25_overall, acc50_overall, acc25_unique, acc50_unique, acc25_multiple, acc50_multiple FROM runs WHERE run_id='v1_mask3d_track_<date>'` returns 6 non-null float values.
6. v1 doc exists with all 10 sections; SOTA table includes at least 5 baselines.
7. README + leaderboard + benchmark/README.md indexes reflect v1.
8. `git status -s` is clean after final commit.

## Risks and mitigations

| risk | mitigation |
|---|---|
| Mask3D `.npz` pcd points lack sufficient density for visibility projection | The visibility-index builder we're calling already handles sparse points. If a Mask3D instance has < `min_visible_points=5` projected pixels in any frame, it's just absent from that frame's visibility — same fallback as NR3D's GT objects. Tested empirically: ZSVG3D's `.npz` averages thousands of points per instance. |
| ScanNet200 class label not in our class_id taxonomy lookup | Use a permissive lookup: any unknown label gets `class_id=-1`. Pack-prep already handles this case (sees label as string only, doesn't gate on label_idx). |
| Wall/floor/ceiling filter drops a real target the user is asking about ("the wall behind the table") | Audit the ScanRefer val for queries whose `object_name` is in `{wall, floor, ceiling}`. If non-zero, document as "queries on background labels are excluded by Camp-A convention" — matches ZSVG3D / SeeGround behavior. Step 1 of impl will measure this. |
| Phase 8 pkl class_name doesn't match ScanRefer JSON's `object_name` for is_unique calc | We're matching on **Phase 8 class_name (Mask3D's ScanNet200 vocabulary in NR3D's GT injection)** to ScanRefer's `object_name`. Both are short class strings. Empirically ScanRefer val class names are mostly direct ScanNet vocabulary; mismatch rate likely < 5%. Pre-impl: dump the unmatched object_names from a 100-utt sample and decide. If too many mismatches, fall back to ScanRefer val's own scene-level class count from the JSON. |
| Mask3D pool bigger than NR3D Phase 8 pool → more proposals in agent prompt → token overflow | Phase 8 averages 30-100 instances per scene; Mask3D averages 30-100 too (per the produced `.npz` we inspected for scene0088_00 = 52 instances). Per-frame visible subset is much smaller. Same pack-prep token bounds apply. |
| 16-hour agent run exhausts API quota or hits GPU-cluster maintenance window | Per-sample checkpoints survive interruptions (NR3D v2 proven). If interrupted, restart picks up only un-cached samples. Run started during off-hours. |
| Per-LLM-call durability gap (carried forward from NR3D v3 design) | Same gap as NR3D. `tool_calls` / `llm_calls` SQLite tables not populated. Tracked in CLAUDE.md, separate fix. |

## Open questions / explicit deferrals

- **SeeGround head-to-head with same backbone (Qwen2-VL-72B)** — defer to v2 once v1 numbers are sane. Need vLLM serving + 8x H100 / API quota.
- **Detection-pool ablations on alternative detectors (BIP3D / V-DETR / GroupFree3D)** — defer to P2 work; out of scope for v1.
- **SR3D integration** — out of scope; SR3D is template-language so the same pipeline fits but evidence on protocol still needed.
- **NR3D v3 + ScanRefer v1 cross-benchmark narrative** — these are different benchmarks; we will NOT mix their numbers. Single-line cross-version note in the v1 doc only.

## Appendix — file map

New files:

- `src/scripts/build_scanrefer_mask3d_cg.py`
- `src/benchmarks/scanrefer_loader.py`
- `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py`
- `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py`
- `src/evaluation/scripts/scanrefer_leaderboard_metrics.py`
- `scripts/ingest_scanrefer_run.py`
- `src/benchmarks/tests/test_scanrefer_loader.py`
- `src/evaluation/scripts/tests/test_scanrefer_leaderboard_metrics.py`
- `src/evaluation/scripts/tests/test_ingest_scanrefer_run.py`
- `docs/benchmark/scanrefer/v1_mask3d_track_<date>.md`
- `docs/benchmark/scanrefer/README.md`
- `docs/benchmark/scanrefer/leaderboard.md`

Modified files:

- `docs/benchmark/README.md` — add ScanRefer row

Untouched (despite being adjacent):

- `src/benchmarks/nr3d_loader.py`
- `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
- `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`
- `src/evaluation/scripts/nr3d_leaderboard_metrics.py`
- `scripts/ingest_nr3d_run.py`
- `src/agents/**` (Stage 2 agent code)
- Phase 8 GT-CG pkl files in `data/nr3d/scannet/<scene>/conceptgraph/` (read-only consumer)

Reference artifacts (read-only):

- `tmp/scanrefer_zsl_code_audit.md` — 2-repo zero-shot code audit
- `tmp/scanrefer_zsl_paper_survey.md` — 6-paper paper survey
- `tmp/seeground_vog_audit_{claude,codex}.md` — SeeGround + VoG cross-validation deep audit
- `docs/benchmark/scanrefer/producer_report_20260502.md` — Linux-side ScanRefer-only-11-scene producer run
- `docs/handoff_2026-05-02_scanrefer_11scenes_linux.md` — handoff for the 11-scene producer run
- `docs/superpowers/specs/2026-05-01-nr3d-fairness-design.md` — NR3D leaderboard track design (template parent)
- `docs/benchmark/nr3d/v3_referit3d_track_20260501.md` — NR3D v3 results (proves the agent can do this protocol family)
