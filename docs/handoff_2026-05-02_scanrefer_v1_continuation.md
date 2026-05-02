# ScanRefer v1 Implementation — Continuation Handoff

> **Audience:** Mac-side Claude Code agent picking up the ScanRefer v1
> implementation after a context reset.
>
> **TL;DR (read this first):**
>
> 1. Branch: `feat/scanrefer-vg-benchmark`, tip pushed at commit `cb3b8fd`.
> 2. ALL planning + audits + data prep is DONE. The NEXT thing to do is
>    **execute Task 1 of the implementation plan** at
>    `docs/superpowers/plans/2026-05-02-scanrefer-implementation.md`.
> 3. Recommended execution path: invoke `superpowers:subagent-driven-development`
>    skill with the plan path. Each task dispatches a fresh subagent. Two-stage
>    review per task (spec compliance, then code quality).
> 4. Agent run (Task 9) is real ~16-hour API call work — budget for it.
> 5. Don't re-do audits or data prep. Everything you need is already on disk.

---

## 1. Where things stand

### What was just shipped (NR3D v3, complete)

A different but related effort just finished: **NR3D v3 leaderboard track**.
That work proved the Stage 1 + Stage 2 + post-aggregator architecture works.

- Spec: `docs/superpowers/specs/2026-05-01-nr3d-fairness-design.md`
- Plan: `docs/superpowers/plans/2026-05-01-nr3d-leaderboard-track.md`
- v3 doc: `docs/benchmark/nr3d/v3_referit3d_track_20260501.md`
- Pool equivalence proof: `docs/benchmark/nr3d/pool_equivalence_log_20260501.md`
- ReferIt3D protocol audit: `docs/benchmark/nr3d/protocol_audit_20260501.md`
- Paper crosscheck: `docs/benchmark/nr3d/paper_crosscheck_20260501.md`

NR3D v3 is **complete and pushed**. Don't touch it. Read it for **template
inspiration** if you want — the ScanRefer plan deliberately mirrors its
file layout, schema, and TDD discipline.

### What's mid-flight (ScanRefer v1, your job)

You are continuing the **ScanRefer detection-mode v1 evaluation**:

- Spec: `docs/superpowers/specs/2026-05-02-scanrefer-design.md` (committed `30354a5`)
- Plan: `docs/superpowers/plans/2026-05-02-scanrefer-implementation.md` (committed `cb3b8fd`, 3503 lines, 12 tasks)
- Producer report (already-built 11 ScanRefer-only scenes on Linux): `docs/benchmark/scanrefer/producer_report_20260502.md`

**Goal**: ship `v1_mask3d_track` ScanRefer evaluation on canonical 9508
val utterances using gpt-5.4-2026-03-05 against Mask3D-predicted
candidate proposals.

---

## 2. Branch + commit state

```
Branch: feat/scanrefer-vg-benchmark
Remote tip: cb3b8fd
Local tip: cb3b8fd (pushed)

Recent commit chain (since master baseline):
  cb3b8fd docs(scanrefer): implementation plan for v1 detection-track
  30354a5 docs(scanrefer): v1 detection-track design spec
  e4bb62d docs(scanrefer): producer report for 11 ScanRefer-only scenes  ← Linux side committed
  594013e docs(scanrefer): handoff for 11 ScanRefer-only scenes Linux producer run
  542342c docs(nr3d): consolidate audits + retract v2 wider-pool caveat
  ef8ce45 docs(nr3d): index updates for v3 leaderboard-track
  df1d9ff docs(nr3d): v3 referit3d-track leaderboard run
  b6f211a feat(nr3d): ingester schema for leaderboard-track metrics
  45be382 feat(nr3d): add leaderboard-track post-aggregator
  7d9b115 docs(nr3d): implementation plan for v3 leaderboard-track
  2fead86 docs(nr3d): pool equivalence log + leaderboard-track design
```

Run `git log --oneline -15` to confirm.

---

## 3. The 10 brainstorming-locked decisions for ScanRefer v1

These decisions are LOCKED. Do NOT reopen them without explicit user
approval. Source: brainstorming sweep on 2026-05-02; recorded in spec §
"Brainstorming-locked decisions".

| # | Decision | Lock |
|---|---|---|
| 1 | Agent input = NR3D-style **real RGB keyframes** (5 frames per query) with Mask3D candidates as 2D bbox overlays | LOCKED |
| 2 | Keyframe selection = top-5 frames where the **target GT instance** is visible (Phase 8 visibility index) | LOCKED |
| 3 | Per-instance text schema = `proposals.jsonl` shape `{id, bbox_3d, score, label, label_idx}`; label = ScanNet200 class name | LOCKED |
| 4 | Mask3D `ins_scores` HIDDEN from agent — uniform `score=1.0` in proposals | LOCKED |
| 5 | Wall / floor / ceiling Mask3D instances FILTERED at converter time (matches ZSVG3D `keep_background=False`) | LOCKED |
| 6 | IoU = axis-aligned 6-DoF, computed via existing `compute_oriented_iou_3d` with Euler=0 | LOCKED |
| 7 | Headline metrics = Acc@0.25 / Acc@0.50 × { Unique, Multiple, Overall } | LOCKED |
| 8 | Test fold = full ScanRefer val (9508 utts on 141 scenes); no `mentions_target_class` or `correct_guess` filter (those are NR3D conventions) | LOCKED |
| 9 | Stage 2 backend = `gpt-5.4-2026-03-05` via internal ModelHub (same as NR3D v3) | LOCKED |
| 10 | Runner concurrency = `workers=32` (NR3D v3 sweet spot for 3-key API rotation) | LOCKED |

If anything feels wrong about a locked decision while you're implementing,
STOP and surface it to the user — don't unilaterally change direction.

---

## 4. Data inventory (everything is already on disk)

This is the most important section. The new agent should NOT re-download
anything. Verify each path exists before doing anything else.

### ScanRefer val annotations (input to loader)

```
data/scanrefer/raw/ScanRefer_filtered_val.json     6.3 MB, 9508 utterances
data/scanrefer/raw/ScanRefer_filtered_train.json   24 MB
data/scanrefer/raw/ScanRefer_filtered_val.txt      141 lines (scene IDs)
data/scanrefer/raw/ScanRefer_filtered.json         30 MB (full)
```

Per-utterance schema:
```json
{
    "scene_id": "scene0011_00",
    "object_id": "5",                  // string; cast to int for ScanNet objectId
    "object_name": "chair",
    "ann_id": "3",
    "description": "the dark brown wooden chair...",
    "token": ["the", "dark", ...]
}
```

### ZSVG3D Mask3D ScanNet200 `.npz` distribution (input to converter)

```
data/scanrefer/Mask3d/scannet200/<scene>.npz   312 ScanNet val scenes (≥ 141 ScanRefer val)
                                               total ~600 MB
```

Per-`.npz` schema:
```python
{
  'ins_pcds':   ndarray (N_inst,) of object dtype, each (N_points, 6) float32 [X,Y,Z,R,G,B]
  'ins_labels': ndarray (N_inst,) of <U16 strings (ScanNet200 class name)
  'ins_scores': ndarray (N_inst,) float32 (Mask3D confidence)
}
```

ZSVG3D's bbox derivation: `(min+max)/2` center, `max-min` extent on `ins_pcds[i][:,:3]`.

### Phase 8 GT-CG pkl (used as GT bbox lookup table)

```
data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz
   141 / 141 ScanRefer val scenes covered (130 NR3D-overlap + 11 ScanRefer-only)
   The 11 ScanRefer-only scenes were built on Linux on 2026-05-02 via
   src/scripts/nr3d_gt_conceptgraph.py and rsync'd back; see
   docs/benchmark/scanrefer/producer_report_20260502.md for details.
```

Per-pkl schema (gunzip + pickle.load):
```python
{
  'objects': [
    {
      'bbox_np':    ndarray (8, 3) float64 axis-aligned 8 corners,
      'class_name': list[str],
      'class_id':   list[int],
      'pcd_np':     ndarray (P, 3) XYZ,
      'pcd_color_np': ndarray (P, 3) RGB or None,
      'is_background': 0/1,
      'num_detections': int,
      ...
    },
    ...
  ],
  'bg_objects': [...],
}
```

NR3D v3 verified target_id ↔ ScanNet objectId 100% match across 8584
utterances. ScanRefer queries use the same `object_id` semantics, so
indexing into this `objects` list works the same way.

### Posed RGB frames (for keyframe rendering)

```
data/nr3d/scannet/<scene>/raw/
├── 000000-rgb.png / 000000-depth.png / 000000.txt (cam-to-world pose)
├── 000010-rgb.png / 000010-depth.png / 000010.txt
├── ...
├── intrinsic_color.txt
├── intrinsic_depth.txt
├── scene_info.json (kept_frame_ids: [0, 10, 20, ...])
└── traj.txt
```

Frame stride = 10. Per-scene 50-300 frames depending on scene size.
141 / 141 scenes covered.

### ScanNet aux (for the converter to validate inputs if it wants)

```
data/nr3d/scannet_aux/<scene>/
├── <scene>.txt                              (axisAlignment matrix)
├── <scene>.aggregation.json                 (instance segmentation)
└── <scene>_vh_clean_2.0.010000.segs.json   (per-vertex segment id)
```

141 / 141 covered. Most pipelines won't need these (Phase 8 pkl already
encodes the same info), but the converter could read them for sanity if
useful.

### What does NOT exist yet (to be created by the plan)

- `data/scanrefer/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz` (created by Task 3)
- `data/scanrefer/scannet/<scene>/conceptgraph/indices/visibility_index.pkl` (created by Task 3)
- `data/scanrefer/scannet/<scene>/conceptgraph/scene_info.json` (created by Task 3)
- `data/scanrefer/scannet/<scene>/pack_scanrefer_v1/` (created by Task 5)
- `tmp/scanrefer_artifacts/full_val_sample_ids.json` (created in Task 9 setup)
- `tmp/scanrefer_eval_v1_full/` (created during Task 9 agent run)

---

## 5. The 12-task implementation plan

The plan is at `docs/superpowers/plans/2026-05-02-scanrefer-implementation.md`.
**Don't read the whole 3503 lines upfront** — too much context.

Read **only the task you're currently executing**, plus the plan header (Goal,
Architecture, File Structure) once at the start.

Task list at a glance:

| Task | Component | Wall time | Output |
|---|---|---|---|
| 1 | Mask3D-CG converter pure helpers (TDD) | 30 min | `src/scripts/build_scanrefer_mask3d_cg.py` (helpers) + tests |
| 2 | Mask3D-CG converter IO + CLI | 1 hour | full converter + integration smoke test |
| 3 | Run converter on 141 scenes | 25-40 min | 141 × pkl + visibility under `data/scanrefer/scannet/` |
| 4 | ScanRefer loader (TDD) | 1 hour | `src/benchmarks/scanrefer_loader.py` + tests |
| 5 | Pack-prep mirror | 1 hour | `src/evaluation/scripts/prepare_pack_v1_inputs_scanrefer.py` + 5-utt smoke |
| 6 | Runner mirror + 20-utt smoke | 1.5 hours | `src/evaluation/scripts/run_scanrefer_vg_side_by_side.py` |
| 7 | Aggregator (TDD) | 1 hour | `src/evaluation/scripts/scanrefer_leaderboard_metrics.py` |
| 8 | Ingester (TDD) | 30 min | `scripts/ingest_scanrefer_run.py` |
| 9 | **Full agent run on 9508 utts** | **~16 hours (background)** | `tmp/scanrefer_eval_v1_full/side_by_side.json` |
| 10 | Aggregate + ingest v1 | 30 min | `runs.sqlite` + `leaderboard_metrics.json` |
| 11 | v1 doc + index files | 1 hour | `docs/benchmark/scanrefer/v1_mask3d_track_20260502.md` + README + leaderboard + benchmark/README |
| 12 | Final regression sweep | 30 min | clean tests, push |

**Total**: ~7 hours dev wall + ~16 hours agent run = ~23 hours elapsed.

The plan is TDD-first throughout. Every code task follows RED → GREEN → COMMIT.
Don't skip the failing-test step.

---

## 6. How to execute (recommended)

### Option A — superpowers:subagent-driven-development (recommended)

Invoke the skill with the plan path. The skill dispatches a fresh subagent per
task, runs spec compliance review, then code quality review, then marks task
done. Loop until all 12 tasks complete.

```
Skill: superpowers:subagent-driven-development
args: Execute the implementation plan at docs/superpowers/plans/2026-05-02-scanrefer-implementation.md.
       12 tasks. Each TDD-first. Use sonnet for mechanical impl tasks; sonnet for review.
       Project uses uv .venv (`source .venv/bin/activate && PYTHONPATH=src pytest ...`).
       Stage 2 backend gpt-5.4-2026-03-05. Reuse pack-v1 / Stage2DeepResearchAgent
       infra unchanged. Branch feat/scanrefer-vg-benchmark, tip cb3b8fd.
```

### Option B — superpowers:executing-plans

Same plan, executed inline in the current session with checkpoints. Use this
if subagent-driven proves wasteful or if you want continuous control.

### Option C — manual one-task-at-a-time

Read Task N, execute its steps, commit, move to Task N+1. Most context-efficient
but slowest. Use when you want full understanding before each commit.

### Hard rule for Task 9

Task 9 is a 16-hour real agent run. Steps:

1. Generate `full_val_sample_ids.json` (Step 9.1)
2. Run pack-prep on 141 scenes (Step 9.2, ~30 min)
3. Launch in tmux session `scanrefer-full` with `--workers 32 --sample-retries 1`
4. Don't poll — let it run. Periodic check via `tmux capture-pane`.
5. When `tmp/scanrefer_eval_v1_full/per_sample/pack_scanrefer_v1/` has 9508 files OR `side_by_side.json` exists, proceed to Task 10.

NR3D v2 ran 8584 utts in 16h50m on the same backend with the same workers
config. ScanRefer 9508 utts ≈ 17-18h. Per-sample checkpoints survive
interruption — runner picks up cached samples on restart.

---

## 7. Critical gotchas (READ BEFORE TOUCHING ANYTHING)

### Mac-only project rules

Per `CLAUDE.md`:

- **Use `uv` + `.venv`** (NOT conda — conda is Linux-side only).
- Activate: `source .venv/bin/activate`.
- Test: `PYTHONPATH=src pytest <path> -v`.
- Long-running tasks (Task 9 specifically): tmux MANDATORY.
- Strict no-fallback rule: any failed import or missing dep → raise, don't silently degrade.

### Branch hygiene

- Don't switch branches — stay on `feat/scanrefer-vg-benchmark`.
- Don't touch NR3D v3 files (already shipped, immutable record).
- Don't touch `src/agents/**` (Stage 2 agent code, fully reused as-is).
- All new files go under the paths specified in the plan's "File Structure" section.

### Things you MUST NOT redo

The following are DONE. Doing them again wastes time and may break things:

- ScanRefer JSON download — already at `data/scanrefer/raw/`
- Mask3D `.npz` download + extract — already at `data/scanrefer/Mask3d/scannet200/`
- Phase 8 GT-CG pkl for 130 NR3D scenes — already in `data/nr3d/scannet/`
- Phase 8 GT-CG pkl for 11 ScanRefer-only scenes (built on Linux 2026-05-02, rsync'd back) — already in `data/nr3d/scannet/`
- Posed RGB frames — already in `data/nr3d/scannet/<scene>/raw/`
- ScanNet aux files — already in `data/nr3d/scannet_aux/<scene>/`
- ScanRefer + SeeGround + VoG audits (3 reports in `tmp/`)
- NR3D v3 leaderboard track (different benchmark, fully shipped, not your concern)
- Brainstorming sweep — 10 decisions locked

### Things to check at the very start

Run these commands to verify the data inventory matches what's documented:

```bash
# Branch + tip
git rev-parse --short HEAD                 # → cb3b8fd
git status -s                              # → only .claude/scheduled_tasks.lock at most

# ScanRefer annotations
ls data/scanrefer/raw/ScanRefer_filtered_val.json   # exists, ~6.3 MB
wc -l data/scanrefer/raw/ScanRefer_filtered_val.txt # 141 scenes

# Mask3D npz
ls data/scanrefer/Mask3d/scannet200/ | wc -l        # 312 scenes

# Phase 8 GT pkl coverage on ScanRefer val
python -c "
from pathlib import Path
val = open('data/scanrefer/raw/ScanRefer_filtered_val.txt').read().split()
have = sum(1 for s in val if (Path(f'data/nr3d/scannet/{s}/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz')).exists())
print(f'{have}/{len(val)} Phase 8 pkl coverage')   # → 141/141
"

# Raw frames coverage on ScanRefer val
python -c "
from pathlib import Path
val = open('data/scanrefer/raw/ScanRefer_filtered_val.txt').read().split()
have = sum(1 for s in val if (Path(f'data/nr3d/scannet/{s}/raw')).is_dir() and any(p.suffix=='.png' for p in Path(f'data/nr3d/scannet/{s}/raw').iterdir()))
print(f'{have}/{len(val)} raw frame coverage')      # → 141/141
"
```

If any check fails, STOP and surface to user.

### Linux producer report (if you need 11-scene context)

The Linux side built the 11 ScanRefer-only scenes' Phase 8 GT-CG packages.
Their report (now durable in repo) flags 3 deviations from my original
handoff:

1. ScanNet aux files needed separate HF download (`zahidpichen/scannet-dataset`)
2. Frame extraction is a separate step before `build-scenes`
3. Producer's `--scannet-root` / `--nr3d-root` are parent dirs, not subdirs

Read `docs/benchmark/scanrefer/producer_report_20260502.md` if you need the
detailed Linux-side notes. **Otherwise skip — that's all done.**

### Plan tasks point at NR3D files for "mirror" patterns

Tasks 5, 6 say "mirror NR3D's pack-prep / runner with these specific deltas".
The plan documents the deltas concretely (function name renames, default
arg changes), but if anything is ambiguous, OPEN the corresponding NR3D
file:
- `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`
- `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
- `src/evaluation/scripts/nr3d_leaderboard_metrics.py`
- `scripts/ingest_nr3d_run.py`

These are the source of truth for the mirror pattern.

---

## 8. cmux worker surfaces (optional, for parallel investigations)

Two worker surfaces are alive in cmux but currently idle:

| Surface | Agent | Last task |
|---|---|---|
| `surface:14` | Claude Code Opus 4.7 max | SeeGround + VoG audit (signed off, idle) |
| `surface:44` | Codex gpt-5.5 xhigh | SeeGround + VoG paper survey (signed off, idle) |

You don't strictly need them for the implementation plan — all 12 tasks are
local code work. But if you hit something that needs deep external research
(e.g., a tricky API question, an unfamiliar codebase), you can dispatch.

Per global supervisor skill, prefer Codex for cost-free investigation,
Claude Code reserved for nuanced instruction-following work.

If you don't need them, no action — they idle harmlessly.

---

## 9. Reference materials (read in priority order)

When picking up:

1. **This handoff doc** (you're reading it) — context, gotchas, locked decisions.
2. **`docs/superpowers/specs/2026-05-02-scanrefer-design.md`** — design context, scope, architecture, risks. ~500 lines, ~15 min read.
3. **`docs/superpowers/plans/2026-05-02-scanrefer-implementation.md` Header + File Structure section only** — task list overview. Skip the per-task details until you start that task.
4. **At Task N start**: read only Task N's section of the plan, plus skim Task N-1 commit message for context if needed.

Don't read these unless specifically referenced:
- Audit reports under `tmp/` (already synthesized into spec/plan)
- NR3D v3 docs (template, already proven; don't need full read)
- Brainstorming history (locked decisions are in spec §"Brainstorming-locked")

---

## 10. Definition of done for the whole effort

You can declare ScanRefer v1 ship'd when ALL of these are true:

- [ ] All 12 tasks committed on `feat/scanrefer-vg-benchmark`.
- [ ] All ~38 new tests pass (`pytest src/scripts/tests/test_build_scanrefer_mask3d_cg.py src/benchmarks/tests/test_scanrefer_loader.py src/evaluation/scripts/tests/test_*scanrefer*.py -v`).
- [ ] No regression on existing NR3D tests (`pytest src/evaluation/scripts/tests/test_*nr3d*.py -v`).
- [ ] `runs.sqlite` has the v1 row with non-null Acc@0.25/0.50 × Unique/Multiple/Overall.
- [ ] `docs/benchmark/scanrefer/v1_mask3d_track_20260502.md` exists with measured numbers substituted (no `[ACC25_OVERALL]` placeholders left).
- [ ] `docs/benchmark/scanrefer/{README, leaderboard}.md` reflect v1 numbers.
- [ ] `docs/benchmark/README.md` Active Benchmarks table has a ScanRefer row.
- [ ] Branch pushed to origin.
- [ ] `git status -s` is empty (other than session-only `.claude/scheduled_tasks.lock`).

---

## 11. If you finish early or need help

- If a task feels harder than the plan estimates, the plan is wrong — surface to user, don't silently expand.
- If a code change requires modifying NR3D files (Task 5/6 mirror), confirm with user before touching them. The intent is to NOT touch NR3D.
- If an agent run goes sideways (Task 9 hangs, Task 6 smoke fails), check NR3D v2's `docs/benchmark/nr3d/v2_phase8_full_20260501.md` — it documents the failure modes we hit there (bg-leakage, sentinel handling, throughput windows) and the same pipe is reused for ScanRefer.
- If you need to consult external knowledge, prefer codex via `/ask codex` (free) over Claude Code (cost). Per global rule.

---

## 12. Final note

This handoff is the only context you need. The plan is exhaustive — its
purpose is exactly so you don't need conversation history. If anything in
this handoff or the plan is unclear, it's a documentation bug — surface it
and the user will fix it.

Good luck. The hard work (audits, decisions, data prep) is all done. What
remains is mechanical TDD implementation following an explicit plan, plus
one 16h overnight agent run.
