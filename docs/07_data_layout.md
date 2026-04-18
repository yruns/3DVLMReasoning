# 07 — Data Layout

The local filesystem vs. what `src/benchmarks/openeqa_loader.py` expects, and how the production pilot bridges the gap via `ensure_runtime_scene`. Line numbers against HEAD = `a8e651e`.

## 7.1 What actually lives on disk

| Path | Role | Present on checkout? |
|---|---|---|
| `data/open-eqa-v0.json` | Official OpenEQA v0 question set (13 031 lines) | **yes** |
| `data/OpenEQA/scannet/<clip_id>/conceptgraph/` | ConceptGraph-prepared scene assets (89 clips, ≈ 37 GB) | **yes** |
| `data/OpenEQA/scannet/<clip_id>/raw/` | Raw `*-rgb.png` / `*-depth.png` / pose files | **yes** |
| `external/open-eqa/` | Upstream Meta/FAIR repo cloned for the official LLM-match evaluator | **yes** |
| `data/frames/<episode_id>/` | Expected by the vanilla `OpenEQADataset` loader | **MISSING** (never existed on this checkout) |
| `data/benchmark/` / `data/benchmarks/` | Referenced in some historical docs | **MISSING** (documentation lag) |

Each prepared scene contains, under `data/OpenEQA/scannet/<clip_id>/conceptgraph/`:

```
bev/                                         # BEV renderings for multimodal Stage-1 parsing
checks/                                      # QA checks produced during prep
enriched_objects.json                        # v10 per-object enrichment (load-hard-fails if missing)
gsa_classes_ram_withbg_allclasses.json       # RAM++ class vocabulary
gsa_classes_ram_withbg_allclasses_colors.json
gsa_detections_ram_withbg_allclasses/        # per-frame 2D detections
gsa_vis_ram_withbg_allclasses/               # per-frame visualisation
indices/visibility_index.pkl                 # Stage-1 view-to-objects index (stride embedded in metadata)
pcd_saves/*_post.pkl.gz                      # ConceptGraph scene graph (SceneObject source)
scene_info.json                              # provenance (may contain stale absolute paths from prep host)
traj.txt                                     # camera trajectory
```

and under `data/OpenEQA/scannet/<clip_id>/raw/`:

```
XXXXXX-rgb.png                                # RGB frames at original stride
XXXXXX-depth.png                              # depth frames
XXXXXX.txt                                    # per-frame pose (if prep produced one)
intrinsic_*.txt                               # camera intrinsics
extrinsic_*.txt                               # extrinsics
mesh.ply                                      # reconstructed mesh (for BEV render)
```

## 7.2 The loader mismatch (flag this to any new agent)

`src/benchmarks/openeqa_loader.py`:

- `OpenEQADataset.from_path(data_root, …)` at `:113-188` expects the JSON at `data_root / "data" / "open-eqa-v0.json"` or `data_root / "open-eqa-v0.json"` (line 135-139). Our local `data/open-eqa-v0.json` satisfies the second form if we pass `data_root = data/`.
- `OpenEQASample.load_frames(max_frames)` at `:48-83` iterates `episode_history` (which it resolves to `data_root / "data" / "frames" / <episode_id>`, line 164-167). **That path does not exist locally.** `load_frames` logs a warning and returns an empty list.
- `src/evaluation/scripts/run_openeqa_stage2_full.py` with `scene_path_provider = data_root / "data" / "frames" / scene_id` (line 407) inherits the same broken assumption; use with `--mock` only.

**Production pilot does NOT use this loader for scene resolution.** It reads `data/open-eqa-v0.json` directly (`load_official_scannet_samples` at `src/agents/examples/openeqa_official_question_pilot.py:244-263`) and filters to `episode_history` values whose clip id exists under `data/OpenEQA/scannet/`. Frame resolution then happens through `ensure_runtime_scene`, §7.3.

`src/agents/adapters/openeqa_adapter.py:96` (`OpenEQAAdapter.get_scene_path`) independently remaps to `scene_data_root / sample.scene_id / "conceptgraph"`, which *is* a valid local path; this adapter is the entry point for any new Benchmark-Adapter-based code.

## 7.3 The runtime-scene overlay (`ensure_runtime_scene`)

Source: `src/agents/examples/openeqa_single_scene_pilot.py:179-217`.

```
data/OpenEQA/scannet/<clip>/conceptgraph/                 raw/
  bev/                                                    000000-rgb.png
  enriched_objects.json                                   000000-depth.png
  gsa_*/                                                  000000.txt
  indices/                                                ...
  pcd_saves/                                              
  traj.txt                                                

                         │ ensure_runtime_scene
                         ▼

tmp/openeqa_runtime_cache/<clip>/          ← or the pilot's --output-root/runtime_cache
  bev -> <orig>/bev                         (symlinks every child of conceptgraph except `results/`)
  enriched_objects.json -> <orig>/…
  gsa_detections_ram_withbg_allclasses -> …
  indices -> …
  pcd_saves -> …
  traj.txt -> …
  results/
    frame000000.jpg -> <orig>/raw/000000-rgb.png       (symlinks renaming raw to Stage-1-compatible names)
    depth000000.png -> <orig>/raw/000000-depth.png
    frame000001.jpg -> ...
    ...
```

Behaviour:

- The function never modifies `data/`; all writes go to `cache_root`.
- Symlinks for each child of `conceptgraph/` (except `results/` itself) are created or refreshed by `ensure_symlink` (`:135-149`).
- The `results/` overlay is rebuilt from scratch (`rebuild_results_overlay`, `:163-176`) whenever the count of `frame*.jpg` / `depth*.png` under `results/` doesn't match the count of `*-rgb.png` / `*-depth.png` under `raw/`, or when `force_rebuild_overlay=True`.
- Frame-name convention: `results/frameXXXXXX.jpg` + `results/depthXXXXXX.png` — this is the name pattern `KeyframeSelector._set_image_paths` expects, so the overlay is the adapter.
- `infer_stride` at `:152-160` reads the stride from `indices/visibility_index.pkl`'s `metadata.stride` field; pass `--stride 0` in pilots to force this inference.

**Parallelism caveat**: `ensure_runtime_scene` writes symlinks and is not internally thread-safe. The pilot acquires a per-clip lock via `_get_scene_lock(clip_id)` (`openeqa_official_question_pilot.py:47-55`) before calling it; any new entry point must replicate this, or the first few workers on the same clip will race on symlink creation.

## 7.4 `scene_info.json` provenance caveat

`scene_info.json` under `conceptgraph/` was generated on the ConceptGraph prep machine and may contain absolute source paths like `/home/ysh/...` which do **not** exist on the evaluation machine. These fields are *provenance only*. Never treat them as runnable paths.

## 7.5 Where each consumer reads from

| Consumer | Reads from |
|---|---|
| `OpenEQADataset.from_path` (legacy / mock) | `data/open-eqa-v0.json` (fine) + `data/frames/<episode>/` (broken locally) |
| `openeqa_official_question_pilot.load_official_scannet_samples` | `data/open-eqa-v0.json` directly, filter to `data/OpenEQA/scannet/` |
| `ensure_runtime_scene` | `data/OpenEQA/scannet/<clip>/conceptgraph/` + `/raw/` |
| `KeyframeSelector.from_scene_path` | the runtime overlay returned by `ensure_runtime_scene` |
| `Stage1BackendCallbacks` | the same `KeyframeSelector` instance, via the pilot's `selector` argument to `run_stage2` |
| `openeqa_official_eval` | `external/open-eqa/openeqa/evaluation/llm_match.py` + the `openeqa-v0.json` items already loaded by the pilot |

## 7.6 Things a new agent will trip over

| Pitfall | What to do |
|---|---|
| Passing `--data-root data/OpenEQA/scannet` to the pilot thinking it is the OpenEQA root | The pilot's `--data-root` is the scene-root; `--json-path` is the question-root. Defaults are correct; only override if you know why. |
| Running `OpenEQADataset.from_path("data/")` and iterating `sample.load_frames()` | Frames list will be empty; you'll silently get zero-image predictions. Use the pilot instead. |
| Deleting `tmp/openeqa_runtime_cache/<clip>/` between runs | Cheap; overlays rebuild in seconds. Do this if you suspect the overlay is stale or corrupted. |
| Editing files under `conceptgraph/` while a run is live | The runtime overlay is symlinked, so your edits are visible immediately — usually undesired. |
| Expecting `data/benchmark/` or `data/benchmarks/` from older docs | They don't exist on this checkout; ignore those paths. |

Cross-ref: `02_architecture.md §2.1` for the two-stage diagram; `09_gotchas.md` for the broader foot-gun list.
