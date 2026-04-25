# EmbodiedScan 3D BBox Proposal Feasibility Design

Date: 2026-04-24

## Goal

Evaluate whether two proposal-generation routes can produce useful 3D bounding boxes from EmbodiedScan RGB-D observations:

1. A 2D detection / segmentation route that lifts 2D proposals into 3D.
2. A ScanNetV2 SOTA 3D detector route that runs on point clouds reconstructed from one or more RGB-D observations, plus higher-quality ScanNet crops.

This study is proposal-only. It does not evaluate downstream VLM reasoning, language grounding, object selection, or agent behavior. For each ground-truth target instance, each method produces a proposal set and receives credit for the best spatial overlap with the GT box.

## Non-Goals

- Do not use referring-expression text to select a predicted box.
- Do not require detector class labels to match GT categories.
- Do not optimize prompts, Stage 2 tools, or agent behavior.
- Do not use GT target boxes to crop detector inputs. GT boxes are only for final scoring.

## Repository Context

Relevant existing assets:

- `src/benchmarks/embodiedscan_loader.py` loads EmbodiedScan VG samples and GT 9-DOF boxes.
- `src/benchmarks/embodiedscan_eval.py` computes oriented 3D IoU for EmbodiedScan boxes.
- `bashes/embodiedscan/*` defines a ConceptGraph-style 2D detection -> 3D object map pipeline.
- `src/agents/examples/embodiedscan_vg_pilot.py` already wires EmbodiedScan VG through Stage 1/2, but this study should avoid Stage 2 and evaluate proposals directly.

Current local data state:

- `data/embodiedscan/` contains train/val/test annotation files and `embodiedscan_infos_*.pkl`.
- `data/embodiedscan/scannet/` has 246 scene directories.
- 243 scenes have `conceptgraph/` and visibility indices.
- Prepared ConceptGraph object maps exist under `conceptgraph/pcd_saves/`.
- `enriched_objects.json` is absent for these scenes. The feasibility evaluator should read object-map PKL files directly instead of depending on `KeyframeSelector`, whose current path hard-requires enrichment.

Operational constraints:

- On Linux, run Python inside the `conceptgraph` conda environment.
- Long-running evaluation, detector inference, or batch preprocessing must run inside tmux.
- Do not use GPU 1.

## Model Selection

The primary 3D detector target is ScanNetV2 closed-set SOTA.

Primary target:

- **DEST based on V-DETR**, because public ScanNetV2 benchmark snapshots list it at `mAP@0.5=67.9` and `mAP@0.25=78.8`. The DEST paper states that its V-DETR-based method sets a new SOTA on ScanNetV2 and SUN RGB-D.

Known reproducibility risk:

- The public DEST3D repository currently releases the core ISSM module and a GroupFree3D-based DEST framework, while the README lists training/evaluation pipeline and V-DETR-based DEST as planned releases. If the V-DETR-based SOTA checkpoint and inference path are unavailable, mark this arm as `model_blocked` rather than silently substituting it.

Fallback models:

- **UniDet3D**, because it provides an available implementation and checkpoint path and reports strong ScanNetV2 results.
- **V-DETR**, because DEST-VDETR builds on it and it is a strong reproducible ScanNetV2 detector.

References:

- DEST paper: https://arxiv.org/abs/2503.14493
- DEST repository: https://github.com/OpenSpaceAI/DEST3D
- ScanNetV2 detection benchmark snapshot: https://opencodepapers-b7572d.gitlab.io/benchmarks/3d-object-detection-on-scannetv2.html
- UniDet3D paper: https://arxiv.org/abs/2409.04234
- UniDet3D repository: https://github.com/filaPro/unidet3d

## Experimental Matrix

### 2D Route

`2D-SF`: Single-frame RGB-D + 2D mask or bbox backprojection. Fit a 3D bbox from the visible depth points.

`2D-MV`: Multi-frame RGB-D + poses. Fuse lifted 2D proposal points across a fixed local frame window, then fit a 3D bbox.

`2D-CG`: Existing ConceptGraph object map. Treat ConceptGraph objects as a stronger multi-frame 2D-driven proposal source. Compute proposals directly from object point clouds and stored `bbox_np` where available.

### 3D Detector Route

`3D-SF-Recon`: Single-frame RGB-D backprojected point cloud sent to the ScanNetV2 detector. This tests a difficult out-of-distribution input.

`3D-MV-Recon`: Multi-frame RGB-D reconstruction from the same sampled frame window sent to the detector. This tests realistic embodied observation quality.

`3D-ScanNet-Crop`: Use the EmbodiedScan `scan_id` to locate the original ScanNetV2 scene. Crop the high-quality ScanNet mesh or point cloud using camera frustums or trajectory coverage from the sampled observations, then run the detector. This isolates reconstruction noise from detector capability.

`3D-ScanNet-Full`: Run the detector on the full original ScanNetV2 scene. This is a ceiling diagnostic, not a fair embodied-observation setting.

Cropping rule: `3D-ScanNet-Crop` may use camera pose, intrinsics, frame frustums, depth coverage, and trajectory extent. It must not use GT target boxes or referring expressions.

## Data Split And Sampling

Dataset: EmbodiedScan val split, `scannet` source subset.

Primary evaluation unit:

```text
(scan_id, target_id, gt_bbox_3d)
```

This removes duplicate weighting from multiple referring expressions that target the same instance.

Recommended scales:

- **Smoke**: 5 scenes, about 50 unique target instances. Validate coordinate systems, bbox conversion, and non-degenerate output.
- **Pilot**: 30 scenes with broad category and size coverage. Compare trends and diagnose failure modes.
- **Full feasible split**: all locally usable EmbodiedScan ScanNet val prepared scenes, deduplicated by `(scan_id, target_id)`.

Observation sampling:

- For single-frame methods, select each target's best visible frame. Because this is a proposal upper-bound study, the selector may use oracle visibility derived from GT projection, detection visibility, or depth validity.
- For multi-frame methods, center a fixed temporal window around each target's best visible frame. Main result uses 10 frames; run 5 and 20 frame windows as sensitivity checks when practical.
- Use the same target-conditioned selected frames and camera frustums for `3D-MV-Recon` and `3D-ScanNet-Crop`.
- Record the observation policy for every evaluation record. `3D-ScanNet-Full` and `2D-CG` may use scene-level proposals, but single-frame, multi-frame, and ScanNet-crop arms are target-observation conditioned.

## Proposal Contract

Each proposal generator writes one file per evaluation record and input condition. For target-conditioned arms, the file includes `target_id` and the selected frame ids. For scene-level arms such as `2D-CG` and `3D-ScanNet-Full`, `target_id` can be `null` and the same scene proposal set can be reused across targets.

```json
{
  "scene_id": "scene0415_00",
  "scan_id": "scannet/scene0415_00",
  "target_id": 37,
  "method": "3d-mv-recon-dest-vdetr",
  "input_condition": "multi_frame_recon_10",
  "observation": {
    "policy": "target_best_visible_centered_window",
    "frame_ids": [120, 122, 124, 126, 128, 130, 132, 134, 136, 138]
  },
  "proposals": [
    {
      "bbox_3d": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
      "score": 0.87,
      "source": "detector",
      "metadata": {}
    }
  ]
}
```

All `bbox_3d` values must use EmbodiedScan 9-DOF format:

```text
[cx, cy, cz, dx, dy, dz, alpha, beta, gamma]
```

For 2D-lifted methods:

- Produce AABB proposals with zero Euler angles as the mandatory baseline.
- If robust, also produce OBB proposals using stored `bbox_np`, PCA, or Open3D OBB fitting.
- Record both AABB and OBB scores; the main table should identify which geometry variant is used.

For detector methods:

- Convert detector-native boxes into EmbodiedScan 9-DOF before scoring.
- Preserve raw detector score and class label in metadata, but do not use class labels for the main upper-bound metric.

## Metrics

Main metric: class-agnostic proposal upper bound.

For each unique GT target instance and method condition, take the maximum oriented 3D IoU over proposals produced for that evaluation record. Scene-level proposal sets can be reused for all target instances in that scene.

```text
best_iou(target) = max_i IoU(proposal_i, gt_bbox_3d)
```

Report:

- `mean_best_iou`
- `median_best_iou`
- `Acc@0.25`
- `Acc@0.5`
- proposal count per scene
- non-degenerate box ratio
- runtime per scene

Use `src/benchmarks/embodiedscan_eval.py` for EmbodiedScan oriented IoU.

## Failure Taxonomy

Each failed or low-quality case should be tagged with one primary failure reason:

- `no_proposal`: method produced no boxes for the scene or input condition.
- `coord_mismatch`: proposals are visibly in the wrong coordinate frame.
- `degenerate_box`: box has NaN, near-zero size, or implausibly large extent.
- `visibility_limited`: proposal captures only visible surfaces and underestimates object extent.
- `overmerge`: multiple objects are merged into one large proposal.
- `fragmentation`: one object is split across several small proposals.
- `detector_ood`: detector produces invalid or low-confidence outputs on reconstructed partial point clouds.
- `model_blocked`: target SOTA detector cannot be run because code, weights, or environment are unavailable.

## Interpretation Rules

- If `2D-MV` or `2D-CG` gets reasonable `Acc@0.25` but low `Acc@0.5`, the 2D route is viable for coarse grounding but weak for precise 3D box recovery.
- If `3D-ScanNet-Crop` is much better than `3D-MV-Recon`, the bottleneck is RGB-D reconstruction quality or observation coverage.
- If `3D-ScanNet-Full` is high but `3D-ScanNet-Crop` is low, the bottleneck is partial observability.
- If reconstructed-point-cloud detector arms are low while ScanNet crop/full arms are high, the SOTA detector is sensitive to local or sparse reconstructed point-cloud distribution.
- If ScanNet crop/full arms are also low, the likely issue is detector coverage of EmbodiedScan targets, coordinate conversion, or bbox convention mismatch.

## Execution Plan Boundary

The implementation plan should start with a smoke pipeline before any large run:

1. Build the unique-target index for one or two scenes.
2. Verify GT bbox self-IoU equals 1.0 after serialization and conversion.
3. Generate and visualize proposals from `2D-CG`.
4. Build a single-frame and multi-frame reconstructed point cloud.
5. Attempt detector inference on one high-quality full ScanNet scene, then on cropped and reconstructed inputs.
6. Only after the smoke checks pass, launch pilot/full jobs in tmux.

This design intentionally ends before implementation details such as script names, CLI flags, detector environment setup, and exact output directory layout. Those belong in the follow-up implementation plan.

## Implementation Status (2026-04-25)

This section records what has actually been built and run on `feat/explore_3dbbox`, where the implementation diverged from the design above, and where it remains incomplete. The authoritative results document is `docs/10_experiment_log/embodiedscan_3d_bbox_feasibility_report/REPORT.md`. The mirrored Lark write-up lives at `https://my.feishu.cn/docx/TzDDd7HYNoEgsgxkE8vclm7dnqb`.

### Code state

The feasibility module is `src/benchmarks/embodiedscan_bbox_feasibility/`. Approximately 39 commits landed on 2026-04-24, plus uncommitted 2026-04-25 changes that wire the V-DETR detector path and the scene-level full-scan materialization.

Modules:

- `models.py` — Pydantic schemas for targets, observations, detector inputs, proposals, evaluation records. Now carries `axis_align_matrix` and `visible_frame_ids` on `EmbodiedScanTarget`.
- `targets.py` — unique `(scan_id, target_id)` index. Pulls `axis_align_matrix` and per-target `visible_frame_ids` from `scene_info`.
- `geometry.py` — AABB fitting, degenerate-box guard, point cloud transform, depth backprojection helpers.
- `observations.py` — target-conditioned frame-window selection.
- `backproject.py` — RGB-D-to-points lifting.
- `conceptgraph.py` — `2D-CG` proposals from prepared `pcd_saves/*.pkl`. Now emits two geometry variants per object (`pcd_aabb` from `pcd_np`, `bbox_np_aabb` from `bbox_np`) and applies `axis_align_matrix` when supplied.
- `pointcloud_inputs.py` — materializes detector inputs for `single_frame_recon`, `multi_frame_recon`, `scannet_pose_crop`, `scannet_full`. Applies axis-alignment so detector PLY files live in `embodiedscan_aligned` frame.
- `detector_runner.py` / `detector_adapter.py` — generic subprocess runner and proposal JSON adapter.
- `vdetr.py` — V-DETR command template, `camera_corners_to_depth_corners`, ScanNet 18-class label table, proposal JSON writer.
- `evaluator.py` — class-agnostic best-IoU + Acc@0.25/0.50 over EmbodiedScan oriented IoU.
- `cli.py` — subcommands: `smoke`, `prepare-inputs`, `run-detector`, `evaluate-records`, plus batch-selection flags (`--target-categories`, `--require-visible-frames`, `--max-scenes`, `--max-targets-per-scene`, `--scene-level-full`).

External integration:

- `scripts/vdetr_export_predictions.py` runs V-DETR from `external/V-DETR` against one prepared PLY and writes proposals JSON in EmbodiedScan 9-DOF.
- Detector checkpoint: `external/V-DETR/checkpoints/scannet_540ep.pth`. Defaults: 40000 points, conf 0.05, top-k 256.

### Coverage vs. the design's experimental matrix

| Designed arm | Implementation | Status |
|---|---|---|
| `2D-SF` (single-frame mask backprojection + fit) | none | not implemented |
| `2D-MV` (multi-frame fused 2D mask + fit) | none | not implemented |
| `2D-CG` (ConceptGraph object map) | `conceptgraph.py` + `2D-CG full-local` | run on 2096 deduplicated targets |
| `3D-SF-Recon` | `single_frame_recon` | run on batch30 (30 targets) |
| `3D-MV-Recon` | `multi_frame_recon` | run on batch30 with **5 frames**, not the design's 10-frame main setting |
| `3D-ScanNet-Crop` | `scannet_pose_crop` | run on batch30, but uses **pose-bounds + padding**, not the designed image-frustum crop |
| `3D-ScanNet-Full` | `scannet_full` (scene-level cache) | run on batch30 (10 scenes) |

### Detector substitution

The design's primary target is DEST-VDETR. In practice the V-DETR base detector is what was actually run, which corresponds to the design's documented fallback. DEST-VDETR was not pursued; the design's `model_blocked` tag would apply if it were tried but is not currently surfaced in any record.

### Implementation findings the design did not anticipate

These are load-bearing for any follow-up plan and should override the design where they conflict:

- **V-DETR coordinate frame conversion**: `outputs.box_corners` from V-DETR are in camera frame; EmbodiedScan oriented IoU expects depth/aligned frame. The required permutation is `camera [x, y, z] -> depth [x, z, -y]`, implemented as `camera_corners_to_depth_corners`. Before this fix, V-DETR mean IoU was near zero despite visually plausible boxes; after the fix, the 3-target full-scene pilot reached mean IoU `0.8613`.
- **Axis alignment as part of input prep**: detector inputs and `2D-CG` proposals must be transformed by `scene_info.axis_align_matrix` before scoring. Records persist `axis_align_applied` and `coordinate_frame` flags in `metadata` so downstream evaluation can verify this was honored.
- **Scene-level reuse for `scannet_full`**: `scannet_full` is materialized once per scene (`<scene_id>_scene.ply`) rather than once per target. Proposals are reused across all targets in the same scene.
- **Bottleneck is detector training distribution, not point-cloud cleanliness**: `scannet_pose_crop` uses clean ScanNet mesh points but still scores `mean IoU 0.18 / Acc@0.5 0.20`, far below `scannet_full` at `0.56 / 0.67`. Higher-quality local points are not sufficient; the detector relies on whole-room context.
- **New failure-mode tags observed**: `category_ood` (e.g. `picture` fails even at full-scene) and `thin_flat_geometry` (e.g. `curtain` reaches Acc@0.25 1.00 but Acc@0.5 0.17). Add these to the failure taxonomy alongside the originally listed tags.

### Headline results (batch30, V-DETR; 2D-CG run on full-local 2096 targets)

| Route / condition | Targets | Mean IoU | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|---:|
| `2D-CG` full-local | 2096 | 0.343 | 0.623 | 0.274 |
| V-DETR `single_frame_recon` | 30 | 0.124 | 0.267 | 0.067 |
| V-DETR `multi_frame_recon` (5 frames) | 30 | 0.131 | 0.233 | 0.133 |
| V-DETR `scannet_pose_crop` | 30 | 0.180 | 0.267 | 0.200 |
| V-DETR `scannet_full` | 30 | 0.558 | 0.933 | 0.667 |

### Gaps relative to the design

- The image-frustum variant of `3D-ScanNet-Crop` is not implemented; only pose-bounds crop with padding is.
- Multi-frame main result is 5 frames, not 10. The 5/10/20-frame sensitivity sweep is open.
- 2D mask-driven routes (`2D-SF`, `2D-MV`) are not implemented; the only 2D route is `2D-CG`.
- DEST-VDETR is not run. V-DETR base only.
- No `model_blocked` records were written; substitution to V-DETR happened upstream of record generation.

### Recommended follow-ups

These come from REPORT.md and the Lark write-up, and supersede the design's "next experiments" list where they overlap:

1. Larger full-scene V-DETR sweep at 50–100 scenes to stabilize the batch30 conclusion.
2. Exact image-frustum crop variant of `3D-ScanNet-Crop` (full image, 2D-bbox, 2D-mask) with V-DETR and with direct geometric fitting.
3. Mask-guided geometric fitting as a single-image fallback that bypasses V-DETR.
4. RGB-D fusion ablation across 1 / 5 / 10 / 20 / all visible frames, optionally with TSDF or voxel fusion, to separate "not enough frames" from "detector unsuited to partial reconstructions."
5. Move proposal selection into Stage 2 (2D evidence + visibility + language) over the cached scene-level V-DETR proposals — this is the recommended system pipeline; it is out of scope for this proposal-only study.
