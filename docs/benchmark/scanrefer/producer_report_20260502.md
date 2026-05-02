# ScanRefer 11-scene GT-CG producer report — 2026-05-02

Linux-side producer run for the 11 ScanRefer-only val scenes that were not
covered by the NR3D-130 leaderboard build. Brings local ScanRefer-val
coverage from 130/141 (94.1% utt) to 141/141 (100%) so v1 can ship as a
full-canonical-val number, not a 130-subset caveat.

Source handoff: `docs/handoff_2026-05-02_scanrefer_11scenes_linux.md` (tip
`594013e`).

## Run identification

| Item | Value |
|---|---|
| Branch | `feat/scanrefer-vg-benchmark` |
| Tip commit | `594013e` (handoff doc) |
| Host | Linux server, `/home/ysh/codecase/3DVLMReasoning` |
| Conda env | `conceptgraph` (Python 3.10) |
| GPU | `7` (only fully-free GPU; 24 GB free, 0% util at launch) |
| Producer wall time | extract 453.0 s + build 241.2 s = 694.2 s (~12 min) |
| Raw artifact dirs | `data/nr3d/scannet/<scene>/{raw,conceptgraph}/` |
| Aux dirs | `data/nr3d/scannet_aux/<scene>/` (see Step 0 below) |
| Scratch logs | `tmp/scanrefer_handoff/{extract.log,producer.log}` |

## Step 0 — acquire ScanNet aux files (handoff omission)

The handoff only listed `.sens`, `.aggregation.json`, and `_vh_clean_2.ply`
under `/home/ysh/Datasets/ScanNet/scans/<scene>/` as the inputs to verify.
On this server, **only `.sens` and `_vh_clean_2.ply` live in the scans
tree** (verified by listing the existing NR3D-130 scene `scene0011_00`
which the producer already successfully built — same shape). The
`{aggregation.json, segs.json, axis-align .txt}` triple lives under
`data/nr3d/scannet_aux/<scene>/`, sourced from
`zahidpichen/scannet-dataset` on Hugging Face (no auth, ~128 MiB total
for the original 130-scene set; matches `MEMORY.md::project_nr3d_gt_conceptgraph_layout`).

The 11 ScanRefer-only scenes had no scannet_aux dirs yet. Downloaded
3 files × 11 scenes = 33 files via `huggingface_hub.hf_hub_download`,
~5.5 MB total, 34.3 s elapsed. Downloaded suffixes:

- `<scene>.txt` (axisAlignment metadata)
- `<scene>.aggregation.json` (segGroups)
- `<scene>_vh_clean_2.0.010000.segs.json` (segIndices)

Then ran `verify-aux` (parses all 3 + checks `.sens` + `.ply` exist on
`scannet-root`); all 11 passed.

## Step 1 — extract `.sens` frames into raw layout

Handoff did **not** mention this stage. The producer's `build-scenes`
expects `data/nr3d/scannet/<scene>/raw/` already populated (RGB + depth
+ pose at stride=10). That stage is `scripts/extract_nr3d_test_frames.py`.

```bash
python scripts/extract_nr3d_test_frames.py \
    --scenes scene0088_00 scene0277_00 scene0304_00 scene0316_00 scene0342_00 \
             scene0354_00 scene0382_00 scene0406_00 scene0414_00 scene0575_00 scene0660_00 \
    --workers 8 \
    --invalid-pose-policy drop
```

Defaults used: `--nr3d-root data/nr3d`, `--scannet-root /home/ysh/Datasets/ScanNet`,
`--stride 10`. Drops stride-selected frames whose pose contains NaN/Inf
(per `data/nr3d/scannet/.frame_extraction_log.json`).

Aggregate result: `1351` kept frames across 11 scenes, `106` dropped for
NaN/Inf pose, `1457` raw-stride total, 453.0 s elapsed (8 workers).

Per-scene `dropped_bad_pose` worth flagging:

- `scene0414_00`: kept=77, dropped=40 of 117 (34.2%)
- `scene0316_00`: kept=77, dropped=24 of 101 (23.8%)
- `scene0354_00`: kept=113, dropped=22 of 135 (16.3%)
- `scene0660_00`: kept=104, dropped=1 of 105 (~0.95%)
- All other scenes: 0 dropped.

All 11 scenes still satisfy the producer's `>= 50 kept frames` floor
(`raw_info.kept_frame_ids` from `raw/scene_info.json`).

## Step 2 — build conceptgraph GT objects

```bash
export CUDA_VISIBLE_DEVICES=7

python -m src.scripts.nr3d_gt_conceptgraph build-scenes \
    --scenes scene0088_00 scene0277_00 scene0304_00 scene0316_00 scene0342_00 \
             scene0354_00 scene0382_00 scene0406_00 scene0414_00 scene0575_00 scene0660_00 \
    --scannet-root /home/ysh/Datasets/ScanNet \
    --nr3d-root data/nr3d \
    --gpu 0 \
    --report tmp/scanrefer_handoff/producer_report.md
```

Path-correction vs handoff: handoff said `--scannet-root
/home/ysh/Datasets/ScanNet/scans` and `--nr3d-root data/nr3d/scannet`,
both one level deep beyond what the producer actually expects. The
producer joins `<scannet-root>/scans/<scene>/` and `<nr3d-root>/scannet_aux/<scene>/`
internally; the corrected paths above match `DEFAULT_SCANNET_ROOT` /
`DEFAULT_NR3D_ROOT`.

Note: `--report tmp/.../producer_report.md` is currently a no-op for
`build-scenes` — `write_report` is defined but never called. This file
is the de facto report instead.

## Step 3 — verify outputs (handoff snippet)

| scene | num_objects | num_rgb_frames | num_visibility_mappings | build_s |
|---|---:|---:|---:|---:|
| scene0088_00 | 36 | 110 | 1493 | 33.0 |
| scene0277_00 | 17 | 107 | 715 | 16.4 |
| scene0304_00 | 16 | 178 | 1017 | 20.7 |
| scene0316_00 | 16 | 77 | 462 | 13.9 |
| scene0342_00 | 26 | 62 | 663 | 17.6 |
| scene0354_00 | 19 | 113 | 931 | 12.7 |
| scene0382_00 | 18 | 87 | 632 | 15.9 |
| scene0406_00 | 30 | 142 | 1189 | 26.6 |
| scene0414_00 | 40 | 77 | 857 | 28.2 |
| scene0575_00 | 24 | 294 | 2574 | 41.9 |
| scene0660_00 | 10 | 104 | 710 | 14.3 |
| **total** | **252** | **1351** | **11243** | **241.2** |

All 11 scenes pass the handoff's verification snippet
(`pkl_path / vis_path / scene_info.json / raw/scene_info.json` exist;
`>= 5 objects`; `bbox_np.shape == (8, 3)`; `>= 50 kept frames`):

```
all 11 scenes verified OK
```

Caveats on scene0660_00: handoff predicted "museum-style scan, 100+
instances". Actual GT count is `10` — smallest of the 11. Still ≥ 5,
so the producer accepts. No action needed; flag if downstream depends
on richer instance density for this scene.

## Step 4 — Mac-side rsync (not executed here)

Per handoff §"How Mac will pull the artifacts" — Mac will pull the 11
new scene dirs under `data/nr3d/scannet/` via rsync. No git push of
artifact data; only this report doc is committed.

## Definition-of-done checklist

- [x] All 11 scenes produced `data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz`
- [x] All 11 scenes produced `data/nr3d/scannet/<scene>/conceptgraph/indices/visibility_index.pkl`
- [x] All 11 scenes produced `data/nr3d/scannet/<scene>/raw/scene_info.json` with non-empty `kept_frame_ids`
- [x] Verification snippet prints `all 11 scenes verified OK`
- [x] Producer report committed under `docs/benchmark/scanrefer/producer_report_20260502.md`

## Things to fix in the producer / handoff before next runs

1. **Handoff missing aux-acquisition step.** The handoff implicitly
   assumed `scannet_aux/<scene>/` would exist. For the 11 new scenes,
   it didn't. Future producer handoffs that introduce new scenes
   should explicitly call out `huggingface_hub.hf_hub_download` from
   `zahidpichen/scannet-dataset` for the 3 aux suffixes.
2. **Handoff missing frame-extraction step.** `extract_nr3d_test_frames.py`
   has to run before `build-scenes`. Producer error message is clear
   (`raw directory not found`) but the handoff promised "no script
   changes needed; just feed it new scene IDs", which suggests build is
   sufficient on its own.
3. **`--report` flag in `build-scenes` is a no-op.** Either wire
   `write_report(...)` into the build path or drop the flag.
4. **Handoff `--scannet-root`/`--nr3d-root` are off by one path
   segment** (described above). Either accept both forms in the
   producer or correct the handoff template.

These are notes for the Mac-side maintainer; nothing here blocked the
run (workarounds applied during this session).
