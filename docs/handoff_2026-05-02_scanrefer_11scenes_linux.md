# ScanRefer 11-Scene Linux Handoff — 2026-05-02

> **Audience:** the Linux-side Claude Code agent on the host that owns
> `/home/ysh/Datasets/ScanNet/scans/` (same machine that produced the
> NR3D Phase 8 GT-CG packages a few days ago).
>
> **TL;DR:** Run the existing Phase 8 GT-CG producer on **11 ScanRefer-only
> ScanNet val scenes**, sanity-check outputs, then rsync them back to the
> Mac. ~10 min of GPU work, ~5-6 GB of artifacts. No code changes needed.

## Why this work exists

We just shipped the NR3D leaderboard track on **130** scenes. ScanRefer
val is **141** scenes — exactly NR3D test ∪ 11 extra scenes. Those 11
scenes don't yet have any local infrastructure on the Mac:

```
data/ScanNet/scans/scene0088_00/   ← empty
data/ScanNet/scans/scene0277_00/   ← empty
... (etc)
```

The Mac side currently has 94.1% utt coverage of canonical ScanRefer val
(8944 / 9508 utts on the 130 NR3D-overlap scenes). After you finish, we'll
have **141/141 = 100%** coverage so v1 can ship as a full-canonical-val
ScanRefer number, not a 130-subset caveat.

## What's already in place (don't redo)

- Branch `feat/scanrefer-vg-benchmark` is at tip `542342c` upstream
  (push from earlier today). Pull this before running anything.
- Mac side has unzipped `data/scanrefer/raw/ScanRefer_filtered_*.json`
  and `data/scanrefer/Mask3d/scannet200/*.npz` (all 312 ScanNet val scenes
  covered, including these 11). You don't need to touch any of that.
- The producer script `src/scripts/nr3d_gt_conceptgraph.py` already
  handles GT injection — same script you used for the original 130 scenes.
  No script changes needed; just feed it new scene IDs.

## The 11 scenes

```
scene0088_00
scene0277_00
scene0304_00
scene0316_00
scene0342_00
scene0354_00
scene0382_00
scene0406_00
scene0414_00
scene0575_00
scene0660_00
```

Verify each one exists in your ScanNet root before running:

```bash
for sc in scene0088_00 scene0277_00 scene0304_00 scene0316_00 scene0342_00 \
          scene0354_00 scene0382_00 scene0406_00 scene0414_00 scene0575_00 scene0660_00; do
    test -f /home/ysh/Datasets/ScanNet/scans/$sc/$sc.sens && \
    test -f /home/ysh/Datasets/ScanNet/scans/$sc/${sc}.aggregation.json && \
    test -f /home/ysh/Datasets/ScanNet/scans/$sc/${sc}_vh_clean_2.ply && \
    echo "OK $sc" || echo "MISSING $sc — please source ScanNet for this scene"
done
```

If any line says `MISSING`, stop and report back — don't try to download
ScanNet under the hood. We need to know which asset is missing so we can
decide.

## Setup

```bash
# Activate conda (NEVER recreate .venv on Linux per CLAUDE.md)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate conceptgraph

# Pull latest
cd ~/codecase/3DVLMReasoning
git fetch origin
git checkout feat/scanrefer-vg-benchmark
git pull origin feat/scanrefer-vg-benchmark

# Sanity check — should print scenes that exist with .sens
which python                                          # → conda env's python
python -c "import torch, open_clip, open3d; print('imports OK')"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader   # avoid GPU 1 (broken)
```

## Run the producer

The producer has three subcommands (`verify-aux | build-scenes |
verify-outputs`). For 11 fresh scenes you only need `build-scenes`.

```bash
# Pick a healthy GPU (NOT 1 — broken on this server, see CLAUDE.md)
export CUDA_VISIBLE_DEVICES=0

mkdir -p tmp/scanrefer_handoff

python -m src.scripts.nr3d_gt_conceptgraph build-scenes \
    --scenes scene0088_00 scene0277_00 scene0304_00 scene0316_00 scene0342_00 \
             scene0354_00 scene0382_00 scene0406_00 scene0414_00 scene0575_00 scene0660_00 \
    --scannet-root /home/ysh/Datasets/ScanNet/scans \
    --nr3d-root data/nr3d/scannet \
    --gpu 0 \
    --report tmp/scanrefer_handoff/producer_report.md \
    2>&1 | tee tmp/scanrefer_handoff/producer.log
```

**Notes on the args:**

- `--nr3d-root data/nr3d/scannet` is intentional — the producer's
  output layout matches NR3D's. Since these 11 scenes don't overlap with
  NR3D test, we can co-locate them under `data/nr3d/scannet/` without
  collision. The Mac side will rsync them back into the same tree.
- `--gpu 0` — change to `2/3/4/5/6/7` if 0 is busy; **NEVER 1** (broken).
- The `--report` path is just for the producer's summary doc; not required
  by downstream code.

**Expected wall time:** ~30-60s per scene (SensReader frame extraction
dominates; CLIP feature extraction comes second). Total ~10 min for 11
scenes on a healthy GPU.

**Expected disk:** ~500 MB of extracted RGB+depth frames per scene + ~10
MB of conceptgraph pkl/json per scene. Total ~5-6 GB.

## Sanity-check outputs

After the producer finishes, run this verification per scene. Every
assertion should pass. If any fails, stop and report back.

```bash
python -c "
import gzip, pickle, json
from pathlib import Path

scenes = ['scene0088_00','scene0277_00','scene0304_00','scene0316_00','scene0342_00',
          'scene0354_00','scene0382_00','scene0406_00','scene0414_00','scene0575_00','scene0660_00']

for sc in scenes:
    base = Path(f'data/nr3d/scannet/{sc}')
    # Required files
    pkl_path = base / 'conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz'
    vis_path = base / 'conceptgraph/indices/visibility_index.pkl'
    info_path = base / 'conceptgraph/scene_info.json'
    raw_info_path = base / 'raw/scene_info.json'
    assert pkl_path.exists(), f'{sc}: missing pkl'
    assert vis_path.exists(), f'{sc}: missing visibility_index'
    assert info_path.exists(), f'{sc}: missing scene_info.json'
    assert raw_info_path.exists(), f'{sc}: missing raw scene_info.json'

    # PKL content
    with gzip.open(pkl_path, 'rb') as f:
        payload = pickle.load(f)
    objs = payload['objects']
    assert len(objs) >= 5, f'{sc}: too few objects ({len(objs)})'
    assert objs[0]['bbox_np'].shape == (8, 3), f'{sc}: bbox_np shape {objs[0][\"bbox_np\"].shape}'

    # Raw frames count
    raw_info = json.loads(raw_info_path.read_text())
    n_kept = len(raw_info['kept_frame_ids'])
    assert n_kept >= 50, f'{sc}: too few kept frames ({n_kept})'

    info = json.loads(info_path.read_text())
    print(f'{sc}: num_objects={info[\"num_objects\"]} '
          f'num_rgb_frames={info[\"num_rgb_frames\"]} '
          f'num_visibility_mappings={info[\"num_visibility_mappings\"]}')
print('all 11 scenes verified OK')
"
```

Expected output: 11 lines like
`scene0088_00: num_objects=52 num_rgb_frames=148 num_visibility_mappings=842`
followed by `all 11 scenes verified OK`.

## How Mac will pull the artifacts

Don't push artifact data to git (data/ is gitignored). The Mac side will
pull via rsync. **You don't need to do anything proactively** — once you
report DONE, the Mac side will run:

```bash
# Mac side, you don't need to run this
rsync -av --include='scene0088_00/***' --include='scene0277_00/***' \
          --include='scene0304_00/***' --include='scene0316_00/***' \
          --include='scene0342_00/***' --include='scene0354_00/***' \
          --include='scene0382_00/***' --include='scene0406_00/***' \
          --include='scene0414_00/***' --include='scene0575_00/***' \
          --include='scene0660_00/***' --exclude='*' \
          <linux-host>:~/codecase/3DVLMReasoning/data/nr3d/scannet/ \
          data/nr3d/scannet/
```

If you want to also push the producer report doc to remote git (so the
Mac side picks it up via `git pull`), commit `tmp/scanrefer_handoff/producer_report.md`
under a new path like `docs/benchmark/scanrefer/producer_report_20260502.md`
and push:

```bash
mkdir -p docs/benchmark/scanrefer
cp tmp/scanrefer_handoff/producer_report.md docs/benchmark/scanrefer/producer_report_20260502.md
git add docs/benchmark/scanrefer/producer_report_20260502.md
git commit -m "docs(scanrefer): producer report for 11 ScanRefer-only scenes"
git push origin feat/scanrefer-vg-benchmark
```

This is optional — the Mac side mainly needs the artifact data, not the
report.

## Gotchas (read before running)

1. **GPU 1 is broken on this server** — `CUDA_VISIBLE_DEVICES=1` causes
   CUDA initialization failures. Use 0 or 2-7. Verify with `nvidia-smi`
   before launching.

2. **Conda env, NOT .venv** — per CLAUDE.md `Package Management (Linux)`,
   bare `python` resolves to conda's python. Do NOT recreate `.venv`. The
   conda `conceptgraph` env has all needed deps (torch, open3d, open_clip,
   SAM, Florence-2). If `import open_clip` or `import open3d` fails, the
   env is broken — stop and report.

3. **Strict no-fallback rule** — if SensReader / ScanNet aggregation
   parsing / Open3D mesh loading fails for any scene, raise immediately.
   Do not silently skip the scene or substitute placeholder data. Per
   project CLAUDE.md `Strict No-Fallback Rule (MANDATORY)`.

4. **One scene is plausibly large** — `scene0660_00` is a museum-style
   scan and may have 100+ instances. Producer should handle it but might
   take 90s instead of 30s.

5. **Don't touch `data/nr3d/scannet/<existing>/` directories** — only
   write to the 11 new scene dirs. Existing 130 scenes are read-only here.

6. **tmux is your friend** — if running in a fresh ssh, kick this off
   inside tmux per CLAUDE.md `Long-Running Tasks: tmux (MANDATORY)`:

   ```bash
   tmux new-session -d -s scanrefer-11 \
     "source ~/miniconda3/etc/profile.d/conda.sh && conda activate conceptgraph && \
      cd ~/codecase/3DVLMReasoning && \
      python -m src.scripts.nr3d_gt_conceptgraph build-scenes --scenes ... 2>&1 | tee /tmp/scanrefer-11.log"
   tmux capture-pane -t scanrefer-11 -p -S -50  # peek progress
   ```

## Definition of done

You can report DONE when:

- [ ] All 11 scenes produced `data/nr3d/scannet/<scene>/conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz`
- [ ] All 11 scenes produced `data/nr3d/scannet/<scene>/conceptgraph/indices/visibility_index.pkl`
- [ ] All 11 scenes produced `data/nr3d/scannet/<scene>/raw/scene_info.json` with `kept_frame_ids` non-empty
- [ ] The verification python snippet above prints `all 11 scenes verified OK`
- [ ] (optional) producer report committed to `docs/benchmark/scanrefer/producer_report_20260502.md`

Report back the per-scene summary (the 11 lines from the verification
snippet) so the Mac side knows what to expect.

## Branch + commit reference

| Item | Value |
|---|---|
| Branch | `feat/scanrefer-vg-benchmark` |
| Tip at handoff | `542342c` (Mac side will push more commits before this lands; you may want `git pull origin feat/scanrefer-vg-benchmark` once before starting and once after to be safe) |
| Mac-side context | `docs/superpowers/specs/2026-05-01-nr3d-fairness-design.md` (NR3D protocol audit, what we already did) |
| ScanRefer audits | `tmp/scanrefer_zsl_code_audit.md`, `tmp/scanrefer_zsl_paper_survey.md`, `tmp/seeground_vog_audit_*.md` (these are gitignored on Mac side; you don't need them — listed only as Mac-side context for completeness) |

## Estimated total wall time

| Step | Time |
|---|---|
| `git pull` + conda activate + sanity imports | < 1 min |
| Producer build-scenes for 11 scenes | ~10 min |
| Verification snippet | < 30 s |
| (optional) producer report commit + push | < 1 min |
| **Total** | **~12 min** |

Faster than NR3D's original 130-scene producer run because we're only
doing 11 scenes (the original took several hours).
