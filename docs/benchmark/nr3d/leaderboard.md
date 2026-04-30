# NR3D Public Leaderboard (reference)

Source: https://referit3d.github.io/benchmarks.html and the NR3D paper.
Numbers are GT-classification overall accuracy unless noted; some recent
papers also report Acc@0.25 / Acc@0.5.

| Method | Setup | Overall | Easy | Hard | View-dep | View-indep | Reference |
|---|---|---:|---:|---:|---:|---:|---|
| ReferIt3DNet (NR3D-only) | classification | 35.6 | 43.6 | 27.9 | 32.5 | 37.1 | Achlioptas et al. ECCV 2020 |
| BUTD-DETR | classification | 54.6 | 60.7 | 48.4 | 46.0 | 58.0 | leaderboard |
| MVT | classification | 55.1 | 61.3 | 49.1 | 54.3 | 55.4 | leaderboard |
| 3D-VisTA | classification | 64.2 | 72.1 | 56.7 | 61.5 | 65.1 | leaderboard |
| MiKASA | classification | 64.4 | 69.7 | 59.4 | 65.4 | 64.0 | leaderboard |
| UniVLG | classification | 65.2 | 73.3 | 57.0 | 55.1 | 69.9 | leaderboard 2026 |

Our numbers, when available, will appear in `README.md` against this
reference table. Until an alternate NR3D test GT source is wired, our headline
metric will be reported on the NR3D **train** split. Note: our setup is
detection-mode (Acc@0.25/0.50) on EmbodiedScan-derived 9-DOF bboxes, so it is
not directly comparable to the classification leaderboard above.
