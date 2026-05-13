# NR3D Public Leaderboard (reference)

Source: https://referit3d.github.io/benchmarks.html and the NR3D paper.
Numbers are GT-classification overall accuracy unless noted; some recent
papers also report Acc@0.25 / Acc@0.5 in a separate detection-mode track.

| Method | Setup | Overall | Easy | Hard | View-dep | View-indep | Reference |
|---|---|---:|---:|---:|---:|---:|---|
| ReferIt3DNet (NR3D-only) | classification | 35.6 | 43.6 | 27.9 | 32.5 | 37.1 | Achlioptas et al. ECCV 2020 |
| BUTD-DETR | classification | 54.6 | 60.7 | 48.4 | 46.0 | 58.0 | leaderboard |
| MVT | classification | 55.1 | 61.3 | 49.1 | 54.3 | 55.4 | leaderboard |
| 3D-VisTA | classification | 64.2 | 72.1 | 56.7 | 61.5 | 65.1 | leaderboard |
| MiKASA | classification | 64.4 | 69.7 | 59.4 | 65.4 | 64.0 | leaderboard |
| UniVLG | classification | 65.2 | 73.3 | 57.0 | 55.1 | 69.9 | leaderboard 2026 |
| **Ours (v5.1, zero-shot RGB+VLM, fair-view)** | classification, no GT-visible keyframes | **68.48** | **78.43** | **59.18** | **57.38** | **74.53** | [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md) |
| **Ours (v3, zero-shot RGB+VLM)** | classification | **80.79** | **86.06** | **75.87** | **72.46** | **85.34** | [v3_referit3d_track_20260501.md](v3_referit3d_track_20260501.md) |

The v5.1 "Ours" row is the latest fair-view measurement: same canonical
full-scene GT-instance pool and leaderboard slicing, but without the
GT-target-visible keyframe shortcut used by the historical v3 run. It reruns
and merges all 241 failed v5 sentinels, recovering 240 completed outputs. The
v3 row is retained as the best historical number and GT-visible evidence
upper-bound. Pool / fold equivalence is empirically verified in
[pool_equivalence_log_20260501.md](pool_equivalence_log_20260501.md). The
remaining asymmetry vs the published rows is paradigm (zero-shot RGB+VLM vs
trained 3D-point-cloud model), not metric slicing.

Reference table for the separate detection-mode track (NR3D Det Acc@0.25 /
Acc@0.50 — distinct from the classification leaderboard above): see
[paper_crosscheck_20260501.md](paper_crosscheck_20260501.md) §Q7 for
UniVLG / MCLN / 3D-VisTA-reeval / BUTD-DETR-reeval / PQ3D numbers.

## Internal Partial Pilots (not leaderboard rows)

These rows use fixed internal subsets and are tracked for ablation/debugging
only. They must not be quoted as public NR3D leaderboard results.

| Run | Setup | Fold | Overall | Easy | Hard | View-dep | View-indep | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Same-fold v1/v3 baseline | GT-visible keyframes + GT-pool classification | 100 | 80.00 | 85.37 | 76.27 | 61.76 | 89.39 | Post-aggregated from `tmp/nr3d_eval_v1_full/side_by_side.json` restricted to v4 random100. |
| [v4 fair-view + guards](v4_agent_guards_fair_views_20260512.md) | query-driven keyframes + TADG/no-match/evidence-frame guards | 100 | 71.00 | 82.93 | 62.71 | 67.65 | 72.73 | Partial diagnostic: -9.00 pp overall vs same-fold baseline; no LLM/service failures. |

## Memory / Throughput Note

The v5 full run exposed that raw ConceptGraph pkl files are not safe to load
at high process concurrency: compressed files of 20MB can expand over 7GB of
unused `mask` arrays during `pickle.load`. The full local scene set contains
about 258.9GB of expanded masks. Post-run hardening adds stripped
`*.light.pkl.gz` sidecar caches and `--ensure-lightweight-cache`; missing
caches are auto-built under a cross-process lock, so cache absence does not
drop or fail benchmark samples.
