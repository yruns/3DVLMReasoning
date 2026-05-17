# NR3D Leaderboard Reference

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
| Ours (v5.1, zero-shot RGB+VLM, invalidated) | classification, projection-only visibility index | 68.48 | 78.43 | 59.18 | 57.38 | 74.53 | [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md) |
| Ours (v3, zero-shot RGB+VLM, invalidated) | classification, GT-target-visible but projection-only visibility index | 80.79 | 86.06 | 75.87 | 72.46 | 85.34 | [v3_referit3d_track_20260501.md](v3_referit3d_track_20260501.md) |

The v5.1 row is retained only as an audit trail. It must not be quoted as a
valid fair-view or SOTA comparison because the NR3D
`view_to_objects` / `object_to_views` files used by the pack were built with
`use_depth=False`; they encode projection/frustum candidates, not
depth-occlusion visibility. A valid comparison requires rebuilding the
visibility indices with depth occlusion, regenerating packs, and rerunning the
full fold. The v3 row is also invalidated as a benchmark claim because its
GT-target-visible shortcut used the same projection-only visibility source.
Candidate-pool and fold equivalence are summarized in [protocol.md](protocol.md).

Detection-mode NR3D rows such as Acc@0.25 / Acc@0.50 are a separate protocol
and should not be mixed with this classification table. See
[protocol.md](protocol.md) for the metric distinction.

## Internal Partial Pilots (not leaderboard rows)

These rows use fixed internal subsets and are tracked for ablation/debugging
only. They must not be quoted as public NR3D leaderboard results.

| Run | Setup | Fold | Overall | Easy | Hard | View-dep | View-indep | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Same-fold v1/v3 baseline | GT-visible keyframes + GT-pool classification | 100 | 80.00 | 85.37 | 76.27 | 61.76 | 89.39 | Invalidated: projection-only visibility source. |
| [v4 fair-view + guards](v4_agent_guards_fair_views_20260512.md) | query-driven keyframes + TADG/no-match/evidence-frame guards | 100 | 71.00 | 82.93 | 62.71 | 67.65 | 72.73 | Invalidated: same projection-only visibility source as v5/v5.1. |
| [v6 inline labels + depth-visible marks](v6_inline_labels_depth_visible_random100_20260513.md) | preserved v4 keyframe ids + depth-aware marked frames + inline `#id category` labels + guards; no NMS | 100 | 71.00 | 78.05 | 66.10 | 70.59 | 71.21 | Depth-aware partial pilot; not a full leaderboard row. |
| [v7 Stage1 callbacks no-CLIP](v7_stage1_callbacks_noclip_random100_20260513.md) | v6 fixed fold/pack + NR3D Stage1 callbacks wired + no per-selector CLIP fallback in `request_more_views`; no NMS | 100 | 73.00 | 82.93 | 66.10 | 70.59 | 74.24 | Depth-aware partial pilot; not a full leaderboard row. |
| [v8 clean initial marked-on-demand](v8_clean_initial_marked_on_demand_random100_20260514.md) | clean initial RGB keyframes + text-only left-to-right proposal inventory; marked images only via `view_keyframe_marked`; same guards; no NMS | 100 | 67.00 | 85.37 | 54.24 | 58.82 | 71.21 | Negative ablation vs v7; not a full leaderboard row. |
| [v9 selective marked images](v9_selective_mark_random100_20260514.md) | v8 clean initial contract + `list_frame_proposals` text inventory + filtered `view_keyframe_marked(categories/proposal_ids)` rendering; same guards; no NMS | 100 | 74.00 | 90.24 | 62.71 | 64.71 | 78.79 | Best depth-aware random100 pilot so far; still not a full leaderboard row. |
| [v9.3 strat600 (v93 UI overhaul + keyframe_selector fix)](v9_3_strat600_20260517.md) | catalog-first pack + working Stage-1 text retrieval + TADG/no-match/evidence-frame guards + v9.3 BEV/mark_frame UI overhaul | **600** (strat) | 66.67 | 73.45 | 60.32 | 54.98 | 73.01 | First v9.3 benchmark-grade run; strat600 is calibrated within ±0.19 pp of FULL 7805 on the v9.1_fix reference, so this row is **the most directly comparable internal pilot** to the public leaderboard rows above. Statistically equivalent to v9.2 text-first FULL (+1.47 pp Overall inside the strat600 ±2.3 pp band); confirms that the v9.3 fixes do not regress agent accuracy on NR3D. |

## Memory / Throughput Note

The v5 full run exposed that raw ConceptGraph pkl files are not safe to load
at high process concurrency: compressed files of 20MB can expand over 7GB of
unused `mask` arrays during `pickle.load`. The full local scene set contains
about 258.9GB of expanded masks. Post-run hardening adds stripped
`*.light.pkl.gz` sidecar caches and `--ensure-lightweight-cache`; missing
caches are auto-built under a cross-process lock, so cache absence does not
drop or fail benchmark samples.

The v7 random100 callback run exposed a second memory boundary: if
`request_more_views` calls `selector.find_objects()` under high concurrency,
each scene selector can lazily load its own CLIP model. NR3D now disables that
object-term CLIP fallback for the callback path and relies on hypothesis
categories plus visibility instead; the 100-worker rerun stayed well below the
15GB RSS guard.

The v8 random100 clean-initial run stayed below the same 15GB guard as well
(observed around 4.1GB RSS), and completed 100/100 checkpoints with no failed
sentinels. It produced 192 retryable ModelHub `429` lines, so the practical
throughput limit on this fold is qpm/tpm rather than local memory.

The v9 selective-mark run also completed 100/100 checkpoints under the 15GB RSS
guard with no failed sentinels. It produced 212 retryable ModelHub `429` lines;
the local memory issue did not recur, and throughput remained LLM-limit bound.
