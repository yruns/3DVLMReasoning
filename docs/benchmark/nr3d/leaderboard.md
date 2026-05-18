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
| **Ours (v9.3 text-first, depth-aware, fair)** | classification, strat600 calibrated fold (≈ FULL ±0.19 pp on v9.1_fix reference) | **66.67** | 73.45 | **60.32** | 54.98 | **73.01** | [v9_3_strat600_20260517.md](v9_3_strat600_20260517.md) — see also [v9_4_gt_leak_postmortem_20260518.md](v9_4_gt_leak_postmortem_20260518.md) |
| Ours (v9.1_fix FULL, **invalidated**) | classification, GT-target-visible RGB seed keyframes silently injected (seed-drain leak at `d5f40ba`, fixed in `8ebf701`) | 82.95 | 88.36 | 77.88 | 78.23 | 85.51 | [v9_1_fix_FULL_REPRO_20260516.md](v9_1_fix_FULL_REPRO_20260516.md) — see [v9_4_gt_leak_postmortem_20260518.md](v9_4_gt_leak_postmortem_20260518.md) |
| Ours (v5.1, zero-shot RGB+VLM, invalidated) | classification, projection-only visibility index | 68.48 | 78.43 | 59.18 | 57.38 | 74.53 | [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md) |
| Ours (v3, zero-shot RGB+VLM, invalidated) | classification, GT-target-visible but projection-only visibility index | 80.79 | 86.06 | 75.87 | 72.46 | 85.34 | [v3_referit3d_track_20260501.md](v3_referit3d_track_20260501.md) |

The **v9.3 text-first 66.67 %** row is the only valid depth-aware comparison
against UniVLG in this archive. It is our current honest "best" on NR3D, with
+1.47 pp Overall headroom and +3 pp on both Hard and V-Indep relative to the
public 65.2 % SOTA. The strat600 fold is calibrated to within ±0.19 pp of the
canonical FULL 7805 on the v9.1_fix reference; bootstrap 90 %-band on Overall
is ±2.3 pp.

The **v9.1_fix FULL 82.95 % row is invalidated** as a fair NR3D claim because
the 5 GT-target-visible seed keyframes that pack-prep writes into
`bundle.keyframes` were silently auto-injected into agent context every
evidence-update turn at commit `d5f40ba` (the runtime commit). The leak was
fixed in `8ebf701`. The diagnostic flag `--restore-stage1-seed-keyframe-drain`
restores the leak on v9.3 code and reproduces this row's number to within
+0.33 pp; see
[v9_4_gt_leak_postmortem_20260518.md](v9_4_gt_leak_postmortem_20260518.md)
for the full audit trail. The 82.95 % row joins v3 (80.79 %) and v5.1 (68.48 %)
in the GT-leak-invalidated category and must not be quoted against public NR3D
SOTA.

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
| [v9.3 strat600 text-first (v93 UI overhaul + keyframe_selector fix)](v9_3_strat600_20260517.md) | catalog-first pack + working Stage-1 text retrieval + TADG/no-match/evidence-frame guards + v9.3 BEV/mark_frame UI overhaul | **600** (strat) | 66.67 | 73.45 | 60.32 | 54.98 | 73.01 | First v9.3 benchmark-grade run; strat600 is calibrated within ±0.19 pp of FULL 7805 on the v9.1_fix reference, so this row is **the most directly comparable internal pilot** to the public leaderboard rows above. Statistically equivalent to v9.2 text-first FULL (+1.47 pp Overall inside the strat600 ±2.3 pp band); confirms that the v9.3 fixes do not regress agent accuracy on NR3D. |
| [v9.3 strat600 catalog-only A/B](v9_3_strat600_notext_20260517.md) | same pack + same guards + `--disable-stage1-text-retrieval` (drops `select_by_text` tool + `_no_text` playbook variants) | **600** (strat) | 64.17 | 73.10 | 55.81 | 56.87 | 68.12 | Paired A/B with the row above. Δ vs text-first paired: −2.50 pp Overall, Hard −4.51, V-Indep −4.89, V-Dep +1.90 — same per-tier direction as v9.2 FULL A/B. **Δ vs v9.1_fix FULL = −18.78 pp** → refutes the "clean catalog-only recovers 83 %" hypothesis; 83 % was the bug-state fall-through artifact, not the no-text policy. |
| [v9.4-A strat600 force-error](v9_4a_strat600_force_error_20260517.md) | same pack + same guards + `--force-stage1-text-retrieval-to-error` (keeps text-first playbook + tool registered, makes tool body short-circuit to ERROR before `KeyframeSelector`) | **600** (strat) | 63.17 | 71.03 | 55.81 | 55.45 | 67.35 | v9.4 Experiment A — tests whether the v9.1_fix cadence is reproducible by forcing the ERR signature on top of the working v9.3 chassis. **Negative result**: −1.00 pp vs v9.3 no-text, −19.83 pp vs v9.1_fix calibrated baseline, 0 / 6 audit magic cases recovered. The 16 pp gap is not in the ERR signature alone; next candidate is the Stage-1 seed-keyframe drain leak at commit `d5f40ba` (fixed in `8ebf701`). |
| [v9.4-D strat600 force-error + restore-drain](v9_4d_strat600_force_error_seed_drain_20260517.md) | same pack + same guards + `--force-stage1-text-retrieval-to-error` + **`--restore-stage1-seed-keyframe-drain`** (the latter bypasses the `initial_keyframe_paths` filter, restoring the GT-target-visible RGB injection that was present at commit `d5f40ba` and fixed in `8ebf701`) | **600** (strat) | **83.33** | 84.48 | **82.26** | 76.78 | 86.89 | v9.4 Experiment D — **isolates the 16 pp**: within +0.33 pp of v9.1_fix calibrated 83.00, Hard +4.20 pp above v9.1_fix, **6/6 magic audit cases recovered**. Conclusion: v9.1_fix's 82.95 % is an **information leak** (5 GT-visible RGBs silently injected per turn), not a fair NR3D result. v9.1_fix FULL row should be re-tagged with a fair-view caveat (same status as v3 / v5.1). Honest v9.3 baseline ≈ 65 % matches public UniVLG SOTA. **Test-time flag only**; never produce a published number. |
| [v10 no-initial-keyframes strat600](v10_no_initial_keyframes_strat600_20260518.md) | branch `feat/remove-initial-keyframes`; Stage 2 receives no pack-prepared initial keyframe evidence; first-person evidence is reachable only through active `select_by_*` / marking tools; same canonical fold + guards | **600** (strat) | 64.33 | 72.41 | 56.77 | 53.55 | 70.18 | Spec-cleanup validation. Runs cleanly with 600 / 600 completed, 0 no-match, 0 tracebacks, eval + metrics exit 0. Accuracy is −2.34 pp vs v9.3 text-first and +0.16 pp vs v9.3 catalog-only, so this is not a new best. Trace audit points to unsupported spatial relation aliases, masked-category `select_by_text` guard blocks, and guard-induced target/anchor corrections as the main case-level failure modes. |
| [v10 tool-trace image metadata strat600](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-tool-trace-image-metadata-cfee0ef) | same branch/fold/guards; first-person tool images are recorded only in `Stage2ToolObservation.image_metadata`, with no bundle/runtime pending-image side channel | **600** (strat) | 61.00 | 67.93 | 54.52 | 52.13 | 65.81 | Architecture validation after removing the pending-image side channel. Eval + metrics exit 0; 600 side-by-side rows; 590 completed and 10 final failed statuses. Net −20 correct vs `220f128` (73 old misses recovered, 93 old positives regressed). Source and artifact scans found no `vg_pending`, `pending_image_paths`, `pending_image_metadata`, or `queue_pending` references. Main failure mode: target-category guard target/anchor misparsing. |
| [v10 target-category guard5 probe](v10_no_initial_keyframes_strat600_20260518.md#target-category-guard-recovery-probe-2ce81df) | diagnostic subset of five prior `cfee0ef` target/anchor guard-deadlock failures; `target_category_guard` head-noun parsing fix, same standard guards | **5** (diagnostic) | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | Not a leaderboard row. All 5 previously failed cases completed and selected the target id, with `target_category_guard_blocked=false`: box/wall, chair/mirror, shelf/plant, storage-bin/clock, monitor/bookshelf. Confirms a local no-GT guard fix only; remaining buckets require separate extractor, spatial-relation, evidence-frame, and candidate-coverage work. |
| [v10 missing-status4 probe](v10_no_initial_keyframes_strat600_20260518.md#missing-status-extractor-recovery-probe-e5617ac) | diagnostic subset of four prior `cfee0ef` synthetic missing-status failures; NR3D extractor resolves direct structured `proposal_id` outputs through the fair proposal pool, same standard guards | **4** (diagnostic) | 100.00 | 100.00 | n/a | 100.00 | 100.00 | Not a leaderboard row. All 4 previously missing-status samples completed, selected the target id, and preserved query/tool trace. Run log had no `prediction missing status`, traceback, or exception matches. Confirms output-normalization parity only; it does not change agent evidence inputs. |
| [v10 no-GT fixes strat600](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-guardextractor-fixes-a6f6077) | same branch/fold/guards; target-category guard head-noun fix + NR3D proposal-only extractor fix; no initial keyframes and no pending-image side channel | **600** (strat) | 62.67 | 71.03 | 54.84 | 53.55 | 67.61 | Standard full strat600 rerun. Eval + metrics exit 0; 600 side-by-side rows; 597 completed and 3 final failed statuses. +10 correct vs `cfee0ef` (79 recovered, 69 regressed). Still below `220f128` and v9.3 text-first; remaining failures are target-category generic/anchor parsing and unsupported `same side as`. |
| [v10 guard-scope 15-case probe](v10_no_initial_keyframes_strat600_20260518.md#guard-scope-15-case-probe-6355781) | diagnostic subset of 15 prior `a6f6077` failures; TCG later-option target-head scoping + EFG room/anchor-side scope fixes, same standard guards | **15** (diagnostic) | 53.33 | n/a | 53.33 | 53.33 | n/a | Not a leaderboard row. All 15 completed and 8 / 15 previously failed cases selected the target id. Confirms local no-GT guard-scope fixes; remaining buckets are candidate closure, role-bound relation evidence, and noun-scoped side parsing. |
| [v10 candidate-closure 15-case probe](v10_no_initial_keyframes_strat600_20260518.md#candidate-closure-15-case-probe-06b8d58) | diagnostic subset of 15 prior `a6f6077` failures; EFG same-category candidate-closure soft block, same standard guards | **15** (diagnostic) | 33.33 | 28.57 | 37.50 | 28.57 | 37.50 | Not a leaderboard row. All 15 completed and 5 / 15 previously failed cases selected the target id. Confirms a no-GT candidate-closure improvement; remaining failures need topology, anchor-subtype grounding, and large row/ordinal tools. |
| [v10 rationale/payload consistency probe](v10_no_initial_keyframes_strat600_20260518.md#rationalepayload-consistency-probe-be3c4fc) | diagnostic subset of 3 prior `a6f6077` stale-payload contradictions; `submit_final` blocks rationale/payload mismatch, same standard guards | **3** (diagnostic) | 66.67 | 50.00 | 100.00 | 0.00 | 100.00 | Not a leaderboard row. Live probe completed 3 / 3 and recovered 2 / 3 audited stale-payload cases. Offline scan over prior strat600 submit traces found 3 / 600 contradictions after false-positive tightening. Remaining miss is category alias + anchor relation, not stale payload. |
| [v10 generic subtype probe](v10_no_initial_keyframes_strat600_20260518.md#generic-subtype-category-probe-dbd9adb) | same 3-case diagnostic subset; asymmetric generic/subtype target-category matching plus VG playbook guidance | **3** (diagnostic) | 66.67 | 50.00 | 100.00 | 0.00 | 100.00 | Not a leaderboard row. Preserves the two recovered stale-payload cases and confirms the remaining chair/office-chair miss is now a nested-anchor flow problem: the agent must resolve "desk closest to the window" before selecting the chair behind it. |
| [v10 nested-anchor flow probe](v10_no_initial_keyframes_strat600_20260518.md#nested-anchor-flow-probe-3b675f9) | same 3-case diagnostic subset; prompt/skill-only contract to resolve nested anchor superlatives before target ranking | **3** (diagnostic) | 66.67 | 50.00 | 100.00 | 100.00 | 50.00 | Not a leaderboard row. Recovers the intended nested-anchor chair case (`scene0663_00`, `#33` -> `#6`) by comparing desk anchors to the window first, but regresses the negated-window pillow case (`#20` -> `#19`). Partial-positive only; needs a follow-up negated-anchor distinction before broader claims. |
| [v10 negated-anchor prompt probe](v10_no_initial_keyframes_strat600_20260518.md#negated-anchor-prompt-probe-fd1a628) | same 3-case diagnostic subset; prompt/skill-only marked-positive counterexample wording for negated anchors | **3** (diagnostic) | 33.33 | 50.00 | 0.00 | 0.00 | 50.00 | Negative diagnostic. Recovers the negated-window pillow case but regresses the intended nested-anchor chair and the outside-door case. Do not treat this prompt wording as an active improvement. |
| [v10 above/below alignment probe](v10_no_initial_keyframes_strat600_20260518.md#abovebelow-alignment-tool-probe-54198ef) | same 15-case diagnostic slice as candidate-closure; `compare_proposals_spatial` ranks `above` / `below` by correct vertical side, horizontal alignment, then vertical magnitude | **15** (diagnostic) | 53.33 | 57.14 | 50.00 | 71.43 | 37.50 | Not a leaderboard row. Deterministic tool behavior is unit-covered. Live slice completes 15 / 15 and is 8 / 15 vs 5 / 15 at `06b8d58`, but traces contain no above/below comparator calls, so treat as no-regression evidence; next work is prompt/skill routing into this tool. |
| [v10 vertical relation routing prompt probe](v10_no_initial_keyframes_strat600_20260518.md#vertical-relation-routing-prompt-probe-61042c3) | same 15-case diagnostic slice; broad playbook prose maps under/below/above to `compare_proposals_spatial` before marked-frame verification | **15** (diagnostic) | 33.33 | 28.57 | 37.50 | 28.57 | 37.50 | Negative diagnostic. Produces the desired `below` comparison on `scene0077` and keeps it correct, but regresses 3 unrelated cases vs `54198ef` and drops the slice from 8 / 15 to 5 / 15. Do not keep this broad wording active; use a narrower guard/tool route. |

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
