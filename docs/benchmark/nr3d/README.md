# NR3D VG Benchmark Archive

This directory is the permanent process archive for NR3D visual grounding
evaluations in this repo. Keep the version docs immutable; use this README as
the current human-facing index.

> **⚠️ Read this first** —
> [**v9_4_gt_leak_postmortem_20260518.md**](v9_4_gt_leak_postmortem_20260518.md)
> The `v9.1_fix FULL` row at 82.95 % was invalidated on 2026-05-17 as a
> GT-target-visible information leak (5 seed keyframes silently injected
> per evidence-update turn at commit `d5f40ba`, fixed in `8ebf701`). The
> honest depth-aware NR3D full-set "best" on this codebase is now **v11
> proposal enrichment FULL at 72.26 % on the canonical filtered 7805-query
> slice**. The latest v20 Codex SDK CLI-only strat600 pilot scored
> **74.50 %**, the best observed valid pilot-fold row so far, but it still
> needs a full-set confirmation before changing the headline full-set result.
> The preceding v11 strat600 pilot scored 71.33 %, recovering most of
> the lower reproduction gap. A 2026-05-20
> high-worker reproduction at `workers=30` completed cleanly but scored
> 70.00 %, and a 2026-05-21 `workers=20` reproduction from the current best
> branch scored 69.17 %. Treat 72.26 as the current best full-set result, with
> the high-worker rows as stability caveats.
> The older v9.3
> text-first row at 66.67 % remains the clean baseline closest to public
> UniVLG SOTA (65.2 %). The post-mortem consolidates
> the 6 investigation sub-docs into one narrative and gives the
> reproduction recipe + diagnostic-flag contracts.

## Entry Points

| File | Purpose |
|---|---|
| **[v9_4_gt_leak_postmortem_20260518.md](v9_4_gt_leak_postmortem_20260518.md)** | **Unified GT-leak post-mortem (read first).** Consolidates the v9.3 → v9.4-D investigation: refutes the cadence hypothesis, isolates the 16 pp v9.1_fix advantage as the Stage-1 seed-keyframe drain leak, lists the two diagnostic flags now in the codebase, and prescribes v9.4's next direction. |
| [leaderboard.md](leaderboard.md) | Public SOTA context plus our invalidated v3 / v5.1 / v9.1_fix audit rows and historical rows. |
| [protocol.md](protocol.md) | Consolidated protocol notes: metric family, fold/filter rules, candidate-pool equivalence, fairness boundary. |
| [depth_visibility_rebuild_20260513.md](depth_visibility_rebuild_20260513.md) | Root-cause record and rebuild summary for depth-aware NR3D visibility indices. |
| [depth_visibility_spotcheck_20260513.html](depth_visibility_spotcheck_20260513.html) | Visual spotcheck frames rendered from rebuilt depth-aware `view_to_objects`. |
| [v9_2_full_select_by_text_ab_20260516.md](v9_2_full_select_by_text_ab_20260516.md) | **Full 8584-utterance A/B** of the `enable_stage1_text_retrieval` toggle. Filtered (n=7805): text-first **65.20 %** vs catalog-first **64.65 %** (Δ **+0.55 pp** overall). Per-tier: text-first wins on **Hard +1.14** and **V-Indep +1.56**; catalog-first wins on **V-Dep −1.31**; Easy is tied. **0** Python errors across 17 168 sample runs; 46 graceful no-match `-1` failures. Random100's 5 pp gap shrinks to 0.55 pp at full scale — the audit hypothesis is now *partially* validated (it holds on view-dep, not on hard / view-indep). **Decision: keep `select_by_text` on by default; v9.3 should route per query type.** |
| [v9_2_full_text_first_trace_20260516.html](v9_2_full_text_first_trace_20260516.html) | VG agent trace viewer (2T+4F) for the **text-first** full run. Cases picked to show variant-specific wins/losses: 2 correct (where text-first was uniquely right), 2 losses where no-text was correct, 2 losses both shared. Covers attribute / superlative / spatial-anchor / view-dep query types. |
| [v9_2_full_no_text_trace_20260516.html](v9_2_full_no_text_trace_20260516.html) | VG agent trace viewer (2T+4F) for the **catalog-first** full run. Mirrors the text-first selection: 2 correct (where catalog-first uniquely won), 2 losses where text-first was correct, 2 losses both shared. Direct side-by-side reading of the same cases (`scannet/scene0249_00::32::36779` appears in both as opposite outcomes). |
| [v9_4d_strat600_force_error_seed_drain_20260517.md](v9_4d_strat600_force_error_seed_drain_20260517.md) | **🆕 v9.4 Experiment D — `--restore-stage1-seed-keyframe-drain` + `--force-stage1-text-retrieval-to-error` combined**. HEAD `a0d802b`, workers=40, 34.5 min wall. **POSITIVE RESULT, ISOLATES THE 16 pp**: Overall **83.33 %** (Easy 84.48 / Hard **82.26** / V-Dep 76.78 / V-Indep 86.89) — within **+0.33 pp** of v9.1_fix strat600 calibrated 83.00, BETTER on Hard by +4.20. **Recovered 6 / 6 magic cases** (vs v9.4-A's 0 / 6). Each evidence-update injects 6 images (1 BEV + 5 GT-target-visible seed keyframes) — exactly the v9.1_fix bug-state shape. **The v9.1_fix +16 pp advantage is fully attributable to the Stage-1 seed-keyframe drain leak fixed in `8ebf701`, NOT cadence or playbook prose.** Implication: v9.1_fix 82.95 % is an information leak (GT-visible RGBs auto-injected), not a fair NR3D result; the honest v9.3 baseline ~65 % matches public UniVLG SOTA 65.2 %. v9.1_fix FULL row should be re-tagged with a fair-view caveat. mark_frame_with_bbox=3.48/sample (matches/exceeds v9.1_fix's 2.90, +1.01 vs v9.4-A) — leak is what triggers the cadence, not vice versa. |
| [v10_no_initial_keyframes_strat600_20260518.md](v10_no_initial_keyframes_strat600_20260518.md) | **v10 spec cleanup — no initial keyframe evidence path.** Branch `feat/remove-initial-keyframes`, canonical strat600. Initial HEAD `220f128` removed pack-prepared initial evidence and completed 600 / 600 at **64.33 %** Overall (Easy 72.41 / Hard 56.77 / V-Dep 53.55 / V-Indep 70.18). Follow-up HEAD `cfee0ef` removed the remaining bundle/runtime pending-image side channel and routes first-person tool visuals only through `tool_trace.image_metadata`; it completed eval + metrics with 590 completed statuses, 10 final failed statuses, and **61.00 %** Overall. Latest full rerun at launch/runtime HEAD `a6f6077` adds the no-GT guard/extractor fixes: **62.67 %** Overall (Easy 71.03 / Hard 54.84 / V-Dep 53.55 / V-Indep 67.61), 597 completed and 3 final failed statuses. Follow-up probe `2f9afe4` fixes generic `object is a/an X` category extraction and rationale-only left/right EFG overreach; it recovers the audited closest-to case but the closed-door visual attribute case remains wrong. Later cabinet-anchor probe `6cb6844` / unit follow-up `c94f5e1` fixes cabinet-vs-fridge target/anchor separation enough to recover one audited kitchen-cabinet case, while group-internal cabinet ranking remains open. View-dependent side-evidence prompt probe `23f68de` recovers 5 / 15 newly audited failures by requiring same-frame viewer/anchor evidence. Follow-up `c5a75ae` fixes explicit-target priority in target-category guard and recovers the audited angled-wall bookshelf case. Guard-scope probe `6355781` fixes later-option target-head parsing plus room/anchor-side EFG scope and recovers 8 / 15 fresh audited failures. Candidate-closure probe `06b8d58` adds small same-category marked-evidence closure in EFG and recovers 5 / 15 additional fresh audited failures. Rationale/payload probe `be3c4fc` recovers 2 / 3 stale-payload cases. Subtype probe `dbd9adb` preserves those 2 / 3 and shows the remaining chair case is a nested-anchor failure, not just category aliasing. Nested-anchor probe `3b675f9` recovers that chair case but regresses a negated-window pillow case, so the change is partial-positive only. Negated-anchor prompt probe `fd1a628` recovers the pillow case but regresses the chair and outside-door cases; it is negative and should not remain active. Above/below alignment tool commit `54198ef` is unit-covered and its same 15-case diagnostic probe is 8 / 15 with no regression vs `06b8d58`, but the live traces did not call the vertical comparator. Broad vertical-routing prompt probe `61042c3` creates the desired `below` call on `scene0077` but drops the slice to 5 / 15, so it is negative. Narrow TADG vertical-relation guard probe `66c6218` also forces the desired relation calls but remains 5 / 15 and regresses three cases vs `54198ef`; do not keep it active. Crop unavailable tool-surface probe shows the old success-like stub (`5193cff`) and hidden-tool fix (`1d3b02d`) both score **13 / 26**, while exposing an ERROR-only crop tool (`cbfb9ed`) drops to **11 / 26**; keep `request_crops` hidden unless a concrete renderer is configured. Compound target-head probe `b00ca7c` fixes the audited `all black desk chair` guard bug (`expected_category=chair`, `#3 office chair`, IoU 1.0) but same 26-case slice remains **13 / 26**, so it is a local correctness patch rather than an aggregate gain. Superlative anchor-coverage probe `71510a0` extends TADG ambiguous-anchor coverage to closest/farthest and runs a fresh 15-case audited slice at **5 / 15**; strict override follow-up `97d60c9` recovers `scene0222` but drops the slice to **3 / 15**, so it is negative and should be reverted or narrowed. Multi-anchor spatial tool probe `80aa8f6` adds `compare_candidates_to_anchors` plus playbook routing for closest/farthest/near/next_to and improves the same 15-case slice to **9 / 15** while preserving every `71510a0` correct case; the standard follow-up at `b671243` reaches **70.00 %** Overall on strat600. TADG binding at runtime `a3ff7f1` reaches the current best fair v10 row, **72.00 %** Overall; a 2026-05-20 high-worker `workers=30` reproduction completed cleanly but scored **70.00 %**. Next targets are candidate/anchor overlap rejection and unresolved `anchor_disagreement` finalization. |
| [v10 target-category guard5 probe](v10_no_initial_keyframes_strat600_20260518.md#target-category-guard-recovery-probe-2ce81df) | **Diagnostic recovery probe, not a leaderboard row.** After 15-case audit, HEAD `2ce81df` fixes target-category guard head-noun parsing without GT inputs. On the five audited guard-deadlock failures from `cfee0ef`, all 5 completed and selected the target id: box/wall, chair/mirror, shelf/plant, storage-bin/clock, monitor/bookshelf. Subset metric 5 / 5 = 100 %. Remaining buckets: missing-status extraction, unsupported behind/front relations, evidence-frame guard overreach, and candidate coverage before no-match. |
| [v10 missing-status4 probe](v10_no_initial_keyframes_strat600_20260518.md#missing-status-extractor-recovery-probe-e5617ac) | **Diagnostic recovery probe, not a leaderboard row.** HEAD `e5617ac` brings the NR3D pack-v1 extractor to ScanRefer parity for direct structured `proposal_id` outputs. On four audited `cfee0ef` synthetic missing-status failures, all 4 completed, selected the target id, and preserved query + tool trace. Run log contained no `prediction missing status`, traceback, or exception. |
| [v10 no-GT fixes strat600](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-guardextractor-fixes-a6f6077) | **Standard canonical strat600 rerun.** Launch/runtime HEAD `a6f6077`; later test-only HEAD `f887917` only adds CI guardrails banning pending-image channels. Overall **62.67 %** (Easy 71.03 / Hard 54.84 / V-Dep 53.55 / V-Indep 67.61), +1.67 pp / +10 correct vs `cfee0ef`, with 597 completed and 3 final failed statuses. Source/artifact scan found no pending-image side-channel keys. |
| [v10 multi-anchor strat600](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-multi-anchor-tool-b671243) | **Standard canonical strat600 rerun.** Launch/runtime HEAD `b671243` adds `compare_candidates_to_anchors` and multi-anchor playbook routing. Overall **70.00 %** (Easy 78.28 / Hard 62.26 / V-Dep 58.77 / V-Indep 76.09), 598 completed and 2 final failed statuses. +44 correct vs `a6f6077`, +34 vs `220f128`, +54 vs `cfee0ef`. Source/artifact scan found no pending-image side-channel keys; next narrow fixes are candidate/anchor overlap rejection and unresolved `anchor_disagreement` finalization. |
| [v10 multi-anchor TADG strat600](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-multi-anchor-tadg-binding-a3ff7f1) | **Standard canonical strat600 rerun.** Launch/runtime HEAD `a3ff7f1` makes `compare_candidates_to_anchors` first-class TADG evidence and blocks unresolved `anchor_disagreement`. Overall **72.00 %** (Easy 82.41 / Hard 62.26 / V-Dep 62.56 / V-Indep 77.12), 600 completed. +12 correct vs `b671243`; high-worker reproduction with `workers=30` completed but scored 70.00 %. |
| [v11 proposal enrichment strat600](v11_enrichment_strat600_20260521.md) | **Standard canonical strat600 rerun with enriched proposal notes.** Launch/runtime HEAD `793cb57` adds compact enrichment to initial `Proposal notes:` and full enrichment to `inspect_proposal`. Overall **71.33 %** (Easy 79.66 / Hard 63.55 / V-Dep 61.61 / V-Indep 76.61), 600 completed. Not a new best, but above the recent 70.00 / 69.17 reproductions of the same family. |
| [v11 proposal enrichment FULL](v11_enrichment_full_20260523.md) | **Full 8584-utterance rerun with enriched proposal notes on all NR3D scenes.** Launch/runtime HEAD `63dc417`; final metrics regenerated after deleting/rerunning API-error sentinels. Filtered Overall **72.26 %** (Easy 81.74 / Hard 63.39 / V-Dep 62.46 / V-Indep 77.60), 8492 completed and 92 final failed statuses, 0 checkpoint error fields. Current best valid full-set row. |
| [v11 select_by_text coverage strat600](v11_select_by_text_strat600_coverage_20260524.md) | **Tool coverage audit, not a leaderboard run.** Launch/runtime HEAD `45a2dae`; 16-way shard run over canonical strat600 using raw NR3D query and production-equivalent `select_by_text(..., k=3, use_visual_context=False)`. Target-id-in-returned-keyframes hit@3 **62.17 %** overall; **83.07 %** when non-empty; empty returned frames **25.17 %**. View-dependent hit@3 is only **50.71 %**, roughly random. |
| [v12 select_by_text viewpoint coverage strat600](v12_select_by_text_viewpoint_strat600_coverage_20260524.md) | **Tool coverage audit after viewpoint-hypothesis schema/executor fixes, not a leaderboard run.** Launch/runtime HEAD `b8582d6`; 16-way shard run over canonical strat600 using raw NR3D query and production-equivalent `select_by_text(..., k=3, use_visual_context=False)`. Target-id-in-returned-keyframes hit@3 **60.50 %** overall; **83.07 %** when non-empty; empty returned frames **27.17 %**. View-dependent hit@3 is **47.87 %**. |
| [v13 select_by_text viewpoint-aware coverage strat600](v13_select_by_text_viewpoint_aware_strat600_coverage_20260524.md) | **Tool coverage audit with `viewpoint_aware=True`, not a leaderboard run.** Launch/runtime HEAD `f8e6d22`; 16-way shard run over canonical strat600 using raw NR3D query and opt-in `select_by_text(..., k=3, use_visual_context=False, viewpoint_aware=True)`. Target-id-in-returned-keyframes hit@3 **69.00 %** overall; **83.98 %** when non-empty; empty returned frames **17.83 %**. View-dependent hit@3 jumps to **69.67 %**, mostly from rank-only demotion of ambiguous directional constraints. |
| [v14 select_by_text viewpoint prompt smoke50](v14_select_by_text_viewpoint_prompt_smoke50_20260524.md) | **Tool coverage smoke, not a leaderboard run.** Launch/runtime HEAD `3688096`; 10-way shard run over the first 50 rows of canonical strat600 after porting viewpoint prompt/schema to the active parser path. Target-id-in-returned-keyframes hit@3 **66.00 %** overall; View-Dep hit@3 **61.11 %**. The key signal is parser emission: **7 / 18** view-dependent samples emitted non-empty `viewpoint_contexts`, with **8** `reference_frame="viewer"` constraints/selectors. |
| [v15 select_by_text viewpoint prompt strat600](v15_select_by_text_viewpoint_prompt_strat600_coverage_20260524.md) | **Tool coverage audit after active-parser viewpoint prompt/schema port, not a leaderboard run.** Launch/runtime HEAD `9c62e51`; 20-way shard run over canonical strat600 using raw NR3D query and opt-in `select_by_text(..., k=3, use_visual_context=False, viewpoint_aware=True)`. Target-id-in-returned-keyframes hit@3 **69.17 %** overall; View-Dep hit@3 **73.93 %**; View-Indep hit@3 **66.58 %**. Parser now emits non-empty `viewpoint_contexts` on **98 / 600** samples, including **96 / 211** view-dependent samples. |
| [v16 select_by_text bbox-spatial strat600](v16_select_by_text_bbox_spatial_strat600_coverage_20260524.md) | **Tool coverage audit after bbox-aware spatial execution repair, not a leaderboard run.** Launch/runtime HEAD `43aed24`; 20-way shard run over canonical strat600 using raw NR3D query and opt-in `select_by_text(..., k=3, use_visual_context=False, viewpoint_aware=True)`. Target-id-in-returned-keyframes hit@3 **71.17 %** overall; empty returned frames drop **106 -> 81** vs v15. Gains are concentrated in View-Indep (**66.58 -> 71.72**), while View-Dep regresses (**73.93 -> 70.14**). |
| [v17 Codex SDK strat600](v17_codex_sdk_strat600_20260605.md) | **Codex Agent SDK integration run, not a new best.** Launch/runtime HEAD `6037c0b`; canonical strat600 through new `--backend codex_sdk`, local ModelHub adapter, `CODEX_AGENT_MODEL=gpt-5.4-2026-03-05` after gpt-5.5 permission failure. Overall **63.33 %** (Easy 70.00 / Hard 57.10 / V-Dep 53.55 / V-Indep 68.64), 600 completed, 0 errors. This SDK entrypoint is a one-turn BEV+catalog selector, not the DeepAgents tool loop. |
| [v18 Codex SDK MCP/text strat600](v18_codex_sdk_mcp_text_strat600_20260605.md) | **Codex Agent SDK MCP/skills/prefix-cache integration run, not a new best.** Launch/runtime HEAD `348ad52`; canonical strat600/case600 with `CODEX_AGENT_ENABLE_MCP_TOOLS=1`, prefix cache session `nr3d_codex_case600_20260605_348ad52_adapter`, synced playbook skills, and lazy `select_by_text` selector support. Overall **63.00 %** (Easy 70.69 / Hard 55.81 / V-Dep 55.45 / V-Indep 67.10), 600 completed, 0 final errors. Actual trace contains only `codex_sdk_turn` 600 times, so no MCP tool calls were exercised. |
| [v19 Codex SDK CLI fallback strat600](v19_codex_sdk_cli_fallback_strat600_20260606.md) | **Codex Agent SDK CLI/MCP tool integration run; former best observed valid strat600 pilot.** Launch/runtime HEAD `c19a747`; canonical strat600/case600 with `CODEX_AGENT_ENABLE_MCP_TOOLS=1`, `CODEX_AGENT_ENABLE_CLI_TOOLS=1`, prefix cache session `nr3d_codex_cli_case600_20260606_c19a747`, synced skills, local ModelHub adapter, and CLI fallback tool calls. Overall **72.50 %** (Easy 80.34 / Hard 65.16 / V-Dep 59.72 / V-Indep 79.43), 600 completed, 0 final errors. Tool trace contains 2592 calls, including `inspect_proposal`, `select_by_text`, `compare_proposals_spatial`, `compare_candidates_to_anchors`, `mark_frame_with_bbox`, and `view_bev`. |
| [v20 Codex SDK CLI-only strat600](v20_codex_sdk_cli_only_strat600_20260606.md) | **Codex Agent SDK CLI-only tool integration run; best observed valid strat600 pilot.** Launch/runtime HEAD `33376e2`; canonical strat600/case600 with `CODEX_AGENT_ENABLE_MCP_TOOLS=0`, `CODEX_AGENT_ENABLE_CLI_TOOLS=1`, prefix cache session `nr3d_codex_cli_only_case600_20260606_33376e2`, synced skills, local ModelHub adapter, and CLI evidence-tool calls. Overall **74.50 %** (Easy 81.38 / Hard 68.06 / V-Dep 64.45 / V-Indep 79.95), 599 completed and 1 final failed status after rerunning 14 transient JSONDecodeError checkpoints. SQLite contains 4621 tool calls, 0 MCP calls, and 6.70 evidence-tool calls/sample. |
| [v20 Codex SDK CLI-only 2T2F trace](v20_codex_sdk_cli_only_trace_2t2f_20260606.html) | Static 2T2F multimodal trace viewer for the v20 run. Includes two correct and two false samples, left-side execution-chain navigation, collapsible per-tool cards, raw input/output, BEV, selector-returned RGB frames, and `mark_frame_with_bbox` evidence images. |
| [v10 guard target-semantics probe](v10_no_initial_keyframes_strat600_20260518.md#guard-target-semantics-probe-2f9afe4) | **Diagnostic probe, not a leaderboard row.** HEAD `2f9afe4` fixes shell-head category parsing (`object is a/an X`) and prevents EFG from deriving left/right target constraints from rationale-only wording. Two audited cases: `closer to TV` recovers from wrong `#3` to target `#2` (IoU 1.0); `fully closed door` no longer hits the guard deadlock but still chooses the wrong door proposal (`#0` vs target `#2`). |
| [v10 cabinet-anchor target probe](v10_no_initial_keyframes_strat600_20260518.md#cabinet-anchor-target-probe-6cb6844) | **Diagnostic probe, not a leaderboard row.** HEAD `6cb6844` fixes cabinet aliases and positional cabinet-head extraction so refrigerator anchors are not accepted as cabinet targets. Two audited cases: kitchen cabinet over/with microwave recovers from wrong `#4` to target `#8` (IoU 1.0); top-left cabinet no longer selects the fridge anchor but still chooses neighboring cabinet `#15` instead of target `#14`. Follow-up HEAD `c94f5e1` adds unit-only coverage for generic `object you are looking for is X` complements. |
| [v10 view-dependent side-evidence probe](v10_no_initial_keyframes_strat600_20260518.md#view-dependent-side-evidence-probe-23f68de) | **Diagnostic probe, not a leaderboard row.** HEAD `23f68de` tightens the shared and VG playbooks for view-dependent left/right: use same marked viewer/anchor frame, prefer `require_all=True` for small candidate sets, and do not combine screen-left/right from different camera viewpoints. On 15 newly audited failures, 5 recover: mice-left, table-facing chair-right, two-closest armchair-right, monitor-keyboard, and nested table relation. Remaining buckets are role-bound composite relation evidence, noun-scoped EFG, target-category operative-clause parsing, and ordinal/superlative candidate closure. |
| [v10 bookshelf target-category guard probe](v10_no_initial_keyframes_strat600_20260518.md#bookshelf-target-category-guard-probe-c5a75ae) | **Diagnostic probe, not a leaderboard row.** HEAD `c5a75ae` makes explicit target actions (`find/select/choose/want`) outrank demonstrative context heads in target-category guard. The audited angled-wall query now expects `bookshelf` instead of `wall` and recovers `scene0208_00::106::14558` from failed to target `#106` (IoU 1.0). |
| [v10 guard-scope 15-case probe](v10_no_initial_keyframes_strat600_20260518.md#guard-scope-15-case-probe-6355781) | **Diagnostic probe, not a leaderboard row.** HEAD `6355781` fixes two no-GT guard-scope bugs: later option heads override earlier context in TCG, and EFG no longer treats anchor/room-side phrases as target image-left/right constraints. On 15 fresh failures from `v10_no_gt_fixes_strat600_20260519`, 8 recover and all 15 complete. Remaining buckets are candidate closure, role-bound relation evidence, and noun-scoped side parsing. |
| [v10 candidate-closure 15-case probe](v10_no_initial_keyframes_strat600_20260518.md#candidate-closure-15-case-probe-06b8d58) | **Diagnostic probe, not a leaderboard row.** HEAD `06b8d58` adds an EFG candidate-closure check: for small same-category sets under left/right, ordinal/superlative, or target-anchor relation cues, every same-category candidate must appear in marked evidence before finalization. On 15 fresh failures, 5 recover and all 15 complete. Remaining buckets are topology, anchor-subtype grounding, and large row/ordinal sets. |
| [v10 rationale/payload consistency probe](v10_no_initial_keyframes_strat600_20260518.md#rationalepayload-consistency-probe-be3c4fc) | **Diagnostic probe, not a leaderboard row.** HEAD `be3c4fc` adds a no-GT `submit_final` consistency guard for stale structured payloads: if rationale rules out the submitted proposal or names a different final id, the tool soft-blocks. Offline scan of the prior strat600 trace found 3 / 600 contradictions after false-positive tightening; the 3-case live probe completes and recovers 2 / 3, while the remaining chair/office-chair case needs alias + relation work. |
| [v10 generic subtype probe](v10_no_initial_keyframes_strat600_20260518.md#generic-subtype-category-probe-dbd9adb) | **Diagnostic probe, not a leaderboard row.** HEAD `dbd9adb` makes generic category checks asymmetric so `chair` can accept subtype labels such as `office chair`, while specific `office chair` queries still reject plain `chair` answers. The same 3-case probe remains 2 / 3: the chair sample still selects `#33` because the agent resolves the nested anchor "desk closest to the window" incorrectly. Next target is nested-anchor flow: resolve anchor superlatives before target ranking. |
| [v10 nested-anchor flow probe](v10_no_initial_keyframes_strat600_20260518.md#nested-anchor-flow-probe-3b675f9) | **Diagnostic probe, not a leaderboard row.** HEAD `3b675f9` adds a nested-anchor dependency-order contract to VG grounding and spatial-disambiguation skills. It recovers `scene0663_00::6::33306` by first comparing desks `#3/#4` to window `#5`, then selecting chair `#6` behind resolved desk `#3`; however, it regresses the negated-window pillow case to `#19`, so this is a partial-positive prompt change rather than a general win. |
| [v10 negated-anchor prompt probe](v10_no_initial_keyframes_strat600_20260518.md#negated-anchor-prompt-probe-fd1a628) | **Diagnostic probe, not a leaderboard row.** HEAD `fd1a628` requires marked positive counterexamples for negated anchor relations. It recovers `scene0222_00::20::35738` but regresses `scene0663_00::6::33306` and `scene0678_00::21::1134`; this prompt wording is negative and should be reverted or replaced by a narrower guard/tool. |
| [v10 above/below alignment probe](v10_no_initial_keyframes_strat600_20260518.md#abovebelow-alignment-tool-probe-54198ef) | **Diagnostic probe, not a leaderboard row.** HEAD `54198ef` changes `compare_proposals_spatial` so `above` / `below` rank by correct vertical side, horizontal alignment, then vertical magnitude. Unit tests cover the deterministic tool behavior. The same 15-case diagnostic slice completes 15 / 15 and is 8 / 15, up from 5 / 15 at `06b8d58`, but none of the live traces called the above/below comparator; treat this as no-regression evidence, not direct live attribution. |
| [v10 vertical relation routing probe](v10_no_initial_keyframes_strat600_20260518.md#vertical-relation-routing-prompt-probe-61042c3) | **Negative diagnostic, not a leaderboard row.** HEAD `61042c3` adds broad playbook prose mapping under/below/above to `compare_proposals_spatial`. It succeeds on the targeted `scene0077` by producing a `below` comparison and relation evidence, but the 15-case slice drops from 8 / 15 to 5 / 15 and regresses three unrelated cases. Do not keep this wording active. |
| [v10 vertical TADG guard probe](v10_no_initial_keyframes_strat600_20260518.md#vertical-relation-tadg-guard-probe-66c6218) | **Negative diagnostic, not a leaderboard row.** HEAD `66c6218` moves the vertical relation route into TADG, requiring explicit under/below/above submissions to bind `compare_proposals_spatial` evidence. It fires on `scene0077` and `scene0063`, but the slice remains 5 / 15 and regresses three cases vs `54198ef`; do not keep this guard active. |
| [v9_4a_strat600_force_error_20260517.md](v9_4a_strat600_force_error_20260517.md) | **v9.4 Experiment A — `--force-stage1-text-retrieval-to-error` flag** keeps text-first playbook + system prompt + `select_by_text` registered but tool body short-circuits to a new `force-disabled` ERROR before touching `KeyframeSelector`. HEAD `5180d60`, workers=40, 32.8 min wall. **NEGATIVE RESULT**: Overall **63.17 %** (Easy 71.03 / Hard 55.81 / V-Dep 55.45 / V-Indep 67.35) — −3.50 pp vs v9.3 text-first, −1.00 pp vs v9.3 no-text, **−19.83 pp vs v9.1_fix strat600 calibrated 83.00**. 174/174 select_by_text calls returned the new force-disabled ERROR signature; 0 reached the selector (wiring confirmed). But the agent only called select_by_text on 29 % of samples (vs v9.1_fix's 100 % in the audit), so the text-first playbook prose by itself doesn't force the v9.1_fix cadence. **Recovered 0 / 6 magic cases.** Refutes the simple "cadence anchor from forced ERR" hypothesis; the 16 pp gap must be in chassis/runtime changes between `d5f40ba` and `aa11922` (not playbook prose + tool surface). Doc proposes Experiment D (restore Stage-1 seed-keyframe drain leak) as next-highest-leverage. |
| [v9_1_fix_vs_v9_3_audit30_20260517.md](v9_1_fix_vs_v9_3_audit30_20260517.md) | **Per-sample audit of the 16 pp gap** — same 30 stratified strat600 ids run on (a) v9.1_fix bug-state worktree `d5f40ba` text-first, (b) v9.3 text-first `ae83fea`, (c) v9.3 no-text `aa11922`. On n=30: 60.0 % / 46.7 % / 53.3 % Overall — v9.1_fix wins **6 cases that BOTH v9.3 variants lose** (the "magic" cases), v9.3 wins 2 cases v9.1_fix loses. Recurring v9.1_fix pattern: `select_by_text→ERR → catalog walk → per-candidate mark → first submit → evidence-frame-guard re-mark → corrected re-submit`. The 16 pp gap is **deliberation cadence**, not policy: v9.1_fix marks 2.90 frames/sample vs v9.3-no's 2.13 (+36 %), `inspect_proposal` rate is identical at 3.07/sample. Concrete v9.4 experiments proposed (A: force select_by_text→ERR while keeping text-first playbook; B: explicit cadence prose in playbook; C: per-query routing + cadence). |
| [v9_3_strat600_notext_20260517.md](v9_3_strat600_notext_20260517.md) | **v9.3 strat600 catalog-only A/B partner** (HEAD `aa11922`, `--disable-stage1-text-retrieval`, workers=40, ~32 min). Overall **64.17 %** (Easy 73.10 / Hard 55.81 / V-Dep 56.87 / V-Indep 68.12). Direct A/B vs `v9_3_strat600_20260517`: **−2.50 pp Overall**, Hard −4.51, V-Indep −4.89, V-Dep **+1.90** — same per-tier direction as v9.2 FULL A/B (text wins Hard+V-Indep, no-text wins V-Dep). **Δ vs v9.1_fix FULL 82.95 = −18.78 pp** → **refutes the "Layer 1 alone recovers 83 %" hypothesis.** v9.1_fix's 83 was *not* a clean catalog-only policy; it was an artifact of (broken `select_by_text` + text-first playbook + agent's fall-through). 0 `select_by_text` calls, 1245 `_no_text` playbook loads (verified). 0 / 600 failed, 0 tracebacks. |
| [v9_3_strat600_20260517.md](v9_3_strat600_20260517.md) | **First v9.3 benchmark-grade run on canonical 600-case strat fold** (HEAD `ae83fea`, workers=40, ~32 min, default text-first). Overall **66.67 %** (Easy 73.45 / Hard 60.32 / V-Dep 54.98 / V-Indep 73.01). +1.47 pp over v9.2 text-first FULL (within strat600 ±2.3 pp 90 % band), −16.28 pp vs v9.1_fix FULL bug-state — re-confirms "Stage-1-actually-working = −17 pp on NR3D" on the new fold. **0** `runtime.keyframe_selector is None` errors over 185 `select_by_text` invocations (v9.3 fix from `32bfd0e` is healthy at production grade). 179 `view_bev(categories=...)` calls + 424 `view_bev(highlight=...)` calls; 1 501 `mark_frame_with_bbox(ids=...)` calls — agents fully adopt the v9.3 UI. 1 / 600 graceful no-match `failed`, 0 tracebacks. |
| [v9_3_strat10_v93ui_smoke_trace_20260517.html](v9_3_strat10_v93ui_smoke_trace_20260517.html) | **10-case smoke validating v9.3 UI overhaul** (BEV on-demand labels + mark_frame_with_bbox top-left colour-matched labels). Overall **60.0 %** (6/10), Easy **83.3 %** (5/6), Hard **25.0 %** (1/4), V-Dep **40.0 %** (2/5), V-Indep **80.0 %** (4/5). Tool usage: `select_by_text` 3× (0 errors, 6 frames), `view_bev` 9× including **3× with new `categories=` argument**, `mark_frame_with_bbox` 23×. **0 None-selector errors** + **0 stale cache hits** (new `_v93` cache tag invalidates the May-16 OLD-style overlays). Also includes a 4-pane visual at `assets/v9_3_strat10_v93ui_smoke_trace_20260517/_summary_4pane.png` showing (A) clean default BEV, (B) view_bev(highlight) with focused labels, (C–D) mark_frame with colour-matched top-left labels. |
| [assets/v9_3_mark_label_ux_20260517/comparison_old_vs_new.png](assets/v9_3_mark_label_ux_20260517/comparison_old_vs_new.png) | **v9.3 `mark_frame_with_bbox` label overhaul** (commit pending). Label is now (a) ~20 % larger (SIMPLEX→DUPLEX, scale floor 0.7→0.84, +20 % at the 1500 px ref width); (b) always anchored at the bbox top-left INSIDE the bbox (with auto fallback to "just above the bbox top" when the bbox is too small); (c) background filled with the bbox stroke colour, text colour auto-picked from black/white by ITU-R BT.601 luminance (>140 → black, else white). Before/after composite on scene0608_00 frame 7 with 4 overlapping bboxes (plant / coffee table / beanbag chair / recliner chair). 7 new tests, 16/16 mark_frame_with_bbox tests green. |
| [assets/v9_3_bev_ux_20260517/bev_4way_comparison.png](assets/v9_3_bev_ux_20260517/bev_4way_comparison.png) | **v9.3 BEV UX overhaul** (commit pending). 4-way visual: (1) OLD = label-every-proposal (cluttered, 1500×1500 with ~30 % wasted whitespace, labels illegible at thumbnail scale); (2) NEW default = mesh + trajectory + small dots, **no text labels** (918×1162 after tighter crop, 53 % area reduction); (3) NEW `view_bev(highlight=[6, 20])` = label only those two proposals on bright-yellow background with bold black text; (4) NEW `view_bev(categories=["door"])` = label every proposal whose category matches. Mirrors `mark_frame_with_bbox` UX. 8 new tests; default-no-labels contract pinned. |
| [v9_3_strat10_smoke_trace_20260517.html](v9_3_strat10_smoke_trace_20260517.html) | **10-case smoke on the strat600 fold** validating the v9.3 "runtime.keyframe_selector is None" root-cause fix (commit `32bfd0e`). Default config (text retrieval ON) + all 3 guards (TADG / no-match / evidence-frame) + `pack_nr3d_v9_catalog_first`. Overall **70.0 %** (7/10), Easy **83.3 %** (5/6), Hard **50.0 %** (2/4), V-Dep **60.0 %** (3/5), V-Indep **80.0 %** (4/5). `select_by_text` invoked **3×** across the 10 cases with **0 ERROR responses** (vs 100/100 in the historical v9.1_fix wrapper-bypass bug); **6 valid frames** returned from real Stage-1; 1 case hit the audit's "empty-pred" wall (spatial-anchor phrasing on a chalkboard query). 145 tool calls rendered, 115 image assets. |
| [v9_3_strat600_subset_design_20260517.md](v9_3_strat600_subset_design_20260517.md) | **600-case stratified NR3D fold** intended as a faster proxy for the canonical filtered 7805. Stratified on `(is_easy, is_view_dep)` with salt search across 4096 candidates to pin the salt-locked metrics to within ±0.19 pp of v9.1_fix FULL on every leaderboard column (Overall 0.8300 vs 0.8295, Easy −0.09, Hard +0.19, V-Dep −0.03, V-Indep +0.09). 1000-trial Monte-Carlo bootstrap confirms the design is unbiased (|bias| < 0.0015 / metric) with ~±2.3 pp 90 % band on Overall. 119 / 130 scenes (91.5 %), 65 / 72 categories (90.3 %). Fold ID: `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`; salt: `nr3d_v9_3_strat600_v291`. |
| [v9_1_fix_FULL_REPRO_20260516.md](v9_1_fix_FULL_REPRO_20260516.md) | 🥇 **v9.1_fix run on FULL 8584 test fold** at the original commit `d5f40ba`. **classification_acc_filtered = 82.95 %** — the highest depth-aware NR3D test result in the project (beats the invalidated v3 80.79 %). +17.75 pp over v9.2 text-first / +18.30 pp over v9.2 no-text on the same 7805-Q fold. Hard +22.80, V-Dep +21.40 — confirms the catalog-only fallback policy dominates "Stage-1 first move" by a large margin, not just at random100 scale. **v9.3 should default to catalog-only and surface Stage-1 only conditionally.** |
| [v9_1_fix_FULL_trace_20260516.html](v9_1_fix_FULL_trace_20260516.html) | VG agent trace viewer (2T+4F) for the v9.1_fix FULL REPRO run. 2 correct cases where v9.1_fix uniquely won vs v9.2 text-first (view-dep "standing in the middle facing the carts", attribute "tall skinny white picture") — show the catalog-enumerate-after-Stage-1-ERROR pattern. 4 wrong cases (2 where v9.2 won, 2 both-wrong) for failure-mode contrast. |
| [v9_1_fix_reproduction_20260516.md](v9_1_fix_reproduction_20260516.md) | **v9.1_fix random100 reproduction** at the original commit `d5f40ba` (worktree). 86 → **84** within ±2 pp single-seed noise band. Confirms the 86 baseline is **not** a measurement artifact, and the v9.1_fix → v9.2 16-18 pp regression is real (8-9× the noise band). Adds the `Pre-run checklist (MANDATORY)` rule to `CLAUDE.md` so every iteration commits + records both head + run-time SHAs. |
| [v9_2_select_by_text_ab_20260516.md](v9_2_select_by_text_ab_20260516.md) | **random100 A/B** (n=100, ~13 min): text-first **68.0 %** vs catalog-first **63.0 %** (Δ +5 pp). Direction same as full set, magnitude inflated by small-fold noise. Useful as a quick smoke before launching the 15h full-set run. |
| [v9_1_select_by_text_audit_20260516.md](v9_1_select_by_text_audit_20260516.md) | Diagnostic — `select_by_text` reliability vs GT frame visibility on the same random100 fold. Headline: **hit@3 = 55 %** (47 % at K=1, plateaus at K=10), driven by a **32 % empty-prediction wall** (parser/executor returns `no_evidence`). When non-empty, hit@3 = **81 %** and `mean_first_hit_rank` ≈ 1.16 — Stage-1 is either great or hopeless. Strongest on hard / rare-target queries (+31 pp vs random), hurts on easy ones. Quantitatively reconciles the v9.1_fix → v9.1_real regression and gives the v9.1 → v9.2 playbook trade. |
| [v9_1_real_stage1_actually_works_20260516.md](v9_1_real_stage1_actually_works_20260516.md) | v9.1_real — Stage-1 wrapper-fix (`de8225f`); `select_by_text` truly reaches Stage-1 in 100/100 samples (Overall **69.00**, Hard **57.63**, View-dep **58.82**). **−17 pp regression** reveals the v9.1 "first move = `select_by_text`" prior is wrong for NR3D. Bug fix is correct; playbook reorder is the next move. |
| [v9_1_real_trace_20260516.html](v9_1_real_trace_20260516.html) | VG agent trace viewer (2T+4F) on the v9.1_real run — failures show how Stage-1 returns geometrically misleading frames for view-dep / attribute queries. |
| [v9_1_fix_keyframe_selector_wiring_20260515.md](v9_1_fix_keyframe_selector_wiring_20260515.md) | v9.1_fix — INCOMPLETE wiring fix; `Stage2DeepResearchAgent.build_agent` wrapper still bypassed the selector. Run scored Overall **86.00** but with `select_by_text` silently erroring in 100/100 samples — score reflects catalog-only fallback, not v9.1 design. Superseded by `v9_1_real_*`. |
| [v9_1_fix_trace_20260515.html](v9_1_fix_trace_20260515.html) | VG agent trace viewer (2T+2F batch 1) on the v9.1_fix run — same 4 sample IDs as the v9.1 broken trace for side-by-side comparison. |
| [v9_1_fix_trace_batch2_20260515.html](v9_1_fix_trace_batch2_20260515.html) | VG agent trace viewer (2T+2F batch 2) on the v9.1_fix run — 4 fresh cases including hard / view-dep wins (scene0164, scene0131) and instructive failures (scene0697, scene0149). |
| [v9_1_selectors_return_images_20260515.md](v9_1_selectors_return_images_20260515.md) | v9.1 initial — selectors-return-RGB + mark_frame_with_bbox + BEV fixes (Overall **81.00**, Hard **79.66**). −5 pp regression caused by a Stage-1 wiring bug; see v9.1_fix for the resolution. |
| [v9_1_selectors_return_images_trace_20260515.html](v9_1_selectors_return_images_trace_20260515.html) | VG agent trace viewer (4 cases) from the v9.1 initial run; mostly useful as a "before-fix" reference. |
| [v9_catalog_first_20260515.md](v9_catalog_first_20260515.md) | v9 catalog-first random100 pilot, leak-fixed (Overall **86.00**, Hard **81.36**). |
| [v9_selective_mark_random100](v9_selective_mark_random100_20260514.md) | Latest clean-initial + selective marked-image random100 pilot. |
| [v9_selective_mark_trace_2t2f_20260514.html](v9_selective_mark_trace_2t2f_20260514.html) | Static VG agent trace viewer for 2 correct and 2 failed v9 random100 cases. |
| [v8_clean_initial_marked_on_demand_random100](v8_clean_initial_marked_on_demand_random100_20260514.md) | Clean-initial/marked-on-demand negative random100 ablation. |
| [v8_clean_initial_trace_2t2f_20260514.html](v8_clean_initial_trace_2t2f_20260514.html) | Static VG agent trace viewer for 2 correct and 2 failed v8 random100 cases. |
| [v7_stage1_callbacks_noclip_random100](v7_stage1_callbacks_noclip_random100_20260513.md) | Best current depth-aware random100 callback-wired pilot. |
| [v6_inline_labels_depth_visible_random100](v6_inline_labels_depth_visible_random100_20260513.md) | Depth-aware random100 no-NMS rerender pilot before NR3D callbacks were wired. |
| [v6_random100_case_studies_20260513.html](v6_random100_case_studies_20260513.html) | Full Stage1+Stage2 visual walkthrough for 2 correct and 2 failed v6 random100 cases. |
| [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md) | Latest full-test audit record; invalidated pending depth-aware rerun. |
| [v5p1_case_studies_20260513.html](v5p1_case_studies_20260513.html) | Visual stage1+stage2 reasoning walkthrough for selected correct and failed cases. |
| [runs.sqlite](runs.sqlite) | Queryable per-run and per-sample metrics. |

## Current Result Status

**Headline honest depth-aware result on the canonical filtered full set**:

**v11 proposal enrichment FULL 72.26 %** — see
[v11_enrichment_full_20260523.md](v11_enrichment_full_20260523.md).
Runtime HEAD `63dc417`, no initial keyframes, no pending-image side channel,
full-scene proposal enrichment available through initial notes and
`inspect_proposal`.

Best observed valid **strat600 pilot**: [v20 Codex SDK CLI-only
strat600](v20_codex_sdk_cli_only_strat600_20260606.md), **74.50 %**
Overall on the canonical case600 fold. Treat this as a pilot result until it is
confirmed on the full filtered 7805-query slice.

| Metric | v11 enrichment FULL (cite this for fair full set) | v20 Codex SDK CLI-only strat600 | UniVLG (public SOTA) | Delta v11 vs UniVLG |
|---|---:|---:|---:|---:|
| Overall   | **72.26** | **74.50** | 65.2 | **+7.06** |
| Easy      | **81.74** | 81.38 | 73.3 | **+8.44** |
| Hard      | 63.39 | **68.06** | 57.0 | **+6.39** |
| V-Dep     | 62.46 | **64.45** | 55.1 | **+7.36** |
| V-Indep   | 77.60 | **79.95** | 69.9 | **+7.70** |

The v11 full row is the current best fair full-set result in this archive:
+7.06 pp over the public UniVLG headline, with **zero initial-keyframe or
pending-image side channel** and 0 final checkpoint `error` fields. The v20
Codex SDK CLI-only 74.50 % strat600 row is the best observed valid
pilot-fold row; the v11 strat600 pilot at 71.33 % was slightly lower, while
the full fold lands at 72.26 %. Keep the high-worker reproduction caveat:
transient ModelHub/API failures must be deleted and rerun before accepting
final metrics.

Latest spec-cleanup run:
[v10_no_initial_keyframes_strat600_20260518.md](v10_no_initial_keyframes_strat600_20260518.md)
removes all initial keyframe evidence from Stage 2 and requires active
`select_by_*` / marking tool calls for first-person evidence. The initial
`220f128` run completed 600 / 600 with 0 no-match and 0 tracebacks, but
scored **64.33 %** Overall. The follow-up `cfee0ef` run additionally removed
the bundle/runtime pending-image queue and routes first-person tool images only
through `tool_trace.image_metadata`; it produced 600 side-by-side rows, 590
completed statuses, 10 final failed statuses, 0 tracebacks, and **61.00 %**
Overall. Later standard follow-ups add a no-GT multi-anchor spatial comparison
tool (`b671243`, **70.00 %**) and TADG binding for unresolved multi-anchor
evidence (`a3ff7f1`, **72.00 %**). Later reproductions on the best branch
returned **70.00 %** at `workers=30` and **69.17 %** at `workers=20`, so use
the original row as the best observed result rather than a deterministic score.

**Previously claimed depth-aware best** (`v9_1_fix_FULL_REPRO_20260516` =
82.95 %) was **invalidated on 2026-05-17** as an information leak: the 5
GT-target-visible seed keyframes that pack-prep writes into
`bundle.keyframes` were silently auto-injected into agent context every
evidence-update turn at commit `d5f40ba`. The leak was plugged in
`8ebf701`; the v9.1_fix FULL run pre-dates that fix. See
[**v9_4_gt_leak_postmortem_20260518.md**](v9_4_gt_leak_postmortem_20260518.md)
for the full investigation. The 82.95 % row joins v3 (80.79 %) and v5.1
(68.48 %) in the "GT-leak-invalidated" category and must not be quoted
against public NR3D SOTA.

The diagnostic flag `--restore-stage1-seed-keyframe-drain` cleanly
reproduces the leak's effect on v9.3 code: it lifts v9.4-A's 63.17 % to
**v9.4-D's 83.33 %** on the same strat600 fold, recovering 6 / 6
audit-isolated magic cases. This is the **GT-visible upper bound**, not
a publishable number.

For provenance: the older v9.x row table below is preserved as audit
trail, but Overall / per-tier columns for `v9_1_fix_FULL_REPRO` are now
treated the same way as v3 and v5.1.

| Metric | v9.1_fix FULL (invalidated, leak) | v3 (invalidated, projection-only viz) | v5.1 (invalidated, projection-only viz) | **v9.3 text-first (fair)** | v9.2 text-first (fair) | UniVLG SOTA |
|---|---:|---:|---:|---:|---:|---:|
| Overall   | ~~82.95~~ | ~~80.79~~ | ~~68.48~~ | **66.67** | 65.20 | 65.2 |
| Easy      | ~~88.36~~ | ~~86.06~~ | ~~78.43~~ | 73.45 | 76.01 | 73.3 |
| Hard      | ~~77.88~~ | ~~75.87~~ | ~~59.18~~ | 60.32 | 55.08 | 57.0 |
| V-Dep     | ~~78.23~~ | ~~72.46~~ | ~~57.38~~ | 54.98 | 56.83 | 55.1 |
| V-Indep   | ~~85.51~~ | ~~85.34~~ | ~~74.53~~ | 73.01 | 69.76 | 69.9 |

### v5 / v5.1 records (older "fair-view" full runs) are still invalidated pending rerun

Their packs were built from NR3D
`visibility_index.pkl` files whose metadata records `use_depth=False`, so
`view_to_objects` / `object_to_views` encoded projection/frustum
candidates rather than depth-occlusion visibility. Those indices are not
valid as agent-visible evidence.

Latest recorded, invalidated fair-view full-test row:

- Version: `v5p1_failed_rerun_full_20260513`
- Branch: `feat/nr3d-v4-agent-guards-fair-views`
- Run-time code commit: `c404536`
- Documentation commit introducing the v5.1 record: `6000e2b`
- Fold: 8584 NR3D test utterances; n_filtered=7805 after canonical
  `mentions_target_class=True`
- Evidence selection: query-driven fair keyframes; target id / GT bbox used
  only for scoring
- Stage 2 backend: `gpt-5.4-2026-03-05`
- Raw merged artifacts: `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- SQLite run id: `v5p1_failed_rerun_full_20260513`
- Case-study HTML:
  [v5p1_case_studies_20260513.html](v5p1_case_studies_20260513.html)

| Metric | v5.1 recorded, invalidated | v5 before failed-rerun | v3 recorded, invalidated | UniVLG public SOTA |
|---|---:|---:|---:|---:|
| Overall | **68.48** | 66.53 | 80.79 | 65.20 |
| Easy | **78.43** | 76.09 | 86.06 | 73.30 |
| Hard | **59.18** | 57.59 | 75.87 | 57.00 |
| View-Dep | **57.38** | 55.74 | 72.46 | 55.10 |
| View-Indep | **74.53** | 72.41 | 85.34 | 69.90 |

The v5.1 run reran all 241 failed sentinels from v5. Of those, 240 recovered
to completed outputs and 1 persisted as no-match. That persistent row is
filtered out of the 7805Q headline because `mentions_target_class=false`.
The rerun adds 152 correct samples inside the filtered fold and moves overall
from 66.53 to 68.48.

Interpretation:

- v5.1 is no longer a valid fair-view NR3D result because the underlying
  object-frame visibility index did not use depth occlusion.
- Do not compare v5.1 against UniVLG or any other SOTA row until the depth-aware
  visibility indices are rebuilt, packs regenerated, and the full run rerun.
- v3 is also invalidated as a benchmark claim because its GT-target-visible
  shortcut used the same projection-only visibility source.

Latest depth-aware partial pilot (catalog-first, leak-fixed):

- Version: `v9_catalog_first` (clean re-run)
- Branch / commit: `feat/v9-catalog-first-scene-exploration` / leak-fix commit
  (see version doc)
- Scope: same v4/v6/v7/v8/v9_selective_mark random100 fold; **BEV image + Cat-B
  text inventory only** as initial evidence (no first-person seed keyframes;
  `Stage2RuntimeState.initial_keyframe_paths` now blocks the pack-prep seed
  drain); 6 catalog-first selectors + unified `view_keyframe(mode='auto')` +
  `view_bev`
- Result: Overall **86.00**, Easy **92.68**, Hard **81.36**, View-Dep **82.35**,
  View-Indep **87.88**
- Raw artifacts: `tmp/nr3d_eval_v9_full_clean_20260515_1442/`
- Note: First *clean* catalog-first random100 result. **+12.00 overall** and
  **+18.65 Hard** vs v9_selective_mark — confirms that a global BEV primer
  plus selector-driven evidence wins decisively over first-person seed
  keyframes on this fold. Earlier `v9_full` (leaky, `tmp/nr3d_eval_v9_full_20260515_1401/`)
  scored 81.00 / 85.37 / 77.97 / 73.53 / 84.85; the +5 / +7.3 / +3.4 / +8.8 /
  +3.0 lift is the post-fix evidence that the leak was *biasing* the agent
  toward the GT-target-visible seeds rather than helping it.

Previous depth-aware partial pilot:

- Version: `v9_selective_mark_random100`
- Branch / commit: `feat/nr3d-v4-agent-guards-fair-views` /
  `4a1fba1-dirty-selective-mark`
- Scope: same v4/v6/v7/v8 random100 fold; clean initial RGB keyframes with
  text-only inventories; `list_frame_proposals` plus filtered
  `view_keyframe_marked(categories/proposal_ids)` for crowded frames
- Result: Overall 74.00, Easy 90.24, Hard 62.71, View-Dep 64.71,
  View-Indep 78.79
- Raw artifacts:
  `tmp/nr3d_eval_v9_selective_mark_random100_20260514/`
- Case-study HTML:
  [v9_selective_mark_trace_2t2f_20260514.html](v9_selective_mark_trace_2t2f_20260514.html)
- Note: Selective marked-image rendering, but kept first-person seed
  keyframes; superseded by v9_catalog_first.

Earlier depth-aware partial pilot:

- Version: `v8_clean_initial_marked_on_demand_random100`
- Branch / commit: `feat/nr3d-v4-agent-guards-fair-views` / `f86c161`
- Scope: same v4/v6/v7 random100 fold; clean initial RGB keyframes with
  text-only left-to-right proposal inventories; marked images only through
  `view_keyframe_marked`
- Result: Overall 67.00, Easy 85.37, Hard 54.24, View-Dep 58.82,
  View-Indep 71.21
- Raw artifacts:
  `tmp/nr3d_eval_v8_clean_initial_marked_on_demand_random100_20260514/`
- Note: Negative ablation vs v7. The change improves Easy but significantly
  hurts Hard and View-Dep; keep v7 as the stronger random100 baseline.

Previous best depth-aware random100 pilot:

- Version: `v7_stage1_callbacks_noclip_random100`
- Branch / commit: `feat/nr3d-v4-agent-guards-fair-views` / `690cbf6`
- Scope: same v4 random100 fold; preserved v4 keyframe frame ids; no selector
  NMS
- Result: Overall 73.00, Easy 82.93, Hard 66.10, View-Dep 70.59,
  View-Indep 74.24
- Raw artifacts:
  `tmp/nr3d_eval_v7_callbacks_noclip_random100_20260513/`
- Note: NR3D Stage1 callbacks are wired; `request_more_views` disables
  per-selector CLIP object-term fallback to stay within the 15GB RSS budget at
  `workers=100`.

## Version Timeline

| Version | Date | Branch / commit | Headline | Scope | Status |
|---|---|---|---:|---|---|
| [v1_phase8_smoke20_mac](v1_phase8_smoke20_mac_20260430.md) | 2026-04-30 | `feat/nr3d-vg-benchmark` / `9115fd7` | Acc@0.50=65.00 | 20Q smoke | Invalidated: projection-only visibility |
| [v2_phase8_full](v2_phase8_full_20260501.md) | 2026-05-01 | `feat/nr3d-vg-benchmark` / `ad8d439` plus in-flight fixes | Acc@0.50=77.62 | 8584Q full | Invalidated: projection-only visibility |
| [v3_referit3d_track](v3_referit3d_track_20260501.md) | 2026-05-01 | `feat/nr3d-vg-benchmark` / `b6f211a` | Overall=80.79 | 8584Q / 7805Q filtered | Invalidated: projection-only GT-visible source |
| [v4_agent_guards_fair_views](v4_agent_guards_fair_views_20260512.md) | 2026-05-12 | `feat/nr3d-v4-agent-guards-fair-views` / `3f1c6d8` | Overall=71.00 | 100Q pilot | Invalidated: projection-only visibility |
| [v5_agent_guards_fair_views_full](v5_agent_guards_fair_views_full_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `7c996ad` | Overall=66.53 | 8584Q / 7805Q filtered | Invalidated: projection-only visibility |
| [v5p1_failed_rerun_full](v5p1_failed_rerun_full_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `c404536` | Overall=68.48 | 8584Q / 7805Q filtered | Invalidated: projection-only visibility |
| [v6_inline_labels_depth_visible_random100](v6_inline_labels_depth_visible_random100_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `41253ad` | Overall=71.00 | 100Q pilot | Depth-aware partial, no NMS |
| [v7_stage1_callbacks_noclip_random100](v7_stage1_callbacks_noclip_random100_20260513.md) | 2026-05-13 | `feat/nr3d-v4-agent-guards-fair-views` / `690cbf6` | Overall=73.00 | 100Q pilot | Depth-aware partial, callbacks wired, no NMS |
| [v8_clean_initial_marked_on_demand_random100](v8_clean_initial_marked_on_demand_random100_20260514.md) | 2026-05-14 | `feat/nr3d-v4-agent-guards-fair-views` / `f86c161` | Overall=67.00 | 100Q pilot | Depth-aware partial, negative clean-initial ablation |
| [v9_selective_mark_random100](v9_selective_mark_random100_20260514.md) | 2026-05-14 | `feat/nr3d-v4-agent-guards-fair-views` / `4a1fba1-dirty-selective-mark` | Overall=74.00 | 100Q pilot | Depth-aware partial, selective marked-image rendering |
| [v9_catalog_first_20260515](v9_catalog_first_20260515.md) | 2026-05-15 | `feat/v9-catalog-first-scene-exploration` / leak-fix | Overall=**86.00** (clean) / 81.00 (leaky) | 100Q pilot | Depth-aware partial, BEV-first + 6 selectors, no first-person seed; Stage-1 seed-keyframe drain leak documented + fixed |
| [v9_2_select_by_text_ab_20260516](v9_2_select_by_text_ab_20260516.md) | 2026-05-16 | `feat/v9-1-selectors-return-images` / `c2c52d0` | text-first **68.00** / catalog-first **63.00** | 100Q A/B | Depth-aware partial; `enable_stage1_text_retrieval` toggle; same code/prompts/pack — only the toggle differs. Audit's catalog-first hypothesis falsified at random100 (−5 pp). |
| [v9_2_full_select_by_text_ab_20260516](v9_2_full_select_by_text_ab_20260516.md) | 2026-05-16 | `feat/v9-1-selectors-return-images` / `c2c52d0` | text-first **65.20** / catalog-first **64.65** | **8584Q / 7805Q filtered A/B** | First full-set v9.x evaluation. 15h28m parallel on Mac, workers=20 each, 0 Python errors. Gap shrinks from random100's 5pp to 0.55pp at full scale; per-tier text-first wins Hard / V-Indep, catalog-first wins V-Dep. Default decision: keep `select_by_text` on. |
| [v9_1_fix_FULL_REPRO_20260516](v9_1_fix_FULL_REPRO_20260516.md) | 2026-05-16 | `feat/v9-1-selectors-return-images` / head `e750d8b`, runtime `d5f40ba` (worktree) | **82.95 / 88.36 / 77.88 / 78.23 / 85.51** | 8584Q / 7805Q filtered | **🥇 New depth-aware NR3D high.** v9.1_fix wrapper-bypass bug reproduces broken Stage-1 catalog fallback on full set. +17.75 / +22.80 / +21.40 pp on Overall / Hard / V-Dep vs v9.2 text-first. workers=40 single side, ~8h32m wall, 0 Python errors. |
| [v9_3_strat600_subset_design_20260517](v9_3_strat600_subset_design_20260517.md) | 2026-05-17 | `feat/v9-1-selectors-return-images` / `4728205` | _no run; fold design_ | 600Q stratified ⊂ 7805Q filtered | Salt-locked 600-case fold pinned to within ±0.19 pp of v9.1_fix FULL on every leaderboard metric (Overall **0.8300** vs ref 0.8295). 1000-trial bootstrap shows design unbiased, 90 % band ±2.3 pp on Overall. Use as a fast proxy for full-set runs (estimated 1h vs 8h on workers=20+). |
| [v9_3_strat600_20260517](v9_3_strat600_20260517.md) | 2026-05-17 | `feat/v9-1-selectors-return-images` / `ae83fea` | Overall **66.67** / Easy **73.45** / Hard **60.32** / V-Dep **54.98** / V-Indep **73.01** | 600Q strat600 (=300Q Easy + 300Q Hard, 211Q V-Dep + 389Q V-Indep) | **First benchmark run on v9.3 HEAD** (keyframe_selector fix + BEV/mark_frame UI overhaul + `_v93` cache tag). 32.2 min wall on workers=40. Re-confirms the v9.2 text-first FULL band (Overall +1.47 pp inside ±2.3 pp; Hard +5.24 pp slightly outside but positive). 0 `keyframe_selector is None` errors, 0 stale-cache hits, 0 tracebacks. 185 `select_by_text` calls all reach real Stage-1 parser; 38 ERROR returns are legitimate `Masked category leak` guard fires (agent self-contradicting hidden_categories). Strong UI adoption: 179 `view_bev(categories=…)` + 424 `view_bev(highlight=…)`, 1 501 `mark_frame_with_bbox(ids=…)`. |
| [v9_3_strat600_notext_20260517](v9_3_strat600_notext_20260517.md) | 2026-05-17 | `feat/v9-1-selectors-return-images` / `aa11922` | Overall **64.17** / Easy **73.10** / Hard **55.81** / V-Dep **56.87** / V-Indep **68.12** | 600Q strat600 (paired A/B with v9_3_strat600_20260517) | **Catalog-only A/B partner**, only `--disable-stage1-text-retrieval` differs. Δ vs text-first paired run: −2.50 pp Overall (just inside ±2.3 pp band), Hard −4.51, V-Indep −4.89, V-Dep **+1.90** — same per-tier direction as v9.2 FULL A/B. **Δ vs v9.1_fix FULL = −18.78 pp** → **refutes "clean catalog-only = 83 %" hypothesis**; the 83 was a bug-state fall-through artifact, not the no-text policy. 0 select_by_text calls, 1245 `_no_text` playbook loads verified; 0 failed, 0 tracebacks. **Next: audit fall-through behaviour or per-query routing (see doc §What's next).** |
| [v9_1_fix_vs_v9_3_audit30_20260517](v9_1_fix_vs_v9_3_audit30_20260517.md) | 2026-05-17 | `d5f40ba` (worktree) + `ae83fea` + `aa11922` | Overall **60.0 / 46.7 / 53.3** (v9.1_fix / v9.3-txt / v9.3-no) | 30Q stratified audit subset of strat600 | **Per-sample audit on the 16 pp gap.** v9.1_fix wins 6 cases both v9.3 variants lose; v9.3 wins 2 cases v9.1_fix loses; net +4 / 30 to v9.1_fix (matches full-fold 16 pp Overall gap within n=30 variance). Recurring v9.1_fix pattern: forced `select_by_text→ERR` → catalog walk → per-candidate marking → evidence-frame-guard-driven re-mark + re-submit cycle. mark_frame_with_bbox rate +36 % vs v9.3-no; inspect_proposal rate is IDENTICAL across all three at 3.07 / sample. **Conclusion: gap is deliberation cadence, not policy or tool surface.** Proposes 3 concrete v9.4 experiments (force ERR + keep text-first playbook is the cheapest). |
| [v9_4a_strat600_force_error_20260517](v9_4a_strat600_force_error_20260517.md) | 2026-05-17 | `feat/v9-1-selectors-return-images` / `5180d60` | Overall **63.17** / Easy **71.03** / Hard **55.81** / V-Dep **55.45** / V-Indep **67.35** | 600Q strat600 (v9.4 Experiment A — force_stage1_text_retrieval_to_error) | **Negative result.** New `--force-stage1-text-retrieval-to-error` flag keeps text-first playbook + tool registered, makes tool body short-circuit to ERROR before selector. 174/174 calls hit new force-disabled ERROR signature (wiring confirmed). −3.50 pp vs v9.3 text-first, −1.00 pp vs v9.3 no-text, **−19.83 pp vs v9.1_fix strat600 calibrated**. Agent only called select_by_text on 29 % of samples (vs v9.1_fix's 100 %), so playbook prose alone doesn't force the cadence. **Recovered 0 / 6 audit magic cases.** Refutes simple "cadence from ERR signature" hypothesis. Next-highest-leverage = Experiment D (restore Stage-1 seed-keyframe drain leak that was present at d5f40ba but fixed in 8ebf701). |
| [v9_4d_strat600_force_error_seed_drain_20260517](v9_4d_strat600_force_error_seed_drain_20260517.md) | 2026-05-17 | `feat/v9-1-selectors-return-images` / `a0d802b` | Overall **83.33** / Easy **84.48** / **Hard 82.26** / V-Dep **76.78** / V-Indep **86.89** | 600Q strat600 (v9.4 Experiment D — force-ERR + restore-drain) | **🎯 ISOLATES THE 16 pp.** Adds `--restore-stage1-seed-keyframe-drain` (bypasses v9.3's seed-filter, restoring the GT-target-visible RGB injection that was present at `d5f40ba` and fixed in `8ebf701`) on top of v9.4-A's force-ERR. Result: within **+0.33 pp of v9.1_fix calibrated 83.00**, Hard +4.20pp ABOVE v9.1_fix. **Recovered 6/6 magic audit cases**. Each evidence-update injects 6 images (1 BEV + 5 GT-visible seeds). Conclusion: v9.1_fix's 82.95 % is **an information leak**, not a fair NR3D number. v9.1_fix FULL row should be re-tagged with fair-view caveat. Honest v9.3 baseline = ~65 % matches public UniVLG SOTA. Test-time flag only; never ship as a published number. |
| [v10_no_initial_keyframes_strat600_20260518](v10_no_initial_keyframes_strat600_20260518.md) | 2026-05-18 | `feat/remove-initial-keyframes` / `220f128` | Overall **64.33** / Easy **72.41** / Hard **56.77** / V-Dep **53.55** / V-Indep **70.18** | 600Q strat600 (no initial keyframe evidence) | Spec-cleanup validation. Removes pack-prepared initial keyframe evidence from Stage 2; first-person evidence is reachable only through active `select_by_*` / marking tools. 600 / 600 completed, 0 no-match, 0 tracebacks, eval + metrics exit 0. Not a new accuracy best; main case-level issues are unsupported spatial aliases, masked-category guard blocks, and guard-induced target/anchor corrections. |
| [v10_tool_trace_image_metadata_strat600](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-tool-trace-image-metadata-cfee0ef) | 2026-05-18 | `feat/remove-initial-keyframes` / `cfee0ef` | Overall **61.00** / Easy **67.93** / Hard **54.52** / V-Dep **52.13** / V-Indep **65.81** | 600Q strat600 (tool images via `tool_trace.image_metadata`) | Architecture validation after removing the remaining bundle/runtime pending-image side channel. `vg_pending_images`, `pending_image_paths`, `pending_image_metadata`, and `queue_pending` are absent from source and artifacts. Eval + metrics exit 0; 600 side-by-side rows; 590 completed and 10 final failed statuses. Net −20 correct vs `220f128` (73 old misses recovered, 93 old positives regressed). Dominant failure mode: target-category guard target/anchor misparsing. |
| [v10_target_category_guard5_20260519](v10_no_initial_keyframes_strat600_20260518.md#target-category-guard-recovery-probe-2ce81df) | 2026-05-19 | `feat/remove-initial-keyframes` / `2ce81df` | Overall **100.00** on n=5 | 5Q diagnostic recovery probe | Not benchmark-grade. Covers five `cfee0ef` guard-deadlock failures after fixing head-noun parsing in `target_category_guard`. All 5 completed and selected the target id with `target_category_guard_blocked=false`; confirms the local root-cause fix without GT inputs. |
| [v10_missing_status4_20260519](v10_no_initial_keyframes_strat600_20260518.md#missing-status-extractor-recovery-probe-e5617ac) | 2026-05-19 | `feat/remove-initial-keyframes` / `e5617ac` | Overall **100.00** on n=4 | 4Q diagnostic recovery probe | Not benchmark-grade. Covers four `cfee0ef` synthetic missing-status failures after the NR3D extractor learned to resolve direct structured `proposal_id` outputs through the fair proposal pool. All 4 completed, selected the target id, and preserved query/tool trace. |
| [v10_no_gt_fixes_strat600_20260519](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-guardextractor-fixes-a6f6077) | 2026-05-19 | `feat/remove-initial-keyframes` / launch+runtime `a6f6077` | Overall **62.67** / Easy **71.03** / Hard **54.84** / V-Dep **53.55** / V-Indep **67.61** | 600Q strat600 (standard guards, no GT inputs) | Standard rerun after target-category guard and NR3D extractor fixes. Eval + metrics exit 0; 600 side-by-side rows; 597 completed and 3 final failed statuses. +10 correct vs `cfee0ef` (79 old misses recovered, 69 old positives regressed). Remaining final failed buckets: target-category generic/anchor parsing and unsupported `same side as`. |
| [v10_viewdep15_probe_20260519](v10_no_initial_keyframes_strat600_20260518.md#view-dependent-side-evidence-probe-23f68de) | 2026-05-19 | `feat/remove-initial-keyframes` / `23f68de` | Overall **33.33** on n=15 | 15Q diagnostic recovery probe | Not benchmark-grade. Prompt/skill-only update after a new 15-case audit: view-dependent left/right must use a same marked viewer/anchor frame and cannot combine screen-left/right across viewpoints. Recovers 5 / 15 old failures; remaining failures motivate role-bound relation evidence, noun-scoped EFG, operative target-head parsing, and candidate-closure gates. |
| [v10_bookshelf_guard_probe_20260519](v10_no_initial_keyframes_strat600_20260518.md#bookshelf-target-category-guard-probe-c5a75ae) | 2026-05-19 | `feat/remove-initial-keyframes` / `c5a75ae` | Overall **100.00** on n=1 | 1Q diagnostic recovery probe | Not benchmark-grade. Fixes target-category guard priority so `Find the bookshelf ... angled wall` targets `bookshelf`, not context `wall`. Probe recovers `scene0208_00::106::14558` from failed to `selected_object_id=106`, IoU 1.0. |
| [v10_guard_scope_probe15_20260519](v10_no_initial_keyframes_strat600_20260518.md#guard-scope-15-case-probe-6355781) | 2026-05-19 | `feat/remove-initial-keyframes` / `6355781` | Overall **53.33** on n=15 | 15Q diagnostic recovery probe | Not benchmark-grade. Fixes TCG later-option target-head scoping and EFG room/anchor-side scope. Probe completes 15 / 15 and recovers 8 / 15 failures from `v10_no_gt_fixes_strat600_20260519`; remaining failures point to candidate closure and role-bound relation evidence. |
| [v10_candidate_closure_probe15_20260519](v10_no_initial_keyframes_strat600_20260518.md#candidate-closure-15-case-probe-06b8d58) | 2026-05-19 | `feat/remove-initial-keyframes` / `06b8d58` | Overall **33.33** on n=15 | 15Q diagnostic recovery probe | Not benchmark-grade. Adds a no-GT EFG candidate-closure soft block over small same-category sets with comparison cues. Probe completes 15 / 15 and recovers 5 / 15 failures from `v10_no_gt_fixes_strat600_20260519`; candidate-closure messages fired in 7 submit traces. |
| [v10_rationale_payload_probe3_20260519](v10_no_initial_keyframes_strat600_20260518.md#rationalepayload-consistency-probe-be3c4fc) | 2026-05-19 | `feat/remove-initial-keyframes` / `be3c4fc` | Overall **66.67** on n=3 | 3Q diagnostic recovery probe | Not benchmark-grade. Adds a no-GT `submit_final` rationale/payload consistency guard. The live probe completes 3 / 3 and recovers 2 / 3 stale-payload-audited cases; offline scan over the prior strat600 `submit_final` traces found 3 / 600 contradictions after false-positive tightening. |
| [v10_subtype_probe3_20260519](v10_no_initial_keyframes_strat600_20260518.md#generic-subtype-category-probe-dbd9adb) | 2026-05-19 | `feat/remove-initial-keyframes` / `dbd9adb` | Overall **66.67** on n=3 | 3Q diagnostic probe | Not benchmark-grade. Adds asymmetric generic/subtype category matching (`chair` may accept `office chair`) and playbook guidance. The probe preserves the 2 / 3 recovered cases but does not fix `scene0663_00::6::33306`; trace shows a nested-anchor failure around "desk closest to the window". |
| [v10_nested_anchor_probe3_20260519](v10_no_initial_keyframes_strat600_20260518.md#nested-anchor-flow-probe-3b675f9) | 2026-05-19 | `feat/remove-initial-keyframes` / `3b675f9` | Overall **66.67** on n=3 | 3Q diagnostic probe | Not benchmark-grade. Prompt/skill-only nested-anchor update recovers `scene0663_00::6::33306` (`#33` -> `#6`) with the intended desk-to-window comparison, but regresses `scene0222_00::20::35738` (`#20` -> `#19`). |
| [v10_negated_anchor_probe3_20260519](v10_no_initial_keyframes_strat600_20260518.md#negated-anchor-prompt-probe-fd1a628) | 2026-05-19 | `feat/remove-initial-keyframes` / `fd1a628` | Overall **33.33** on n=3 | 3Q diagnostic probe | Negative. Marked-positive negated-anchor prompt recovers the pillow case (`#19` -> `#20`) but regresses the nested-chair and outside-door cases (`#6` -> `#33`, `#21` -> `#8`). |
| [v10_below_align_probe15_20260519](v10_no_initial_keyframes_strat600_20260518.md#abovebelow-alignment-tool-probe-54198ef) | 2026-05-19 | `feat/remove-initial-keyframes` / `54198ef` | Overall **53.33** on n=15 | 15Q diagnostic probe | Not benchmark-grade. Tool-level above/below horizontal-alignment fix is unit-covered. Live same-slice probe completes 15 / 15 and is 8 / 15 vs 5 / 15 at `06b8d58`, but traces contain no above/below `compare_proposals_spatial` calls, so next work is prompt/skill routing. |
| [v10_vertical_route_probe15_20260519](v10_no_initial_keyframes_strat600_20260518.md#vertical-relation-routing-prompt-probe-61042c3) | 2026-05-19 | `feat/remove-initial-keyframes` / `61042c3` | Overall **33.33** on n=15 | 15Q diagnostic probe | Negative. Broad playbook wording routes `scene0077` into the fixed `below` comparator and preserves that case, but drops the slice from 8 / 15 to 5 / 15 with regressions on cabinet/window-facing/chalkboard cases. |
| [v10_vertical_tadg_probe15_20260519](v10_no_initial_keyframes_strat600_20260518.md#vertical-relation-tadg-guard-probe-66c6218) | 2026-05-19 | `feat/remove-initial-keyframes` / `66c6218` | Overall **33.33** on n=15 | 15Q diagnostic probe | Negative. TADG forces `below` relation evidence on `scene0077` and `scene0063`, but only preserves the already-correct printer case; the chair-under-TV case and two unrelated cases regress vs `54198ef`. |
| [v10_desk_chair_probe26_20260519](v10_no_initial_keyframes_strat600_20260518.md#compound-target-head-guard-probe-2a5a241---b00ca7c) | 2026-05-19 | `feat/remove-initial-keyframes` / `b00ca7c` | Overall **50.00** on n=26 | 26Q diagnostic probe | Not benchmark-grade. Fixes compound explicit target-head parsing so `Choose the all black desk chair` expects generic `chair` and allows `office chair`. Recovers `scene0518_00::3::1731` from failed sentinel to target `#3`, IoU 1.0; same-slice accuracy remains 13 / 26 vs `1d3b02d`, after four recoveries and four regressions. |
| [v10_multi_anchor_strat600_20260519](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-multi-anchor-tool-b671243) | 2026-05-19 | `feat/remove-initial-keyframes` / launch+runtime `b671243` | Overall **70.00** / Easy **78.28** / Hard **62.26** / V-Dep **58.77** / V-Indep **76.09** | 600Q strat600 (standard guards, no GT inputs) | Standard rerun after `compare_candidates_to_anchors` and playbook routing for multi-anchor closest/farthest/near/next_to. Eval + metrics exit 0; 600 side-by-side rows; 598 completed and 2 final failed statuses. +44 correct vs `a6f6077`, +34 vs `220f128`, +54 vs `cfee0ef`. Source/artifact scans found no pending-image side-channel keys. |
| [v10_multi_anchor_tadg15_20260519](v10_no_initial_keyframes_strat600_20260518.md#multi-anchor-tadg-binding-probe-99dffcc) | 2026-05-19 | `feat/remove-initial-keyframes` / `99dffcc` | Overall **53.33** on n=15 | 15Q diagnostic probe | Not benchmark-grade. Teaches TADG to bind `compare_candidates_to_anchors` evidence and block unresolved `anchor_disagreement`. On 15 failed cases from `b671243`, completes 15 / 15 and recovers 8 / 15; `TADG_MULTI_ANCHOR_UNRESOLVED` fires in 9 traces and every blocked trace continues. |
| [v10_multi_anchor_tadg_strat600_20260519](v10_no_initial_keyframes_strat600_20260518.md#full-strat600-rerun-after-multi-anchor-tadg-binding-a3ff7f1) | 2026-05-19 | `feat/remove-initial-keyframes` / launch+runtime `a3ff7f1` | Overall **72.00** / Easy **82.41** / Hard **62.26** / V-Dep **62.56** / V-Indep **77.12** | 600Q strat600 (standard guards, no GT inputs) | Standard rerun after TADG learned `compare_candidates_to_anchors` evidence and blocks unresolved `anchor_disagreement`. Eval + metrics exit 0; 600 side-by-side rows; 600 completed. +12 correct vs `b671243`, +56 vs `a6f6077`, +46 vs `220f128`. Source/artifact scans found no pending-image side-channel keys. |
| [REPRO_20260520_v10_multi_anchor_tadg_strat600_20260519_w30](v10_no_initial_keyframes_strat600_20260518.md#high-worker-reproduction-of-multi-anchor-tadg-binding-a3ff7f1--workers30) | 2026-05-20 | `best/v10-multi-anchor-tadg-72-a3ff7f1` / runtime `a3ff7f1` | Overall **70.00** / Easy **78.62** / Hard **61.94** / V-Dep **61.61** / V-Indep **74.55** | 600Q strat600 high-worker reproduction | Negative reproduction of the 72.00 row with `workers=30` and `MODELHUB_AK_WEIGHTS=1,1,1`. Eval + metrics exit 0; 600 side-by-side rows; 600 completed; 49 recoveries and 61 regressions vs the original run. Log shows 2724 retryable 429s, 19 retryable 403s, 15 attempt-2 retries, and no fatal errors. |
| [REPRO_20260521_v10_multi_anchor_tadg_strat600_20260519_w20](v10_no_initial_keyframes_strat600_20260518.md#workers20-reproduction-of-multi-anchor-tadg-binding-7203e18) | 2026-05-21 | `best/v10-multi-anchor-tadg-72-a3ff7f1` / runtime `7203e18` (docs-only drift from `a3ff7f1`) | Overall **69.17** / Easy **79.31** / Hard **59.68** / V-Dep **62.56** / V-Indep **72.75** | 600Q strat600 workers=20 reproduction | Negative reproduction of the 72.00 row using the original worker count. Eval + metrics exit 0; 600 side-by-side rows; 598 completed / 2 failed; 45 recoveries and 62 regressions vs the original run. Log shows 294 retryable 403s, 56 retryable 429s, 35 attempt-2, 6 attempt-3, 1 attempt-4, and no traceback. |
| [v11_enrichment_strat600_20260521](v11_enrichment_strat600_20260521.md) | 2026-05-21 | `best/v10-multi-anchor-tadg-72-a3ff7f1` / launch+runtime `793cb57` | Overall **71.33** / Easy **79.66** / Hard **63.55** / V-Dep **61.61** / V-Indep **76.61** | 600Q strat600 with proposal enrichment | Standard rerun after adding compact enrichment to initial proposal notes and full enrichment to `inspect_proposal`. Eval + metrics exit 0; 600 side-by-side rows; 600 completed. Stabilization-positive but not a new strat600 best. |
| [v11_enrichment_full_20260523](v11_enrichment_full_20260523.md) | 2026-05-23 | `best/v10-multi-anchor-tadg-72-a3ff7f1` / launch+runtime `63dc417` | Overall **72.26** / Easy **81.74** / Hard **63.39** / V-Dep **62.46** / V-Indep **77.60** | 8584Q full / 7805Q filtered with proposal enrichment | Current best valid full-set row. Eval + metrics regenerated after deleting/rerunning API-error sentinels; final 8584 side-by-side rows, 8492 completed / 92 final failed statuses, and 0 checkpoint error fields. |
| [v11_select_by_text_strat600_coverage_20260524](v11_select_by_text_strat600_coverage_20260524.md) | 2026-05-24 | `best/v10-multi-anchor-tadg-72-a3ff7f1` / launch+runtime `45a2dae` | select_by_text hit@3 **62.17** / non-empty hit@3 **83.07** / empty **25.17** | 600Q strat600 tool coverage audit | Not a leaderboard row. Raw-query `select_by_text(..., k=3, use_visual_context=False)` coverage with 16 shard workers. Direct pack visibility and Phase-8 visibility matched on all 600 rows; 0 script errors. View-dependent hit@3 is **50.71 %**, so this tool should remain a prior rather than sole evidence for spatial/view-dependent queries. |
| [v14_select_by_text_viewpoint_prompt_smoke50_20260524](v14_select_by_text_viewpoint_prompt_smoke50_20260524.md) | 2026-05-24 | `analysis/nr3d-select-by-text-coverage` / launch+runtime `3688096` | select_by_text hit@3 **66.00** / View-Dep **61.11** / viewpoint contexts **7 / 50** | 50Q smoke from strat600 first50 | Not a leaderboard row. Validates that the active parser path now emits real viewpoint contexts after the prompt/schema port. Same-50 v13 comparison is 68.00 -> 66.00 hit@3, so there is no metric-gain claim from this smoke. |
| [v15_select_by_text_viewpoint_prompt_strat600_20260524](v15_select_by_text_viewpoint_prompt_strat600_coverage_20260524.md) | 2026-05-24 | `analysis/nr3d-select-by-text-coverage` / launch+runtime `9c62e51` | select_by_text hit@3 **69.17** / View-Dep **73.93** / viewpoint contexts **98 / 600** | 600Q strat600 tool coverage audit | Not a leaderboard row. Overall is effectively flat vs v13 (+0.17 pp), but parser viewpoint emission is now real and View-Dep improves +4.27 pp. View-Indep drops -2.06 pp, so next audit should inspect context recoveries/regressions before claiming a net selector improvement. |
| [v18_codex_sdk_mcp_text_strat600_20260605](v18_codex_sdk_mcp_text_strat600_20260605.md) | 2026-06-05 | `feat/intro-codex-agent-sdk` / launch+runtime `348ad52` | Overall **63.00** / Easy **70.69** / Hard **55.81** / V-Dep **55.45** / V-Indep **67.10** | 600Q strat600/case600 with Codex SDK MCP/skills/prefix-cache config | Integration validation after adding the Codex SDK MCP tool server, syncing DeepAgents playbooks into skills, and enabling lazy `select_by_text` selector initialization. Eval + metrics exit 0; final side-by-side has 600 completed rows and 0 errors. MCP was enabled in per-sample input, but the observed tool trace has only `codex_sdk_turn` 600 times, so this did not exercise MCP tool calls. |

## Protocol Summary

- Public NR3D leaderboard headline is classification accuracy:
  `selected_object_id == target_id`.
- Canonical local full fold: 8584 utterances, 130 scenes.
- Canonical filtered fold: 7805 utterances after `mentions_target_class=True`.
- Easy/Hard split: `n_objects <= 2`.
- View-Dep split: canonical 10-token literal set from ReferIt3D.
- Candidate pool: full-scene GT segmented / object proposals; the old
  "target-type-only public pool" assumption is retracted.
- Fairness boundary: v3 uses GT-target-visible keyframes; v5/v5.1 use
  query-driven keyframes but are invalidated because their object-frame
  visibility was projection-only. v6/v7/v8/v9 are depth-aware but partial and
  preserve v4 keyframe choices to isolate Stage2/runtime changes before NMS.

See [protocol.md](protocol.md) for the consolidated evidence and caveats.

## Reproduction

### Canonical pilot fold (use this for all small-batch validation)

**`tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`** (600 samples,
stratified on `(is_easy × is_view_dep)`, salt-locked to within ±0.19 pp
of the FULL filtered 7805 on the v9.1_fix reference; bootstrap 90 % band
±2.3 pp on Overall). Use this fold for every iteration / A/B / smoke run
that doesn't need full-7805 statistical power.

```bash
# 1. (one-time) regenerate the fold from data/nr3d/ — byte-stable; MD5 = 12a69d8d14a81519024bbe00d6334434
PYTHONPATH=src python scripts/build_nr3d_strat600_fold.py

# 2. (one-time) prep the pack on the 600 ids (~10 min on Mac)
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --split test \
    --keyframe-mode query_driven --ensure-lightweight-cache

# 3. run the agent on the 600 ids
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v9_catalog_first \
    --output-dir tmp/nr3d_eval_<run_id>/ \
    --workers 20

# 4. aggregate the canonical 5-column leaderboard metrics on the 600 ids
PYTHONPATH=src python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
    --side-by-side tmp/nr3d_eval_<run_id>/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --output docs/benchmark/nr3d/assets/<run_id>_strat600_leaderboard.json
```

Design + bootstrap validation: see
[v9_3_strat600_subset_design_20260517.md](v9_3_strat600_subset_design_20260517.md).

**Variance budget when comparing two strat600 runs**: Overall ±2.3 pp 90 %,
Easy / V-Indep ±2.7–2.9 pp, Hard ±3.5 pp, V-Dep ±4.5 pp (n=119). Any
delta inside its band should be confirmed on the FULL 7805 before being
claimed as a real change.

The older `v4_agent_guards_fair_views_random100_sample_ids.json` fold is
**deprecated for new comparisons** — preserved only for reproducing the
historical v4–v9.2 random100 runs. Per-tier ±5–10 pp noise on random100
was the source of the v9.1 → v9.2 misread before this fold existed.

### Other reproduction recipes

Download the raw NR3D annotation files:

```bash
bash scripts/download_nr3d.sh data/nr3d
```

Load test samples with Phase 8 GT-CG boxes:

```bash
PYTHONPATH=src python -c "
from benchmarks.nr3d_loader import Nr3dDataset
ds = Nr3dDataset.from_path(
    data_root='data/nr3d',
    split='test',
    bbox_source='phase8_gt_cg',
    phase8_data_root='data/nr3d/scannet',
    max_samples=200,
)
print(f'loaded={len(ds)} stats={ds.stats}')
"
```

Ingest a future side-by-side output:

```bash
PYTHONPATH=src python scripts/ingest_nr3d_run.py \
    --output-dir tmp/nr3d_eval_<run> \
    --run-id <run> \
    --branch feat/nr3d-vg-benchmark \
    --commit <short_sha> \
    --backend pack_v1 \
    --judge-model none \
    --notes "NR3D pack_v1 run"
```

Prepare a Phase 8 pack:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --split test
```

For high-concurrency query-driven prep, build lightweight ConceptGraph object
caches once, then enable the ensure flag during prep. If a cache is still
missing, it is built under a cross-process lock and the sample continues:

```bash
PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --build-lightweight-cache-only

PYTHONPATH=src python src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/full_test_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v4_agent_guards_fair_views \
    --split test \
    --keyframe-mode query_driven \
    --ensure-lightweight-cache
```

Run the pack-v1 Stage 2 runner:

```bash
PYTHONPATH=src python src/evaluation/scripts/run_nr3d_vg_side_by_side.py \
    --sample-ids tmp/nr3d_artifacts/v1_smoke20_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --output-dir tmp/nr3d_eval_v1_smoke20 \
    --workers 1
```

Run the v5.1 reproduction path from the version doc:

- Failed rerun subset:
  `tmp/nr3d_artifacts/v5_failed241_sample_ids_20260513.json`
- Base full output:
  `tmp/nr3d_eval_v4_agent_guards_fair_views_full/`
- Failed rerun output:
  `tmp/nr3d_eval_v5_failed_rerun_20260513/`
- Merged output:
  `tmp/nr3d_eval_v5_failed_rerun_merged_20260513/`
- Detailed CLI and ingest commands:
  [v5p1_failed_rerun_full_20260513.md](v5p1_failed_rerun_full_20260513.md)

## SQLite

Canonical DB: `docs/benchmark/nr3d/runs.sqlite`

```sql
SELECT run_id, n,
       printf('%.4f', classification_acc_filtered) AS acc_filtered,
       n_filtered,
       printf('%.4f', mean_iou) AS mean_iou,
       printf('%.4f', acc25) AS acc25,
       printf('%.4f', acc50) AS acc50
FROM runs;
```

Current headline query:

```sql
SELECT run_id, n, n_filtered,
       printf('%.4f', classification_acc_filtered) AS overall,
       printf('%.4f', acc_easy) AS easy,
       printf('%.4f', acc_hard) AS hard,
       printf('%.4f', acc_view_dep) AS vdep,
       printf('%.4f', acc_view_indep) AS vind
FROM runs
WHERE run_id='v5p1_failed_rerun_full_20260513';
-- v5p1_failed_rerun_full_20260513|8584|7805|0.6848|0.7843|0.5918|0.5738|0.7453
```

## Retained / Deleted Docs

Retained:

- README, leaderboard, protocol, runs.sqlite
- one version doc per evaluated run: v1, v2, v3, v4, v5, v5.1

Deleted during consolidation:

- `protocol_audit_20260501.md`
- `paper_crosscheck_20260501.md`
- `pool_equivalence_log_20260501.md`

Their durable conclusions were merged into [protocol.md](protocol.md),
[leaderboard.md](leaderboard.md), and the current-result section above.
