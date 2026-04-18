# 00 — Research Manifest

Load-bearing academic claims of this project, each stated as a falsifiable sentence with a full evidence trail. An agent mining this repo for publications should treat this file as the **compressed list of defensible contributions**; everything else in `docs/` exists to support or contextualise them.

All HEAD spot-checks below are against commit `a8e651e` on `feat/embodiedscan-grounding`. Full per-version primary evidence lives in `10_experiment_log/`. The six paper-angle framings live in `11_academic_angles_catalog.md`.

## Claim Schema

Every claim below follows the same template. The supervisor-mandated fields are:

- **Status**: `load-bearing` (paper cannot exist without it), `supporting` (strengthens but not central), `speculative` (plausible but not yet tested).
- **Evidence trail**: commit hashes (with date + message), code anchors (file:line), quantitative numbers (fold-annotated), pointers into `10_experiment_log/`.
- **Generalization frontier**: the broader claim this specialisation sits under.
- **Prior work position**: at least two named works with a concrete delta line.
- **Novelty / Risk** — integer ratings 1–5.
- **Honest critique**: the strongest one-sentence attack we have already spotted.
- **REQUIRES for publication**: the minimum experiments or analyses that must land before submission.

---

## Claim 1: Scene graphs are high-recall, low-precision priors that must be verified by pixels rather than consumed as oracles.

**Status**: load-bearing (this is the architectural spine of the B1 angle and the implicit claim behind the v9 → v14 trajectory).

**Evidence trail**
- Commit: `fbd642e` (2026-04-04) "docs: update results summary with v14 — MNAS 73.1, #1 on OpenEQA ScanNet"
- Code anchors:
  - `src/agents/runtime/base.py:355–362` — explicit system-prompt line *"Stage 1 is a high-recall evidence retriever, not ground truth. Stage 2 must verify, repair, or reject Stage-1 hypotheses using pixels."*
  - `src/agents/stage1_adapters.py:54` — `build_stage2_evidence_bundle()` packages Stage 1 as a `Stage1HypothesisSummary` explicitly labelled as prior, not answer.
  - `src/agents/runtime/deepagents_agent.py:570` — the iterative evidence-refinement `run()` loop is the mechanism by which the prior is verified.
- Quantitative:
  - End-to-end trajectory: MNAS `46.5 (100Q, v9) → 55.4 (100Q, v10) → 62.6 (100Q, v11) → 65.0 (100Q, v12) → 71.4 (1050Q, v13) → 73.1 (1050Q, v14)`. +26.6 MNAS (+57 % relative) on comparable tail (v9 → v14). See `10_experiment_log/README.md`.
  - Against published baselines on OpenEQA ScanNet: +21.8 MNAS over GPT-4V 500Q subset (51.3), +35.3 over GPT-4 + ConceptGraphs (37.8); see `10_experiment_log/leaderboard.md`. Judge caveat: ours = Gemini 2.5 Pro vs. others' GPT-4 → not directly comparable.
- Failure evidence: `10_experiment_log/v14_inventory_20260404.md` shows the 1050Q per-category breakdown; even the best category (Object State 84.7) is well below human 87.7 → verification step matters.

**Generalization frontier**
Any pipeline that combines a symbolic/structured scene representation with a neural vision-language model should, at the current capability tier, treat the symbolic layer as a **soft prior to verify**, not an authoritative knowledge base. This extends to navigation-plan and manipulation tasks once we instrument a comparable verify-loop.

**Prior work position**
- vs **OpenEQA (CVPR 2024, arXiv:2312.15857)**: their scene-memory baselines (GPT-4 + ConceptGraphs 37.8, GPT-4 + Sparse Voxel Maps 40.9) consume the scene representation as KB and issue a single VLM query. Our delta is the explicit verify-loop; numerical lift vs. their best single-shot baseline is +21.8 MNAS on ScanNet (same benchmark, different judge).
- vs **ConceptGraphs (in OpenEQA Table 2)**: treats the graph as ground truth. Our Stage 1 reuses ConceptGraphs' detection pipeline verbatim but demotes the graph to a hypothesis; empirically, v14 recovers +35.3 MNAS relative to that formulation.
- vs **3D-Mem (CVPR 2025, arXiv:2411.17735)**: 3D-Mem maintains a memory bank consumed by a VLM; no explicit "memory may lie" contract. Our stance is that the memory contents are untrusted and must be re-grounded in pixels before being used.

**Novelty**: 3/5 (architectural stance; several contemporaneous works share the flavour).
**Risk**: 4/5 (scooping; SOTA treadmill — Gemini 2.5 Pro / a future backbone may erase the gap).

**Honest critique**
Our headline 73.1 MNAS uses `gpt-5.4-2026-03-05` as backbone; a hostile reviewer will attribute most of the +26.6 lift to model progress, not the verify-loop.

**REQUIRES for publication**
- `C-a` Same-backbone single-shot baseline (`gpt-5.4` called once on OpenEQA episodic-memory frames with no Stage 1 and no tools).
- `C-b` Stage-1-only ablation (Stage 1 retrieves *k* frames, `gpt-5.4` one-shot without tools or E2E nudge).
- `C-e` Per-category MNAS gap between `C-a` / `C-b` / full v14 — shows the gap is not concentrated in one skill.
- Matched-judge re-evaluation with GPT-4 (OpenEQA's judge) on the 1050Q split; `10_experiment_log/leaderboard.md §ToDo` already flags this.

---

## Claim 2: When a VLM-agent can retrieve privileged symbolic context through a tool, it systematically under-uses that tool; pre-injecting the same context into the system prompt dominates tool-gated access at the current VLM capability tier.

**Status**: load-bearing (this is the central empirical finding of the v14 change and of the C2 paper angle).

**Evidence trail**
- Commits:
  - `b4197a1` (2026-04-04) "feat: inject enrichment object inventory into Stage 2 system prompt" — landed the change.
  - `f122b07` (2026-03-30) "feat: LLM object enrichment + pipeline integration (v10)" — introduced the enrichment data as a tool-gated payload.
  - `1887e03` (2026-04-04) "docs: v13 full 1050Q failure analysis" — the diagnostic that motivated the switch (note: the original doc file is deleted from the workspace but preserved in the commit).
- Code anchors:
  - `src/agents/runtime/base.py:395` — `_format_scene_inventory(object_context)` call inside `build_system_prompt`.
  - `src/agents/runtime/base.py:296–310` — `_format_scene_inventory` implementation.
  - `src/agents/runtime/deepagents_agent.py:110–116` — the `retrieve_object_context` tool whose under-use this fix diagnoses.
  - `src/query_scene/keyframe_selector.py:352–358` — the `enriched_objects.json` load site (fails hard if missing; no silent fallback).
- Quantitative:
  - v14 vs. v13 on the 1050Q split (same backbone, same judge, same retrieval): **MNAS 71.4 → 73.1 (+1.8)**, `Score=5` count 573 → 600 (+27), `Score=1` 195 → 178 (−17), mean tool-calls/Q 1.36 → 1.33 (−0.03) — see `10_experiment_log/v14_inventory_20260404.md`.
  - Per-category MNAS delta: Attribute +3.4, Functional +2.6, Spatial +2.6, Object State +1.9, World Knowledge +1.9, Object Recognition +1.8, Localization −1.6. (Single-category regression flagged.)
  - Diagnostic pre-condition: 51 % of low-score v13 failures had the ground-truth object already present in `enriched_objects.json` (cited from `1887e03`).

**Generalization frontier**
The finding is one instance of a broader conjecture: **tool under-use is a first-class failure mode for VLM-agents when the tool provides information the model cannot cheaply decide to need.** If corroborated at larger backbones or on different agent stacks, it reframes tool-design guidance away from "clean retrievable interfaces" toward "bake privileged signals into the context until the agent learns to ask".

**Prior work position**
- vs **ReAct (Yao et al.)**: unconditional tool selection, no measurement of under-use.
- vs **Toolformer (Schick et al.)**: trains the model to *insert* tool calls; does not study the failure mode where the model fails to call an always-available tool.
- vs **ToolBench / Gorilla (Patil et al., arXiv:2305.15334)**: tool-selection accuracy benchmarks; no diagnostic for "correctly available, never invoked".
- vs **CoV — Chain-of-View (arXiv:2601.05172)**: evidence-seeking via explicit verification prompts; CoV moves information into the prompt once per verification loop, analogous to our inventory injection but scoped to a single object and per-turn. Our delta: persistent scene-level inventory injection plus a failure-mode diagnostic that motivates it.

**Novelty**: 3/5 (prompt engineering with a measurement rigor wedge).
**Risk**: 4/5 (concurrent arXiv submissions on agentic scene understanding may surface the same observation).

**Honest critique**
As VLM tool-use quality improves, the gap between "context-as-tool" and "context-in-prompt" should shrink — the finding may age poorly; a defensible paper must state this reservation explicitly and demonstrate the gap at at least two backbone tiers.

**REQUIRES for publication**
- Sweep over injected-token budget on the 1050Q split: 0 / quarter-inventory (filtered by Stage-1 hypothesis) / half / full / per-category-filtered. Measures the information-per-token trade-off.
- Cross-backbone measurement: repeat v13-vs-v14 delta with at least one strictly weaker backbone (e.g., `gpt-5.2`) and one strictly stronger (e.g., a 2026-Q3 backbone when available) to show how the gap scales.
- Tool-use rate before and after injection: how often does the agent *still* call `retrieve_object_context` once the inventory is in the prompt? If it drops to near zero, the tool should be removed, not merely made redundant.

---

## Claim 3: A single scalar confidence signal, combined with an explicit `insufficient_evidence` status, is sufficient to gate three orthogonal control points in a VLM-agent — and doing so outperforms fixed-budget ReAct on evidence-hungry 3D QA.

**Status**: load-bearing (this is the A2 angle; also underwrites v9, v12, v13 prompt-policy fixes).

**Evidence trail**
- Commits:
  - `b6a8aa6` (2026-03-24) "feat: parallel eval, confidence guard, LLM query rewrite for OpenEQA pilot" — introduced `--confidence-guard 0.6`.
  - `80ebf21` (2026-03-27) "feat: E2E nudge loop, eval report tool, 100-case analysis" — introduced the `_build_evidence_nudge` follow-up on `insufficient_evidence`.
  - `4d92316` (2026-03-28) "feat: trigger E2E on low-confidence Stage2 completions" — extended the E2E trigger from `status != completed` to also cover `confidence < guard`.
  - `44b9600` (2026-04-02) "feat: v13 prompt — mandatory crops, overconfidence calibration, bias correction" — added the explicit *"Do NOT report confidence > 0.7 if you used zero tools"* policy.
- Code anchors:
  - `src/agents/runtime/base.py:415–490` — `apply_uncertainty_stopping` downgrades `COMPLETED` with `confidence < threshold` to `INSUFFICIENT_EVIDENCE` when no more evidence can be acquired.
  - `src/agents/runtime/deepagents_agent.py:635–655` — the nudge branch: on low-confidence stall, inject a follow-up urging tool use.
  - `src/agents/examples/openeqa_official_question_pilot.py:519–555` — the E2E rerun is gated on `status != completed` OR `confidence < --confidence-guard` (default 0.6).
  - `src/agents/core/agent_config.py:58` — runtime-level `confidence_threshold = 0.4`.
- Quantitative:
  - v9 introduction of `--confidence-guard 0.6`: +10.5 MNAS on the 5-scene pilot (per `b6a8aa6` commit message), separable from `--llm-rewrite` which separately lifted `direct_grounded` rate 26 % → 33 %.
  - v12 → v13 delta on 100Q (`65.0 → 71.4` on the 1050Q fold-change boundary): `10_experiment_log/v13_calibration_20260330.md`; the v13 change set is prompt-only and includes the confidence cap alongside mandatory-crops.
  - 1050Q v13 → v14: the nudge + cap + E2E-guard triad remained unchanged; the +1.8 MNAS lift is attributed to Claim 2, not this one — useful as a factoring argument.

**Generalization frontier**
Any agent whose tool budget is not tight can benefit from a *single* uncertainty signal driving *multiple* orthogonal control points (intra-turn tool-invocation, inter-turn continuation, inter-run rerun). The specialisation to VLM + 3D QA is tight but the design pattern is not.

**Prior work position**
- vs **ReAct (Yao et al.)**: fixed or model-chosen tool budget without a scalar gate.
- vs **Reflexion (Shinn et al., arXiv:2303.11366)**: uses self-verbalised reflection to trigger retries; we instead use a scalar + status pair, which is cheaper to inspect and compose.
- vs **Chain-of-Verification (Dhuliawala et al., arXiv:2309.11495)**: CoV re-asks for verification unconditionally; ours conditions on the scalar and downgrades completion status when appropriate.
- vs the OpenEQA paper's own Episodic Memory baselines: these do not gate on confidence at all; single-shot generation.

**Novelty**: 4/5 (systematic gating of three orthogonal points by a single scalar, demonstrated at benchmark scale, appears to be open).
**Risk**: 3/5 (RLHF-style calibration work may overlap).

**Honest critique**
The scalar is VLM self-reported and we already know it is miscalibrated — that is precisely why v13 had to hand-cap it at 0.7 with zero tools. Defending "gate on a miscalibrated signal" requires showing the gate survives under quantified miscalibration, not merely that it improves aggregate MNAS.

**REQUIRES for publication**
- Factorial ablation on HEAD (not git history): all 2³ = 8 combinations of {completion-downgrade, nudge, E2E guard} on the 1050Q split.
- Confidence-vs-correctness calibration curve (ECE) before and after the v13 cap.
- At least one cross-backbone replication — to check that the gating policy is not over-fit to `gpt-5.4`'s particular self-confidence profile.

---

## Claim 4: Tool under-use is a first-class failure mode in multimodal agents, and its rate is measurable *ex post* from failure-case traces.

**Status**: supporting (generalises Claim 2 into a methodological contribution; would anchor a diagnostic paper like angle C1 or C2 framed broadly).

**Evidence trail**
- Commits:
  - `80ebf21` (2026-03-27) — first formal 100-case analysis that identifies tool under-use.
  - `1887e03` (2026-04-04) — full 1050Q failure taxonomy (96 cases analysed) that quantifies it at scale.
  - `b4197a1` (2026-04-04) — corrective change that directly targets the under-use rate.
- Code anchors:
  - `src/agents/trace.py` / `src/agents/trace_server.py` — tool-trace recorder whose output `tool_trace` list is the primary data source.
  - `src/evaluation/batch_eval.py:110–114` — `tool_trace: list[dict]` recorded per `EvalSampleResult`.
  - `src/agents/runtime/base.py:51–64` — `Stage2RuntimeState.record()` is the single choke point; every tool call is logged with its input/output and available for post-hoc analysis.
- Quantitative:
  - From v13 failure analysis: of 96 low-score cases, 51 % had GT object already in `enriched_objects.json` but `retrieve_object_context` was never called (Claim 2 also cites this).
  - Under-use rate is per-tool: for `request_more_views`, v10 invoked it 0 times across the 100Q fold; v11's callbacks-enabled fix lifted that to 61 invocations (`10_experiment_log/v11_callbacks_20260330.md`).

**Generalization frontier**
If every agent benchmark reported a per-tool-invocation breakdown for its failure set, we could define **tool-recall** (per-tool, per-failure-category) as a first-class metric alongside task accuracy. This project is one instance showing that such a metric is computable post-hoc without instrumenting new experiments.

**Prior work position**
- vs **ToolBench / Gorilla**: measure tool-selection accuracy on correctly-needing-a-tool queries; do not quantify "should have called but didn't".
- vs **AgentBench (Liu et al., arXiv:2308.03688)**: reports task-level success but not per-tool invocation stats on failure cohorts.
- vs **MINT (Wang et al., arXiv:2309.10691)**: multi-turn interaction benchmark; does not decompose failure by tool-use pattern.

**Novelty**: 2/5 as a claim, 3/5 if packaged as a proposed metric with a reproducible recipe.
**Risk**: 3/5 (easy to scoop if an agent-benchmark paper adopts the metric quickly).

**Honest critique**
"Tool-recall" is only well-defined when ground-truth tool necessity is observable — we can observe it *ex post* on our benchmark because we control the enrichment data, but the metric will not transport cleanly to settings without such privileged supervision.

**REQUIRES for publication**
- A clean operational definition of *necessary tool* per sample (not always obvious — "should have called `request_crops`" is easier than "should have called `switch_or_expand_hypothesis`").
- The metric computed across ≥ 2 agents (ours + a published baseline) so the paper can argue the metric discriminates.
- A small user-study or held-out validation that the taxonomy (perception / retrieval / visibility / under-use) is reproducible by independent annotators.

---

## Claim 5: A typed hypothesis tree with rank-ordered fallback (`DIRECT_GROUNDED > PROXY_GROUNDED > CONTEXT_ONLY`) exposes a machine-observable "recall mode" signal that downstream consumers can exploit.

**Status**: speculative (the typed tree exists and is used in Stage 1 today; its *downstream value* is asserted, not yet measured).

**Evidence trail**
- Commit: `598bbfe` added multi-label category index which supports the PROXY fallback (`pre-v9`); `9be20f1` preserves BETWEEN anchors inside PROXY (`pre-v9`).
- Code anchors:
  - `schema/hypothesis_output_v1.json` — the formal schema for the hypothesis-output JSON.
  - `src/query_scene/core/hypotheses.py` — `HypothesisKind` and `ParseMode` enums.
  - `src/query_scene/keyframe_selector.py:1570–1731` — `select_keyframes_v2()` runs the ranked executor.
  - `src/agents/examples/openeqa_official_question_pilot.py:362–448` — `run_stage1_ranked` walks query rewrites, commits to first `direct_grounded` result, else best-by-rank.
  - `src/agents/runtime/deepagents_agent.py:94–108` — `inspect_stage1_metadata` tool exposes `hypothesis_kind` / `hypothesis_rank` to the VLM agent; the agent's system prompt at `base.py:355–362` instructs it to use this as a prior signal.
- Quantitative:
  - `--llm-rewrite` (v9) raised `direct_grounded` rate from 26 % → 33 % on the 30-scene fold (`b6a8aa6` commit message).
  - v11 open-ended mode (`02ea2f3`) raised recall-when-target-UNKNOW without collapsing the typed distinction (`10_experiment_log/v11_callbacks_20260330.md`).
  - **No paper-quality ablation exists yet** of Stage 2 performance conditioned on Stage 1 hypothesis kind — see REQUIRES.

**Generalization frontier**
Retrieval components that expose a typed recall-mode signal (not only a scalar score) may enable downstream consumers to adapt their behaviour — for example, skipping tool calls when the retrieval is `DIRECT_GROUNDED`, or escalating to pixel verification when it is `CONTEXT_ONLY`.

**Prior work position**
- vs OpenEQA frame-uniform baselines: no typed signal at all.
- vs CLIP-retrieval baselines (CoV and similar): scalar similarity only; no interpretable recall mode.
- vs structured retrievers in language QA (e.g., DPR, Fusion-in-Decoder): none expose a typed recall mode to the consumer at inference time.

**Novelty**: 3/5.
**Risk**: 3/5 (several retrieval works expose coarse labels; our wedge is the *downstream-observable* rank, which is untested).

**Honest critique**
The fallback ordering is hand-designed; a cleaner method paper would derive or learn the rank from evidence quality, not fix it by convention.

**REQUIRES for publication**
- Per-hypothesis-kind MNAS decomposition on 1050Q: MNAS given `DIRECT_GROUNDED`, given `PROXY_GROUNDED`, given `CONTEXT_ONLY`, weighted by kind frequency.
- A Stage-2 ablation where the agent is denied the `inspect_stage1_metadata` tool (or where `hypothesis_kind` is blanked) — quantifies whether the typed signal actually changes agent behaviour or is merely passively logged.
- Information-theoretic argument: measure the mutual information between `hypothesis_kind` and final correctness on held-out samples; if it is low, Claim 5 reduces to Claim 1 and should be retired from the manifest.

---

## Cross-claim structure (for the agent consuming this file)

| Claim | Role in a B1-led paper | Would headline | Enough alone for a paper? |
|---|---|---|---|
| 1 | Architectural thesis | §2 Framing, §3 Method | Yes, but needs Claim 3 to stand up in ablation |
| 2 | Central empirical finding + Claim 4's proof case | §4 Main experiment, §5.1 Ablation | Yes — a standalone EMPIRICAL paper |
| 3 | Policy formalisation that defends Claim 1 in ablation | §3.3 Control policy, §5.2 Ablation | Yes — a standalone METHOD paper |
| 4 | Methodological spin-off / secondary contribution | §5.3 Diagnostic, Appendix | No — needs ≥ 2 agents and a clean metric definition |
| 5 | Stage-1 narrative continuity | §3.1 Retrieval | No — currently under-measured |

See `11_academic_angles_catalog.md` for how these claims map onto the six paper angles (A1, A2, B1, B2, C1, C2).
