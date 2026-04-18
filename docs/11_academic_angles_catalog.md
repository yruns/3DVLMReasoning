# 11 — Academic Angles Catalog (6 candidates + literature positioning)

This file is the working catalog of paper framings derived from the OpenEQA pipeline. It is the companion to `00_research_manifest.md`: the manifest lists claims; this file lists stories built from combinations of claims. Every angle carries an explicit literature positioning grounded in the published baselines from `10_experiment_log/leaderboard.md` plus adjacent contemporaneous work.

Axes: **METHOD** (single-technique), **SYSTEM** (architecture), **EMPIRICAL** (diagnostic / ablation). Two candidates per axis.

---

## 1. Comparison Matrix

| # | Axis | Short label | Core claim (1 line) | Novelty | Risk | Venue (1st / 2nd) | Cleanest v9–v14 evidence anchor |
|---|---|---|---|:---:|:---:|---|---|
| A1 | METHOD | Ranked hypothesis retrieval | For 3D task-conditioned retrieval, parsing the query into a typed hypothesis tree (DIRECT > PROXY > CONTEXT) with rank-ordered fallback beats single-pass CLIP. | 3 | 3 | NeurIPS / ICLR | `02ea2f3` (v11) open-ended + callbacks +17.5 % E2E over v9; `b6a8aa6` LLM rewrite 26 % → 33 % direct-grounded |
| A2 | METHOD | Uncertainty-gated tool use | One scalar confidence + explicit `insufficient_evidence` status is sufficient to gate invocation, continuation, and rerun control points. | 4 | 3 | NeurIPS / ICLR | `44b9600` (v13) cap 0.7 + `80ebf21` nudge + `b6a8aa6` `--confidence-guard 0.6` (+10.5 on 5-scene pilot) |
| B1 | SYSTEM | Two-stage evidence-seeking | Treat a 3D scene graph as a **high-recall untrustworthy prior** and let a VLM agent verify/repair/reject with pixels. | 3 | 4 | CVPR / ICCV | MNAS 46.5 → 73.1 (100Q / 1050Q, v9 → v14); `10_experiment_log/leaderboard.md` |
| B2 | SYSTEM | Unified multi-task policy (QA + VG) | One agent infrastructure with task-conditional tool gating handles OpenEQA QA and EmbodiedScan VG from shared scene assets. | 3 | 3 | CoRL / CVPR | MNAS 73.1 (QA) + VG smoke Acc@0.25 66.7 % (`9988be9`, 3 samples) |
| C1 | EMPIRICAL | Anatomy of a +26.6-point MNAS lift | A longitudinal diagnostic decomposing v9 → v14 into six discrete capability additions reveals which matters when and under which failure mode. | 2 (standalone) / 4 (as insights) | 3 | NeurIPS D&B / TMLR | Full v9 → v14 numeric trail + 96-case failure taxonomy (`1887e03`) |
| C2 | EMPIRICAL | Symbolic-inventory-in-prompt | Tool-gated privileged context is systematically under-used; pre-injecting it strictly dominates, +1.8 MNAS on 1050Q, Attribute +3.4. | 3 | 4 | EMNLP / ACL Findings | `b4197a1` (v14) / `1887e03` (51 % GT-in-inventory-but-untapped) |

Legend — Novelty: 5 = sharply new formulation, 3 = non-trivial, 1 = trivial extension. Risk: 5 = high scooping / data-requirements / reviewer-pushback risk, 1 = low.

---

## 2. Per-Angle Detail Sheets

Each sheet follows the same order: *title → claim → delta → v9–v14 evidence → venue → novelty/risk → honest critique → REQUIRES → literature positioning*. Literature positioning grounds every `vs X` delta line in a concrete file:line or commit.

### Angle A1 — Ranked Hypothesis Retrieval

| Field | Content |
|---|---|
| Candidate title | *"Don't Just Retrieve, Hypothesise: Typed-Fallback Retrieval for Task-Conditioned 3D Scenes"* |
| Core claim | Parsing a natural-language 3D query into a typed hypothesis tree (`DIRECT_GROUNDED`, `PROXY_GROUNDED`, `CONTEXT_ONLY`) and executing candidates by rank produces strictly higher-quality keyframe evidence than single-pass CLIP or BM25 retrieval, and exposes an interpretable *recall mode* signal that downstream consumers can exploit. |
| Delta (high level) | Other retrievers expose a scalar similarity; ours exposes a typed, rank-ordered fallback that Stage 2 can inspect via `inspect_stage1_metadata`. |
| v9-v14 evidence | (i) `02ea2f3` v11 open-ended + callback enablement = +17.5 % on E2E over v9 (100Q). (ii) `run_stage1_ranked` at `src/agents/examples/openeqa_official_question_pilot.py:362–448` iterates over query rewrites and returns best-by-rank. (iii) `b6a8aa6` v9 `--llm-rewrite` pushed direct-grounded rate 26 % → 33 % (30-scene fold). |
| Target venue | **NeurIPS** (method + ablation), 2nd **ICLR**. |
| Novelty | 3/5 — structured retrieval and multi-query expansion exist; the unique wedge is *typed-fallback with downstream-observable rank*. |
| Risk | 3/5 — risk is reviewer claiming "multi-query with labels". Mitigation: quantify information-theoretic gain of the type label for the Stage-2 decision policy. |
| Honest critique | Fallback ordering is hand-designed; a cleaner method paper would formalise the policy that assigns rank (e.g., calibrated classifier over parse confidence + match evidence). |
| REQUIRES | (1) Per-hypothesis-kind MNAS decomposition on 1050Q; (2) Stage-2 ablation where `inspect_stage1_metadata` is denied; (3) information-theoretic MI between `hypothesis_kind` and final correctness. |

**Literature positioning (A1)**
- **OpenEQA (CVPR 2024, arXiv:2312.15857)** — baseline uses frame-uniform sampling (50f) or scene-memory KB; no structured hypothesis tree. Numerical reference point: GPT-4V 500Q subset scores 51.3 MNAS on ScanNet (see `10_experiment_log/leaderboard.md`). Our Stage 1 exposes the typed tree to the consumer rather than returning only frames.
- **ConceptGraphs (in OpenEQA Table 2; Gu et al. arXiv:2309.16650)** — scene graph without a typed query interface; consumed as KB. Code-level: our `KeyframeSelector._load_objects_from_pcd` (`src/query_scene/keyframe_selector.py:374–459`) reuses ConceptGraphs' object format but wraps it in the ranked executor.
- **Chain-of-View / CoV (arXiv:2601.05172)** — multi-query verification over views; the queries are free-form, not typed. Our delta: typed rank labels survive into Stage 2 at `runtime/base.py:98–108`.
- **3D-Mem (CVPR 2025, arXiv:2411.17735)** — memory bank of scene snippets retrieved by similarity; no fallback typing. Reported ALL score 57.2; does not report ScanNet-only.
- **GraphPad (arXiv:2506.01174)** — inference-time 3D scene-graph updates, retrieval scored by graph-match; again no typed fallback surfaced to the VLM.
- **R-EQA (Embodied AI Workshop 2025)** — retrieval-augmented generation on OpenEQA; ScanNet score 49.1 with Qwen2.5-VL. Their retrieval is dense-vector; our delta is the typed rank plus downstream exposure.
- **LLaVA-3D (arXiv:2409.18125) / 3D-LLM (arXiv:2307.12981)** — feature-space approaches with no explicit retrieval stage; not comparable at the retrieval level, but provide the single-stage baseline that our two-stage pipeline aspires to outperform.

### Angle A2 — Uncertainty-Gated Tool Use

| Field | Content |
|---|---|
| Candidate title | *"Look Again, or Stop Guessing: Scalar-Confidence Gating for VLM Tool-Using Agents"* |
| Core claim | A single uncertainty signal — scalar `confidence ∈ [0,1]` combined with an explicit `insufficient_evidence` status — is sufficient to drive three complementary control points (tool-invocation, turn continuation, external E2E rerun), outperforming fixed-budget ReAct on evidence-hungry 3D QA. |
| Delta (high level) | ReAct and descendants use unconditional or model-chosen tool budgets; we formalise three gates instantiated from one scalar and show all three matter at the same benchmark. |
| v9-v14 evidence | (i) `44b9600` v13 explicit cap *"do not report conf > 0.7 with zero tools"* at `src/agents/runtime/base.py:385`. (ii) `80ebf21` `_build_evidence_nudge` at `src/agents/runtime/deepagents_agent.py:355`, called at `:644`. (iii) `b6a8aa6` v9 `--confidence-guard 0.6` cited as +10.5 MNAS on 5-scene pilot. (iv) `apply_uncertainty_stopping` at `src/agents/runtime/base.py:415–490` downgrades `COMPLETED` to `INSUFFICIENT_EVIDENCE` when evidence is exhausted. |
| Target venue | **NeurIPS** (uncertainty is a core topic), 2nd **ICLR**. |
| Novelty | 4/5 — systematic three-gate policy from one scalar with public 3D-QA evidence is relatively open. |
| Risk | 3/5 — RLHF-style calibration papers may overlap. |
| Honest critique | The confidence is VLM self-reported and is miscalibrated (v13 was introduced *because* of over-confidence); defending the policy under quantified miscalibration is mandatory. |
| REQUIRES | Factorial 2³ ablation on HEAD over {downgrade, nudge, E2E guard} on 1050Q; confidence-vs-correctness calibration curve before/after v13 cap; cross-backbone replication with at least one non-`gpt-5.4` model. |

**Literature positioning (A2)**
- **ReAct (Yao et al., arXiv:2210.03629)** — unconditional tool selection, no gating signal. Our wedge: `src/agents/runtime/deepagents_agent.py:570-692` is a ReAct descendant but the loop exit condition is the scalar gate, not a model-chosen stop.
- **Reflexion (Shinn et al., arXiv:2303.11366)** — self-verbalised reflection triggers retries; expensive and language-based. Our gate is scalar, cheaper to compose, and has a public ablation footprint.
- **Chain-of-Verification / CoV (Dhuliawala et al., arXiv:2309.11495; the 3D variant at arXiv:2601.05172)** — unconditional verification passes. Our delta: condition on the scalar, downgrade status when appropriate, and differ from CoV by explicitly declaring when evidence is *insufficient* rather than always producing an answer.
- **OpenEQA Episodic-Memory baselines (CVPR 2024)** — no confidence gating, one-shot generation. Numerical baseline: GPT-4V 500Q subset 51.3 vs. our v14 73.1 on 1050Q (different judge caveat).
- **AgentBench (Liu et al., arXiv:2308.03688) / MINT (Wang et al., arXiv:2309.10691)** — report task-level accuracy, do not decompose by confidence regime; our per-category MNAS × confidence breakdown in `10_experiment_log/v13_calibration_20260330.md` addresses a gap they leave.
- **Calibration literature (ECE, temperature scaling)** — addresses classifier calibration but not agent tool-use gating; our contribution would sit at the intersection.

### Angle B1 — Two-Stage Evidence-Seeking Agent

| Field | Content |
|---|---|
| Candidate title | *"Evidence, Not Oracle: A Two-Stage Agent that Verifies 3D Scene Graphs with Pixels"* |
| Core claim | On 3D scene QA, treating a detection-based scene graph as a **high-recall soft prior to verify** rather than an authoritative KB — and letting a VLM agent actively acquire pixel evidence — yields **MNAS 73.1 (1050Q, v14)** on OpenEQA ScanNet EM-EQA, #1 among methods reporting ScanNet-only scores (excluding AlanaVLM). |
| Delta (high level) | OpenEQA scene-memory baselines consume the graph as KB; we demote it to a prior and add an evidence-seeking loop. |
| v9-v14 evidence | (i) MNAS trajectory 46.5 (100Q, v9) → 73.1 (1050Q, v14). (ii) System-prompt line `src/agents/runtime/base.py:355–362` *"Stage 1 is a high-recall evidence retriever, not ground truth. Stage 2 must verify, repair, or reject."*. (iii) `build_stage2_evidence_bundle` at `src/agents/stage1_adapters.py:54` packages Stage 1 as prior. (iv) `runtime.run()` loop at `src/agents/runtime/deepagents_agent.py:570` is the verification mechanism. (v) 1050Q leaderboard at `10_experiment_log/leaderboard.md`. |
| Target venue | **CVPR** (SOTA + 3D visual reasoning), 2nd **ICCV**, alt **NeurIPS** with method emphasis. |
| Novelty | 3/5 — architecture in the zeitgeist; committing to *Stage-1-is-not-truth* and validating at 1050Q is meaningful. |
| Risk | 4/5 — highest among the six: scooping, SOTA treadmill, backbone attribution. |
| Honest critique | Our headline 73.1 uses `gpt-5.4-2026-03-05`; a reviewer will attribute most of the +26.6 MNAS lift to model progress unless we run an apples-to-apples same-backbone baseline. |
| REQUIRES | C-a same-backbone single-shot; C-b Stage-1-only baseline; C-e per-category gap decomposition; matched-judge re-evaluation with GPT-4 (see `10_experiment_log/leaderboard.md §ToDo`). |
| Intro re-framing sentence (from ADDENDUM, M1) | *"Rather than a system engineering report, this paper isolates the decision policy that lets a VLM agent treat a symbolic scene graph as a high-recall but untrustworthy prior, and provides the first quantitative answer to when such a prior should be verified, overridden, or silently consumed — measured against a controlled series of policy ablations on a 1050-question benchmark."* |

**Literature positioning (B1)**
- **OpenEQA (CVPR 2024, arXiv:2312.15857)** — same benchmark, same split (ScanNet EM-EQA). Reports four baselines using scene representations as KB (`GPT-4 + ConceptGraphs` 37.8, `GPT-4 + Sparse Voxel Maps` 40.9, `GPT-4 + LLaVA-1.5` 45.4, `GPT-4V 500Q subset` 51.3). Our headline +21.8 MNAS over the strongest of those (same benchmark, different judge). Our system never invokes the scene representation directly for an answer; it supplies it only as a prior.
- **ConceptGraphs (arXiv:2309.16650; appears in OpenEQA Table 2)** — the baseline representation we *reuse* at the data level and *demote* at the semantic level. Code-level: `src/query_scene/keyframe_selector.py:374–459` consumes `pcd_saves/*_post.pkl.gz` verbatim in the ConceptGraphs format; what changes is that this graph enters Stage 2 as `Stage1HypothesisSummary`, not as scripted QA context.
- **3D-Mem (CVPR 2025, arXiv:2411.17735)** — memory bank of 3D snippets for embodied exploration and reasoning; closest in spirit, but memory is consumed directly. Reported ALL score 57.2 (no ScanNet breakdown) vs. our 73.1 ScanNet.
- **GraphPad (arXiv:2506.01174)** — updates the scene graph at inference time; still consumes the graph for the answer rather than verifying it. Reported ALL score 55.3 (with Gemini 2.0 Flash). Our delta: graph updates come from *pixel verification by a VLM agent*, not from reasoning over the graph.
- **CoV — Chain-of-View (arXiv:2601.05172)** — evidence-seeking over views; strongest pattern match to our approach. Reports ALL scores 59.2–67.7 across different VLMs; no ScanNet breakdown. Our delta: CoV's loop is unconditional per-verification; ours is scalar-gated (Claim 3) and ours treats the symbolic prior as *demotable*.
- **R-EQA (Embodied AI Workshop 2025)** — retrieval-augmented generation for EQA, ScanNet 49.1 with Qwen2.5-VL. Retrieves to condition generation; no verify/reject structure.
- **LLaVA-3D (arXiv:2409.18125) / 3D-LLM (arXiv:2307.12981)** — feature-space end-to-end models. Not directly comparable under the same benchmark fold but represent the single-stage baseline philosophy we argue against. Their implicit claim ("one neural model can handle 3D scene QA") is the null hypothesis for B1.
- **SeeDo / SQA3D-style agents** — multi-step agentic 3D reasoning; share the evidence-seeking flavour but not the "graph as untrusted prior" contract. Candidate for inclusion in final Related Work once we confirm specific arXiv ids.

### Angle B2 — Unified Multi-Task Policy (QA + VG)

| Field | Content |
|---|---|
| Candidate title | *"One Agent, Two Worlds: Task-Conditional Tool Gating for Joint 3D Question Answering and Visual Grounding"* |
| Core claim | A single Stage-2 agent stack with task-conditional tool registration (`Stage2TaskType` gates `select_object` / `spatial_compare` for VG, leaves the QA tool set untouched) reaches competitive performance on both OpenEQA QA and EmbodiedScan VG from shared scene assets, without per-task architectural surgery. |
| Delta (high level) | Most 3D-LLM papers specialise to one task; our wedge is shared `DeepAgentsStage2Runtime`, shared `KeyframeSelector`, shared evidence bundle — only the tool *set* changes per task type. |
| v9-v14 evidence | (i) OpenEQA MNAS 73.1 on 1050Q (QA, v14). (ii) EmbodiedScan VG smoke results Acc@0.25 66.7 %, Acc@0.50 33.3 %, mean IoU 0.298 on 3 samples (`9988be9`, v4-plus-phase-VG) versus baseline Acc@0.25 6.2 %. (iii) Tool gating at `src/agents/runtime/deepagents_agent.py:204–247`. (iv) OpenEQA compatibility adapter `60d6f51` proves legacy entry unaffected. **REQUIRES**: full-split VG eval (mini ≥ ~500, full ≥ several k) before the paper can claim competitiveness. |
| Target venue | **CoRL** (multi-task policy for 3D), 2nd **CVPR** / **ICCV** (VG track). |
| Novelty | 3/5 — unified policies exist in manipulation; our wedge is *unified tool-set gating* for 3D evidence-seeking. |
| Risk | 3/5 — reviewer risk moderate (may see VG as loosely attached). |
| Honest critique | Only 3 VG smoke samples are currently public; the unified claim is structurally true but empirically thin until the VG pilot hits mini-split scale. |
| REQUIRES | EmbodiedScan VG mini and full splits; matched-backbone QA+VG number on the same `gpt-5.4`; ablation with the VG tools enabled on QA tasks (control for "shared stack = shared gain"). |

**Literature positioning (B2)**
- **EmbodiedScan (Wang et al., CVPR 2024)** — provides the VG dataset and an SR3D/NR3D-style evaluation protocol. Our pipeline reuses its annotations via `src/benchmarks/embodiedscan_loader.py` and its evaluation at `src/benchmarks/embodiedscan_eval.py`.
- **3D-VisTA (Zhu et al., ICCV 2023) / BUTD-DETR (Jain et al., ECCV 2022)** — VG-specialised architectures with no QA path. Our delta is the shared-infra claim; we do not dispute that a VG-specialised model can score higher in isolation.
- **SeeDo** — multi-step agentic VG; no unified QA story in published version. Our delta: same infrastructure serves both tasks; commit-level anchor at `src/agents/runtime/deepagents_agent.py:204–247` gating + `src/agents/tools/select_object.py` + `src/agents/tools/spatial_compare.py` (commit `9988be9`).
- **LLaVA-3D / 3D-LLM** — feature-space models with multi-task training; their "unified" claim is at the *training* level (single model, multiple heads). Ours is at the *inference infrastructure* level (single graph, tool-set gated by task type). These are different claims and should be positioned as such.
- **PaLM-E (Driess et al., arXiv:2303.03378) / SayCan (Ahn et al., arXiv:2204.01691)** — robotics multi-task policies; inspiration for the framing but on different task families.

### Angle C1 — Anatomy of a +26.6-Point MNAS Lift

| Field | Content |
|---|---|
| Candidate title | *"Twenty-Seven Points in Five Months: A Diagnostic Dissection of a 3D Question-Answering Agent"* (title rounding reflects 46.5 → 73.1 = +26.6; "twenty-seven" is a permissible rounding for the title) |
| Core claim | The v9 → v14 OpenEQA engineering trajectory — six commits spanning enrichment, callbacks, prompt calibration, model upgrade, and inventory injection — provides a **controlled natural experiment** whose per-commit deltas reveal which capability matters in which recall regime of Stage 1 and on which question category. |
| Delta (high level) | Most 3D-QA SOTA papers present a single model number; we present a longitudinal commit-grounded ablation plus failure-case linkage. |
| v9-v14 evidence | The dataset *is* the experiment: v9 100Q 46.5 → v10 100Q 55.4 → v11 100Q 62.6 → v12 100Q 65.0 → v13 1050Q 71.4 → v14 1050Q 73.1. Per-category deltas in `10_experiment_log/v14_inventory_20260404.md`. 96-case failure taxonomy in the commit message of `1887e03`. |
| Target venue | **NeurIPS Datasets & Benchmarks**, 2nd **TMLR**, 3rd **ACL Findings**. |
| Novelty | 2/5 standalone (ablations aren't novel); **4/5 as an insights paper** with a methodological contribution: *how to design an ablation using version history as the control*. |
| Risk | 3/5 — reviewer risk "engineering post-mortem". Mitigation: anchor around a *question* with a falsifiable answer. |
| Honest critique | Each v9 → v14 commit bundles multiple changes; per-commit delta is NOT the same as per-capability delta. Paper-quality analysis needs a proper factorial ablation on HEAD, which is not yet run. |
| REQUIRES | Full factorial on HEAD that turns off one axis at a time (retrieval / enrichment / callbacks / prompt rules / inventory injection / model) on the 1050Q fold; explicit reconciliation between 100Q and 1050Q numbers; per-category-per-axis contribution table. |

**Literature positioning (C1)**
- **OpenEQA (CVPR 2024)** — reports per-category scores but not per-capability ablation of a single agent. Our delta: the longitudinal decomposition gives both axes.
- **BLINK (Fu et al., arXiv:2404.12390) / MME-RealWorld (Zhang et al., arXiv:2408.13257) / MM-Vet (Yu et al., arXiv:2308.02490)** — static benchmarks with static model lists. They motivate *why* a diagnostic paper is valuable (they show fast progress), but do not provide the ablation framework we propose.
- **AgentBench (arXiv:2308.03688)** — agent-level evaluation across tasks; does not provide per-capability decomposition on a single agent.
- **Analysis papers in NLP (e.g., "What does BERT look at?" style)** — methodological inspiration; our contribution would port that analysis tradition to VLM-agents for 3D.

### Angle C2 — Symbolic-Inventory-in-Prompt Study

| Field | Content |
|---|---|
| Candidate title | *"The Tool the Agent Never Called: Why Pre-Injecting Scene Inventory Beats Retrieve-on-Demand"* |
| Core claim | Making privileged symbolic context available behind a tool call systematically under-uses that context — 51 % of failures in a 1050-question 3D-QA benchmark had the ground-truth object already in the enrichment JSON yet the retrieval tool was never called; direct injection into the system prompt recovers most of that gap for +1.8 MNAS on 1050Q and category-wise gains up to +3.4. |
| Delta (high level) | Recent agent-design literature emphasises *tool selection*; we study **tool under-use** as a first-class failure mode with a quantitative remedy. |
| v9-v14 evidence | (i) `b4197a1` v14 inventory injection: +1.8 MNAS (1050Q), Score=1 195 → 178, Score=5 573 → 600, per-category deltas (Attribute +3.4, Spatial +2.6) in `10_experiment_log/v14_inventory_20260404.md`. (ii) v13 analysis (`1887e03`): 51 % of low-score failures had GT in enrichment. (iii) Injection site `src/agents/runtime/base.py:395`; formatter `:296–310`. |
| Target venue | **EMNLP**, 2nd **ACL Findings**, alt **CoLM** (Conference on Language Models). |
| Novelty | 3/5 — prompt-engineering study; novelty carried by the *under-use* diagnostic, not the fix itself. |
| Risk | 4/5 — highest scooping risk; "put context in prompt" is an obvious direction. Mitigation: lead with the *under-use diagnostic*, give the fix as a corollary. |
| Honest critique | At enough VLM tool-use quality, the finding may age poorly; a good paper must state this and offer a falsifiable prediction across backbone tiers. |
| REQUIRES | Injected-token-budget sweep on 1050Q (0 / quarter / half / full / category-filtered); cross-backbone replication (at least one strictly weaker, one strictly stronger); post-injection tool-use rate measurement. |

**Literature positioning (C2)**
- **Toolformer (Schick et al., arXiv:2302.04761)** — trains the model to insert tool calls; our observation is orthogonal — we study under-use of an always-available tool, not selection.
- **ReAct (arXiv:2210.03629) / ToolBench (Qin et al., arXiv:2307.16789) / Gorilla (Patil et al., arXiv:2305.15334)** — tool-selection accuracy benchmarks; do not measure "should have called but didn't".
- **CoV — Chain-of-View (arXiv:2601.05172)** — moves information into the prompt per verification pass; the closest precedent and the strongest comparator. Our delta: persistent scene-level inventory injection (not per-verification) plus the diagnostic that motivates it.
- **Retrieval-augmented generation (Lewis et al., arXiv:2005.11401) generally** — pre-populates context from a retriever unconditionally; at first glance our finding aligns with RAG orthodoxy, but the agentic setting we study has an explicit *choice not to retrieve*, which is the piece we diagnose as broken.
- **OpenEQA (CVPR 2024)** — their baselines either embed the scene representation into the prompt (scene-memory KB style) or do not use it at all; our finding refines that design choice by quantifying when "in prompt" is strictly better than "on demand".

---

## 3. Synthesis & Recommendation

### Feasibility × Impact scatter (subjective, from Fischbach-Walsh X-axis/Y-axis framing)

```
Impact ^
  5     B1                 ← high-impact, high-risk (SOTA framing)
  4     A2, C2
  3     A1, B2, C1 (as insights)
  2     C1 (as raw ablation)
  1
        +-----+-----+-----+-----+-----+---> Feasibility
              1     2     3     4     5
                          ^           ^
                        B2 VG-full required   A1 / A2 / C1 / C2 mostly ready
```

### Angle merge proposal (the preferred packaging)

- **B1 ⊕ C1 ⊕ C2** — the recommended primary submission. B1 carries the architectural thesis (Claim 1). C2 provides the central empirical finding (Claim 2). C1 provides the methodological rigor that defends B1 against the "engineering post-mortem" critique by turning the commit trail into a controlled ablation (Claims 4 and 5 ride as appendix).
- **A2 standalone** — recommended as **top-2** / low-risk insurance submission, especially if the scooping window on B1 closes. Uses Claim 3 alone; the cross-backbone ablation is modest and achievable.
- **B2 as case-study extension inside B1** — not a separate paper; include EmbodiedScan VG numbers in an appendix once the mini-split run lands. Standalone it is too thin (3-sample smoke only).
- **A1 absorbed into B1** — appears as §3.1 (Stage 1) with the downstream information-theoretic argument; no separate publication.

### Angles I am least bullish on

- **C1 standalone**: the dataset IS the experiment is a seductive framing, but without the clean factorial on HEAD the per-commit deltas cannot stand up to a harsh reviewer. Keep C1 as a section inside B1, not a paper.
- **B2 standalone (short term)**: the EmbodiedScan VG numbers are 3-sample smoke. A separate VG-only paper is premature.

### Angles most at risk of "engineering contribution, insufficient novelty"

- **B1** — mitigation is the Intro re-framing sentence quoted in its detail sheet (M1 ADDENDUM).
- **C1** — mitigation is to anchor around a falsifiable question ("what fraction of the gain is retrievable by prompt engineering alone?") and answer it on HEAD with a proper factorial, not a narrative.
- **C2** — mitigation is to lead with the *under-use diagnostic* (Claim 4) rather than the fix (the inventory injection), so the contribution is a methodology, not a prompt trick.

### Risks the user must confirm before we commit to the top-1

1. **Scooping window** — B1's architectural claim is on-trend. If the CVPR deadline is > 8 weeks away we should additionally begin A2 as insurance.
2. **Backbone identity** — B1 stands on `gpt-5.4-2026-03-05`; the minimum defensible control set is C-a + C-b + C-e (ADDENDUM M1).
3. **VG scope** — whether to include B2 claims inside B1 or save for follow-on depends on whether the EmbodiedScan mini/full runs land before submission.

### Cross-claim mapping (bridges this file to `00_research_manifest.md`)

| Angle | Primary claim | Supporting claims | Paper role of each claim |
|---|---|---|---|
| A1 | 5 | (none) | Claim 5 headlines §3.1 |
| A2 | 3 | 4 | Claim 3 headlines, Claim 4 supports the "gate on miscalibrated signal" defence |
| B1 | 1 | 2, 3 | Claim 1 is the thesis; Claims 2 and 3 form the method / ablation |
| B2 | 1 | 4 | Claim 1 extends to VG; Claim 4 is the shared-infra implication |
| C1 | (none headline) | 1, 2, 3, 4 | All four claims decomposed across the v9 → v14 trajectory |
| C2 | 2 | 4 | Claim 2 is the result; Claim 4 is its methodological wrapper |

Ready for top-1 selection. Once confirmed, M2 (literature positioning deep dive for the selected angle) and M3 (full outline) proceed from this catalog.
