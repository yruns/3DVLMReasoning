# Navigation-Planning Benchmark Survey (High-Level, SayPlan-Style Plans)

*Last updated: 2026-04-21*

## Goal

We need an evaluation benchmark for a 3D VLM agent that emits **high-level** navigation plans
with schema:

```json
{
  "target_region": "string (e.g. bedroom, kitchen island)",
  "landmarks":     ["string", ...],
  "waypoints":     ["string" | {"region": "...", "reason": "..."}],
  "risks":         ["string"]
}
```

We are **not** doing R2R-style low-level step-by-step action trajectories. Plans are closer
in spirit to SayPlan (Rana et al. CoRL 2023): semantic, region-aware, landmark-grounded.

The schema above is the **patent contract** (乐书豪-专利交底书 §5.6 step 5, nav_plan output). A
benchmark is "schema-aligned" only if each candidate's gold annotation can be projected
onto all four fields without inventing dominant components.

Our local scene corpus is 89 ScanNet v2 scenes (listed in `/tmp/our_scannet_scenes.txt` and
already prepared under `data/OpenEQA/scannet/<scene_id>/conceptgraph/`). A benchmark is
useful to us if either:

1. at least 5 of its scenes overlap our 89-scene list **and** its gold annotations can be
   projected to the patent schema without synthesising the majority of fields, **or**
2. it is a synthetic/oracle-generated benchmark that can be grounded on our scenes directly
   with exact schema match.

Five candidates are surveyed below: SayPlan, SG3D, HM-EQA, R2R/REVERIE, and a synthetic
GPT-oracle pattern.

## Important Fact Established During Research

Our 89 ScanNet scenes are **not a random subset** — they are the exact OpenEQA ScanNet
split. Of the 1,636 OpenEQA questions, 1,079 land on ScanNet scenes, and those 1,079
cover 89 unique ScanNet scene IDs that match `/tmp/our_scannet_scenes.txt` 1:1. This
matters because it raises the baseline expected overlap for any ScanNet-derived
benchmark.

We also directly downloaded the SG3D dataset and computed overlap:
- SG3D total tasks: 22,346 (train 20.4k / test 1.96k), 4,895 scenes
- SG3D ScanNet subset: 693 unique scenes, 3,174 tasks
- **Overlap of SG3D's ScanNet scenes with our 89 scenes: 34 scenes, 129 tasks**

Per-scene overlap counts (SG3D tasks per scene, top of list):

| scene            | SG3D tasks |
|------------------|-----------:|
| scene0077_00     | 5 |
| scene0406_00     | 5 |
| scene0426_00     | 5 |
| scene0494_00     | 5 |
| scene0704_00     | 5 |
| scene0256_00     | 5 |
| scene0050_00     | 5 |
| scene0354_00     | 5 |
| scene0084_00     | 5 |
| scene0696_00     | 5 |
| ... (24 more)    | 1–5 |

Full overlap list: `scene0050_00, scene0077_00, scene0084_00, scene0131_00,
scene0193_00, scene0207_00, scene0222_00, scene0256_00, scene0354_00, scene0356_00,
scene0406_00, scene0426_00, scene0435_00, scene0461_00, scene0462_00, scene0494_00,
scene0496_00, scene0500_00, scene0518_00, scene0527_00, scene0535_00, scene0550_00,
scene0578_00, scene0583_00, scene0593_00, scene0598_00, scene0633_00, scene0643_00,
scene0645_00, scene0647_00, scene0648_00, scene0696_00, scene0699_00, scene0704_00`.

This gives SG3D the strongest *scene* alignment of any public dataset. But scene
alignment is only half the story: SG3D's native task is **sequential object grounding**
(gold = `target_id` boxes per step), not plan generation. The Schema-Alignment Analysis
further below compares each candidate against the patent contract
`{target_region, landmarks, waypoints, risks}` and is the actual deciding factor.

---

## 1. SayPlan (Rana et al., CoRL 2023)

### Summary

SayPlan is a method paper, not a public benchmark dataset. It demonstrates LLM-based
long-horizon task planning grounded on 3D scene graphs in two custom environments
(office floor, home) spanning up to 3 floors / 36 rooms / ~140 objects.

### Facts

- **Paper / project page**: https://sayplan.github.io/ — arXiv https://arxiv.org/abs/2307.06135,
  PMLR proceedings https://proceedings.mlr.press/v229/rana23a.html
- **Dataset availability**: No public dataset release. Project page lists video demos but
  **no downloadable test set, no GitHub code repo, no split definition**. Confirmed by
  checking the project page: "The webpage does not mention a code repository or dataset
  download link."
- **License**: N/A (no dataset released)
- **Scene base**: Custom 3D scene graphs (not ScanNet, not Matterport, not HM3D). The two
  environments are pre-constructed 3D scene graphs of an office floor and a home floor,
  authored by the paper authors.
- **Scene overlap with our 89 ScanNet scenes**: **0**
- **Output schema alignment**: SayPlan's planner does emit SayPlan-style structured plans
  (a natural-language plan followed by a grounded action sequence over scene-graph node
  IDs, with iterative replanning). The exact plan schema is described qualitatively in
  the paper but not released as a standard. Our `{target_region, landmarks, waypoints,
  risks}` schema is **inspired by SayPlan but is ours** — it is not a published SayPlan
  field set.
- **Task granularity**: High-level (semantic search over scene-graph hierarchy + iterative
  replanning). Delegates low-level path execution to a classical planner.
- **Typical sample count**: Paper reports qualitative evaluations on ~dozens of task
  instructions across the 2 environments — not a large-scale public test set.

### Verdict

SayPlan is the **conceptual blueprint** for our output schema, but **not a usable
benchmark** for us. Zero scenes to ground on, zero public instructions, no license
framework. Cite it as prior work, do not use it as an evaluation set.

---

## 2. SG3D / SG3D-Nav (Task-oriented Sequential Grounding and Navigation in 3D Scenes, NeurIPS D&B 2024)

### Summary

SG3D is the most relevant public benchmark. It contains GPT-4-generated multi-step task
instructions over 3D scene graphs (built on SceneVerse), with each step referencing a
target object in the scene. A v2 arXiv version is titled "Task-oriented Sequential
Grounding **and Navigation** in 3D Scenes" — the navigation extension evaluates the
tasks inside a simulator.

### Facts

- **Paper**: https://arxiv.org/abs/2408.04034 (v1 Aug 2024, v2 adds navigation)
- **Project page**: https://sg-3d.github.io/
- **GitHub**: https://github.com/sg-3d/sg3d (MIT license)
- **Dataset**: https://huggingface.co/datasets/ZhuofanZhang/SG3D (CC-BY-4.0, public, no gating)
- **Dataset size**: 22,346 tasks, 112,236 steps across 4,895 scenes. Hugging Face files:
  `train.json` (13.7 MB LFS, 20.4k tasks), `test.json` (1.22 MB, ~1.96k tasks).
- **Scene breakdown** (from Table 2 of the paper, verified by per-source record counts we
  computed from the raw JSON):
  - HM3D: 2,038 scenes / 9,036 tasks
  - ARKitScenes: 1,575 scenes / 7,395 tasks
  - ScanNet: 693 scenes / 3,174 tasks
  - 3RScan: 472 scenes / 2,194 tasks
  - MultiScan: 117 scenes / 547 tasks
- **Overlap with our 89 scenes**: **34 scenes / 129 tasks** (directly measured on
  `/tmp/sg3d_train.json` + `/tmp/sg3d_test.json`). Well above the ≥5 threshold.
- **Record schema** (from raw HF data):
  ```json
  {
    "task_description": "Water the potted plant near the decoration.",
    "action_steps": [
      {"action": "1. Pick up the bucket ...", "target_id": "21",  "label": "bucket"},
      {"action": "2. Fill it with water ...",  "target_id": "193", "label": "faucet"},
      ...
    ],
    "scan_id": "ScanNet_scene0077_00"  // or "HM3D_00814-...", "3RScan_...", etc.
  }
  ```
- **Output schema alignment with our `{target_region, landmarks, waypoints, risks}`**:
  - `target_region`: **missing**. SG3D tasks are object-centric, not room-centric. Would
    have to synthesize by looking up the room/region of the final `target_id` via our
    enriched scene graph.
  - `landmarks`: **partially present** as intermediate `target_id` / `label` references.
    Each step names an object that functions as a landmark.
  - `waypoints`: **partially present** as the sequence of `action_steps` — but each is a
    mixed nav+manipulation instruction, not a pure region/waypoint.
  - `risks`: **entirely absent**. Would have to synthesize with an LLM.
- **Task granularity**: Mid-level procedural actions, mixing navigation and manipulation
  (e.g., "Walk to the cabinet and fetch a towel", "Turn on the shower head"). **Not pure
  navigation plans**. This is the main schema mismatch.
- **Nav subset**: The v2 paper adds a simulator-based navigation evaluation, but the
  public HF dataset does **not** appear to contain separate nav annotations — it's the
  same `action_steps` records, evaluated in a sim. Confirmed by HF file listing:
  only `train.json` and `test.json`.
- **GitHub citation**:
  ```
  @inproceedings{zhang2024sg3d,
    title={Task-oriented Sequential Grounding and Navigation in 3D Scenes},
    author={Zhang, Zhuofan and ...},
    booktitle={NeurIPS Datasets and Benchmarks Track},
    year={2024}
  }
  ```

### Verdict

SG3D is the best *scene-aligned* public dataset we found. Cost of adoption:
- We get 129 tasks grounded on our scenes for free.
- We must **transform** each SG3D task into our plan schema: derive `target_region` from
  the final target object's room (available via our enriched scene graph), convert
  `action_steps` into `landmarks` + `waypoints`, and synthesize `risks` via LLM.
- SG3D tasks mix manipulation with navigation — roughly half of steps are purely
  navigational ("Walk to the cabinet", "Go to the shower area"). Navigation-only steps
  can be filtered.

This is directly usable but requires a one-time conversion script, not a drop-in loader.

---

## 3. OpenEQA (A-EQA / EM-EQA) — "HM-EQA" clarification

The user asked about "HM-EQA / A-EQA" as one candidate. These are **two different
benchmarks** that get conflated:

- **OpenEQA's A-EQA split** (Majumdar et al., CVPR 2024) — Meta FAIR, ScanNet + HM3D.
- **HM-EQA** (Ren et al., RSS 2024, "Explore until Confident") — Stanford ILIAD, HM3D only.

Both are embodied QA benchmarks. Neither was built for navigation-plan evaluation, but
both have scene-grounded questions that *imply* navigation to evidence locations.

### 3a. OpenEQA / A-EQA

- **Paper**: https://openaccess.thecvf.com/content/CVPR2024/papers/Majumdar_OpenEQA_...pdf
- **Project page**: https://open-eqa.github.io/
- **GitHub**: https://github.com/facebookresearch/open-eqa (MIT, archived Nov 2025, read-only)
- **Dataset**: `data/open-eqa-v0.json` in the repo, ~476 KB, public
- **Size**: 1,636 questions total — 1,079 on ScanNet (89 unique scenes), 557 on HM3D (92
  HM3D scenes across 000-hm3d-… IDs), covering "over 180 real-world environments"
- **Scene base**: ScanNet (v2 val+test) and HM3D
- **Overlap with our 89 scenes**: **89 / 89 = 100%** — our local corpus IS the OpenEQA
  ScanNet split. All 1,079 ScanNet questions are usable.
- **Schema** (from raw JSON):
  ```json
  {
    "question": "What is the white object on the wall above the TV?",
    "answer":   "Air conditioning unit",
    "category": "object recognition",
    "question_id": "...",
    "episode_history": "scannet-v0/<scene_id>"  // or "hm3d-v0/000-hm3d-..."
  }
  ```
- **Output schema alignment with our plan schema**:
  - `target_region`: **missing**. OpenEQA items have no target room — just a question.
  - `landmarks`: **missing**. No annotated landmarks.
  - `waypoints`: **missing**. A-EQA evaluates exploration trajectories against answer
    correctness, but there are no *reference* waypoint annotations.
  - `risks`: **missing**.
- **Task granularity**: Q/A, not plans. A-EQA measures whether an agent's exploration
  produces a correct final answer.
- **Download size**: JSON is small; episode histories are large (ScanNet RGB ~62 GB,
  RGB-D ~70 GB — we already have the relevant scenes prepared).
- **License**: MIT.

**Verdict**: OpenEQA is perfectly scene-aligned but is **not a navigation-plan
benchmark** — it's a QA benchmark. Usable only if we synthesize plans from questions
(GPT-oracle pattern, see §5) using the question text as the task instruction.

### 3b. HM-EQA (Stanford ILIAD)

- **Paper**: "Explore until Confident", RSS 2024. https://arxiv.org/abs/2403.15941
- **Project page**: https://explore-eqa.github.io/
- **GitHub**: https://github.com/Stanford-ILIAD/explore-eqa
- **Size**: 500 questions over 267 HM3D scenes, GPT-4V-generated, five question categories.
- **Scene base**: **HM3D only, not ScanNet**.
- **Overlap with our 89 scenes**: **0** (different scene universe).
- **Output schema alignment**: Q/A only, same caveats as OpenEQA — no plan fields.
- **License**: Unclear (not stated on project page or GitHub README); repo is public.

**Verdict**: HM-EQA is ruled out by scene-base mismatch. We would have to adopt HM3D to
use it, which breaks our ScanNet-focused pipeline.

---

## 4. R2R / REVERIE (Matterport3D)

### Summary

Two canonical Vision-and-Language-Navigation benchmarks on Matterport3D. R2R is fine-
grained step-by-step, REVERIE is high-level ("go to the kitchen and clean the coffee
table"). Different scene universe from ours.

### 4a. R2R (Room-to-Room, Anderson et al., CVPR 2018)

- **Paper / project**: https://bringmeaspoon.org/ — https://arxiv.org/abs/1711.07280
- **GitHub**: https://github.com/peteanderson80/Matterport3DSimulator (MIT code; dataset gated)
- **Size**: 21,567 instructions (~22k) over 90 Matterport3D buildings, avg 29 words.
- **Splits**: train / val_seen / val_unseen / test (exact splits published with dataset).
- **Scene base**: **Matterport3D, not ScanNet**.
- **Overlap with our 89 scenes**: **0**.
- **Instruction style**: **Fine-grained step-by-step** ("Walk past the dining table, turn
  left at the end of the hallway, stop in front of the red door"). Opposite of our
  high-level SayPlan-style plans.
- **Schema**: path (list of viewpoint IDs), instructions (3 per path), heading, distance.
- **License**: Code MIT; Matterport3D data under Matterport3D Terms of Use (academic,
  gated — must request access).
- **Output schema alignment**: **Poor**. R2R has viewpoint-level paths, not
  `{target_region, landmarks, waypoints, risks}`. Instructions are fine-grained, not
  plan-level.

**Verdict**: R2R is the wrong granularity and the wrong scene base. Cannot repurpose.

### 4b. REVERIE (Qi et al., CVPR 2020)

- **Paper**: https://arxiv.org/abs/1904.10151 / CVPR 2020 open-access PDF
- **GitHub**: https://github.com/YuankaiQi/REVERIE
- **Dataset page**: https://yuankaiqi.github.io/REVERIE_Challenge/dataset.html
- **Size**: 21,702 instructions / 10,318 panoramas across 86 Matterport3D buildings /
  4,140 target objects. Splits:
  - Train: 60 scenes, 10,466 instructions, 2,353 objects
  - Val Seen: 46 scenes, 1,423 instructions, 440 objects
  - Val Unseen: 10 scenes, 3,521 instructions, 513 objects
  - Test: 16 scenes, 6,292 instructions, 834 objects
- **Scene base**: Matterport3D.
- **Overlap with our 89 scenes**: **0**.
- **Instruction style**: **High-level** ("Fold the towel in the bathroom with the fishing
  theme"). Closer to our plan style than R2R is, but target_region is implicit, not a
  field.
- **Schema (from repo)**: `{scan, id, path_id, path (viewpoint sequence), heading,
  distance, objId, instructions, instructions_l}`. No explicit `target_region`,
  `landmarks`, or `risks`.
- **License**: Matterport3D terms of use (gated); code license not clearly stated in repo.
- **Output schema alignment**: **Partial**. Could derive `target_region` from the
  building's room segmentation + the viewpoint where objId sits. No landmarks, no risks.
  Repurposing just the instructions (ignoring the Matterport scenes) is possible but then
  the instructions lose grounding — they reference building-specific objects.

**Verdict**: REVERIE's high-level instruction style is philosophically close but we
cannot reuse the data without adopting Matterport3D. Zero overlap with our scenes.
Cannot repurpose the instructions alone because they reference scene-specific objects
that do not exist in our ScanNet scenes.

---

## 5. Synthetic GPT-Oracle Pattern

### Summary

Generate `(instruction, gold_plan)` pairs by prompting a strong LLM conditioned on each
scene's enriched scene graph:
`data/OpenEQA/scannet/<scene_id>/conceptgraph/enriched_objects.json` + `scene_info.json`.

### Facts

- **Availability**: Entirely local. No external dataset license. Costs only LLM tokens.
- **License**: Ours, can be whatever we choose.
- **Scene base**: Our 89 prepared ScanNet scenes — 100% overlap by construction.
- **Output schema alignment**: **Exact.** We prompt the LLM to emit
  `{target_region, landmarks, waypoints, risks}` directly. No post-hoc transformation.
- **Task granularity**: Exactly what we want — high-level SayPlan-style plans.
- **Typical sample count**: Fully configurable. Pilot target: 3–5 instructions per scene
  × 89 scenes ≈ 270–450 samples, which matches SG3D's scene coverage but at our exact
  schema.
- **Risks of this approach**:
  - Gold plans are GPT-generated, not human-verified. Need a small human QA pass
    (e.g., authors check a random 30 plans for correctness) before calling it gold.
  - Distribution skew: whatever LLM we use will bias toward patterns it has seen. A
    second-LLM cross-check (e.g., Claude judges GPT's gold, or vice versa) mitigates.
  - Contamination risk if the evaluator VLM is the same family that generated gold.
- **Comparable precedents**:
  - HM-EQA itself uses GPT-4V to generate questions from sampled views (§3b above).
  - SG3D uses GPT-4 + human verification for task generation (§2 above).
  - OpenEQA relies on human-written Qs but uses LLM-as-judge for scoring.

### Verdict

Synthetic GPT-oracle is the **only** option that gives an exact schema match with zero
external data dependencies. It is also the only option that scales with our scene set
instead of being capped by external overlap. The scientific caveats (gold fidelity,
distribution skew) are real but manageable with cross-model validation + sampled human
QA.

---

## Comparison Table

| Benchmark               | Public?         | License        | Scene base         | Overlap w/ our 89 | Schema match          | Granularity        | Samples                | Usable for us?                          |
|-------------------------|-----------------|----------------|--------------------|------------------:|-----------------------|--------------------|------------------------|-----------------------------------------|
| SayPlan (CoRL'23)       | No dataset      | N/A            | Custom (2 envs)    | 0                 | Conceptual match only | High-level         | Qualitative only       | No — method paper, not a benchmark      |
| **SG3D**                | **Yes (HF)**    | **CC-BY-4.0 (data) + MIT (code)** | ScanNet + HM3D + ARKit + 3RScan + MultiScan | **34 scenes / 129 tasks** | Partial — needs transformation | Mid-level (nav+manip mix) | 22,346 tasks total; 129 on our scenes | **Yes, with a conversion script** |
| OpenEQA (A-EQA)         | Yes             | MIT            | ScanNet + HM3D     | **89 scenes / 1,079 Q** (100%) | None — it's Q/A, not plans | QA, not plans      | 1,079 ScanNet Qs       | Only as seed for synthetic plans        |
| HM-EQA                  | Yes             | Unclear        | HM3D only          | 0                 | None — Q/A only       | QA                 | 500 Qs / 267 HM3D scenes | No — wrong scene base                   |
| R2R                     | Code MIT; data gated | Matterport3D ToU | Matterport3D       | 0                 | Poor — fine-grained trajectories | Low-level step-by-step | ~22k instr over 90 bldgs | No                                      |
| REVERIE                 | Code public; data gated | Matterport3D ToU | Matterport3D       | 0                 | Partial — high-level instr, no plan fields | High-level goal    | 21,702 instr / 86 bldgs | No — zero scene overlap                 |
| **Synthetic GPT-oracle**| By construction | Ours           | Our 89 ScanNet     | **89 / 89**       | **Exact**             | High-level plan    | Configurable (~270–450 pilot) | **Yes, the schema-exact option**        |

---

## Schema-Alignment Analysis (Patent Contract)

The patent contract is `{target_region, landmarks, waypoints, risks}`. Each candidate is
scored per-field as **exact** / **derivable** / **synthesise** / **missing**:

- **exact** = the benchmark's native field is the patent field as-is.
- **derivable** = a deterministic lookup against our enriched scene graph recovers it
  (e.g. final `target_id` → room label via `enriched_objects.json`).
- **synthesise** = an LLM has to invent the field from the instruction; the gold
  signal in that field is then LLM-authored, not dataset-authored.
- **missing** = no reasonable projection; the field cannot be constructed without a
  completely parallel annotation pass.

| Benchmark              | `target_region` | `landmarks` | `waypoints` | `risks`     | Dominant mode |
|------------------------|-----------------|-------------|-------------|-------------|---------------|
| SayPlan (paper)        | missing         | missing     | missing     | missing     | No gold at all |
| **SG3D**               | derivable       | derivable   | derivable (but object-centric, not region-centric) | synthesise | ≥1 synthesised field, several forced through object-→region projection |
| OpenEQA A-EQA          | missing         | missing     | missing     | missing     | QA only, no plan fields |
| HM-EQA                 | missing         | missing     | missing     | missing     | QA only + scene-base mismatch |
| R2R                    | missing         | missing     | low-level path, wrong granularity | missing | Trajectory-level |
| REVERIE                | derivable (from objId's room) | missing | missing (viewpoint-seq, wrong granularity) | missing | High-level goal, but no landmark or risk fields |
| **Synthetic oracle**   | **exact**       | **exact**   | **exact**   | **exact**   | **Generated directly into patent schema** |

Key observation raised during supervisor review (Codex-consultant critique): **SG3D is
sequential object grounding, not plan generation**. Its gold signal lives in the
`target_id` sequence (3D bounding boxes in the scene graph), which evaluates a grounding
model. Projecting it to `{target_region, landmarks, waypoints, risks}` only uses the
`label` and the final-step room — the SG3D ground truth becomes a weak supervision
signal, and the interesting field (`risks`) is entirely LLM-synthesised. This undercuts
the "real-world anchor" claim that originally motivated preferring SG3D.

## Recommendation

### Primary: **Synthetic GPT-oracle** on our 89 ScanNet scenes

After the schema-shape analysis (and the Codex-consultant critique folded into this
revision), synthetic is the only candidate that is both scene-aligned and schema-aligned.
The earlier preference for SG3D as primary assumed its gold was usable as-is for plan
evaluation; in fact, SG3D's gold evaluates object grounding, and three of the four patent
fields must be derived or synthesised.

Rationale:

1. **Schema match is exact.** The oracle prompt emits `{target_region, landmarks,
   waypoints, risks}` directly. No post-hoc projection, no field invention away from the
   gold signal. This matches the patent contract `{target_region, landmarks, waypoints,
   risks}` (NOT the current `{subgoals, landmarks, risks}` shape at
   `src/agents/runtime/base.py:106-111`, which is a known bug to be fixed in M4).

2. **Scene alignment is 100 %.** Generation is conditioned on
   `data/OpenEQA/scannet/<scene>/conceptgraph/{enriched_objects,scene_info}.json`, so every
   instruction references objects / regions that actually exist in the scene. No
   Matterport-vs-ScanNet / HM3D-vs-ScanNet adoption cost.

3. **Granularity match is exact.** We prompt the oracle for SayPlan-style plans (region,
   landmark references, waypoints, risks) — not for manipulation steps (SG3D) or
   viewpoint-level trajectories (R2R/REVERIE).

4. **Control over risk profile.** We can stratify generated plans by difficulty (single-
   room, cross-room, obstacle-avoidance, multi-landmark) and by plan length. External
   datasets do not let us dial this.

5. **Scale.** 3–5 plans per scene × 89 scenes = 270–445 samples, comfortably above the
   pilot threshold.

### Secondary (optional, keep in scope but do not block on): SG3D as "real-world anchor"

The 129 SG3D-on-our-scenes tasks remain useful as a secondary sanity check: running our
agent on them and computing whatever subset of our metrics survives the projection (e.g.
landmark recall on the `label` list, region match on final target) gives reviewers an
externally-sourced number that is not subject to the "self-generated gold" critique.

This is **optional** for the pilot. If the SG3D→patent-schema conversion script takes
more than ~1 day, drop it; the synthetic corpus stands alone.

### Controls to address the main synthetic-benchmark critique

The obvious objection to synthetic gold is "it's just LLM talking to itself". Mitigations:

- **Cross-model generation vs evaluation.** Generate with GPT-4.1-class, judge with a
  different family (Claude Opus 4.x, or the SG3D-converted real-anchor subset when
  available).
- **Human QA on a stratified sample.** Authors verify 30–50 plans across difficulty
  buckets before locking the gold.
- **Release the generation prompts, scene-graph inputs, and judge prompts** so the
  benchmark is deterministically reproducible.
- **Report both synthetic and SG3D numbers separately** whenever SG3D is in scope, so
  reviewers can track the gap between self-gold and external-gold.

### Why not the others

- **SayPlan**: no dataset, zero scenes. Cite as prior work only.
- **OpenEQA / HM-EQA**: Q/A, not plans. OpenEQA is already our EM-EQA benchmark for the
  Stage-2 agent; reusing it for plan evaluation would conflate the tasks. HM-EQA is
  additionally ruled out by the HM3D scene base.
- **R2R / REVERIE**: Matterport3D, zero scene overlap, wrong granularity (R2R) or
  schema (REVERIE). Scene-base adoption cost is prohibitive.

## Next Steps

1. **Land the GPT-oracle generator** under `src/benchmarks/navplan_synth.py` (or
   `synthetic_nav_plans.py`) that:
   - Reads `data/OpenEQA/scannet/<scene>/conceptgraph/{enriched_objects,scene_info}.json`.
   - Prompts the configured LLM for N plans per scene.
   - Validates each plan against the JSON schema and retries on schema failure.

2. **Fix the NAV_PLAN schema in `src/agents/runtime/base.py:106-111`** — update
   `default_payload_schema(Stage2TaskType.NAV_PLAN)` from `{subgoals, landmarks, risks}`
   to `{target_region, landmarks, waypoints, risks}`, and update
   `default_output_instruction(NAV_PLAN)` to match. This is the patent-contract fix
   flagged by the supervisor; without it the agent emits a shape the evaluator cannot
   score.

3. **Optional SG3D anchor loader** under `src/benchmarks/sg3d_loader.py` that:
   - Loads `train.json` / `test.json` from HF (cached locally).
   - Filters to `scan_id` prefix `ScanNet_` and our 89-scene list.
   - Exposes a transform-to-plan function using `enriched_objects.json` for region lookup.
   - Gated behind a flag; not required for the M5 pilot.

4. **Stand up an evaluator** in `src/benchmarks/navplan_eval.py` that scores predicted
   plans against gold on each field (region match, landmark recall, waypoint
   reachability, risk LLM judge).

## Sources

- SayPlan: https://sayplan.github.io/ · https://arxiv.org/abs/2307.06135 ·
  https://proceedings.mlr.press/v229/rana23a.html
- SG3D: https://sg-3d.github.io/ · https://arxiv.org/abs/2408.04034 ·
  https://github.com/sg-3d/sg3d · https://huggingface.co/datasets/ZhuofanZhang/SG3D
- OpenEQA: https://open-eqa.github.io/ · https://github.com/facebookresearch/open-eqa
- HM-EQA: https://explore-eqa.github.io/ · https://arxiv.org/abs/2403.15941 ·
  https://github.com/Stanford-ILIAD/explore-eqa
- R2R: https://bringmeaspoon.org/ · https://github.com/peteanderson80/Matterport3DSimulator ·
  https://arxiv.org/abs/1711.07280
- REVERIE: https://yuankaiqi.github.io/REVERIE_Challenge/dataset.html ·
  https://github.com/YuankaiQi/REVERIE · https://arxiv.org/abs/1904.10151
- SceneVerse (used by SG3D as scene-graph substrate):
  https://github.com/scene-verse/SceneVerse

## Appendix: Overlap Computation Provenance

Measured on 2026-04-21 from:
- `/tmp/our_scannet_scenes.txt` (89 lines, scene IDs like `scene0015_00`)
- `/tmp/sg3d_train.json` (13,650,680 bytes) and `/tmp/sg3d_test.json` (1,215,706 bytes)
  downloaded via `curl https://huggingface.co/datasets/ZhuofanZhang/SG3D/resolve/main/...`
- `/tmp/openeqa.json` (476,360 bytes) downloaded via
  `curl https://raw.githubusercontent.com/facebookresearch/open-eqa/main/data/open-eqa-v0.json`

Reproducibility: run the Python snippet in `Appendix A` of the Stage-2 evaluation doc
(`docs/05_evaluation.md`) or re-derive with the following (same commands used during
this survey):

```python
import json, re
ours = set(open('/tmp/our_scannet_scenes.txt').read().split())
sg3d = [*json.load(open('/tmp/sg3d_train.json')),
        *json.load(open('/tmp/sg3d_test.json'))]
sn_overlap = sorted({r['scan_id'].removeprefix('ScanNet_')
                     for r in sg3d if r['scan_id'].startswith('ScanNet_')} & ours)
print(len(sn_overlap), sn_overlap)
```
