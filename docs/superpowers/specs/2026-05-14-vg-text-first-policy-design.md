# VG Text-First Candidate Policy Design - 2026-05-14

## Goal

Make all visual-grounding benchmark agents, including NR3D and ScanRefer, behave
more like Transcrib3D at the start of each case:

1. use structured proposal metadata first;
2. narrow to category-compatible candidates;
3. use deterministic geometry/spatial tools when the language supports it;
4. request marked frames/crops only when structured evidence is insufficient.

The policy must not erase the benchmark difference between NR3D and ScanRefer.
NR3D uses ScanNet annotated object boxes, so text/geometry can often solve the
case. ScanRefer uses Mask3D-style detected proposals, so labels and boxes are
noisier and visual confirmation remains mandatory for most final decisions.

## Design

### Shared VG Policy

The VG system prompt and `vg-grounding-playbook` should make the first phase
explicitly text-first:

1. Read the `Scene Proposal Inventory`.
2. Identify the focal object category and compatible synonyms.
3. Call `find_proposals_by_category(...)` for the focal class before viewing
   more frames when the query gives an identifiable object type.
4. If the candidate set is one item, treat it as the leading hypothesis.
5. If the query contains size/height/elevation superlatives, call
   `rank_proposals_by_geometry(...)` on the same-category shortlist.
6. If the query contains anchor-based relations, call
   `compare_proposals_spatial(...)` after collecting target and anchor
   candidates.
7. Only request `view_keyframe_marked`, `request_more_views`,
   `request_crops`, or `switch_or_expand_hypothesis` after the structured
   pass leaves ambiguity or the language requires visual evidence.

### Benchmark-Specific Guardrails

The prompt must distinguish two candidate-pool regimes:

- **Clean GT/object-pool VG**: NR3D-style or EmbodiedScan-style object pools
  with reliable proposal identities. If category filtering and deterministic
  geometry produce a unique answer for non-visual language, the agent may
  submit after inspecting the proposal metadata. Visual confirmation is still
  useful, but not required for every pure text/geometry case.
- **Noisy detector-pool VG**: ScanRefer-style Mask3D proposal pools. The agent
  should still perform the same structured candidate pass first, but it must
  visually confirm the final proposal in a marked frame or crop unless the case
  is a clear no-match/OOD decision. This avoids over-trusting detector labels
  and large noisy boxes.

### When Visual Evidence Is Required

All VG benchmarks should require visual evidence before final submission when
the referring expression depends on:

- color, material, shape, texture, open/closed state, or object appearance;
- left/right/front/back language that is frame- or viewer-dependent;
- occlusion, containment, support, or "on top of" relations that can be wrong
  in 3D center-only comparisons;
- label-noisy candidate sets where category labels contradict visible marks;
- ScanRefer detector-pool cases, even if the structured pass yields one strong
  candidate.

### Implementation Targets

The expected implementation should touch only prompt/tool-policy surfaces:

- `src/agents/runtime/base.py`: strengthen the VG system prompt to define the
  text-first structured pass and detector-pool visual-confirmation rule.
- `src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md`: rewrite
  the decision tree so the structured pass comes before frame acquisition, with
  explicit ScanRefer/noisy-pool handling.
- `src/agents/core/agent_config.py`: bump `chassis_tools_version` because
  prompt behavior changes.
- tests under `src/agents/tests/` and
  `src/agents/packs/vg_embodiedscan/tests/`: assert that the prompt/playbook
  includes the structured-first steps, deterministic ranking rules, and
  ScanRefer visual-confirmation caveat.

No new retrieval algorithm, proposal pool, selector, or benchmark runner should
be added in this change. The first implementation should be prompt-policy only,
then evaluated on NR3D random100, NR3D first300, and ScanRefer random100.

## Success Criteria

- The prompt/playbook clearly says: structured candidate filtering and
  deterministic geometry/spatial comparison happen before more visual evidence.
- ScanRefer is explicitly covered: text-first candidate filtering applies, but
  final answers should be visually confirmed because Mask3D candidates are
  noisy.
- Existing VG tool surfaces remain unchanged.
- Relevant prompt snapshot/unit tests pass.
- Evaluation plan after implementation:
  - NR3D fixed random100 should recover from v10's 69.00 regression toward v9
    or v7.1's 71.00.
  - NR3D Transcrib3D first300 should not lose the v10 76.87 matched-fold gain.
  - ScanRefer random100 should not regress from the current documented
    ScanRefer v3.x baseline before any full-val run is attempted.

## Risks

- A text-first policy can overfit NR3D and underuse images on ScanRefer. The
  detector-pool visual-confirmation rule is the main mitigation.
- More explicit prompt rules may increase token count. The change should
  replace the current decision tree rather than append another long section.
- Deterministic geometry helps size/height language but can hurt view-dependent
  cases. The playbook must state that view-dependent relations require marked
  frame evidence before final submission.

## Spec Self-Review

- No placeholders remain.
- Scope is limited to VG prompt/playbook policy and tests.
- The design covers both clean object-pool VG and noisy detector-pool
  ScanRefer.
- Success criteria include both behavior checks and benchmark checks.
