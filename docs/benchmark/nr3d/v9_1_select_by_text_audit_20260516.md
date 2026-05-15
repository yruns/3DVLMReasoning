# v9.1 `select_by_text` reliability audit — NR3D random100

- **Date**: 2026-05-16
- **Branch / tip**: `feat/v9-1-selectors-return-images` @ `1b515ba`
- **Script**: `scripts/audit_select_by_text_nr3d.py`
- **Sample list**: `tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json`
- **Raw output**: `docs/benchmark/nr3d/assets/select_by_text_audit_random100_20260516.json`
- **Stdout log**:   `docs/benchmark/nr3d/assets/select_by_text_audit_random100_20260516.log`

## What this measures

After the v9.1 wiring fix (`de8225f`), `select_by_text` is the first tool in the v9.1 playbook and is actually executing Stage-1. The previous two NR3D runs disagreed wildly on accuracy:

- `v9.1_fix` (broken Stage-1, silent fallback to catalog): **86%**
- `v9.1_real` (Stage-1 actually working, called as first move): **69%**

Before we redesign the playbook, we need to know how reliable `select_by_text` *itself* is on raw NR3D queries — independent of what Stage-2 does with the frames it returns.

**Question**: if we feed the raw NR3D query directly into Stage-1 (`KeyframeSelector.select_keyframes_v2`), how often does the top-K frame set contain at least one frame where the GT target object is actually visible (depth-aware)?

This is the *upper bound* on what `select_by_text` can contribute as the agent's first move.

## Method

For each of the 100 frozen NR3D `random100` samples:

1. Read the prepared sample artifact (`pack_nr3d_v1`) → natural-language `query` + `target_id` + `scene_id`.
2. Load the scene's Phase 8 depth-aware visibility index → `gt_frames = {frame_id : target visible}`.
3. Build (and cache per scene) a `KeyframeSelector` with `stride=1`, `llm_model=gemini-2.5-pro` — identical to the production runner config.
4. Call `selector.select_keyframes_v2(query=query, k=10, use_visual_context=False)`. Note: the production `select_by_text` tool caps `k` at 3; we used `k=10` so we can compute hit@1/3/5/10 from a single LLM call per sample.
5. Compute `hit@K = 1` iff `pred_top_K ∩ gt_frames ≠ ∅`. Also `recall@K`, `first_hit_rank`, and `gt_coverage = |gt_frames| / |valid_frames|` (a difficulty proxy).

Random baseline for hit@K (sampling K frames uniformly from the scene's valid frames, with replacement approximation): `1 - (1 - gt_coverage)^K`, averaged over the fold.

All 100 samples succeeded. No errors, no unmeasurable targets (every GT target is visible in at least 1 frame). 58 unique scenes, scene cache sized to 2 to keep memory bounded.

## Headline numbers

| K | Stage-1 hit@K | Random baseline | Δ vs random |
|---|---:|---:|---:|
| 1  | **47.0 %** | 21.0 % | **+26.0 pp** |
| 3  | **55.0 %** | 48.0 % | **+ 7.0 pp** |
| 5  | **55.0 %** | 63.9 % | **− 8.9 pp** |
| 10 | **55.0 %** | 82.8 % | **−27.8 pp** |

- `mean_recall@3` = **0.42** (when there is a hit, the hit is dense)
- `mean_first_hit_rank` (when hit@10) = **1.16** — when Stage-1 hits, it almost always hits at rank 1.
- `mean_gt_coverage` over the fold = **0.21**.

Two things are immediately striking:

1. **The hit rate plateaus at 55 % from K=3 onwards.** Increasing the budget from 3 frames to 10 frames buys *nothing*. The 45 % of queries that miss at K=3 still miss at K=10.
2. **Stage-1 is strongly precise at K=1** (+26 pp over random) but its absolute ceiling is well below the random baseline once K ≥ 5.

The plateau is fully explained by **empty predictions**.

## The 32% empty-pred wall

| Bucket | n | hit@1 | hit@3 |
|---|---:|---:|---:|
| Stage-1 returned ≥1 frame | 68 / 100 | 47 / 68 = **69.1 %** | 55 / 68 = **80.9 %** |
| Stage-1 returned 0 frames (`parse=no_evidence`) | 32 / 100 | 0 / 32 | 0 / 32 |

When Stage-1 returns *anything* it is genuinely good — **81 % hit@3** — and the hit is almost always at rank 1. But on 32 of 100 random NR3D queries Stage-1's query-executor returns `no_evidence`: the parser produced hypotheses, but the executor could not ground them. For the agent these manifest as the `select_by_text` tool returning `{"frames": []}`, which is strictly less useful than three uniformly-random valid frames.

## Difficulty stratification

| Difficulty (gt_coverage) | n | Stage-1 hit@3 | Random hit@3 | Δ |
|---|---:|---:|---:|---:|
| Very easy (≥ 0.30) | 16 | 68.8 % | 76.1 % | − 7.4 pp |
| Medium (0.10 – 0.30) | 66 | 53.0 % | 49.1 % | + 4.0 pp |
| Hard   (< 0.10)    | 18 | 50.0 % | 19.3 % | **+30.7 pp** |

This is the most actionable picture:

- **On hard queries (rare targets, < 10 % frame coverage), Stage-1 is dominant (+31 pp).** A random selector almost never sees a rare target; Stage-1 catches it in half the cases.
- **On medium queries, Stage-1 and random are roughly tied.** The 32% empty-pred toll cancels out the K=1 precision win.
- **On easy queries, Stage-1 *hurts*.** Common targets are everywhere — random hits 76 % already, and the 32% empty-pred queries drag Stage-1 to 69 %.

## Empty-pred failure modes

Of the 32 empty-pred queries, heuristic tagging of the raw text (queries can belong to multiple buckets):

| Tag | Count | Example |
|---|---:|---|
| Has spatial-anchor phrasing ("X next to Y", "X under Y", "X across from Y") | 26 | "cabinet touching the wall farthest from entrance door" |
| Uses right / left | 13 | "Choose the chair on the right." |
| View-dependent ("facing X", "standing at Y", "enter the room") | 8 | "Facing fridge, this cabinet is middle of the three upper cabinets to the right." |
| Over-specific target term ("Apple keyboard", "Padded folding chair", "Upper kitchen cabinet") | 6 | "Choose the chair on the right." (parser emits `Padded folding chair`) |
| Uses middle / between | 5 | "'The lamp between the two beds.'" |
| Uses explicit negation | 1 | "Pick the cabinet under the desk that does NOT have a red chair" |

The dominant failure is the **spatial-anchor grammar**: the parser finds the target category fine, but the executor cannot resolve "the entrance door", "the wall", "the bunk bed", "the bookshelf in the corner", etc. into a unique ConceptGraph anchor — so the joint-coverage step gets no anchor view and falls back to "no_evidence". A secondary failure is the parser **over-specifying the target category** (`Padded folding chair` from "Choose the chair on the right.") so that even matching the simple category fails.

All 32 empty-pred queries are listed in
`docs/benchmark/nr3d/assets/select_by_text_audit_random100_20260516.json` under
`samples[*]` with `pred_top_k == []` and `parse_status == "no_evidence"`.

## Implications

This audit reconciles the v9.1_fix vs v9.1_real gap:

- v9.1_fix accidentally ran with `runtime.keyframe_selector == None`, so `select_by_text` always errored and the agent silently fell back to catalog-driven selectors. The catalog has a 100 % "frame containing the target" rate by construction (proposals carry frame_views) — so for 86 % of samples the agent saw a useful frame.
- v9.1_real has Stage-1 wired correctly *and* prioritises `select_by_text` as the first move. On ≈ 1/3 of queries this first move returns nothing, leaving the agent with only the unmarked BEV and no concrete frame in flight; on another fraction it returns the *wrong* frame for easy queries that random would have nailed. The net is the observed 17 pp regression.

For deciding the v9.1 → v9.2 playbook the relevant policy options are now:

| Option | Expected | Rationale |
|---|---|---|
| A. Catalog-first; `select_by_text` only for hard queries (mention of "facing", "between", multi-clause spatial) | Recover toward 86 %; keep Stage-1 value on the hard tail | Stratification shows hard-queries are exactly where Stage-1 dominates. |
| B. Task-conditioned: VG → catalog-first, QA / OpenEQA → text-first | Hold OpenEQA gains while restoring NR3D | OpenEQA "where is the kitchen counter" queries match Stage-1's strength (rare-target retrieval). |
| C. Patch the executor — drop the over-specific category (`Padded folding chair → chair`) and add a soft anchor-resolution fallback (skip unresolved anchors and return top-K by target alone) | Cuts empty-pred rate from 32 % → < 15 %; lifts ceiling from 55 % to ~70 % | The 32 empty-preds are not deep semantic failures — they are parser/executor coverage gaps. |
| D. `select_by_text` returns coverage-fallback frames when `no_evidence` | Eliminates the "actively worse than random" failure mode | Cheap one-line patch in the tool; downstream agent still sees 3 candidate frames. |

The audit is consistent with the v9.1_real handoff: **catalog-first + text-first-only-when-needed** (Option A + B) is the lowest-risk fix for accuracy. Option C+D is the v9.2 investment that turns Stage-1 from "either great or hopeless" into "always at least as good as random".

## Reproduce

```bash
source .venv/bin/activate
python scripts/audit_select_by_text_nr3d.py \
    --sample-ids tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json \
    --data-root data/nr3d/scannet \
    --pack-name pack_nr3d_v1 \
    --k 1,3,5,10 \
    --max-selector-cache-size 2 \
    --output tmp/audit_select_by_text_nr3d/random100.json
```

Runtime on this Mac: ~33 min wall-clock (LLM-bound). Each query parse + Stage-1 retrieval is 10-50 s, dominated by `gemini-2.5-pro` latency in `QueryParser`.

## Caveats

- The audit calls `select_keyframes_v2` with `use_visual_context=False` to match what `select_by_text` does at runtime. Turning on BEV-context might lift the parser's accuracy on view-dependent queries; not measured here.
- "GT visible" uses the Phase 8 depth-aware visibility index. Other definitions ("target is the largest object in the frame") would shift hit@K slightly but not change the qualitative picture — the 32 empty-pred cases are unmeasurable under *any* hit definition because `pred_top_K = ∅`.
- The fold is fixed `random100` from `v4_agent_guards_fair_views`. Different folds may have a different easy/medium/hard mix.
- LLM determinism: `select_keyframes_v2` is partially LLM-driven, so reruns may produce different hits on borderline queries. Single-pass run; not averaged.
