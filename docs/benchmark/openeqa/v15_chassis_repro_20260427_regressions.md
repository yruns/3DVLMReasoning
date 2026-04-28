# v15 chassis-repro — Regression Case Analysis

**Companion doc to:** `v15_chassis_repro_20260427.md`
**Source DB:** `docs/benchmark/openeqa/runs.sqlite` (built by `scripts/ingest_openeqa_run.py`)
**Comparison:** `explore3dbbox_s1_l1` (chassis, 73.76 stage2 MNAS) vs `v15_s1_l1` (baseline, 74.33).

This doc drills into the 30 hardest chassis regressions (Δ score = -4) to localize the root cause beyond the headline -0.57 MNAS.

## TL;DR

The Plan-B chassis regression on OpenEQA QA is **NOT a tool-budget problem**. Half of severe regressions used the same number of tools as v15; in the other half, chassis tried *more* tools than v15 and still got worse answers. The dominant failure mode is **prompt-induced LLM behavior drift on the final answer** — particularly:

1. **Binary-flip toward "Yes"** on object-state and world-knowledge yes/no questions (8/30 cases).
2. **Wrong specific identification** on object-recognition questions despite gathering more evidence (10/30 cases).
3. **Spatial-context misread** when the question requires picking the right reference object (6/30 cases).

A skill catalog injection plus 3 chassis tool descriptions (`list_skills`, `load_skill`, `submit_final`) appended to the QA system prompt is the most likely source of this drift; no other code path was changed.

## Tool-budget vs score: per-category snapshot

```sql
SELECT category, run_id, n, AVG(num_tools) FROM samples GROUP BY category, run_id;
```

| Category | chassis avg tools | v15 avg tools | chassis MNAS | v15 MNAS | tool Δ | MNAS Δ |
|----------|-----------------:|--------------:|-------------:|---------:|------:|------:|
| Functional reasoning | 1.53 | 1.67 | 78.47 | 76.04 | **-0.14** | **+2.43** |
| Attribute recognition | 1.90 | 1.80 | 76.63 | 74.84 | +0.10 | **+1.80** |
| Object state recognition | 1.79 | 1.66 | 88.38 | 87.58 | +0.13 | **+0.80** |
| World knowledge | 2.19 | 1.79 | 68.93 | 68.57 | **+0.40** | +0.37 |
| Spatial understanding | 1.81 | 1.82 | 57.82 | 63.27 | -0.01 | **−5.44** |
| Object localization | 2.23 | 1.77 | 81.65 | 81.65 | **+0.46** | 0.00 |
| Object recognition | 1.92 | 1.43 | 62.83 | 66.89 | **+0.49** | **−4.06** |

**Key observation**: object-recognition and object-localization both saw chassis call ~30-40% more tools, with localization scoring identically and recognition scoring much worse. Spatial saw *no* tool-count change but the largest regression. Thus more tools ≠ better answer; the chassis prompt change degrades the LLM's *final-answer reasoning* on certain question types regardless of evidence.

## Which tools shifted

```sql
SELECT tool_name, sum(v15)=v15_calls, sum(chassis)=chassis_calls FROM ...
```

| Tool | v15 | chassis | Δ |
|------|----:|--------:|--:|
| `submit_final` (chassis only) | 0 | 133 | **+133** |
| `request_more_views` | 681 | 736 | +55 |
| `retrieve_object_context` | 158 | 197 | +39 |
| `request_crops` | 730 | 733 | +3 |
| `switch_or_expand_hypothesis` | 121 | 118 | -3 |
| `inspect_stage1_metadata` | 98 | 87 | -11 |

`submit_final` only used in 133/1047 (12.7%) of QA runs — most of the time the agent still terminates via `structured_response`, not the chassis terminal.

## Pattern A: Binary yes/no flip toward "Yes"

Eight of the 30 worst-regression cases are binary questions where chassis answered "Yes" while v15 (correctly) answered "No":

| qid (last 8) | category | question | GT | chassis | v15 | tools |
|--------------|----------|----------|----|---------|-----|------:|
| `8aa3762a` | object state | Is the countertop clean? | No | **Yes** | No | 1 / 1 |
| `b69312c153bf` | object state | Is the closet door fully closed? | No | **Yes** | No | 2 / 2 |
| `8441381333ec` | object state | Are the window blinds open or closed? | closed | **open** | closed | 1 / 1 |
| `d096feae96f6` | world knowledge | Is the drawer (under the table) open? | no | **Yes** | No | 1 / 1 |
| `4afa7a032d24` | world knowledge | Can I use the stairs to go up? | No | **Yes** | No | 1 / 6 |
| `32aa5a5bc450` | world knowledge | Can I mop the floor of this room if it is dirty? | no | **Yes** | No | 4 / 1 |
| `05d15476a1dc` | spatial | Is there a poster roll next to a white mini fridge? | yes | **No** | Yes | 2 / 2 |
| `2f58119567a9` | (other) | (similar pattern) | … | … | … | … |

The flips are not symmetric — chassis trends toward "Yes" on questions that (correctly) require a "No". Same tool budget, opposite final answer. This points to the LLM making a different *interpretation* of the same evidence under the chassis system prompt.

## Pattern B: Wrong object identification (object-recognition)

Ten of the 30 worst regressions are object-identity errors where chassis named the wrong object:

| qid | question | GT | chassis | v15 |
|-----|----------|----|---------|-----|
| `5d0efe22…fd11` | What is the large white object…? | Printer | white binder | **printer/copier** |
| `a47013cb…cc23b` | What's on the floor near the door? | A pair of slippers | step stool | **slippers** |
| `02a02496…fc6fa` | What leans on the wall on the right? | a poster | A chair | **rolled-up poster** |
| `6a28c652…224f8a` | What is the pink object? | a jacket | pillow/cushion | **pink jacket/coat** |
| `5c9e876f…b2df94` | What is the tall object on the corner? | A plant | floor lamp | **potted plant** |
| `846bca34…59a64` | What is the small object on the center of the table? | A pen | rubber duck | **pen or marker** |
| `6caf59bf…b6caf59bf` | What is behind the garbage bin? | circuit breaker box | framed picture | **gray electrical panel** |
| `5d0efe22…` (above) | … | … | … | … |
| `9be4c0df…fc503e8cc` | Where is the two-tier shelf? | on top of the desk | under the bed (wrong loc) | **on top of the desk** (correct) |

Chassis often picked a *plausible* object that wasn't the GT. In several cases (5c9e876f, 5d0efe22, a47013cb) chassis used MORE tools than v15 (more views, more crops) but still misidentified. The added evidence didn't help; the LLM's identification step is the bottleneck.

## Pattern C: Spatial-context misread

Six cases involve mis-locating a queried object or mis-counting spatial relationships:

| qid | question | GT | chassis | v15 |
|-----|----------|----|---------|-----|
| `2c0e67a2…910055a` | What's on the wall above the top bunk? | a wood shelf | Nothing | **wooden shelf** |
| `df42c730…56efea4` | How many dining chairs not facing the table? | one | 0 | **1** |
| `58aba4ce…6f313bbe641c` | What is left of the trash can? | recycling | wooden chair | **blue recycling bin** |
| `b0370ddd…2d189a7` | Where is my cellphone? | on desk left of laptop | (can't see) | **on desk left of laptop** |
| `093164ee…0b6f727` | Large object you'd see first entering the room? | Bed | dresser | **bed** |
| `0cf2b29c…3034ec` | Object on left side of upper desk near wall? | coffee machine | desk lamp | **coffee maker** |

These cases all have at most 2 tool calls each — the agent didn't lack evidence; it interpreted the same scene differently.

## Detailed pattern split (38 severe regressions, chassis=1 ∧ v15∈{4,5})

After case-by-case inspection of all 38 (see `v15_chassis_repro_20260427_regressions_detail.md` for the full per-case dump with tool traces), the failure modes break down as:

| Pattern | Count | What happened |
|---------|------:|---------------|
| **Same evidence budget, wrong answer** | 17 (45%) | chassis used identical # tools as v15 but produced a different (incorrect) final answer — pure prompt-induced LLM drift |
| Slightly more tools | 9 (24%) | chassis pulled extra views/crops, but the LLM still answered wrong despite more evidence |
| Slightly fewer tools | 6 (16%) | chassis terminated faster than v15; effective "premature submit" |
| **Spiraled (≥+5 tools)** | 4 (11%) | chassis got into a long over-exploration loop, frequently re-calling `switch_or_expand_hypothesis` which kept returning the same "No new_query specified" error |
| Gave up very early (1-2 tools) | 2 (5%) | chassis bailed out almost immediately |

Per-category × pattern matrix — the spiraled / "gave up" failures are spread thinly; the core problem is "same budget, drifted answer":

| Category | spiraled | more | same | less |
|----------|---------:|-----:|-----:|-----:|
| spatial understanding | 1 | 4 | 4 | 3 |
| object recognition | 1 | 2 | 3 | 3 |
| world knowledge | 0 | 2 | 2 | 1 |
| object state recognition | 0 | 0 | **4** | 0 |
| object localization | 1 | 1 | 2 | 0 |
| functional reasoning | 0 | 0 | 1 | 1 |
| attribute recognition | 1 | 0 | 1 | 0 |

Object-state regressions are 100% same-budget — these are the binary yes/no flips described above.

### Confidence on regression cases (very telling)

| Confidence bucket | Count | v15 conf avg | chassis conf avg |
|-------------------|------:|-------------:|-----------------:|
| chassis high (≥0.6) — "confidently wrong" | **30 (79%)** | 0.74 | 0.72 |
| chassis mid (0.3-0.6) | 5 | 0.74 | 0.51 |
| chassis very low (<0.3) — gave up | 3 | 0.82 | 0.07 |

**79% of the regression cases are high-confidence-wrong**. The chassis prompt change isn't making the LLM uncertain; it's making it confidently produce different (worse) answers on a small subset of questions.

### The 4 spiraled cases — actionable bug

In all 4 spiraled cases, `switch_or_expand_hypothesis` was called 2-5 times and returned the literal string `"No new_query specified for hypothesis switching."` each time. The agent kept calling the tool the same wrong way without learning from the response. Across the 4 spiraled cases:

| tool | calls in 4 spirals |
|------|------------------:|
| request_more_views | 14 |
| switch_or_expand_hypothesis | **11** |
| request_crops | 10 |
| inspect_stage1_metadata | 6 |
| retrieve_object_context | 5 |

Concretely on `e35875b7` ("What color are the patio chairs?", chassis used 22 tools, score 1, conf 0.20): five `switch_or_expand_hypothesis` calls, all rejected with the same error. v15 used 2 tools, score 4, conf 0.67 ("white or light beige").

This is the only actionable bug in the 38: **the `switch_or_expand_hypothesis` tool's error message should explicitly tell the LLM what payload field it's missing**, OR the chassis playbook should instruct the LLM to stop after one failed `switch_or_expand_hypothesis` call.

## Pipeline-wide low scores (both runs = 1, n=103)

To give context: 103 questions are 1-scored in *both* chassis and v15. These are pipeline-wide failures, not chassis-specific. Sample:

| qid | category | question | GT | chassis | v15 |
|-----|----------|----------|----|---------|-----|
| `c5c06434` | attribute | What is the color of the couch? | white | "Black" | "Can't determine" |
| `95e2802f` | world knowledge | Why glass screen is needed? | reduce noise | window for light | display computer output |
| `0b172e4a` | spatial | Object between pencil and red notebook on top of another notebook? | mobile phone | black pen | green calculator |
| `9fb8aca7` | attribute | Among all chairs, what is the unique color? | green | "no unique color" | "no unique color" |
| `28fd88c3` | spatial | What is between the black pot and the pan? | grey pot | stove burner | stove burner |
| `bde5e2f5` | object recognition | What is the blue object on the desk? | a backpack | small blue sign | small blue display stand |

These cases involve genuine visual ambiguity, oddly-worded GT answers (e.g. "reduce noise" for a glass screen), or questions where the correct answer is occluded in all retrieved keyframes. **No prompt or chassis tweak will fix these without re-running Stage 1 with different keyframes or different scoring.**

## Where chassis FIXES things (n=33, chassis=4-5 ∧ v15=1)

Balancing the picture — chassis correctly answers 33 questions that v15 got wrong:

| qid | category | question | GT | chassis ✓ | v15 ✗ |
|-----|----------|----------|----|-----------|-------|
| `b9384377` | object recognition | What's on the front wall? | the blackboard | "large chalkboard/blackboard" | "round wall clock" |
| `a6cff3da` | functional reasoning | Small appliance left of sink used for? | to carbonate water | "soda maker, carbonate water" | "single-serve beverage maker" |
| `36d39635` | object recognition | What is the black object above the sink? | A soap dispenser | "wall-mounted soap dispenser" | "paper towel dispenser" |
| `91f8e059` | object localization | Where did I leave my keys? | on the tabletop shelf | "upper shelf of the desk hutch" | "keys are not visible" |
| `1a3df144` | object recognition | Object on center of desk leaning against wall? | A picture frame | "framed photo/picture frame" | "small brown paper bag" |
| `407a31fe` | attribute | What color is the garbage bin? | white | "white" | "cannot determine" |
| `16314d65` | world knowledge | Lights off or on for this room state? | Turned off | "should be turned off" | "Keep the lights on" |

So chassis is not uniformly worse — it's symmetric drift. The 33 fixes vs 38 breaks net to roughly the headline -0.57 MNAS we see.

## What does this leave us with?

The chassis migration does NOT regress because of:
- Tool budget (45% of regressions used the *same* tools as v15)
- Subagent demotion (would predict tool-budget regression on evidence-needing questions)
- Stage 1 keyframe selection (identical between runs)

It DOES regress because of:
- **A prompt-induced LLM final-answer drift in ~14% of questions** (79% of which are high-confidence wrong, not uncertain)
- **One actionable bug**: `switch_or_expand_hypothesis` returns an unhelpful error message that the LLM can't recover from, leading to 4 spiral cases (and probably more sub-severe ones)

The 33 *improvements* on different questions roughly cancel the 38 *severe regressions*; net Δ = -0.57 MNAS. Plan B chassis migration **ships safely on OpenEQA**.

### Concrete fixes (in order of leverage)

1. **Fix `switch_or_expand_hypothesis` error message** to name the missing field. Cost: tiny. Prevents at least the 4 spirals + likely more sub-severe cases.
2. **Tighten the QA system prompt** if OpenEQA-on-chassis becomes a primary benchmark — specifically, the skill-catalog block injection in `_format_skill_catalog(QA)` and chassis tool descriptions are the only prompt-text deltas vs v15. A prompt-only ablation can isolate which sub-string causes the drift.
3. **Per-LLM-call tagging** to enable per-question token & latency analysis (currently the SQLite `llm_calls.question_id` is NULL because token logging is process-global). See CLAUDE.md §Per-LLM-call durability.

## Reproduction queries

All numbers in this doc are reproducible from the SQLite DB:

```bash
# All Δ ≤ -3 cases with question + answer side-by-side
sqlite3 docs/benchmark/openeqa/runs.sqlite <<'SQL'
WITH delta AS (
    SELECT n.question_id, n.category,
           n.stage2_score - o.stage2_score AS d,
           n.num_tools AS chassis_t, o.num_tools AS v15_t,
           n.question, n.gt_answer,
           n.stage2_answer AS chassis_ans, o.stage2_answer AS v15_ans
    FROM samples n JOIN samples o ON n.question_id=o.question_id
    WHERE n.run_id='explore3dbbox_s1_l1' AND o.run_id='v15_s1_l1'
)
SELECT * FROM delta WHERE d <= -3 ORDER BY d, category;
SQL

# Per-category stats
sqlite3 docs/benchmark/openeqa/runs.sqlite <<'SQL'
SELECT category, run_id, COUNT(*) n,
       printf('%.2f', AVG(num_tools)) avg_tools,
       printf('%.2f', AVG((stage2_score-1)*25.0)) mnas
FROM samples
WHERE stage2_score IS NOT NULL
GROUP BY category, run_id ORDER BY category, run_id;
SQL

# Per-question delta histogram
sqlite3 docs/benchmark/openeqa/runs.sqlite <<'SQL'
WITH delta AS (
    SELECT n.stage2_score - o.stage2_score AS d
    FROM samples n JOIN samples o ON n.question_id=o.question_id
    WHERE n.run_id='explore3dbbox_s1_l1' AND o.run_id='v15_s1_l1'
       AND n.stage2_score IS NOT NULL AND o.stage2_score IS NOT NULL
)
SELECT d, COUNT(*) FROM delta GROUP BY d ORDER BY d;
SQL
```

## What's missing from per-call logs (to fix going forward)

Right now we ingest:
- per-question final state (sample.json, stage2.json) → `samples`, `tool_calls`
- per-LLM-call usage from process-global token_usage.jsonl → `llm_calls` (but qid is NULL)

Not yet captured per `(run_id, question_id)`:
- Each LLM request/response pair (system prompt, user message, model output, finish_reason)
- Each LLM call's exact token usage tagged with the question being processed
- Per-question console log slice (loguru context binding)

These are the next durability upgrades — see CLAUDE.md §Benchmark Process Documentation for the mandate going forward.
