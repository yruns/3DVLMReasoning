# v11.1 Failure Analysis: 29 Low-Score Cases

**Date**: 2026-03-30
**Branch**: `feat/v11-evidence-seeking-fixes`
**Baseline**: v11.1 E2E mean 3.505 (+22.6% vs v9), 97/100 completed
**Analyzed**: All 29 cases with E2E score <= 2 (28 score=1, 1 score=2)
**Method**: 6 parallel reviewer agents, each examining 10+ raw images per scene

---

## 1. Comparison with v10

| Metric | v10 | v11.1 | Change |
|--------|:---:|:-----:|:------:|
| Low-score cases (<=2) | 36 | 29 | -7 |
| Fixed from v10 | — | 12 | — |
| New in v11.1 | — | 5 | — |
| Persistent | — | 24 | — |

**12 cases FIXED** by v11.1's callbacks + min keyframes + open-ended mode.
**5 NEW failures** introduced (marked [NEW] below).

---

## 2. Root Cause Distribution

| Root Cause | Count | % | Description |
|------------|:-----:|:-:|-------------|
| RETRIEVAL_FAILURE | 18 | 62% | Wrong keyframes, wrong spatial coverage |
| REASONING_ERROR | 12 | 41% | VLM misidentified objects/colors/spatial relations |
| VISIBILITY_ISSUE | 5 | 17% | Dark scenes, tiny objects below detection threshold |
| AMBIGUOUS_GT | 3 | 10% | GT color/object debatable |
| LABELING_ERROR | 2 | 7% | GT answer appears factually incorrect |

---

## 3. Systemic Issues (ranked by impact)

### Issue 1: "Between X and Y" Spatial Queries — 5 cases

Cases: 18, 19, 21, 23, 28

Stage 1 retrieves keyframes showing X and Y independently but NOT the space between them. The agent needs a frame showing the spatial corridor between two landmarks.

Example: Case 23 "What is between sofa and kitchen counter?" — Stage 1 retrieved sofa views (frames 20, 48) and counter views (frame 582) but not frames 530-540 where the stools sit between them.

**Fix**: Spatial midpoint retrieval — when query has "between X and Y", compute the geometric midpoint between anchors and select views covering that region.

### Issue 2: Tool Under-Utilization — 13 cases

Cases: 1, 3, 13, 14, 17, 18, 19, 22, 23, 24, 25, 26, 28

Stage 2 agent has 0 tool calls even with confidence as low as 0.18-0.62. The agent answers immediately without seeking additional evidence. In v11.1 with callbacks enabled, the agent CAN use tools but often doesn't.

Worst examples:
- Case 28: conf=0.85, 0 tool calls, answered wrong corner (toolboxes vs desk)
- Case 25: conf=0.55, 0 tool calls, "no non-black chairs" when red chairs exist in 70% of frames
- Case 13: conf=0.90, 0 tool calls, said "Black" jacket when cream jacket is clearly underneath

**Fix**: Lower the confidence threshold for tool triggering; or add a prompt instruction: "When your answer contradicts the question's premise or when confidence < 0.7, ALWAYS request additional views."

### Issue 3: Spatial Preposition Blindness — 4 cases

Cases: 1, 2, 3, 20

"Under the table" queries consistently return desk-top views because CLIP co-visibility ranks frames where the table surface is visible, not where the under-table area is visible. Same issue for "behind" and "underneath."

**Fix**: Spatial-preposition-aware view reranking — for "under X", prefer frames with lower camera elevation angles; for "behind X", prefer frames from the opposite side.

### Issue 4: Keyframe Spatial Clustering — 3 cases

Cases: 9, 25, and partially 5

Stage 1 selects all keyframes from the same trajectory segment. Case 25: 35 chairs detected, but all 3 keyframes from views 5-28 (first 10% of trajectory) showing only black chairs. Red chairs visible from view 60+ were completely missed.

**Fix**: Enforce trajectory diversity — divide the view sequence into N segments and sample at least one keyframe from each relevant segment.

### Issue 5: request_more_views Returns Empty — 4 cases

Cases: 4, 5, 7, 10

When the agent correctly identifies insufficient evidence and calls `request_more_views`, the tool returns "No additional views available" despite 300 views existing. The tool implementation likely only serves views where the queried object was detected, but for undetected objects (toothbrush, hat), no views are available.

**Fix**: Fallback to trajectory-based view serving — when no object-specific views are available, serve spatially diverse frames from unexplored trajectory regions.

### Issue 6: Small Object Detection Gap — 5 cases

Cases: 3, 4, 15, 26, 27

Objects below the ConceptGraph detection threshold: toothbrush (013), trash bin (100), phone (109), calculator (109). These are never segmented, so Stage 1 cannot ground them.

**Fix**: Expand GSA/RAM vocabulary + run a secondary small-object detector for common household items.

---

## 4. NEW Failures in v11.1 (5 cases)

| Case | Scene | Question | Root Cause | Why New? |
|------|-------|----------|------------|----------|
| 9 | 037 | Which side of closet is emptier? | **REGRESSION** | Query parser degraded: `target_term` changed from `"closet shelf"` → `"UNKNOW"`, lost critical comparison keyframe |
| 18 | 102 | Blue object between table and fridge? | RETRIEVAL_FAILURE | "Between" spatial query not handled; agent too conservative |
| 21 | 104 | Between fireplace and sofa? | RETRIEVAL_FAILURE | Same "between" issue; "door" not in proxy vocabulary |
| 22 | 105 | Color of can next to monitor? | REASONING_ERROR | Brown→black color confusion; 0 tool calls at conf 0.58 |
| 23 | 105 | Between sofa and kitchen counter? | RETRIEVAL_FAILURE | Endpoints retrieved but not spatial midpoint |

**Case 9 is the most critical** — it's a regression caused by the v11.1 open-ended query mode. The query parser's `UNKNOW` target with `open_ended=True` changed the keyframe selection strategy, dropping the comparison keyframe that v10 selected correctly.

---

## 5. GT/Evaluator Issues Confirmed

| Case | Issue | Detail |
|------|-------|--------|
| 5, 7 | **GT LABELING_ERROR** | GT says "vacuum cleaner" but object is fire extinguisher (enriched_objects confirms) |
| 29 | **GT LABELING_ERROR** | GT says "coffee machine" but red object is KitchenAid stand mixer; actual coffee maker is black |
| 6 | **AMBIGUOUS_GT** | "Green" sofa appears gray/olive under dim lighting; enriched_objects says "beige" |
| 16 | **UNANSWERABLE** | "What color is my hat?" — camera wearer's hat not visible in any frame |
| 11, 12 | **BEYOND CAPABILITY** | Patio furniture outside scan boundary, only visible through glass |

---

## 6. Priority Fixes for v12

| Priority | Fix | Cases Affected | Complexity |
|:--------:|-----|:--------------:|:----------:|
| **P0** | Fix Case 9 regression: investigate open_ended parser interaction | 1 | Low |
| **P0** | Lower tool-trigger threshold / prompt instruction to use tools more aggressively | 13 | Low |
| **P1** | "Between X and Y" spatial midpoint retrieval | 5 | Medium |
| **P1** | Spatial preposition view reranking (under→low-angle, behind→reverse-side) | 4 | Medium |
| **P1** | Trajectory diversity enforcement for high-count targets | 3 | Medium |
| **P2** | Fix request_more_views fallback to serve trajectory-diverse frames | 4 | Medium |
| **P2** | Small object detection vocabulary expansion | 5 | High |
| **P3** | Semantic flexibility (armchair = sofa in small rooms) | 1 | Low |

**Conservative estimate: P0 fixes (regression + tool usage) could improve 10+ cases, pushing E2E from 3.505 toward ~3.7+.**

---

## 7. Per-Case Root Cause Table

| # | Score | Scene | Category | Question | Root Cause | NEW? | Persistent? |
|---|:-----:|-------|----------|----------|------------|:----:|:-----------:|
| 1 | 1 | 003 | spatial | Space under table by window? | RETRIEVAL_FAILURE | | v10 also |
| 2 | 1 | 013 | object | What's underneath table? | RETRIEVAL+VISIBILITY | | v10 also |
| 3 | 1 | 013 | spatial | In front of clothes pile? | VISIBILITY+REASONING | | v10 also |
| 4 | 1 | 013 | attribute | Toothbrush color? | RETRIEVAL_FAILURE | | v10 also |
| 5 | 1 | 014 | object | Red object next to sofa? | RETRIEVAL+LABELING | | v10 also |
| 6 | 1 | 014 | attribute | Sofa color near table? | AMBIGUOUS_GT | | v10 also |
| 7 | 1 | 014 | spatial | Behind vacuum cleaner? | RETRIEVAL_FAILURE | | v10 also |
| 8 | 1 | 031 | spatial | Where to find other shoe? | REASONING_ERROR | | v10 also |
| 9 | 1 | 037 | spatial | Which closet side emptier? | **REGRESSION** | **YES** | |
| 10 | 1 | 046 | object | Room number? | RETRIEVAL_FAILURE | | v10 also |
| 11 | 1 | 047 | attribute | Patio chair color? | BEYOND_CAPABILITY | | v10 also |
| 12 | 1 | 047 | object | Large object on patio? | BEYOND_CAPABILITY | | v10 also |
| 13 | 1 | 048 | attribute | Bottom jacket color? | REASONING_ERROR | | v10 also |
| 14 | 1 | 100 | spatial | Right of brown table? | REASONING_ERROR | | v10 also |
| 15 | 1 | 100 | object | Object near tables to left? | RETRIEVAL+VISIBILITY | | v10 also |
| 16 | 1 | 101 | attribute | Hat color? | UNANSWERABLE | | v10 also |
| 17 | 1 | 102 | object | White+green object on table? | RETRIEVAL_FAILURE | | v10 also |
| 18 | 1 | 102 | spatial | Blue object between table+fridge? | RETRIEVAL_FAILURE | **YES** | |
| 19 | 1 | 102 | spatial | Between two fridges? | REASONING_ERROR | | v10 also |
| 20 | 1 | 104 | spatial | More trash in gray bin? | RETRIEVAL_FAILURE | | v10 also |
| 21 | 1 | 104 | spatial | Between fireplace and sofa? | RETRIEVAL_FAILURE | **YES** | |
| 22 | 1 | 105 | attribute | Can color next to monitor? | REASONING_ERROR | **YES** | |
| 23 | 1 | 105 | spatial | Between sofa and counter? | RETRIEVAL_FAILURE | **YES** | |
| 24 | 1 | 106 | spatial | Behind the sofa? | REASONING_ERROR | | v10 also |
| 25 | 1 | 108 | attribute | Non-black chair color? | RETRIEVAL_FAILURE | | v10 also |
| 26 | 1 | 109 | spatial | Between pencil and notebook? | REASONING_ERROR | | v10 also |
| 27 | 1 | 109 | spatial | On top of calculator? | RETRIEVAL_FAILURE | | v10 also |
| 28 | 1 | 110 | spatial | Corner between two doors? | REASONING_ERROR | | v10 also |
| 29 | 2 | 105 | object | Red object on counter? | REASONING+AMBIGUOUS_GT | | v10 also |
