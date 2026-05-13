# NR3D Leaderboard-Track Fairness — Design

**Date:** 2026-05-01  
**Branch at design time:** `feat/nr3d-vg-benchmark`  
**Tip commit at design time:** `f9783b9`  
**Authors:** Shuhao Yue (designer / executor), Codex worker (paper crosscheck), Claude Code worker (referit3d code audit)

## Goal

Make our NR3D test evaluation **strictly comparable** to the public NR3D
leaderboard (ReferIt3DNet, 3D-VisTA, MVT, MiKASA, UniVLG GT-track, etc.) so we
can publish a 5-column row (Overall / Easy / Hard / View-Dep / View-Indep)
that lines up against published numbers without protocol caveats.

The current `v2_phase8_full` headline (Acc@0.25 = 77.67 %, Acc@0.50 = 77.62 %,
mean IoU = 0.7800 on 8584 utts) is **not directly comparable** to public
numbers because it scores 9-DoF oriented IoU on a GT-pool, while the public
leaderboard scores classification accuracy on the same pool. This design
delivers the missing classification-accuracy track.

## Background

The canonical NR3D protocol is now summarized in
`docs/benchmark/nr3d/protocol.md`, merged from the older ReferIt3D code audit,
paper crosscheck, and pool-equivalence notes. That consolidated note records
the empirical verification that our Phase 8 GT-CG pool and 130-scene fold match
the canonical NR3D GT-track dimensions (8584 utts, 100 % `target_id` class
alignment).

Key takeaways from the audits:

1. Canonical pool = full-scene segmented instances (target + same-class +
   shuffled clutter), capped at `max_test_objects=88`. **Not** target-type-only.
2. Canonical metric = classification accuracy (`predicted_id == target_id`),
   per-utterance binary score. **No IoU threshold.**
3. Canonical filter chain at test time =
   `bad-context blacklist + clothes drop + mentions_target_class_only=True +
   max_seq_len=24 + drop unique-target test utts`. `correct_guess=True` is
   **not** canonical — it's a UniVLG-specific add-on.
4. Easy ⟺ `n_objects ≤ 2` (single same-class distractor); Hard ⟺ `n_objects > 2`.
5. View-dep ⟺ `set(tokens) ∩ {front, behind, back, right, left, facing,
   leftmost, rightmost, looking, across} ≠ ∅` (per-token, not substring).
6. Detection-mode papers (BUTD-DETR, UniVLG-Det, MCLN) report a separate
   Acc@0.25 / Acc@0.50 track on detector proposals — out of scope for this
   design.

The pool-equivalence log proves we share canonical's pool and fold without
modification; the only deltas are at the protocol layer (filter, metric,
slicing).

## Scope

### In scope

- A new aggregator module that consumes existing `side_by_side.json` outputs
  and emits leaderboard-track metrics.
- Schema additions to the SQLite ingester to record the new metrics.
- A new dated version doc `docs/benchmark/nr3d/v3_referit3d_track_<YYYYMMDD>.md`
  reporting 5-column results against the public SOTA table.
- Updates to `docs/benchmark/nr3d/README.md` and `docs/benchmark/README.md`.
- Tests for the aggregator covering the canonical filter, easy/hard partition,
  view-dep keyword matching, and inner-join completeness.

### Out of scope (explicit non-goals)

- Detector-pool variant (V-DETR / BIP3D / ConceptGraph detection) — separate
  P3 work, mirrors the EmbodiedScan v4-v6 detector-pool work.
- Tokenization re-alignment (running `referit3d/data_generation/nr3d/tokenization.py::pre_process_text`
  on our queries before the agent sees them). The agent gets raw natural
  language; this is a paradigm choice (zero-shot RGB+VLM vs trained 3D
  models), not a protocol violation. Documented as a caveat in the v3 doc.
- SR3D dataset integration.
- Re-running the Stage 2 agent. We reuse all 8584 per-sample checkpoints
  from `tmp/nr3d_eval_v1_full/per_sample/`.
- View-dep keyword set being configurable. Hardcoded literal set, matching
  `referit3d/analysis/utterances.py:103-105`.

## Architecture and data flow

```
NR3D CSV  +  Phase 8 GT-CG packages
       │
       │  Nr3dDataset.from_path(mentions_target_class_only=True)
       ▼
canonical fold (≈7805 utts after filter; verified empirically)
       │
       │  inner-join on sample_id
       ▼
     ┌─────────────────────────────────────────┐
     │  side_by_side.json (from v2 run)        │  ←  reused as-is
     │    per_sample[*].selected_object_id     │
     │    per_sample[*].sample_id              │
     └────────────────┬────────────────────────┘
                      │
                      ▼
        nr3d_leaderboard_metrics.compute_leaderboard_metrics
                      │
                      ▼
            leaderboard_metrics.json
                {n_full, n_filtered,
                 classification_acc_full,
                 classification_acc_filtered,   ← headline
                 acc_easy / acc_hard,
                 acc_view_dep / acc_view_indep,
                 per_sample[]}
                      │
                      ▼
            ingest_nr3d_run.py (extended)
                      │
                      ▼
            docs/benchmark/nr3d/runs.sqlite
                      │
                      ▼
        docs/benchmark/nr3d/v3_referit3d_track_<date>.md
                      │
                      ▼
        README.md / leaderboard.md updates
```

Phase 8 packages, the keyframe selector, the runner, and the agent are
**unmodified**. The aggregator is a pure post-processing step that turns
already-recorded `selected_object_id` predictions into the leaderboard
columns.

## Components

### Component 1 — `src/evaluation/scripts/nr3d_leaderboard_metrics.py` (new)

Public function:

```python
def compute_leaderboard_metrics(
    side_by_side_path: Path,
    nr3d_data_root: Path,
    phase8_data_root: Path,
    canonical_filter: bool = True,
) -> dict:
    ...
```

Returned dict keys (all required):

| key | type | meaning |
|---|---|---|
| `n_full` | int | utts in side_by_side (8584 for v2 fold) |
| `n_filtered` | int | utts after `mentions_target_class_only=True` |
| `classification_acc_full` | float | acc on n_full denominator |
| `classification_acc_filtered` | float | **headline** — acc on canonical fold |
| `acc_easy` | float | acc on `n_objects ≤ 2` ∩ filtered |
| `acc_hard` | float | acc on `n_objects > 2` ∩ filtered |
| `acc_view_dep` | float | acc on view-dep ∩ filtered |
| `acc_view_indep` | float | acc on view-indep ∩ filtered |
| `n_easy` / `n_hard` / `n_view_dep` / `n_view_indep` | int | per-slice denominators |
| `per_sample` | list[dict] | one row per sample with all flags |

The view-dep keyword set is hardcoded:

```python
_VIEW_DEP_TOKENS = frozenset({
    "front", "behind", "back", "right", "left",
    "facing", "leftmost", "rightmost", "looking", "across",
})
# ↑ source: referit3d/analysis/utterances.py:103-105
```

CLI entry point:

```
python src/evaluation/scripts/nr3d_leaderboard_metrics.py \
    --side-by-side tmp/nr3d_eval_v1_full/side_by_side.json \
    --nr3d-data-root data/nr3d \
    --phase8-data-root data/nr3d/scannet \
    --output tmp/nr3d_eval_v1_full/leaderboard_metrics.json
```

#### Invariants (enforced as runtime asserts inside the aggregator)

1. Every sample_id from the canonical filtered fold appears in
   `side_by_side_path` (raise on missing — no silent partial denominators).
2. `n_filtered ≤ n_full`.
3. `n_easy + n_hard == n_filtered`.
4. `n_view_dep + n_view_indep == n_filtered`.
5. `per_sample` length equals `n_full`; each row carries `is_filtered_out`,
   `is_easy`, `is_view_dep`, `is_correct` booleans.

### Component 2 — `src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py` (new)

Pytest module covering:

- `test_classification_acc_all_correct` — 4 hits → 1.0
- `test_classification_acc_all_wrong` — 4 misses → 0.0
- `test_easy_hard_partition` — `n_objects ∈ {2, 3, 5}` produces correct
  easy / hard denominators
- `test_view_dep_token_per_token_not_substring` — `["frontier", "of"]`
  does NOT trigger; `["front", "of"]` does
- `test_view_dep_lowercase_assumption` — uppercase token does NOT match
  (asserts upstream lowercase invariant)
- `test_filtered_fold_completeness_raises_on_missing` — missing sample_id in
  `side_by_side` raises rather than silently scoring 0
- `test_canonical_filter_off_makes_filtered_equal_full` — sanity for
  `canonical_filter=False`

Each test creates synthetic `side_by_side.json` and a synthetic
`mentions_target_class` lookup; no real Phase 8 packages required.

### Component 3 — `scripts/ingest_nr3d_run.py` (extend)

#### Schema additions

`runs` table — new columns (NULLable for backward compatibility):

```sql
ALTER TABLE runs ADD COLUMN classification_acc_full REAL;
ALTER TABLE runs ADD COLUMN classification_acc_filtered REAL;
ALTER TABLE runs ADD COLUMN acc_easy REAL;
ALTER TABLE runs ADD COLUMN acc_hard REAL;
ALTER TABLE runs ADD COLUMN acc_view_dep REAL;
ALTER TABLE runs ADD COLUMN acc_view_indep REAL;
ALTER TABLE runs ADD COLUMN n_filtered INTEGER;
```

`samples` table — new columns:

```sql
ALTER TABLE samples ADD COLUMN is_easy INTEGER;          -- 0/1
ALTER TABLE samples ADD COLUMN is_view_dep INTEGER;      -- 0/1
ALTER TABLE samples ADD COLUMN is_filtered_out INTEGER;  -- 0/1
ALTER TABLE samples ADD COLUMN classification_correct INTEGER;  -- 0/1
```

#### CLI extension

```
--leaderboard-metrics PATH    optional; if provided, populate the new columns
                              from leaderboard_metrics.json
```

When `--leaderboard-metrics` is absent, the new columns stay NULL — old
runs ingested without it remain valid.

### Component 4 — `docs/benchmark/nr3d/v3_referit3d_track_<YYYYMMDD>.md` (new)

Sections (in order):

1. **Run identity** — branch, commit, reused `tmp/nr3d_eval_v1_full/`
   directory, aggregator commit, run id `v3_referit3d_track_<YYYYMMDD>`.
2. **Methodology** — explicit statement that we reuse v2 per-sample agent
   outputs and post-aggregate, why this is mathematically equivalent to
   re-running.
3. **Fold** — 130 scenes, n_full=8584, n_filtered=<measured>, easy/hard/v-dep/v-indep splits.
4. **Headline 5-column table** — Overall / Easy / Hard / V-Dep / V-Indep
   (all on canonical filtered fold).
5. **SOTA comparison** — same 5 columns alongside ReferIt3DNet, 3D-VisTA,
   MVT, MiKASA, UniVLG GT-track.
6. **Internal cross-check table** — `classification_acc_full` (8584
   denominator) vs `classification_acc_filtered` (≈7805 denominator), and
   the v2 IoU-based numbers as a separate row, with explanation that
   `Acc@0.50 ≈ classification_acc_full` is the GT-pool collapse.
7. **Caveats** —
   - Input modality asymmetry: zero-shot RGB+VLM vs trained 3D point-cloud
     methods (paradigm difference, not protocol violation).
   - 287 v2 sentinel failures still scored 0; their absolute count and
     impact on the filtered fold should be reported.
   - GT-pool inflation effects on the IoU buckets are no longer relevant
     since the headline metric is classification.
8. **SQLite reproduction query** — single SQL printing the 5 columns from
   `runs` table.
9. **Cross-version comparison** (NR3D-only) — v1 smoke20 / v2 full /
   v3 referit3d-track.

### Component 5 — index updates

- `docs/benchmark/nr3d/README.md` — add v3 row to the timeline table; update
  the "Current Interpretation" section to lead with the v3 leaderboard
  number; mark v2's caveats as "superseded by v3".
- `docs/benchmark/README.md` — replace the NR3D row's highlight with v3
  numbers (e.g. "v3 leaderboard-track: Overall = X %, Easy = Y %, ...").

## Acceptance criteria

The design is complete when ALL of these are true:

1. `python src/evaluation/scripts/nr3d_leaderboard_metrics.py --side-by-side tmp/nr3d_eval_v1_full/side_by_side.json --nr3d-data-root data/nr3d --phase8-data-root data/nr3d/scannet --output tmp/nr3d_eval_v1_full/leaderboard_metrics.json` runs to completion in under 2 minutes and writes a non-empty `leaderboard_metrics.json` matching the schema in Component 1.
2. `pytest src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py -v` passes all 7 tests.
3. `python scripts/ingest_nr3d_run.py --output-dir tmp/nr3d_eval_v1_full --run-id v3_referit3d_track_<YYYYMMDD> --leaderboard-metrics tmp/nr3d_eval_v1_full/leaderboard_metrics.json --branch feat/nr3d-vg-benchmark --commit $(git rev-parse --short HEAD) --backend pack_v1 --judge-model none --notes "..."` populates the new columns; querying `SELECT classification_acc_filtered, acc_easy, acc_hard, acc_view_dep, acc_view_indep FROM runs WHERE run_id='v3_referit3d_track_<YYYYMMDD>'` returns 5 non-null values.
4. `docs/benchmark/nr3d/v3_referit3d_track_<YYYYMMDD>.md` exists, contains all 9 sections, and the SOTA table includes at least 4 published baselines.
5. `docs/benchmark/nr3d/README.md` timeline shows v3 row; `docs/benchmark/README.md` NR3D row highlights the v3 number.
6. `git status -s` is clean (no leftover artifacts) after a final commit.

## Implementation order

| step | deliverable | dependency | est. wall time |
|---|---|---|---|
| 1 | failing tests in `test_nr3d_leaderboard_metrics.py` | — | 30 min |
| 2 | `nr3d_leaderboard_metrics.py` until tests pass | step 1 | 60 min |
| 3 | run aggregator on `tmp/nr3d_eval_v1_full/`; capture metrics | step 2 | 10 min |
| 4 | extend `ingest_nr3d_run.py` schema + CLI | step 3 (need real metrics file shape) | 30 min |
| 5 | write v3 doc using actual numbers from step 3 | step 3, step 4 | 60 min |
| 6 | README.md / leaderboard.md updates | step 5 | 15 min |

Total: ~3.5 hours, no API calls, no agent re-runs.

## Risks and mitigations

| risk | mitigation |
|---|---|
| `mentions_target_class_only=True` retains < 90 % of utts (i.e. our local CSV release differs from canonical) | The pool-equivalence log already measured 7805 / 8584 = **90.92 %**, within 0.7 pp of the ReferIt3D paper's 91.6 % figure (p.9). Step 1 of implementation re-confirms this number against the loader's actual filter behavior. If the new measurement diverges from 90.92 % by more than 1 pp, document and proceed (still canonical-aligned, just minor CSV-version variation). |
| sample_id missing from `side_by_side.json` after filter | Aggregator raises immediately (Component 1 invariant 1). The 287 v2 sentinel failures are NOT missing — they have status=`failed` and selected_object_id=null, which the aggregator scores as `is_correct=False`. |
| view-dep token matching produces unexpected counts | Tests cover the per-token-not-substring case; if real-world counts look surprising, dump a sample of utts and compare against published view-dep / view-indep N from 3D-VisTA Table 4 caption. |
| ingester schema migration breaks the existing single v2 row | New columns are NULLable; v2 row simply has NULLs in those columns. |
| v3 numbers are still uncomparable due to a missed protocol detail | Mitigated by two-source consensus from the audit (code + 2 papers); residual risk is low. The version doc explicitly enumerates everything we DID align and everything still asymmetric (input modality). |

## Open questions / explicit deferrals

- **Tokenization alignment**: not pursued in this design; documented as a
  caveat. If empirically v3 numbers underperform expectation in a way the
  pre-process_text difference could explain, revisit in v4.
- **Detector-pool track**: separate future work; the current design only
  delivers the GT-track leaderboard column.
- **SR3D**: out of scope.

## Appendix — file map

New files:

- `src/evaluation/scripts/nr3d_leaderboard_metrics.py`
- `src/evaluation/scripts/tests/test_nr3d_leaderboard_metrics.py`
- `docs/benchmark/nr3d/v3_referit3d_track_<YYYYMMDD>.md`

Modified files:

- `scripts/ingest_nr3d_run.py` — schema + CLI flag

Untouched (despite being adjacent):

- `src/benchmarks/nr3d_loader.py`
- `src/evaluation/scripts/run_nr3d_vg_side_by_side.py`
- `src/evaluation/scripts/prepare_pack_v1_inputs_nr3d.py`
- `src/agents/**` (Stage 2 agent code)
- Phase 8 packages under `data/nr3d/scannet/<scene>/conceptgraph/`

Reference artifact:

- `docs/benchmark/nr3d/protocol.md` — consolidated protocol, paper, and
  pool/fold-equivalence summary
