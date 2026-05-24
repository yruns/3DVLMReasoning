# NR3D v11 select_by_text strat600 coverage audit

This audit measures the current `select_by_text` tool's first-hop visual
coverage on the canonical NR3D strat600 fold. It is not an agent accuracy or
leaderboard run. The question is:

> If `select_by_text` is called once with the raw NR3D query, do the returned
> keyframes contain the GT target proposal id?

The production tool caps `k` at 3, so the headline metric is hit@3: at least
one of the returned frames has `target_id` in `visible_proposal_ids`.

## Pre-run Checklist

| Item | Value |
|---|---|
| Pending tracked changes committed before launch | yes |
| Branch | `best/v10-multi-anchor-tadg-72-a3ff7f1` |
| Head commit at launch | `45a2dae` |
| Run-time code commit | `45a2dae` - no worktree drift |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Fold size | 600 canonical stratified samples |
| Pack name | `pack_nr3d_v9_catalog_first` |
| Output dir | `tmp/nr3d_select_by_text_audit_current_strat600_20260524/` |
| Merged summary | `tmp/nr3d_select_by_text_audit_current_strat600_20260524/merged_summary.json` |
| Started | `2026-05-24T02:32:05+08:00` |
| Last shard completed | `2026-05-24T02:47:48+08:00` |
| Workers | 16 shard processes |
| Tool-equivalent call | `select_by_text(query=<raw NR3D query>, k=3, hidden_categories=[], use_visual_context=False)` |
| Judge model | none - deterministic coverage against visibility indices |

## Exact Commands

The 600 samples were split into 16 scene-sorted shard files under:

```bash
tmp/nr3d_select_by_text_audit_current_strat600_20260524/shards/
```

Each shard ran:

```bash
PYTHONPATH=src PYTHONUNBUFFERED=1 MODELHUB_AK_WEIGHTS=1,1,1 \
.venv/bin/python scripts/audit_select_by_text_nr3d.py \
  --sample-ids tmp/nr3d_select_by_text_audit_current_strat600_20260524/shards/shard_XX.json \
  --data-root data/nr3d/scannet \
  --pack-name pack_nr3d_v9_catalog_first \
  --k 1,2,3 \
  --output tmp/nr3d_select_by_text_audit_current_strat600_20260524/outputs/shard_XX.json \
  --max-selector-cache-size 2 \
  --resume
```

The tmux runner was:

```bash
tmp/nr3d_select_by_text_audit_current_strat600_20260524/run_parallel.sh
```

All 16 shard logs ended with `exit=0`.

## Headline Coverage

| Slice | n | hit@1 | hit@2 | hit@3 | empty frames | hit@3 when non-empty | random hit@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 600 | 47.33 | 58.67 | **62.17** | 25.17 | 83.07 | 51.00 |
| Easy | 290 | 55.86 | 64.83 | **67.93** | 22.76 | 87.95 | 51.64 |
| Hard | 310 | 39.35 | 52.90 | **56.77** | 27.42 | 78.22 | 50.41 |
| View-Dep | 211 | 34.60 | 46.92 | **50.71** | 36.97 | 80.45 | 50.52 |
| View-Indep | 389 | 54.24 | 65.04 | **68.38** | 18.77 | 84.18 | 51.26 |

`random hit@3` is the per-sample expectation from uniformly sampling three
valid frames from that scene, using the sample's GT target visibility coverage.

## Status Breakdown

| Field | Count |
|---|---:|
| Samples | 600 |
| Missing samples after merge | 0 |
| Duplicate sample rows | 0 |
| Script errors | 0 |
| Empty `pred_top_k` / no returned frames | 151 |
| Non-empty responses | 449 |

Parser / executor statuses:

| Status | Count |
|---|---:|
| `direct_grounded` | 369 |
| `no_evidence` | 151 |
| `proxy_grounded` | 50 |
| `context_only` | 30 |

The direct pack-visibility check and the Phase-8 depth-aware visibility-index
check matched exactly for all 600 samples (`pack_vs_phase8_diff_count = 0`).

## Case Spotchecks

- Success: `scannet/scene0019_00::19::19821`, query `door closer to the
  vending machine`, returned frames `[38, 45, 32]`, target `#19 door` visible in
  frame 38. Marked image:
  `data/nr3d/scannet/scene0019_00/pack_nr3d_v9_catalog_first/bev/filtered_marks/frame_38_ids_9_19.png`.
- Non-empty miss: `scannet/scene0015_00::47::10339`, query `The longer
  whiteboard on the wall furthest away from the doorway.`, returned frames
  `[79, 75, 11]`; none contains target `#47 whiteboard`. A GT-visible frame is
  165. Marked images inspected:
  `frame_79_ids_2_6_32_33_34.png` and `frame_165_ids_47.png`.
- Empty response: `scannet/scene0011_00::28::4659`, query `The cabinet next to
  the doors. furtherest from the tv`, returned `[]` with `parse_status =
  no_evidence`; target `#28 cabinet` is visible in frame 33.

For these three cases, the ConceptGraph point-cloud pickle
`conceptgraph/pcd_saves/full_pcd_gt_axisaligned_post.pkl.gz` was loaded and the
target proposal object was present.

## Interpretation

Current `select_by_text` is useful when it returns frames: non-empty hit@3 is
83.07 %. The overall production-tool coverage is lower, 62.17 %, because 151 /
600 raw queries return no frames. The view-dependent slice is the weak point:
hit@3 is 50.71 %, roughly random, and empty responses rise to 36.97 %.

This supports the current policy of treating `select_by_text` as a good first
move for rare / direct targets, but not as a reliable sole source for spatial
or view-dependent NR3D queries.

## Why 151 Queries Return No Evidence

`status = no_evidence` is produced in
`KeyframeSelector.execute_hypotheses()`: hypotheses are executed by rank, and if
every hypothesis either returns an empty `ExecutionResult` or is skipped because
of an `UNKNOW` anchor, the selector returns no frames.

For the 151 empty responses:

| Root cause bucket | n | Share |
|---|---:|---:|
| Spatial relation filter reduced candidates to 0 | 150 | 99.34 % |
| Select constraint reduced candidates to 0 | 1 | 0.66 % |

Category lookup itself was not the dominant problem:

| Check | n |
|---|---:|
| Root categories matched at least one scene object | 151 / 151 |
| Root category match set contained the GT target id | 116 / 151 |
| Root category match set did not contain the GT target id | 35 / 151 |

The 35 root-category misses are mostly catalog / alias coverage rather than
pure target-head mistakes: 31 / 35 have a lexical match between the benchmark
category and parsed root categories, but the GT object's local labels were not
returned by `_find_by_categories`. The 4 clear target-head inversions are:

| Sample | GT category | Parsed root | Query pattern |
|---|---|---|---|
| `scene0353_00::5::27624` | chair | `t-shirt` | "orange shirt on the black chair" |
| `scene0535_00::1::33634` | window | `cabinet` | "cabinet full of junk in front of the window" |
| `scene0565_00::23::30788` | cart | `wall` | "green wall right above this cart" |
| `scene0598_00::9::25057` | chair | `backpack` | "backpack on the desk in front of this chair" |

The spatial-zero relations observed in the execution logs were:

| Relation whose filter/check returned 0 | Count |
|---|---:|
| `on` | 57 |
| `next_to` | 18 |
| `right_of` | 14 |
| `in_front_of` | 12 |
| `left_of` | 11 |
| `between` | 10 |
| `inside` | 10 |
| `behind` | 9 |
| `below` | 8 |
| `above` | 8 |
| `near` | 6 |
| `in` | 2 |
| `beside` | 1 |

Important mechanics:

- 144 / 151 no-evidence samples had only one parsed hypothesis. There is usually
  no category-only fallback hypothesis after the hard spatial constraint fails.
- 38 / 151 were eliminated by `QuickFilters` before full spatial checking,
  mostly `on`, `next_to`, `below`, and `near`.
- The remaining spatial-zero cases reached full spatial checking. Directional
  relations such as `left_of`, `right_of`, `in_front_of`, and `behind` use the
  scene's global X/Y axes, not the viewer frame described in many NR3D queries.
- `on` is mapped to `on_top_of`, which requires the target centroid to be above
  the anchor and horizontally close to the anchor centroid. This is brittle for
  expressions like "on the floor", "on the wall", "on the desk", or "on the
  bed", especially with large support surfaces whose centroid is far from the
  target.

In short: most empty `select_by_text` responses are not parser failures in the
sense of "no target category". They are hard-constraint failures: the parser
often finds the target class, but the executor treats spatial/support language
as mandatory geometry and returns empty when the relation checker is too strict
or viewpoint-insensitive.
