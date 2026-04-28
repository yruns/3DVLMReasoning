# `docs/benchmark/` — Per-Benchmark Process Archive

One sub-directory per benchmark. Each holds the *immutable* trail of what was run, on what code commit, against which fold, and what the LLM judge returned. Treat any file under this tree as evidence — never silently overwrite past results.

## Directory Layout (Pattern)

```
docs/benchmark/
├── README.md                    ← this index
└── <benchmark>/
    ├── README.md                ← per-benchmark version timeline + leaderboard
    ├── leaderboard.md           ← extended public leaderboard
    ├── <vN>_<short>_<date>.md   ← one per evaluated version
    ├── *.html                   ← optional dashboards / case studies
    └── …
```

When adding a new benchmark, mirror this layout. The version-doc filename
convention is `<vN>_<one-or-two-word-tag>_<YYYYMMDD>.md`, where `vN` is the
internal version of *our* pipeline at the time of the run (not the
benchmark's version). Always include:

- The git branch + tip commit at run time.
- The exact CLI invocation (or launcher script path).
- Raw artifact directory under `tmp/…/` so numbers can be re-derived.
- The judge model name (LLM-as-judge benchmarks).
- The fold size and selection mechanism (e.g. `--force-selection /path/to/frozen.json`).

## Active Benchmarks

| Benchmark | Status | Path | Highlight |
|-----------|--------|------|-----------|
| **OpenEQA ScanNet EM-EQA** | live | [`openeqa/`](openeqa/) | v15 stage2 MNAS **74.33** (`s1_l1`, frozen 1050Q, Gemini 2.5 Pro) |
| **EmbodiedScan VG** | smoke | [`embodiedscan/`](embodiedscan/) | v1 GT-pool pack_v1 smoke: 68Q, Acc@0.25/0.50 **91.18** |

## Pending Migrations

These bodies of work currently live elsewhere and should adopt this layout
when their next results land:

- **SQA3D / ScanRefer / Nav Plan** — not yet evaluated; create their own
  sub-folders when the first evaluation runs.

## Why This Layout

Process documentation is at risk of disappearing into commit messages,
ad-hoc handoff files, and `tmp/` directories that get garbage-collected.
Forcing every benchmark result into a versioned, dated, tracked file
under `docs/benchmark/<name>/` keeps a permanent paper trail across
branch deletions, tmp cleanups, and team handoffs. If a result is
worth quoting in a paper, it lives here; if it's a dead-end ablation,
it still lives here as a trace of what was tried.

Cross-references from the curriculum-style framework docs
(`docs/00_…11_*.md`) point into this tree as immutable evidence.
