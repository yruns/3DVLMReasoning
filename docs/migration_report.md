# Migration Report: concept-graphs to 3DVLMReasoning

**Date:** 2026-03-22
**Migration Version:** 1.0.0
**Overall Grade:** B+ (82.7%)
**Status:** FUNCTIONALLY COMPLETE - TEST INFRASTRUCTURE NEEDS WORK

> **Note:** This is a copy of the canonical report at `/MIGRATION_REPORT.md` in the repository root.
> See that file for the authoritative version.

---

## Executive Summary

The migration from `concept-graphs/conceptgraph` to `3DVLMReasoning` is **functionally complete**. Core modules are operational, the adapter-based architecture is implemented, and equivalence testing infrastructure exists. However, test infrastructure has known issues that prevent reaching the 95% pass rate target.

### Current State (Verified 2026-03-22)

| Area | Status | Details |
|------|--------|---------|
| Core Modules | ✓ 5/5 Operational | query_scene, dataset, agents, evaluation, config |
| Migration Tests | ✓ 76/76 Passed | After TASK-420 schema compatibility fixes |
| Integration Tests | ~ 12 passed, 21 skipped | Environment/data-dependent |
| Unit Test Collection | ✗ Errors | Import path issues in some test files |
| Lint/Format | ✗ Not Clean | `ruff` and `black` checks have violations |

### Key Achievements

- **All core modules importable** and functional
- **Migration equivalence tests: 76/76 passing** (100% after schema fixes)
- **Dataset adapter architecture** implemented for Replica and ScanNet
- **Evaluation framework** fully migrated with scripts and tests
- **Ground truth validation** infrastructure in place

---

## Quick Reference

### Verified Test Commands

```bash
# Config + Evaluation tests (469 passed)
.venv/bin/python -m pytest tests/config tests/evaluation -q

# Integration tests (12 passed, 21 skipped)
.venv/bin/python -m pytest tests/integration -q

# Migration equivalence tests (76 passed)
.venv/bin/python -m pytest tests/migration -q
```

### Module Import Health

All modules import successfully:
- `query_scene` - Query-driven keyframe retrieval
- `dataset` - Dataset adapters (Replica, ScanNet)
- `agents` - VLM agentic reasoning
- `evaluation` - Benchmark evaluation framework
- `config` - Configuration management

### Scorecard Summary

From `scorecard_final.json`:
- **Parsing accuracy:** 100%
- **Keyframe recall:** 100% at all k
- **Module health:** 5/5 operational
- **Performance:** Within tolerance

---

## Detailed Information

For the full migration report including:
- Phase-by-phase breakdown
- Schema evolution notes
- File change summary
- Remaining work items

See the canonical report at **`/MIGRATION_REPORT.md`** in the repository root.

---

*Last verified: 2026-03-22*
