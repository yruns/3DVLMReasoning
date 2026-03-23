# Generate Migration Test Data Script

**Location**: `scripts/generate_migration_test_data.py`
**Purpose**: Generate reproducible ground truth data for migration equivalence testing
**Status**: ✅ Ready for execution (TASK-401)

## Quick Start

```bash
# Generate ground truth in default location
python scripts/generate_migration_test_data.py

# Generate in custom location
python scripts/generate_migration_test_data.py --output-dir /path/to/output
```

## Output Files

The script generates four JSON files in `tests/migration/ground_truth/`:

1. **parsing.json** (100+ cases)
   - Query parsing test cases
   - Expected parse modes, hypothesis kinds, categories
   
2. **keyframes.json** (50+ cases)
   - Keyframe selection test cases
   - Expected frame indices with k-tolerance
   
3. **hypotheses.json** (30+ cases)
   - Hypothesis execution test cases
   - Full HypothesisOutputV1 structures
   
4. **manifest.json**
   - Summary with counts and metadata

## Test Coverage

### Query Parsing (100+ cases)
- Simple object queries (20)
- Spatial relations: on, near, in, next_to, under, above, behind, in_front (30)
- Counting queries: "all X", "number of X" (20)
- Multi-object: "X and Y" (15)
- Complex nested: "X on Y near Z" (15)

### Keyframe Selection (50+ cases)
- Living room scenes (3+)
- Bedroom scenes (3+)
- Kitchen scenes (3+)
- Bathroom scenes (3+)
- Office scenes (3+)
- Spatial queries (10+)
- Simple queries (10+)
- Multi-object queries (10+)

### Hypothesis Execution (30+ cases)
- Direct success (5+)
- Direct failure (5+)
- Fallback cases (5+)
- Multi-mode (5+)
- Spatial validation (5+)
- Counting (5+)

## Script Structure

```python
# Data Models
ParsingTestCase      # Query parsing test case
KeyframeTestCase     # Keyframe selection test case
HypothesisTestCase   # Hypothesis execution test case

# Generation Functions
generate_parsing_test_cases()      # 100+ parsing cases
generate_keyframe_test_cases()     # 50+ keyframe cases
generate_hypothesis_test_cases()   # 30+ hypothesis cases
generate_all_ground_truth()        # Main orchestration
```

## Dependencies

Requires installed packages:
- `pydantic` - For data validation
- `query_scene` module - For query structures

Install with:
```bash
uv pip install -e ".[dev]"
```

## Next Steps

1. **TASK-401**: Run script to generate actual ground truth
2. **TASK-410**: Use data for keyframe equivalence tests
3. **TASK-411**: Use data for parsing equivalence tests

## Related Files

- `TASK-400-COMPLETION.md` - Task completion report
- `MIGRATION_PLAN.md` - Original specification
- `run_migration_scorecard.py` - Validation script

---

*Generated for TASK-400 - 2026-03-22*
