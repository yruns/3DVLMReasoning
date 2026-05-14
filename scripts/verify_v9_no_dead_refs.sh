#!/usr/bin/env bash
# F.5 verification: assert no v9-deleted tool names linger in the Stage-2
# agent runtime / packs / skills. This is the v9 production surface.
#
# Eval scripts under src/evaluation/{scripts,ablations}/ are intentionally
# excluded because they still drive pre-v9 ablations against historical
# benchmarks; they will be reaped in a separate cleanup pass.
#
# Docs are intentionally excluded entirely (they are immutable historical
# records per CLAUDE.md "Benchmark Process Documentation").

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

FAIL=0
DEAD_NAMES=(
    'request_more_views'
    'switch_or_expand_hypothesis'
    'find_proposals_by_category'
    'list_keyframes_with_proposals'
    'inspect_stage1_metadata'
    'enable_temporal_fan'
    'enable_stage1_callback'
)

# Allowlist: files where dead names are *expected* to appear (regression tests
# that assert the names are gone, guards that accept the old name as a trace
# alias for back-compat).
ALLOWLIST=(
    'src/agents/skills/tests/test_no_evidence_scouting.py'
    'src/agents/skills/tests/test_scene_exploration_playbook_loadable.py'
    'src/agents/tests/test_migration_no_callback_tools.py'
    'src/agents/tests/test_stage2_deep_agent.py'
    'src/agents/packs/vg_embodiedscan/skills/tests/test_playbook_v9_consistency.py'
    'src/agents/packs/qa_default/skills/tests/test_qa_playbook_v9_consistency.py'
    'src/agents/runtime/tests/test_build_system_prompt_v9.py'
    'src/agents/core/tests/test_config_no_dead_flags.py'
    'src/agents/tests/test_openeqa_official_question_pilot.py'
    'src/agents/tests/test_tadg.py'
    'src/agents/tests/test_no_match_guard.py'
    'src/agents/tests/test_evidence_frame_guard.py'
    'src/agents/tests/test_verify_no_dead_refs.py'
    'src/agents/tests/integration/test_v9_vg_end_to_end_mock.py'
    'src/agents/tests/integration/test_v9_qa_end_to_end_mock.py'
    'src/agents/skills/no_match_guard.py'
    'src/agents/skills/evidence_frame_guard.py'
    'src/agents/skills/tadg.py'
)

ALLOW_RE=$(printf '%s|' "${ALLOWLIST[@]}")
ALLOW_RE=${ALLOW_RE%|}

# Scope: only the v9 production surface (agents runtime + packs + skills +
# query_scene scene_bev_builder + catalog adapters + agent tools).
SCAN_ROOTS=(
    'src/agents/'
    'src/query_scene/scene_bev_builder.py'
    'src/query_scene/keyframe_selector.py'
)

for name in "${DEAD_NAMES[@]}"; do
    matches=$(rg --no-messages -l "$name" "${SCAN_ROOTS[@]}" || true)
    if [[ -z "$matches" ]]; then
        continue
    fi
    bad=$(echo "$matches" | grep -E -v "^(${ALLOW_RE})$" || true)
    if [[ -n "$bad" ]]; then
        echo "FAIL: dead name '$name' found in v9 production surface:"
        echo "$bad" | sed 's/^/  /'
        FAIL=1
    fi
done

if [[ $FAIL -eq 0 ]]; then
    echo "PASS: no v9-deleted tool names found in production surface"
fi
exit $FAIL
