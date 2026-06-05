#!/usr/bin/env python3
"""Sync DeepAgents playbooks into Codex Agent SDK skill files."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKILLS: tuple[tuple[str, str, str], ...] = (
    (
        "scene-exploration-playbook",
        "Use when solving NR3D/VG samples with text-first and catalog scene exploration tools plus first-person evidence checks.",
        "src/agents/skills/shared_skills/scene_exploration_playbook.md",
    ),
    (
        "vg-grounding-playbook",
        "Use when selecting one proposal_id for an NR3D visual grounding query with text-first and catalog VG tools.",
        "src/agents/packs/vg_embodiedscan/skills/vg_grounding_playbook.md",
    ),
    (
        "vg-spatial-disambiguation",
        "Use when an NR3D/VG query contains spatial relations, anchors, ordering, closest/farthest, left/right, or nested relations.",
        "src/agents/packs/vg_embodiedscan/skills/vg_spatial_disambiguation.md",
    ),
)


def normalize_for_codex_sdk(body: str) -> str:
    """Remove DeepAgents-only skill loading instructions from Codex skill text."""
    body = body.replace(
        "Prerequisite: `load_skill('scene-exploration-playbook')` first (it loads\n"
        "the catalog-first variant automatically in this run).",
        "Codex SDK prerequisite: `scene-exploration-playbook` is already mounted\n"
        "as a skill input and the scene-exploration tool gate is preloaded.",
    )
    return body.replace(
        "Prerequisite: `load_skill('scene-exploration-playbook')` first.",
        "Codex SDK prerequisite: `scene-exploration-playbook` is already "
        "mounted as a skill input and the scene-exploration tool gate is "
        "preloaded.",
    )


def render_skill(name: str, description: str, source_rel: str) -> str:
    source = PROJECT_ROOT / source_rel
    body = normalize_for_codex_sdk(source.read_text(encoding="utf-8").strip())
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        f"# {name}\n\n"
        "This Codex Agent SDK skill is synchronized from the DeepAgents "
        "playbook used by the Stage-2 VG runtime. Regenerate with "
        "`scripts/sync_codex_playbook_skills.py` after editing the source "
        "playbook.\n\n"
        "Codex SDK note: this skill is mounted as a `SkillInput`. The "
        "`nr3d_tools` MCP server and the `nr3d_tools_cli.py` fallback expose "
        "the same tool names used below. If MCP tools are not visible in the "
        "SDK turn, call the CLI command printed in the task prompt with "
        "`call <tool_name> '<json args>'`.\n\n"
        f"Source playbook: `{source_rel}`\n\n"
        f"<!-- BEGIN_SYNCED_PLAYBOOK: {source_rel} -->\n"
        f"{body}\n"
        "<!-- END_SYNCED_PLAYBOOK -->\n"
    )


def main() -> None:
    for name, description, source_rel in SKILLS:
        target = PROJECT_ROOT / ".agents" / "skills" / name / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            render_skill(name, description, source_rel),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
