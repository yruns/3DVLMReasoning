"""Smoke test: evidence_scouting skill should no longer be registered or shipped."""

from pathlib import Path

from agents.core.agent_config import Stage2TaskType
from agents.skills.registry import skills_for

# Trigger pack registration.
import agents.packs.vg_embodiedscan.registration  # noqa: F401
import agents.packs.qa_default.registration  # noqa: F401


def test_no_evidence_scouting_skill_files():
    vg_md = (
        Path(__file__).resolve().parents[3]
        / "packs"
        / "vg_embodiedscan"
        / "skills"
        / "evidence_scouting.md"
    )
    qa_md = (
        Path(__file__).resolve().parents[3]
        / "packs"
        / "qa_default"
        / "skills"
        / "evidence_scouting.md"
    )
    assert not vg_md.exists()
    assert not qa_md.exists()


def test_no_evidence_scouting_registered():
    for task_type in (Stage2TaskType.VISUAL_GROUNDING, Stage2TaskType.QA):
        names = {s.name for s in skills_for(task_type)}
        assert "evidence-scouting" not in names
