"""Task-pack fields on Stage2RuntimeState."""
from __future__ import annotations

from agents.core.task_types import Stage2EvidenceBundle
from agents.runtime.base import Stage2RuntimeState


def test_task_ctx_defaults_to_none() -> None:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    assert rs.task_ctx is None


def test_skills_loaded_defaults_empty_set() -> None:
    rs = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    assert rs.skills_loaded == set()


def test_skills_loaded_is_per_instance() -> None:
    a = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    b = Stage2RuntimeState(bundle=Stage2EvidenceBundle())
    a.skills_loaded.add("vg-grounding-playbook")
    assert "vg-grounding-playbook" not in b.skills_loaded
