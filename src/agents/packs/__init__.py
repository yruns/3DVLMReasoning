"""Stage-2 task packs. Importing this module triggers deterministic
pack imports so they self-register into agents.skills.PACKS."""
from agents.packs import qa_default as _qa_default  # noqa: F401
from agents.packs import vg_embodiedscan as _vg_embodiedscan  # noqa: F401


def ensure_default_packs_registered() -> None:
    """Re-register default packs after tests or tools clear PACKS."""
    import importlib

    from agents.core.agent_config import Stage2TaskType
    from agents.skills import PACKS

    if Stage2TaskType.QA not in PACKS:
        importlib.reload(_qa_default)
    if Stage2TaskType.VISUAL_GROUNDING not in PACKS:
        importlib.reload(_vg_embodiedscan)


__all__ = ["ensure_default_packs_registered"]
