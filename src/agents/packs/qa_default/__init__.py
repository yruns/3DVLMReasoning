"""Default QA pack - auto-registers on import."""
from agents.packs.qa_default.registration import QA_PACK, register

register()

__all__ = ["QA_PACK"]
