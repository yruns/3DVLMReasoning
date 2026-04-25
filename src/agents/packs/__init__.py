"""Stage-2 task packs. Importing this module triggers deterministic
pack imports so they self-register into agents.skills.PACKS."""
from agents.packs import vg_embodiedscan as _vg_embodiedscan  # noqa: F401

__all__ = []
