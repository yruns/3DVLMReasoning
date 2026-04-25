"""EmbodiedScan VG pack — auto-registers on import."""
from agents.packs.vg_embodiedscan.registration import VG_PACK, register

register()

__all__ = ["VG_PACK"]
