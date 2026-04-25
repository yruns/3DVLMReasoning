"""EmbodiedScan VG tools. All bodies FAIL-LOUD on missing primary skill."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

PRIMARY_SKILL = "vg-grounding-playbook"


def _gate(runtime: Any) -> str | None:
    """Return ERROR string if skill not loaded, else None."""
    if PRIMARY_SKILL not in runtime.skills_loaded:
        return f"ERROR: load_skill({PRIMARY_SKILL!r}) before calling this tool."
    return None


def build_vg_tools(runtime: Any) -> list[BaseTool]:
    ctx = runtime.task_ctx  # VgEmbodiedScanCtx

    @tool
    def list_keyframes_with_proposals() -> str:
        """VG tool. Detailed usage in skill 'vg-grounding-playbook'."""
        gate = _gate(runtime)
        if gate is not None:
            runtime.record("list_keyframes_with_proposals", {}, gate)
            return gate
        items = []
        for kf in runtime.bundle.keyframes:
            fid = kf.frame_id
            visible = ctx.frame_index.get(fid, []) if fid is not None else []
            items.append(
                {
                    "keyframe_idx": kf.keyframe_idx,
                    "frame_id": fid,
                    "visible_proposal_ids": visible,
                    "n_proposals": len(visible),
                    "annotated_image": str(ctx.annotated_image_dir / f"frame_{fid}.png"),
                }
            )
        text = json.dumps(items, ensure_ascii=False)
        runtime.record("list_keyframes_with_proposals", {}, text)
        return text

    return [list_keyframes_with_proposals]


__all__ = ["build_vg_tools", "PRIMARY_SKILL"]
