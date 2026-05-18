"""Dump detailed side-by-side comparison of chassis-vs-v15 regression cases.

Reads the SQLite DB built by ingest_openeqa_run.py PLUS reaches into the raw
artifact dirs to pull full tool_input / response_text + summary / uncertainties
for each (qid) where chassis=1 and v15 in (4,5).

Output: docs/benchmark/openeqa/v15_chassis_repro_20260427_regressions_detail.md
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB = Path("docs/benchmark/openeqa/runs.sqlite")
OUT_MD = Path("docs/benchmark/openeqa/v15_chassis_repro_20260427_regressions_detail.md")
NEW_RUN = "explore3dbbox_s1_l1"
OLD_RUN = "v15_s1_l1"

conn = sqlite3.connect(str(DB))
cur = conn.cursor()

cur.execute(
    """
    SELECT n.question_id, n.category, n.scene, n.question, n.gt_answer,
           o.stage2_score, n.stage2_score,
           o.artifact_dir, n.artifact_dir
    FROM samples n JOIN samples o ON n.question_id=o.question_id
    WHERE n.run_id=? AND o.run_id=?
      AND n.stage2_score=1 AND o.stage2_score IN (4,5)
    ORDER BY n.category, n.question_id
    """,
    (NEW_RUN, OLD_RUN),
)
rows = cur.fetchall()
print(f"found {len(rows)} severe regression cases")


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _short_tools(trace: list, max_resp: int = 120) -> str:
    """Render a compact tool-trace summary."""
    if not trace:
        return "(no tools)"
    lines = []
    for i, t in enumerate(trace, 1):
        name = t.get("tool_name", "?")
        inp = t.get("tool_input") or {}
        # Pull out the most informative input field
        if name == "request_more_views":
            key = inp.get("request_text") or inp.get("object_terms") or ""
            if isinstance(key, list):
                key = ", ".join(key)
            sig = f"mode={inp.get('mode')} | {str(key)[:80]}"
        elif name == "request_crops":
            key = inp.get("text") or inp.get("objects") or ""
            if isinstance(key, list):
                key = ", ".join(key)
            sig = str(key)[:80]
        elif name == "retrieve_object_context":
            sig = str(inp.get("object_terms", inp))[:80]
        elif name == "switch_or_expand_hypothesis":
            sig = str(inp.get("rationale", inp))[:80]
        elif name == "submit_final":
            sig = str(inp.get("rationale", inp.get("payload", "")))[:80]
        else:
            sig = str(inp)[:80]
        resp = (t.get("response_text") or "")[:max_resp].replace("\n", " ")
        lines.append(f"  {i}. **{name}** — `{sig}`\n     → {resp}")
    return "\n".join(lines)


md_lines = [
    "# v15 chassis-repro — Detailed Severe Regression Cases",
    "",
    "**Companion to:** `v15_chassis_repro_20260427_regressions.md` (which gave the high-level patterns).",
    "**Source:** SQLite `docs/benchmark/openeqa/runs.sqlite` + raw stage2.json artifacts.",
    "**Scope:** the 38 cases where chassis stage2_score=1 AND v15 stage2_score ∈ {4,5} — the questions chassis *broke* that v15 had right.",
    "",
    "Each case shows: question + GT, the v15 vs chassis trace (tools used, confidence, final answer, uncertainties). Pattern bucket assigned by post-hoc inspection.",
    "",
    "## Index",
    "",
]

# Group by category for readability
by_cat: dict[str, list] = {}
for row in rows:
    qid, cat, scene, q, gt, v15s, ns, v15dir, ndir = row
    by_cat.setdefault(cat, []).append(row)

for cat in sorted(by_cat):
    md_lines.append(f"- **{cat}**: {len(by_cat[cat])} cases")
md_lines.append(f"- **TOTAL**: {len(rows)} cases")
md_lines.append("")

for cat in sorted(by_cat):
    md_lines.append(f"## Category: {cat}")
    md_lines.append("")
    for row in by_cat[cat]:
        qid, _, scene, q, gt, v15s, ns, v15dir, ndir = row
        try:
            v15_s2 = _load(Path(v15dir) / "stage2.json")
            new_s2 = _load(Path(ndir) / "stage2.json")
        except Exception as e:
            md_lines.append(f"### `{qid}` (load failed: {e})\n")
            continue

        v15_ans = (
            (v15_s2.get("payload") or {}).get("answer") or v15_s2.get("summary") or ""
        )
        new_ans = (
            (new_s2.get("payload") or {}).get("answer") or new_s2.get("summary") or ""
        )
        v15_unc = v15_s2.get("uncertainties") or []
        new_unc = new_s2.get("uncertainties") or []

        md_lines += [
            f"### `{qid}`  (v15={v15s} → chassis={ns})",
            "",
            f"- **scene**: {scene}",
            f"- **question**: {q}",
            f"- **GT answer**: {gt}",
            "",
            f"**v15** (visuals {v15_s2.get('initial_stage1_frames')} -> {v15_s2.get('final_tool_visuals')}, "
            f"conf {v15_s2.get('confidence', 0):.2f}):",
            "",
            _short_tools(v15_s2.get("tool_trace") or []),
            "",
            f"  **answer**: {v15_ans[:300]}",
        ]
        if v15_unc:
            md_lines.append(
                f"  **uncertainties**: {' | '.join(str(u)[:120] for u in v15_unc[:3])}"
            )
        md_lines += [
            "",
            f"**chassis** (visuals {new_s2.get('initial_stage1_frames')} -> {new_s2.get('final_tool_visuals')}, "
            f"conf {new_s2.get('confidence', 0):.2f}):",
            "",
            _short_tools(new_s2.get("tool_trace") or []),
            "",
            f"  **answer**: {new_ans[:300]}",
        ]
        if new_unc:
            md_lines.append(
                f"  **uncertainties**: {' | '.join(str(u)[:120] for u in new_unc[:3])}"
            )
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text("\n".join(md_lines))
print(f"wrote {OUT_MD}")

conn.close()
