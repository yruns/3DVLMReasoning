#!/usr/bin/env python3
"""Generate a polished HTML evaluation report from OpenEQA batch results.

Usage:
    python scripts/generate_eval_report.py --eval-dir tmp/openeqa_eval_5x4_v2
    python scripts/generate_eval_report.py --eval-dir tmp/openeqa_eval_5x4_v2 --open
"""

from __future__ import annotations

import argparse
import base64
import html as html_mod
import json
import sys
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Thumbnail helper
# ---------------------------------------------------------------------------

def make_thumb_b64(path: str | Path, size: int = 280) -> str | None:
    """Generate a base64 data-url thumbnail."""
    try:
        from PIL import Image
    except ImportError:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = Image.open(p).convert("RGB")
        img.thumbnail((size, size))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=78)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


def esc(text: str) -> str:
    return html_mod.escape(str(text))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_data(eval_dir: Path) -> dict[str, Any]:
    """Load all evaluation artifacts from an eval run directory."""
    summary_path = eval_dir / "official_batch_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary found at {summary_path}")

    summary = json.loads(summary_path.read_text())

    s2_metrics_path = eval_dir / "official_predictions_stage2-metrics.json"
    e2e_metrics_path = eval_dir / "official_predictions_e2e-metrics.json"
    s2_metrics = json.loads(s2_metrics_path.read_text()) if s2_metrics_path.exists() else {}
    e2e_metrics = json.loads(e2e_metrics_path.read_text()) if e2e_metrics_path.exists() else {}

    # Load per-question detail files
    cases: list[dict[str, Any]] = []
    for row in summary.get("results", []):
        qid = row["question_id"]
        clip_id = row["clip_id"]
        run_dir = eval_dir / "runs" / clip_id / qid

        case = {**row}
        case["s2_score"] = s2_metrics.get(qid)
        case["e2e_score"] = e2e_metrics.get(qid)

        for fname in ("sample.json", "stage1.json", "stage2.json", "e2e.json"):
            fpath = run_dir / fname
            if fpath.exists():
                case[fname.replace(".json", "")] = json.loads(fpath.read_text())

        cases.append(case)

    return {
        "summary": summary,
        "s2_metrics": s2_metrics,
        "e2e_metrics": e2e_metrics,
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def score_color(score: int | None) -> str:
    if score is None:
        return "#94a3b8"
    return {1: "#ef4444", 2: "#f97316", 3: "#eab308", 4: "#22c55e", 5: "#10b981"}.get(
        score, "#94a3b8"
    )


def score_bg(score: int | None) -> str:
    if score is None:
        return "#f1f5f9"
    return {1: "#fef2f2", 2: "#fff7ed", 3: "#fefce8", 4: "#f0fdf4", 5: "#ecfdf5"}.get(
        score, "#f1f5f9"
    )


def status_badge(status: str) -> str:
    colors = {
        "completed": ("#059669", "#ecfdf5"),
        "direct_grounded": ("#2563eb", "#eff6ff"),
        "proxy_grounded": ("#7c3aed", "#f5f3ff"),
        "context_only": ("#d97706", "#fffbeb"),
        "insufficient_evidence": ("#dc2626", "#fef2f2"),
        "failed": ("#dc2626", "#fef2f2"),
    }
    fg, bg = colors.get(status, ("#64748b", "#f1f5f9"))
    return f'<span style="display:inline-block;padding:2px 10px;border-radius:9999px;font-size:11px;font-weight:600;color:{fg};background:{bg};letter-spacing:.02em">{esc(status)}</span>'


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(data: dict[str, Any]) -> str:
    cases = data["cases"]
    s2_scores = list(data["s2_metrics"].values())
    e2e_scores = list(data["e2e_metrics"].values())
    n = len(cases)
    scenes = sorted(set(c["clip_id"] for c in cases))

    s2_mean = sum(s2_scores) / len(s2_scores) if s2_scores else 0
    e2e_mean = sum(e2e_scores) / len(e2e_scores) if e2e_scores else 0
    s2_pass = sum(1 for s in s2_scores if s >= 3)
    e2e_pass = sum(1 for s in e2e_scores if s >= 3)

    case_details = "\n".join(_render_case(c, i) for i, c in enumerate(cases))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenEQA Evaluation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700;900&display=swap" rel="stylesheet">
{_css()}
</head>
<body>
<div class="page">

  <!-- Hero -->
  <header class="hero">
    <div class="hero-inner">
      <h1 class="hero-title">OpenEQA Evaluation</h1>
      <p class="hero-sub">{n} questions across {len(scenes)} scenes &middot; Stage2 + E2E pipeline</p>
    </div>
  </header>

  <!-- Score overview cards -->
  <section class="score-overview">
    <div class="score-card">
      <div class="score-card-label">Stage 2 Mean</div>
      <div class="score-card-value" style="color:{score_color(round(s2_mean))}">{s2_mean:.2f}<span class="score-card-max">/5</span></div>
      <div class="score-card-sub">{s2_pass}/{n} pass (&ge;3)</div>
    </div>
    <div class="score-card">
      <div class="score-card-label">E2E Mean</div>
      <div class="score-card-value" style="color:{score_color(round(e2e_mean))}">{e2e_mean:.2f}<span class="score-card-max">/5</span></div>
      <div class="score-card-sub">{e2e_pass}/{n} pass (&ge;3)</div>
    </div>
    <div class="score-card">
      <div class="score-card-label">Questions</div>
      <div class="score-card-value">{n}</div>
      <div class="score-card-sub">{len(scenes)} unique scenes</div>
    </div>
    <div class="score-card">
      <div class="score-card-label">E2E Lift</div>
      <div class="score-card-value" style="color:{'#059669' if e2e_mean > s2_mean else '#64748b'}">+{e2e_mean - s2_mean:.2f}</div>
      <div class="score-card-sub">vs Stage 2 alone</div>
    </div>
  </section>

  <!-- Score heatmap table -->
  <section class="section">
    <h2 class="section-title">Score Matrix</h2>
    <div class="heatmap-scroll">
      <table class="heatmap">
        <thead>
          <tr>
            <th>Scene</th>
            <th>Question</th>
            <th>Category</th>
            <th>S1 Status</th>
            <th>S2</th>
            <th>E2E</th>
            <th>&Delta;</th>
          </tr>
        </thead>
        <tbody>
          {"".join(_render_heatmap_row(c) for c in cases)}
        </tbody>
      </table>
    </div>
  </section>

  <!-- Per-case details -->
  <section class="section">
    <h2 class="section-title">Case Details</h2>
    {case_details}
  </section>

</div>

{_modal_html()}
{_js()}
</body>
</html>"""


def _render_heatmap_row(c: dict) -> str:
    s2 = c.get("s2_score")
    e2e = c.get("e2e_score")
    delta = (e2e or 0) - (s2 or 0) if s2 is not None and e2e is not None else 0
    delta_color = "#059669" if delta > 0 else "#dc2626" if delta < 0 else "#94a3b8"
    delta_str = f"+{delta}" if delta > 0 else str(delta)
    guarded = ' <span style="font-size:10px;color:#7c3aed" title="Confidence guard applied">G</span>' if c.get("e2e_guarded") else ""

    clip_short = c["clip_id"].split("-")[0]
    q_short = esc(c["question"][:50]) + ("..." if len(c["question"]) > 50 else "")

    return f"""<tr class="heatmap-row" data-case-id="case-{c['question_id']}" onclick="scrollToCase('case-{c['question_id']}')">
  <td class="hm-scene">{clip_short}</td>
  <td class="hm-question">{q_short}</td>
  <td><span class="cat-tag">{esc(c.get('category', ''))}</span></td>
  <td>{status_badge(c.get('stage1_status', ''))}</td>
  <td style="background:{score_bg(s2)};color:{score_color(s2)};font-weight:700;text-align:center">{s2 or '?'}</td>
  <td style="background:{score_bg(e2e)};color:{score_color(e2e)};font-weight:700;text-align:center">{e2e or '?'}{guarded}</td>
  <td style="text-align:center;color:{delta_color};font-weight:600">{delta_str}</td>
</tr>"""


def _render_case(c: dict, idx: int) -> str:
    qid = c["question_id"]
    s2 = c.get("s2_score")
    e2e_sc = c.get("e2e_score")
    stage1 = c.get("stage1", {})
    stage2 = c.get("stage2", {})
    e2e = c.get("e2e", {})

    # Keyframe thumbnails
    kf_paths = stage1.get("keyframe_paths", [])
    kf_html = ""
    for kf_path in kf_paths:
        thumb = make_thumb_b64(kf_path)
        if thumb:
            fname = Path(kf_path).name
            kf_html += f'<div class="kf-thumb" onclick="openModal(this)"><img src="{thumb}" alt="{fname}"><div class="kf-label">{fname}</div></div>'
        else:
            kf_html += f'<div class="kf-thumb kf-missing"><span>{Path(kf_path).name}</span></div>'

    # Stage 2 details
    s2_evidence = ""
    for ev in stage2.get("evidence_items", []):
        frames = ", ".join(str(f) for f in ev.get("frame_indices", []))
        s2_evidence += f'<div class="evidence-item"><span class="ev-frames">frames [{frames}]</span> {esc(ev.get("claim", ""))}</div>'

    s2_uncertainties = "".join(
        f"<li>{esc(u)}</li>" for u in stage2.get("uncertainties", [])
    )

    # E2E tool trace
    tool_trace_html = ""
    for tc in e2e.get("tool_trace", []):
        tool_input_json = esc(json.dumps(tc.get("tool_input", {}), indent=2, ensure_ascii=False))
        resp_text = esc(tc.get("response_text", "")[:600])
        tool_trace_html += f"""<div class="tool-call">
  <div class="tool-header">
    <span class="tool-name">{esc(tc.get('tool_name', ''))}</span>
  </div>
  <details class="tool-details">
    <summary>Input</summary>
    <pre class="code-block">{tool_input_json}</pre>
  </details>
  <details class="tool-details">
    <summary>Response</summary>
    <pre class="code-block">{resp_text}</pre>
  </details>
</div>"""

    if not tool_trace_html:
        tool_trace_html = '<div class="no-tools">No tool calls</div>'

    # E2E evidence
    e2e_evidence = ""
    for ev in e2e.get("evidence_items", []):
        frames = ", ".join(str(f) for f in ev.get("frame_indices", []))
        e2e_evidence += f'<div class="evidence-item"><span class="ev-frames">frames [{frames}]</span> {esc(ev.get("claim", ""))}</div>'

    guarded_badge = '<span class="guarded-badge">GUARDED</span>' if c.get("e2e_guarded") else ""

    # Payload
    s2_payload = stage2.get("payload", {})
    e2e_payload = e2e.get("payload", {})
    s2_answer = s2_payload.get("answer", stage2.get("summary", "N/A")) if isinstance(s2_payload, dict) else str(s2_payload)
    e2e_answer = e2e_payload.get("answer", e2e.get("summary", "N/A")) if isinstance(e2e_payload, dict) else str(e2e_payload)

    return f"""
<div class="case-card" id="case-{qid}">
  <div class="case-header" onclick="this.parentElement.classList.toggle('expanded')">
    <div class="case-header-left">
      <div class="case-num">#{idx + 1}</div>
      <div>
        <div class="case-question">{esc(c['question'])}</div>
        <div class="case-meta">
          <span class="scene-tag">{esc(c['clip_id'].split('-')[0])}</span>
          <span class="cat-tag">{esc(c.get('category', ''))}</span>
          {status_badge(c.get('stage1_status', ''))}
          {guarded_badge}
        </div>
      </div>
    </div>
    <div class="case-header-right">
      <div class="case-scores">
        <div class="mini-score" style="background:{score_bg(s2)};color:{score_color(s2)}">S2: {s2 or '?'}</div>
        <div class="mini-score" style="background:{score_bg(e2e_sc)};color:{score_color(e2e_sc)}">E2E: {e2e_sc or '?'}</div>
      </div>
      <div class="expand-icon">&#9662;</div>
    </div>
  </div>

  <div class="case-body">
    <!-- Ground truth vs predictions -->
    <div class="answer-grid">
      <div class="answer-box gt-box">
        <div class="answer-label">Ground Truth</div>
        <div class="answer-text">{esc(c.get('answer', 'N/A'))}</div>
      </div>
      <div class="answer-box s2-box">
        <div class="answer-label">Stage 2 <span class="conf-tag">conf {stage2.get('confidence', 0):.2f}</span></div>
        <div class="answer-text">{esc(s2_answer)}</div>
      </div>
      <div class="answer-box e2e-box">
        <div class="answer-label">E2E <span class="conf-tag">conf {e2e.get('confidence', 0):.2f}</span> {guarded_badge}</div>
        <div class="answer-text">{esc(e2e_answer)}</div>
      </div>
    </div>

    <!-- Stage 1: Keyframes -->
    <div class="pipeline-section">
      <h3 class="pipe-title"><span class="pipe-num">1</span> Keyframe Retrieval</h3>
      <div class="pipe-meta">
        Query: <code>{esc(stage1.get('query', ''))}</code>
        &middot; Target: <code>{esc(stage1.get('target_term', ''))}</code>
        &middot; Anchor: <code>{esc(stage1.get('anchor_term', '') or 'none')}</code>
        &middot; Kind: {status_badge(stage1.get('hypothesis_kind', ''))}
      </div>
      <div class="kf-grid">{kf_html or '<div class="no-tools">No keyframes</div>'}</div>
      <details class="stage-details">
        <summary>Query candidates ({len(stage1.get('stage1_query_candidates', []))})</summary>
        <ul class="query-list">{"".join(f'<li>{esc(q)}</li>' for q in stage1.get('stage1_query_candidates', []))}</ul>
      </details>
    </div>

    <!-- Stage 2: Reasoning -->
    <div class="pipeline-section">
      <h3 class="pipe-title"><span class="pipe-num">2</span> VLM Reasoning (Stage 2)</h3>
      <div class="pipe-meta">
        Status: {status_badge(stage2.get('status', ''))}
        &middot; Confidence: <strong>{stage2.get('confidence', 0):.2f}</strong>
        &middot; Tool calls: {len(stage2.get('tool_trace', []))}
      </div>
      {f'<div class="evidence-list"><h4>Evidence</h4>{s2_evidence}</div>' if s2_evidence else ''}
      {f'<div class="uncertainties"><h4>Uncertainties</h4><ul>{s2_uncertainties}</ul></div>' if s2_uncertainties else ''}
      <details class="stage-details">
        <summary>Full payload</summary>
        <pre class="code-block">{esc(json.dumps(s2_payload, indent=2, ensure_ascii=False))}</pre>
      </details>
    </div>

    <!-- E2E: Tool-augmented reasoning -->
    <div class="pipeline-section">
      <h3 class="pipe-title"><span class="pipe-num">3</span> E2E Evidence Seeking</h3>
      <div class="pipe-meta">
        Status: {status_badge(e2e.get('status', ''))}
        &middot; Confidence: <strong>{e2e.get('confidence', 0):.2f}</strong>
        &middot; Tool calls: <strong>{len(e2e.get('tool_trace', []))}</strong>
        &middot; Final keyframes: {e2e.get('final_keyframes', '?')}
      </div>
      <div class="tool-trace">{tool_trace_html}</div>
      {f'<div class="evidence-list"><h4>Evidence</h4>{e2e_evidence}</div>' if e2e_evidence else ''}
      <details class="stage-details">
        <summary>Full payload</summary>
        <pre class="code-block">{esc(json.dumps(e2e_payload, indent=2, ensure_ascii=False))}</pre>
      </details>
    </div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def _css() -> str:
    return """<style>
:root {
  --bg: #fafaf9;
  --surface: #ffffff;
  --surface-alt: #f5f5f4;
  --border: #e7e5e4;
  --border-light: #f0eeec;
  --text: #1c1917;
  --text-2: #57534e;
  --text-3: #a8a29e;
  --accent: #1d4ed8;
  --accent-light: #eff6ff;
  --green: #059669;
  --red: #dc2626;
  --orange: #d97706;
  --purple: #7c3aed;
  --radius: 12px;
  --shadow-sm: 0 1px 2px rgba(0,0,0,.04);
  --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow-lg: 0 4px 12px rgba(0,0,0,.07), 0 1px 3px rgba(0,0,0,.04);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
code, pre, .code-block {
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
}

/* ---- Page ---- */
.page { max-width: 1200px; margin: 0 auto; padding: 0 24px 80px; }

/* ---- Hero ---- */
.hero {
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  margin: 0 -24px;
  padding: 56px 48px 48px;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 70% 20%, rgba(59,130,246,.15) 0%, transparent 60%),
              radial-gradient(ellipse at 20% 80%, rgba(124,58,237,.1) 0%, transparent 50%);
}
.hero-inner { position: relative; max-width: 1200px; margin: 0 auto; }
.hero-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-weight: 900;
  font-size: clamp(2rem, 5vw, 3.2rem);
  color: #f8fafc;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.hero-sub {
  margin-top: 12px;
  font-size: 15px;
  color: #94a3b8;
  font-weight: 300;
  letter-spacing: 0.01em;
}

/* ---- Score overview ---- */
.score-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-top: -32px;
  position: relative;
  z-index: 1;
}
.score-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  text-align: center;
  box-shadow: var(--shadow-lg);
}
.score-card-label { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--text-3); font-weight: 500; }
.score-card-value { font-size: 36px; font-weight: 700; margin: 4px 0 2px; line-height: 1.2; }
.score-card-max { font-size: 16px; color: var(--text-3); font-weight: 400; }
.score-card-sub { font-size: 13px; color: var(--text-2); }

/* ---- Sections ---- */
.section { margin-top: 48px; }
.section-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 20px;
  letter-spacing: -0.01em;
}

/* ---- Heatmap table ---- */
.heatmap-scroll { overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border); box-shadow: var(--shadow); }
.heatmap {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  background: var(--surface);
}
.heatmap th {
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--text-3);
  background: var(--surface-alt);
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0;
}
.heatmap td {
  padding: 8px 14px;
  border-bottom: 1px solid var(--border-light);
  vertical-align: middle;
}
.heatmap-row { cursor: pointer; transition: background .15s; }
.heatmap-row:hover { background: var(--accent-light); }
.hm-scene { font-weight: 600; color: var(--accent); white-space: nowrap; }
.hm-question { max-width: 320px; }

/* ---- Tags ---- */
.scene-tag {
  display: inline-block; padding: 1px 8px; border-radius: 6px;
  font-size: 11px; font-weight: 600; color: var(--accent);
  background: var(--accent-light); margin-right: 6px;
}
.cat-tag {
  display: inline-block; padding: 1px 8px; border-radius: 6px;
  font-size: 11px; font-weight: 500; color: var(--text-2);
  background: var(--surface-alt); border: 1px solid var(--border);
}
.guarded-badge {
  display: inline-block; padding: 1px 7px; border-radius: 6px;
  font-size: 10px; font-weight: 700; color: var(--purple);
  background: #f5f3ff; letter-spacing: .04em;
}
.conf-tag {
  font-size: 11px; color: var(--text-3); font-weight: 400;
}

/* ---- Case cards ---- */
.case-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 16px;
  box-shadow: var(--shadow-sm);
  overflow: hidden;
  transition: box-shadow .2s;
}
.case-card:hover { box-shadow: var(--shadow); }
.case-card.expanded .case-body { display: block; }
.case-card.expanded .expand-icon { transform: rotate(180deg); }

.case-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  cursor: pointer;
  user-select: none;
  gap: 16px;
}
.case-header:hover { background: var(--surface-alt); }
.case-header-left { display: flex; align-items: center; gap: 14px; min-width: 0; flex: 1; }
.case-header-right { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
.case-num {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 8px; background: var(--surface-alt);
  font-size: 13px; font-weight: 700; color: var(--text-2);
  flex-shrink: 0;
}
.case-question {
  font-size: 14px; font-weight: 600; color: var(--text);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.case-meta { display: flex; align-items: center; gap: 6px; margin-top: 4px; flex-wrap: wrap; }
.case-scores { display: flex; gap: 6px; }
.mini-score {
  padding: 3px 10px; border-radius: 8px;
  font-size: 12px; font-weight: 700;
}
.expand-icon {
  font-size: 14px; color: var(--text-3);
  transition: transform .2s;
}

.case-body {
  display: none;
  padding: 0 20px 24px;
  border-top: 1px solid var(--border-light);
}

/* ---- Answer grid ---- */
.answer-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin: 20px 0;
}
.answer-box {
  padding: 14px 16px;
  border-radius: 10px;
  border: 1px solid var(--border);
}
.gt-box { background: #f0fdf4; border-color: #bbf7d0; }
.s2-box { background: #eff6ff; border-color: #bfdbfe; }
.e2e-box { background: #f5f3ff; border-color: #ddd6fe; }
.answer-label {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .06em; color: var(--text-3); margin-bottom: 6px;
}
.answer-text { font-size: 14px; color: var(--text); line-height: 1.5; }

/* ---- Pipeline sections ---- */
.pipeline-section {
  margin-top: 24px;
  padding: 16px;
  background: var(--surface-alt);
  border-radius: 10px;
  border: 1px solid var(--border-light);
}
.pipe-title {
  display: flex; align-items: center; gap: 10px;
  font-size: 14px; font-weight: 700; color: var(--text);
  margin-bottom: 10px;
}
.pipe-num {
  width: 24px; height: 24px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%; background: var(--accent); color: white;
  font-size: 12px; font-weight: 700; flex-shrink: 0;
}
.pipe-meta { font-size: 13px; color: var(--text-2); margin-bottom: 12px; }
.pipe-meta code {
  background: var(--surface); padding: 1px 6px; border-radius: 4px;
  font-size: 12px; border: 1px solid var(--border);
}

/* ---- Keyframes ---- */
.kf-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
  margin: 10px 0;
}
.kf-thumb {
  position: relative; border-radius: 8px; overflow: hidden;
  cursor: pointer; border: 2px solid transparent;
  transition: border-color .15s, transform .15s;
  box-shadow: var(--shadow-sm);
}
.kf-thumb:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: var(--shadow); }
.kf-thumb img { width: 100%; height: auto; display: block; }
.kf-label {
  position: absolute; bottom: 0; left: 0; right: 0;
  background: linear-gradient(transparent, rgba(0,0,0,.7));
  color: white; font-size: 11px; padding: 16px 8px 6px;
  font-family: 'JetBrains Mono', monospace;
}
.kf-missing {
  background: var(--surface); border: 2px dashed var(--border);
  padding: 32px; text-align: center; font-size: 12px; color: var(--text-3);
}

/* ---- Tool calls ---- */
.tool-trace { margin-top: 8px; }
.tool-call {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--orange);
  border-radius: 0 8px 8px 0;
  margin: 8px 0;
  padding: 12px 14px;
}
.tool-header { display: flex; align-items: center; gap: 8px; }
.tool-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px; font-weight: 600; color: var(--orange);
}
.tool-details { margin-top: 8px; }
.tool-details summary {
  font-size: 12px; font-weight: 600; color: var(--text-2);
  cursor: pointer; padding: 4px 0;
}
.no-tools {
  font-size: 13px; color: var(--text-3); font-style: italic;
  padding: 8px 0;
}

/* ---- Evidence ---- */
.evidence-list { margin-top: 12px; }
.evidence-list h4 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); margin-bottom: 6px; }
.evidence-item {
  font-size: 13px; color: var(--text-2); padding: 6px 0;
  border-bottom: 1px solid var(--border-light);
}
.ev-frames {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--accent); font-weight: 500; margin-right: 6px;
}

.uncertainties h4 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); margin: 12px 0 6px; }
.uncertainties ul { padding-left: 18px; }
.uncertainties li { font-size: 13px; color: var(--text-2); margin: 3px 0; }

/* ---- Code block ---- */
.code-block {
  background: #1e293b;
  color: #e2e8f0;
  border-radius: 8px;
  padding: 14px 16px;
  font-size: 12px;
  line-height: 1.5;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 6px 0;
}

/* ---- Details/summary ---- */
.stage-details { margin-top: 10px; }
.stage-details summary {
  font-size: 12px; font-weight: 600; color: var(--accent);
  cursor: pointer; padding: 4px 0;
}
.query-list { padding-left: 18px; margin-top: 6px; }
.query-list li { font-size: 12px; color: var(--text-2); margin: 2px 0; font-family: 'JetBrains Mono', monospace; }

/* ---- Image modal ---- */
.modal-overlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.85);
  z-index: 9999;
  justify-content: center; align-items: center;
  backdrop-filter: blur(4px);
}
.modal-overlay.active { display: flex; }
.modal-overlay img {
  max-width: 92vw; max-height: 92vh;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 20px 60px rgba(0,0,0,.5);
}
.modal-close {
  position: absolute; top: 20px; right: 28px;
  color: white; font-size: 32px; cursor: pointer;
  opacity: .7; transition: opacity .15s;
}
.modal-close:hover { opacity: 1; }

/* ---- Responsive ---- */
@media (max-width: 768px) {
  .hero { padding: 36px 24px 32px; }
  .score-overview { margin-top: -24px; }
  .answer-grid { grid-template-columns: 1fr; }
}
</style>"""


def _modal_html() -> str:
    return """<div class="modal-overlay" id="imgModal" onclick="closeModal()">
  <span class="modal-close">&times;</span>
  <img id="modalImg" src="">
</div>"""


def _js() -> str:
    return """<script>
function scrollToCase(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('expanded');
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function openModal(thumbEl) {
  event.stopPropagation();
  const img = thumbEl.querySelector('img');
  if (!img) return;
  document.getElementById('modalImg').src = img.src;
  document.getElementById('imgModal').classList.add('active');
}

function closeModal() {
  document.getElementById('imgModal').classList.remove('active');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});
</script>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OpenEQA evaluation HTML report.")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Eval output directory")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    args = parser.parse_args()

    eval_dir = args.eval_dir
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir

    data = load_eval_data(eval_dir)
    html_content = build_html(data)

    output = args.output or (eval_dir / "report.html")
    output.write_text(html_content, encoding="utf-8")
    print(f"Report generated: {output}")
    print(f"  {len(data['cases'])} cases, {len(set(c['clip_id'] for c in data['cases']))} scenes")

    if args.open:
        webbrowser.open(f"file://{output.resolve()}")


if __name__ == "__main__":
    main()
