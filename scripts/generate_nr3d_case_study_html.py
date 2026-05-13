"""Generate an NR3D stage1+stage2 case-study HTML report.

The report is intentionally built from persisted benchmark artifacts only:

- stage1 pack JSON under data/nr3d/scannet/<scene>/<pack>/samples/
- annotated keyframes under data/nr3d/scannet/<scene>/<pack>/annotated/
- per-sample stage2 checkpoints under tmp/nr3d_eval_.../per_sample/
- leaderboard_metrics.json for correctness / split metadata

No model calls are made by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_CASES = [
    (
        "scannet/scene0474_00::9::27195",
        "Correct after no-match guard recovery",
        "Backpack on a sectional sofa; the agent first considered no-match, then the guard forced more evidence and it recovered proposal 9.",
    ),
    (
        "scannet/scene0030_00::50::32546",
        "Correct hard view-dependent spatial case",
        "Small white table farthest from windows, touching a wall, near chairs and a chalkboard; this stresses spatial constraints.",
    ),
    (
        "scannet/scene0655_00::19::3585",
        "Correct low-confidence chair case",
        "Same target object as the no-match example below, but the expression mentions chair, so category grounding succeeds.",
    ),
    (
        "scannet/scene0435_00::18::29303",
        "Wrong view-dependent door case",
        "Door on the left wall when facing the window and curtains; the agent selected a different door.",
    ),
    (
        "scannet/scene0655_00::19::8270",
        "Persistent no-match / filtered-out expression",
        "The query says box, but the target object is a chair and the proposal pool has no box-like category candidate.",
    ),
]


@dataclass(frozen=True)
class CaseSpec:
    sample_id: str
    title: str
    why: str


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pretty(value: Any, max_chars: int = 6000) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"
    return esc(text)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_short() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def safe_sample_prefix(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def parse_sample_id(sample_id: str) -> tuple[str, int, str]:
    parts = sample_id.split("::")
    if len(parts) < 3:
        raise ValueError(f"bad sample_id: {sample_id}")
    scene = parts[0].split("/", 1)[1]
    return scene, int(parts[1]), parts[2]


def find_checkpoint(per_sample_dir: Path, sample_id: str) -> Path:
    prefix = safe_sample_prefix(sample_id)
    matches = sorted(per_sample_dir.glob(prefix + "_*.json"))
    if not matches:
        raise FileNotFoundError(f"no checkpoint for {sample_id} under {per_sample_dir}")
    if len(matches) > 1:
        raise ValueError(f"multiple checkpoints for {sample_id}: {matches}")
    return matches[0]


def load_proposals(pack_dir: Path) -> dict[int, dict[str, Any]]:
    path = pack_dir / "proposals.jsonl"
    payload = read_json(path)
    proposals = payload.get("proposals", payload)
    return {int(p["id"]): p for p in proposals}


def load_visibility(pack_dir: Path) -> dict[int, list[int]]:
    raw = read_json(pack_dir / "visibility.json")
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


def try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def collect_image_refs(response_text: str) -> list[tuple[int | None, str]]:
    refs: list[tuple[int | None, str]] = []
    parsed = try_json(response_text)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("annotated_image"):
                refs.append((item.get("frame_id"), str(item["annotated_image"])))
    elif isinstance(parsed, dict) and parsed.get("annotated_image"):
        refs.append((parsed.get("frame_id"), str(parsed["annotated_image"])))

    for match in re.finditer(r"frame_id=(\d+)\s+marked image at ([^;]+)", response_text):
        refs.append((int(match.group(1)), match.group(2)))
    return refs


def make_thumb(
    src: Path,
    *,
    assets_dir: Path,
    html_dir: Path,
    prefix: str,
    max_size: tuple[int, int] = (760, 540),
) -> str:
    if not src.exists():
        return ""
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:8]
    out = assets_dir / f"{prefix}_{src.stem}_{digest}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB")
        img.thumbnail(max_size)
        img.save(out, "JPEG", quality=86, optimize=True)
    return str(out.relative_to(html_dir))


def bbox_summary(bbox: Any) -> str:
    if not isinstance(bbox, list) or len(bbox) < 6:
        return "-"
    vals = [f"{float(x):.3f}" for x in bbox[:6]]
    return f"center=({', '.join(vals[:3])}); size=({', '.join(vals[3:6])})"


def proposal_label(proposals: dict[int, dict[str, Any]], proposal_id: Any) -> str:
    if proposal_id is None:
        return "-"
    prop = proposals.get(int(proposal_id))
    if not prop:
        return "missing"
    return str(prop.get("label") or prop.get("category") or "-")


def visible_labels(
    frame_id: int,
    visibility: dict[int, list[int]],
    proposals: dict[int, dict[str, Any]],
    limit: int = 20,
) -> str:
    ids = visibility.get(frame_id, [])
    labels = [f"{pid}:{proposal_label(proposals, pid)}" for pid in ids[:limit]]
    suffix = "" if len(ids) <= limit else f" ... +{len(ids) - limit}"
    return ", ".join(labels) + suffix


def candidate_ids_from_trace(
    trace: list[dict[str, Any]],
    target_id: int,
    selected_id: int | None,
) -> list[int]:
    ids: list[int] = [target_id]
    if selected_id is not None:
        ids.append(int(selected_id))
    for call in trace:
        tool = call.get("tool_name")
        inp = call.get("tool_input") or {}
        parsed = try_json(str(call.get("response_text") or ""))
        if tool == "inspect_proposal" and isinstance(inp, dict):
            if inp.get("proposal_id") is not None:
                ids.append(int(inp["proposal_id"]))
        if tool == "find_proposals_by_category" and isinstance(parsed, dict):
            for pid in parsed.get("proposal_ids") or []:
                ids.append(int(pid))
    seen = set()
    ordered = []
    for pid in ids:
        if pid not in seen:
            ordered.append(pid)
            seen.add(pid)
    return ordered


def summarize_tool(call: dict[str, Any]) -> str:
    tool = str(call.get("tool_name") or "")
    inp = call.get("tool_input") or {}
    text = str(call.get("response_text") or "")
    parsed = try_json(text)

    if tool == "list_skills" and isinstance(parsed, list):
        return "Available skills: " + ", ".join(str(x.get("name")) for x in parsed if isinstance(x, dict))
    if tool == "load_skill":
        return f"Loaded skill: {inp.get('skill_name') if isinstance(inp, dict) else inp}"
    if tool == "list_keyframes_with_proposals" and isinstance(parsed, list):
        bits = []
        for kf in parsed:
            if isinstance(kf, dict):
                bits.append(
                    f"frame {kf.get('frame_id')} ({kf.get('n_proposals')} proposals)"
                )
        return "Initial marked evidence: " + "; ".join(bits)
    if tool == "find_proposals_by_category" and isinstance(parsed, dict):
        return (
            f"Category search '{parsed.get('category')}' -> "
            f"{parsed.get('proposal_ids') or []}"
        )
    if tool == "inspect_proposal" and isinstance(parsed, dict):
        return (
            f"Proposal {parsed.get('proposal_id')} "
            f"label={parsed.get('category')} "
            f"frames={len(parsed.get('frames_appeared') or [])}"
        )
    if tool == "view_keyframe_marked":
        refs = collect_image_refs(text)
        frame = refs[0][0] if refs else (inp.get("frame_id") if isinstance(inp, dict) else "?")
        cats = re.search(r"categories=(\[.*?\])", text)
        cat_text = cats.group(1) if cats else ""
        if len(cat_text) > 220:
            cat_text = cat_text[:220] + "..."
        return f"Viewed marked frame {frame}. {cat_text}"
    if tool == "request_more_views":
        return "Requested more evidence. Response: " + text[:260]
    if tool == "submit_final":
        payload = inp.get("payload") if isinstance(inp, dict) else None
        rationale = inp.get("rationale") if isinstance(inp, dict) else ""
        return f"submit_final payload={payload}; rationale={str(rationale)[:420]}"
    if tool == "inspect_stage1_metadata":
        return "Inspected stage1 metadata / proposal pool."
    return text[:420] if text else "(no response text)"


def render_table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows
    )
    return f'<table class="kv">{body}</table>'


def render_image_grid(
    refs: list[tuple[int | None, str, str]],
    *,
    case_prefix: str,
    assets_dir: Path,
    html_dir: Path,
    visibility: dict[int, list[int]],
    proposals: dict[int, dict[str, Any]],
    target_id: int,
    selected_id: int | None,
) -> str:
    cards = []
    seen = set()
    for idx, (frame_id, path, source) in enumerate(refs):
        if path in seen:
            continue
        seen.add(path)
        src = Path(path)
        rel = make_thumb(
            src,
            assets_dir=assets_dir,
            html_dir=html_dir,
            prefix=f"{case_prefix}_{idx:02d}",
        )
        if not rel:
            cards.append(
                f'<div class="image-card missing"><div>Missing image</div><code>{esc(path)}</code></div>'
            )
            continue
        fid = int(frame_id) if frame_id is not None else None
        ids = visibility.get(fid, []) if fid is not None else []
        chips = []
        if target_id in ids:
            chips.append('<span class="chip target">target visible</span>')
        if selected_id is not None and int(selected_id) in ids:
            chips.append('<span class="chip selected">selected visible</span>')
        cards.append(
            f"""
            <figure class="image-card">
              <img src="{esc(rel)}" alt="{esc(path)}" loading="lazy">
              <figcaption>
                <div><strong>frame {esc(fid if fid is not None else '?')}</strong> <span class="muted">{esc(source)}</span></div>
                <div>{''.join(chips)}</div>
                <div class="small">{esc(visible_labels(fid, visibility, proposals) if fid is not None else '')}</div>
                <code>{esc(path)}</code>
              </figcaption>
            </figure>
            """
        )
    return '<div class="image-grid">' + "\n".join(cards) + "</div>"


def render_proposal_table(
    proposal_ids: list[int],
    proposals: dict[int, dict[str, Any]],
    target_id: int,
    selected_id: int | None,
) -> str:
    rows = []
    for pid in proposal_ids:
        prop = proposals.get(pid, {})
        role = []
        if pid == target_id:
            role.append("target")
        if selected_id is not None and pid == int(selected_id):
            role.append("selected")
        rows.append(
            "<tr>"
            f"<td>{pid}</td>"
            f"<td>{esc(', '.join(role) or '-')}</td>"
            f"<td>{esc(prop.get('label') or prop.get('category') or '-')}</td>"
            f"<td>{esc(prop.get('score', '-'))}</td>"
            f"<td>{esc(bbox_summary(prop.get('bbox_3d') or prop.get('bbox_3d_9dof')))}</td>"
            "</tr>"
        )
    return (
        '<table><tr><th>ID</th><th>Role</th><th>Label</th><th>Score</th><th>BBox</th></tr>'
        + "\n".join(rows)
        + "</table>"
    )


def render_tool_timeline(
    trace: list[dict[str, Any]],
    *,
    case_prefix: str,
    assets_dir: Path,
    html_dir: Path,
    visibility: dict[int, list[int]],
    proposals: dict[int, dict[str, Any]],
    target_id: int,
    selected_id: int | None,
) -> str:
    blocks = []
    for i, call in enumerate(trace, start=1):
        tool = str(call.get("tool_name") or "")
        inp = call.get("tool_input")
        response = str(call.get("response_text") or "")
        image_refs = collect_image_refs(response)
        image_html = ""
        if image_refs:
            image_html = render_image_grid(
                [(fid, path, f"tool #{i}: {tool}") for fid, path in image_refs],
                case_prefix=f"{case_prefix}_tool{i:02d}",
                assets_dir=assets_dir,
                html_dir=html_dir,
                visibility=visibility,
                proposals=proposals,
                target_id=target_id,
                selected_id=selected_id,
            )
        blocks.append(
            f"""
            <details class="tool" open>
              <summary><span class="tool-index">{i:02d}</span> <span class="tool-name">{esc(tool)}</span> <span>{esc(summarize_tool(call))}</span></summary>
              <div class="tool-body">
                <div class="cols">
                  <div><h4>Input</h4><pre>{pretty(inp, 2500)}</pre></div>
                  <div><h4>Response summary / raw</h4><pre>{pretty(response, 7000)}</pre></div>
                </div>
                {image_html}
              </div>
            </details>
            """
        )
    return '<div class="timeline">' + "\n".join(blocks) + "</div>"


def outcome_text(
    *,
    metric: dict[str, Any],
    checkpoint: dict[str, Any],
    proposals: dict[int, dict[str, Any]],
) -> str:
    target_id = int(metric["target_id"])
    selected_id = metric.get("selected_object_id")
    correct = bool(metric["is_correct"])
    target_label = proposal_label(proposals, target_id)
    selected_label = proposal_label(proposals, selected_id)
    if correct:
        return (
            f"The selected proposal is the target: id {selected_id} ({selected_label}). "
            f"The GT and predicted boxes match with IoU {float(checkpoint.get('iou') or 0):.3f}."
        )
    if selected_id is None:
        return (
            f"The agent did not submit a proposal. The target is id {target_id} ({target_label}), "
            "but the stage2 trace ends in no-match. This is counted as failed in the full fold; "
            "for the official filtered headline this row may be excluded if mentions_target_class=false."
        )
    return (
        f"The agent selected id {selected_id} ({selected_label}), but the target is "
        f"id {target_id} ({target_label}). The classification metric is therefore false "
        f"and IoU is {float(checkpoint.get('iou') or 0):.3f}."
    )


def build_case_html(
    idx: int,
    spec: CaseSpec,
    *,
    data_root: Path,
    per_sample_dir: Path,
    metric: dict[str, Any],
    assets_dir: Path,
    html_dir: Path,
    pack_name: str,
) -> str:
    scene_id, target_id, _ = parse_sample_id(spec.sample_id)
    prefix = safe_sample_prefix(spec.sample_id)
    case_prefix = f"case{idx:02d}_{scene_id}_{target_id}"
    checkpoint = read_json(find_checkpoint(per_sample_dir, spec.sample_id))
    pack_dir = data_root / scene_id / pack_name
    sample_path = pack_dir / "samples" / f"{prefix}.json"
    sample = read_json(sample_path)
    proposals = load_proposals(pack_dir)
    visibility = load_visibility(pack_dir)
    trace = checkpoint.get("tool_trace") or []
    selected_id = checkpoint.get("selected_object_id")
    correct = bool(metric["is_correct"])
    filtered = not bool(metric["is_filtered_out"])
    status_class = "ok" if correct else ("failed" if selected_id is None else "wrong")
    status_label = "CORRECT" if correct else ("NO MATCH" if selected_id is None else "WRONG")

    stage1_refs = [
        (kf.get("frame_id"), str(kf.get("image_path")), "stage1 initial keyframe")
        for kf in sample.get("keyframes") or []
    ]
    stage2_refs: list[tuple[int | None, str, str]] = []
    for i, call in enumerate(trace, start=1):
        for fid, path in collect_image_refs(str(call.get("response_text") or "")):
            stage2_refs.append((fid, path, f"stage2 tool #{i}: {call.get('tool_name')}"))

    target_initial = sum(
        1
        for fid, _, _ in stage1_refs
        if fid is not None and int(target_id) in visibility.get(int(fid), [])
    )
    selected_initial = (
        sum(
            1
            for fid, _, _ in stage1_refs
            if fid is not None
            and selected_id is not None
            and int(selected_id) in visibility.get(int(fid), [])
        )
        if selected_id is not None
        else 0
    )
    tool_counts = Counter(str(t.get("tool_name") or "") for t in trace)
    proposal_ids = candidate_ids_from_trace(trace, target_id, selected_id)

    overview = render_table(
        [
            ("sample_id", spec.sample_id),
            ("query", checkpoint.get("query")),
            ("scene_id", scene_id),
            ("category / target label", f"{sample.get('category')} / {proposal_label(proposals, target_id)}"),
            ("target_id", target_id),
            ("selected_object_id", selected_id),
            ("selected label", proposal_label(proposals, selected_id)),
            ("status", checkpoint.get("status")),
            ("confidence", checkpoint.get("confidence")),
            ("IoU", f"{float(checkpoint.get('iou') or 0):.4f}"),
            ("filtered headline row", filtered),
            ("easy", metric.get("is_easy")),
            ("view-dependent", metric.get("is_view_dep")),
        ]
    )
    stage1_table = render_table(
        [
            ("pack sample", sample_path),
            ("keyframe_mode", sample.get("keyframe_mode")),
            ("uses_gt_target", sample.get("keyframe_selection_uses_gt_target")),
            ("used_non_gt_fallback", sample.get("keyframe_selection_used_fallback")),
            ("initial keyframes", ", ".join(str(k.get("frame_id")) for k in sample.get("keyframes") or [])),
            ("target visible in initial keyframes", f"{target_initial}/{len(stage1_refs)}"),
            ("selected visible in initial keyframes", f"{selected_initial}/{len(stage1_refs)}"),
        ]
    )

    return f"""
    <section class="case" id="case-{idx}">
      <div class="case-head">
        <div>
          <div class="eyebrow">Case {idx}</div>
          <h2>{esc(spec.title)}</h2>
          <p class="why">{esc(spec.why)}</p>
        </div>
        <div class="status {status_class}">{status_label}</div>
      </div>

      <h3>Outcome</h3>
      <p>{esc(outcome_text(metric=metric, checkpoint=checkpoint, proposals=proposals))}</p>
      {overview}

      <h3>Stage 1: query-driven evidence pack</h3>
      <p>
        Stage 1 produced the marked keyframe entry points below. The pack
        explicitly records <code>keyframe_selection_uses_gt_target=false</code>;
        target id and GT bbox are kept for scoring, not for view selection.
      </p>
      {stage1_table}
      {render_image_grid(stage1_refs, case_prefix=case_prefix + "_stage1", assets_dir=assets_dir, html_dir=html_dir, visibility=visibility, proposals=proposals, target_id=target_id, selected_id=selected_id)}

      <h3>Candidate proposals touched by Stage 2</h3>
      <p>The table includes the target, final selected proposal if any, category-search hits, and inspected proposals.</p>
      {render_proposal_table(proposal_ids, proposals, target_id, selected_id)}

      <h3>Stage 2: tool-augmented reasoning trace</h3>
      <p>
        Tool calls: {len(trace)}. Tool mix:
        {esc(', '.join(f'{k}={v}' for k, v in tool_counts.items()))}.
        Every call below is rendered in chronological order with input, response,
        and any referenced marked frame image.
      </p>
      {render_image_grid(stage2_refs, case_prefix=case_prefix + "_stage2refs", assets_dir=assets_dir, html_dir=html_dir, visibility=visibility, proposals=proposals, target_id=target_id, selected_id=selected_id)}
      {render_tool_timeline(trace, case_prefix=case_prefix, assets_dir=assets_dir, html_dir=html_dir, visibility=visibility, proposals=proposals, target_id=target_id, selected_id=selected_id)}
    </section>
    """


def build_html(
    *,
    cases: list[CaseSpec],
    metrics: dict[str, dict[str, Any]],
    data_root: Path,
    per_sample_dir: Path,
    output: Path,
    assets_dir: Path,
    pack_name: str,
) -> str:
    html_dir = output.parent
    case_html = []
    nav = []
    for i, spec in enumerate(cases, start=1):
        metric = metrics[spec.sample_id]
        case_html.append(
            build_case_html(
                i,
                spec,
                data_root=data_root,
                per_sample_dir=per_sample_dir,
                metric=metric,
                assets_dir=assets_dir,
                html_dir=html_dir,
                pack_name=pack_name,
            )
        )
        status = "correct" if metric["is_correct"] else "failed"
        nav.append(f'<a href="#case-{i}"><span>{i}</span>{esc(spec.title)}<em>{status}</em></a>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NR3D v5.1 Stage1+Stage2 Case Studies</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --ink: #172026;
      --muted: #5b6770;
      --line: #d9dee4;
      --panel: #ffffff;
      --ok: #117a4b;
      --wrong: #b54708;
      --failed: #b42318;
      --accent: #315c96;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); line-height: 1.5; }}
    header {{ background: #101820; color: white; padding: 28px 40px; }}
    header h1 {{ margin: 0 0 10px; font-size: 30px; letter-spacing: 0; }}
    header p {{ margin: 4px 0; color: #d3dce5; max-width: 1180px; }}
    main {{ max-width: 1420px; margin: 0 auto; padding: 24px; }}
    nav {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin: 18px 0 28px; }}
    nav a {{ display: flex; gap: 10px; align-items: center; padding: 10px 12px; background: var(--panel); color: var(--ink); text-decoration: none; border: 1px solid var(--line); border-radius: 6px; }}
    nav span {{ font-weight: 700; color: var(--accent); }}
    nav em {{ margin-left: auto; color: var(--muted); font-style: normal; font-size: 12px; }}
    .case {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 22px; margin: 0 0 28px; }}
    .case-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 20px; border-bottom: 1px solid var(--line); padding-bottom: 14px; }}
    .eyebrow {{ color: var(--accent); text-transform: uppercase; font-size: 12px; font-weight: 700; letter-spacing: .08em; }}
    h2 {{ margin: 4px 0 6px; font-size: 24px; }}
    h3 {{ margin-top: 24px; border-left: 4px solid var(--accent); padding-left: 10px; }}
    h4 {{ margin: 0 0 8px; }}
    .why {{ margin: 0; color: var(--muted); max-width: 920px; }}
    .status {{ padding: 8px 12px; border-radius: 6px; color: white; font-weight: 700; white-space: nowrap; }}
    .status.ok {{ background: var(--ok); }}
    .status.wrong {{ background: var(--wrong); }}
    .status.failed {{ background: var(--failed); }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; vertical-align: top; text-align: left; }}
    th {{ background: #edf1f5; }}
    table.kv th {{ width: 230px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin: 12px 0; }}
    .image-card {{ margin: 0; border: 1px solid var(--line); background: #fafbfc; border-radius: 6px; overflow: hidden; }}
    .image-card img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ padding: 10px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #0f1720; color: #edf3f8; padding: 12px; border-radius: 6px; max-height: 360px; overflow: auto; }}
    .small {{ color: var(--muted); font-size: 12px; margin-top: 5px; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .chip {{ display: inline-block; margin: 4px 5px 0 0; padding: 2px 6px; border-radius: 5px; color: white; font-size: 12px; }}
    .chip.target {{ background: var(--ok); }}
    .chip.selected {{ background: var(--accent); }}
    details.tool {{ border: 1px solid var(--line); border-radius: 6px; margin: 10px 0; background: #fbfcfd; }}
    details.tool summary {{ cursor: pointer; padding: 10px 12px; display: flex; gap: 10px; align-items: baseline; }}
    .tool-index {{ font-weight: 700; color: var(--accent); }}
    .tool-name {{ font-weight: 700; min-width: 190px; }}
    .tool-body {{ padding: 0 12px 12px; }}
    .cols {{ display: grid; grid-template-columns: minmax(240px, 0.9fr) minmax(300px, 1.5fr); gap: 12px; }}
    @media (max-width: 860px) {{
      header {{ padding: 22px; }}
      main {{ padding: 14px; }}
      .case-head {{ display: block; }}
      .status {{ display: inline-block; margin-top: 12px; }}
      .cols {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>NR3D v5.1 Stage1 + Stage2 Reasoning Case Studies</h1>
  <p>Generated from persisted artifacts only. No model calls are made by this report.</p>
  <p>Run branch: <code>feat/nr3d-v4-agent-guards-fair-views</code>; run-time code commit: <code>c404536</code>; report-generation repo commit: <code>{esc(git_short())}</code>.</p>
  <p>Source run: <code>v5p1_failed_rerun_full_20260513</code>; output: <code>tmp/nr3d_eval_v5_failed_rerun_merged_20260513/</code>; pack: <code>{esc(pack_name)}</code>.</p>
</header>
<main>
  <nav>{''.join(nav)}</nav>
  {''.join(case_html)}
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard-metrics",
        type=Path,
        default=Path("tmp/nr3d_eval_v5_failed_rerun_merged_20260513/leaderboard_metrics.json"),
    )
    parser.add_argument(
        "--per-sample-dir",
        type=Path,
        default=Path(
            "tmp/nr3d_eval_v5_failed_rerun_merged_20260513/per_sample/pack_nr3d_v4_agent_guards_fair_views"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/nr3d/scannet"))
    parser.add_argument(
        "--pack-name",
        default="pack_nr3d_v4_agent_guards_fair_views",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark/nr3d/v5p1_case_studies_20260513.html"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("docs/benchmark/nr3d/assets/v5p1_case_studies_20260513"),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Optional sample_id override. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_payload = read_json(args.leaderboard_metrics)
    metrics = {
        str(item["sample_id"]): item
        for item in metrics_payload.get("per_sample", [])
        if isinstance(item, dict) and item.get("sample_id")
    }
    if args.case:
        cases = [CaseSpec(sample_id=sid, title=sid, why="User-selected case.") for sid in args.case]
    else:
        cases = [CaseSpec(*row) for row in DEFAULT_CASES]
    missing = [case.sample_id for case in cases if case.sample_id not in metrics]
    if missing:
        raise ValueError(f"sample ids missing from metrics: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    html_text = build_html(
        cases=cases,
        metrics=metrics,
        data_root=args.data_root,
        per_sample_dir=args.per_sample_dir,
        output=args.output,
        assets_dir=args.assets_dir,
        pack_name=args.pack_name,
    )
    args.output.write_text(html_text, encoding="utf-8")
    print(f"[nr3d-case-study] wrote {args.output}")
    print(f"[nr3d-case-study] assets {args.assets_dir}")
    print(f"[nr3d-case-study] cases {len(cases)}")


if __name__ == "__main__":
    main()
