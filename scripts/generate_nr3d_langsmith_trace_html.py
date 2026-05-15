"""Render NR3D VG tool traces as a LangSmith-style static HTML viewer.

The viewer is built from persisted benchmark artifacts only:

- ``leaderboard_metrics.json`` for correctness and split labels.
- per-sample checkpoints for Stage-2 tool traces.
- pack samples for initial keyframe image paths.

It does not call the agent or any model. The saved checkpoints do not include
the full LangChain message stream, so the report reconstructs the visible run
tree from the recorded tool calls and evidence image paths.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def safe_sample_prefix(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def parse_sample_id(sample_id: str) -> tuple[str, int, str]:
    parts = sample_id.split("::")
    if len(parts) < 3:
        raise ValueError(f"bad sample_id: {sample_id}")
    return parts[0].split("/", 1)[1], int(parts[1]), parts[2]


def try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def pretty(value: Any, *, max_chars: int = 10_000) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"
    return esc(text)


def compact_text(text: str, *, max_chars: int = 12_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


def find_checkpoint(per_sample_dir: Path, sample_id: str) -> Path:
    prefix = safe_sample_prefix(sample_id)
    matches = sorted(per_sample_dir.glob(prefix + "_*.json"))
    if not matches:
        raise FileNotFoundError(f"no checkpoint for {sample_id} under {per_sample_dir}")
    if len(matches) > 1:
        raise ValueError(f"multiple checkpoints for {sample_id}: {matches}")
    return matches[0]


def load_pack_sample(data_root: Path, pack_name: str, sample_id: str) -> dict[str, Any]:
    scene_id, _, _ = parse_sample_id(sample_id)
    sample_path = (
        data_root
        / scene_id
        / pack_name
        / "samples"
        / f"{safe_sample_prefix(sample_id)}.json"
    )
    return read_json(sample_path)


def make_thumb(
    src: Path,
    *,
    assets_dir: Path,
    html_dir: Path,
    prefix: str,
    max_size: tuple[int, int] = (900, 620),
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


def collect_image_refs(response_text: str) -> list[tuple[int | None, str]]:
    refs: list[tuple[int | None, str]] = []
    parsed = try_json(response_text)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("annotated_image"):
                refs.append((item.get("frame_id"), str(item["annotated_image"])))
    elif isinstance(parsed, dict) and parsed.get("annotated_image"):
        refs.append((parsed.get("frame_id"), str(parsed["annotated_image"])))

    for match in re.finditer(
        r"frame_id=(\d+)\s+(?:filtered\s+)?marked image at ([^;]+)",
        response_text,
    ):
        refs.append((int(match.group(1)), match.group(2)))
    return refs


def load_proposal_labels(data_root: Path, pack_name: str, scene_id: str) -> dict[int, str]:
    path = data_root / scene_id / pack_name / "proposals.jsonl"
    if not path.exists():
        return {}
    payload = read_json(path)
    proposals = payload.get("proposals", payload)
    labels: dict[int, str] = {}
    if isinstance(proposals, list):
        for proposal in proposals:
            if not isinstance(proposal, dict) or proposal.get("id") is None:
                continue
            labels[int(proposal["id"])] = str(
                proposal.get("label") or proposal.get("category") or ""
            )
    return labels


def tool_label(tool: str) -> str:
    labels = {
        "list_skills": "列出可用技能",
        "load_skill": "加载推理 Playbook",
        "list_keyframes_with_proposals": "读取初始帧候选清单",
        "list_frame_proposals": "读取单帧候选清单",
        "view_keyframe_marked": "查看带框关键帧",
        "find_proposals_by_category": "按类别搜索候选",
        "inspect_proposal": "检查候选属性",
        "compare_proposals_spatial": "几何关系排序",
        "request_more_views": "请求更多视角",
        "request_crops": "请求局部裁剪",
        "switch_or_expand_hypothesis": "切换/扩展 Stage1 假设",
        "submit_final": "提交最终答案",
        "inspect_stage1_metadata": "检查 Stage1 元数据",
        "retrieve_object_context": "检索对象上下文",
    }
    return labels.get(tool, tool)


def node_kind(tool: str) -> str:
    if tool in {"submit_final"}:
        return "final"
    if tool in {"view_keyframe_marked", "request_more_views", "request_crops"}:
        return "evidence"
    if tool in {"list_skills", "load_skill"}:
        return "setup"
    if tool in {
        "find_proposals_by_category",
        "inspect_proposal",
        "compare_proposals_spatial",
        "list_frame_proposals",
        "list_keyframes_with_proposals",
    }:
        return "tool"
    return "tool"


def summarize_tool(call: dict[str, Any]) -> str:
    tool = str(call.get("tool_name") or "")
    inp = call.get("tool_input") or {}
    text = str(call.get("response_text") or "")
    parsed = try_json(text)

    if tool == "list_skills" and isinstance(parsed, list):
        names = [str(x.get("name")) for x in parsed if isinstance(x, dict)]
        return "可用技能：" + "、".join(names)
    if tool == "load_skill":
        return f"加载技能：{inp.get('skill_name') if isinstance(inp, dict) else inp}"
    if tool == "list_keyframes_with_proposals" and isinstance(parsed, list):
        frames = [
            f"{x.get('frame_id')}({x.get('n_proposals')})"
            for x in parsed
            if isinstance(x, dict)
        ]
        return "初始帧候选数：" + " / ".join(frames)
    if tool == "list_frame_proposals" and isinstance(parsed, dict):
        ltr = parsed.get("left_to_right") or []
        suffix = "" if len(ltr) <= 8 else f" ... +{len(ltr) - 8}"
        return (
            f"frame {parsed.get('frame_id')} 可见 {len(parsed.get('visible_proposal_ids') or [])} 个候选；"
            + "，".join(map(str, ltr[:8]))
            + suffix
        )
    if tool == "find_proposals_by_category" and isinstance(parsed, dict):
        return f"类别 {parsed.get('category')} -> {parsed.get('proposal_ids') or []}"
    if tool == "inspect_proposal" and isinstance(parsed, dict):
        frames = parsed.get("frames_appeared") or []
        return (
            f"候选 #{parsed.get('proposal_id')}，类别 {parsed.get('category')}，"
            f"出现帧数 {len(frames)}"
        )
    if tool == "compare_proposals_spatial" and isinstance(parsed, dict):
        return (
            f"关系 {parsed.get('relation')}，anchor #{parsed.get('anchor_id')}，"
            f"排序 {parsed.get('ranked_ids')}"
        )
    if tool == "view_keyframe_marked":
        refs = collect_image_refs(text)
        frame_id = refs[0][0] if refs else (inp.get("frame_id") if isinstance(inp, dict) else "?")
        filters = []
        if isinstance(inp, dict):
            if inp.get("categories"):
                filters.append(f"categories={inp.get('categories')}")
            if inp.get("proposal_ids"):
                filters.append(f"proposal_ids={inp.get('proposal_ids')}")
        filter_text = "；过滤 " + " ".join(filters) if filters else "；未过滤"
        return f"查看 frame {frame_id} 的标注图{filter_text}"
    if tool == "request_more_views":
        return "请求更多视角：" + text[:220]
    if tool == "request_crops":
        return "请求局部裁剪：" + text[:220]
    if tool == "switch_or_expand_hypothesis":
        return "重新请求 Stage1：" + text[:220]
    if tool == "submit_final" and isinstance(inp, dict):
        return f"提交 {inp.get('payload')}；理由：{str(inp.get('rationale') or '')[:260]}"
    return text[:260] if text else "无返回文本"


def render_image_cards(
    refs: list[tuple[int | None, str, str]],
    *,
    assets_dir: Path,
    html_dir: Path,
    prefix: str,
) -> str:
    cards: list[str] = []
    seen: set[str] = set()
    for idx, (frame_id, path, caption) in enumerate(refs):
        if path in seen:
            continue
        seen.add(path)
        rel = make_thumb(
            Path(path),
            assets_dir=assets_dir,
            html_dir=html_dir,
            prefix=f"{prefix}_{idx:03d}",
        )
        if not rel:
            cards.append(
                f'<div class="image-card missing"><strong>图片缺失</strong><code>{esc(path)}</code></div>'
            )
            continue
        cards.append(
            f"""
            <figure class="image-card">
              <img src="{esc(rel)}" alt="{esc(path)}" loading="lazy">
              <figcaption>
                <strong>frame {esc(frame_id if frame_id is not None else "?")}</strong>
                <span>{esc(caption)}</span>
                <code>{esc(path)}</code>
              </figcaption>
            </figure>
            """
        )
    if not cards:
        return '<div class="empty">没有记录到可直接复现的图片路径。</div>'
    return '<div class="image-grid">' + "\n".join(cards) + "</div>"


def compact_tool_response(tool: str, response: str) -> str:
    if tool == "load_skill":
        return compact_text(response, max_chars=5_000)
    if tool in {"list_skills", "list_keyframes_with_proposals", "list_frame_proposals"}:
        return compact_text(response, max_chars=8_000)
    return compact_text(response, max_chars=12_000)


def render_tool_card(
    call: dict[str, Any],
    *,
    step_index: int,
    case_prefix: str,
    assets_dir: Path,
    html_dir: Path,
) -> str:
    tool = str(call.get("tool_name") or "")
    inp = call.get("tool_input")
    response = str(call.get("response_text") or "")
    image_refs = [
        (frame_id, path, f"工具 #{step_index:02d} {tool_label(tool)}")
        for frame_id, path in collect_image_refs(response)
    ]
    image_html = render_image_cards(
        image_refs,
        assets_dir=assets_dir,
        html_dir=html_dir,
        prefix=f"{case_prefix}_tool{step_index:02d}",
    ) if image_refs else ""
    return f"""
    <details class="call-card {esc(node_kind(tool))}" id="{esc(case_prefix)}-step-{step_index}" open>
      <summary>
        <span class="step-no">{step_index:02d}</span>
        <span class="tool-pill">{esc(tool)}</span>
        <span class="step-title">{esc(tool_label(tool))}</span>
        <span class="step-summary">{esc(summarize_tool(call))}</span>
      </summary>
      <div class="call-body">
        <div class="io-grid">
          <section>
            <h4>模型发起的工具输入（原文）</h4>
            <pre>{pretty(inp, max_chars=8_000)}</pre>
          </section>
          <section>
            <h4>工具返回 / 可视证据引用（原文）</h4>
            <pre>{pretty(compact_tool_response(tool, response), max_chars=13_000)}</pre>
          </section>
        </div>
        {image_html}
      </div>
    </details>
    """


def render_tree(trace: list[dict[str, Any]], *, case_prefix: str) -> str:
    rows = [
        f"""
        <a class="tree-node root" href="#{esc(case_prefix)}-stage1">
          <span class="tree-dot"></span>
          <span>Stage 1 证据包</span>
        </a>
        """
    ]
    for idx, call in enumerate(trace, start=1):
        tool = str(call.get("tool_name") or "")
        rows.append(
            f"""
            <a class="tree-node {esc(node_kind(tool))}" href="#{esc(case_prefix)}-step-{idx}">
              <span class="tree-dot"></span>
              <span class="tree-index">{idx:02d}</span>
              <span class="tree-tool">{esc(tool_label(tool))}</span>
            </a>
            """
        )
    return '<div class="trace-tree">' + "\n".join(rows) + "</div>"


def status_label(metric: dict[str, Any]) -> tuple[str, str]:
    if bool(metric.get("is_correct")):
        return "正确", "ok"
    return "错误", "wrong"


def build_case(
    *,
    index: int,
    metric: dict[str, Any],
    data_root: Path,
    per_sample_dir: Path,
    pack_name: str,
    assets_dir: Path,
    html_dir: Path,
) -> tuple[str, str]:
    sample_id = str(metric["sample_id"])
    scene_id, target_id, _ = parse_sample_id(sample_id)
    case_prefix = f"case{index:03d}_{scene_id}_{target_id}"
    checkpoint = read_json(find_checkpoint(per_sample_dir, sample_id))
    sample = load_pack_sample(data_root, pack_name, sample_id)
    labels = load_proposal_labels(data_root, pack_name, scene_id)
    trace = checkpoint.get("tool_trace") or []
    selected_id = checkpoint.get("selected_object_id")
    label_text, label_class = status_label(metric)

    initial_refs = [
        (
            keyframe.get("frame_id"),
            str(keyframe.get("image_path")),
            f"初始 clean RGB keyframe #{keyframe.get('keyframe_idx')}",
        )
        for keyframe in sample.get("keyframes") or []
    ]
    stage1_images = render_image_cards(
        initial_refs,
        assets_dir=assets_dir,
        html_dir=html_dir,
        prefix=f"{case_prefix}_stage1",
    )

    tool_counts = Counter(str(call.get("tool_name") or "") for call in trace)
    selected_label = labels.get(int(selected_id), "") if selected_id is not None else ""
    target_label = labels.get(int(target_id), "")
    split_bits = [
        "Easy" if metric.get("is_easy") else "Hard",
        "View-Dep" if metric.get("is_view_dep") else "View-Indep",
    ]
    nav = f"""
    <button class="run-link {label_class}" data-run="{esc(sample_id)}">
      <span class="run-title">{esc(sample_id)}</span>
      <span class="run-query">{esc(checkpoint.get("query"))}</span>
      <span class="run-meta">{esc(label_text)} · selected={esc(selected_id)} · gt={esc(target_id)}</span>
    </button>
    """
    cards = "\n".join(
        render_tool_card(
            call,
            step_index=idx,
            case_prefix=case_prefix,
            assets_dir=assets_dir,
            html_dir=html_dir,
        )
        for idx, call in enumerate(trace, start=1)
    )
    html_case = f"""
    <article class="run-panel" data-run="{esc(sample_id)}">
      <header class="run-header">
        <div>
          <div class="eyebrow">Run {index:03d} · NR3D visual grounding</div>
          <h2>{esc(checkpoint.get("query"))}</h2>
          <p class="sample-id">{esc(sample_id)}</p>
        </div>
        <div class="verdict {label_class}">{esc(label_text)}</div>
      </header>

      <section class="summary-grid">
        <div><span>GT</span><strong>#{esc(target_id)} {esc(target_label)}</strong></div>
        <div><span>Selected</span><strong>#{esc(selected_id)} {esc(selected_label)}</strong></div>
        <div><span>Confidence</span><strong>{esc(checkpoint.get("confidence"))}</strong></div>
        <div><span>IoU</span><strong>{float(checkpoint.get("iou") or 0):.4f}</strong></div>
        <div><span>Split</span><strong>{esc(" / ".join(split_bits))}</strong></div>
        <div><span>Tool calls</span><strong>{len(trace)}</strong></div>
      </section>

      <section class="trace-layout">
        <aside>
          <h3>调用链条</h3>
          {render_tree(trace, case_prefix=case_prefix)}
          <div class="tool-mix">
            <h4>工具分布</h4>
            {''.join(f'<span>{esc(k)}={v}</span>' for k, v in tool_counts.items())}
          </div>
        </aside>
        <main>
          <section class="stage1-card" id="{esc(case_prefix)}-stage1">
            <h3>Stage 1 输入给 Agent 的初始证据</h3>
            <p>这里是 Agent 第一轮看到的 clean RGB 图像。候选 id / category 的文字清单在 prompt 中给出，图上不直接画 bbox。</p>
            <dl>
              <dt>pack sample</dt><dd><code>{esc(data_root / scene_id / pack_name / "samples" / (safe_sample_prefix(sample_id) + ".json"))}</code></dd>
              <dt>keyframe_mode</dt><dd>{esc(sample.get("keyframe_mode"))}</dd>
              <dt>uses_gt_target</dt><dd>{esc(sample.get("keyframe_selection_uses_gt_target"))}</dd>
            </dl>
            {stage1_images}
          </section>
          <section class="call-stack">
            <h3>Stage 2 Agent 调用栈</h3>
            <p>checkpoint 只保存工具调用轨迹，不保存完整 LangChain message stream；下列节点按真实工具调用顺序复原。工具输入/输出保持原文，其他说明为中文。</p>
            {cards}
          </section>
        </main>
      </section>
    </article>
    """
    return nav, html_case


def build_html(
    *,
    metrics_payload: dict[str, Any],
    metrics: list[dict[str, Any]],
    data_root: Path,
    per_sample_dir: Path,
    pack_name: str,
    output: Path,
    assets_dir: Path,
    title: str,
    run_id: str,
    run_commit: str,
    run_output: str,
) -> str:
    html_dir = output.parent
    nav_parts: list[str] = []
    case_parts: list[str] = []
    for index, metric in enumerate(metrics, start=1):
        nav, case = build_case(
            index=index,
            metric=metric,
            data_root=data_root,
            per_sample_dir=per_sample_dir,
            pack_name=pack_name,
            assets_dir=assets_dir,
            html_dir=html_dir,
        )
        nav_parts.append(nav)
        case_parts.append(case)

    n = metrics_payload.get("n_filtered", len(metrics))
    overall = float(metrics_payload.get("classification_acc_filtered") or 0)
    easy = float(metrics_payload.get("acc_easy") or 0)
    hard = float(metrics_payload.get("acc_hard") or 0)
    view_dep = float(metrics_payload.get("acc_view_dep") or 0)
    view_indep = float(metrics_payload.get("acc_view_indep") or 0)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-2: #fbfcfe;
      --ink: #172026;
      --muted: #66717c;
      --line: #d9dee5;
      --line-strong: #b8c1cc;
      --blue: #2457a6;
      --green: #0f7a4d;
      --red: #b42318;
      --amber: #b54708;
      --code: #111827;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.45; }}
    .app {{ display: grid; grid-template-columns: 390px minmax(0, 1fr); min-height: 100vh; }}
    .sidebar {{ position: sticky; top: 0; height: 100vh; overflow: auto; border-right: 1px solid var(--line); background: #111820; color: white; padding: 18px; }}
    .sidebar h1 {{ font-size: 20px; margin: 0 0 8px; letter-spacing: 0; }}
    .sidebar p {{ color: #c9d3df; margin: 6px 0; font-size: 13px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 16px 0; }}
    .metric {{ background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 8px; padding: 9px; }}
    .metric span {{ display: block; color: #b9c7d6; font-size: 12px; }}
    .metric strong {{ font-size: 20px; }}
    .controls {{ display: grid; grid-template-columns: 1fr; gap: 8px; margin: 14px 0; }}
    .controls input, .controls select {{ width: 100%; border: 1px solid #3a4654; background: #0b1118; color: white; border-radius: 6px; padding: 9px; }}
    .run-list {{ display: grid; gap: 8px; }}
    .run-link {{ width: 100%; text-align: left; border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.06); color: white; border-radius: 8px; padding: 10px; cursor: pointer; }}
    .run-link:hover, .run-link.active {{ background: rgba(255,255,255,.14); border-color: rgba(255,255,255,.35); }}
    .run-link.ok {{ border-left: 4px solid var(--green); }}
    .run-link.wrong {{ border-left: 4px solid var(--red); }}
    .run-title, .run-query, .run-meta {{ display: block; }}
    .run-title {{ font-size: 12px; color: #d5dfeb; overflow-wrap: anywhere; }}
    .run-query {{ margin: 4px 0; font-weight: 650; }}
    .run-meta {{ font-size: 12px; color: #b8c6d6; }}
    .content {{ min-width: 0; padding: 22px; }}
    .run-panel {{ display: none; }}
    .run-panel.active {{ display: block; }}
    .run-header {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }}
    .eyebrow {{ color: var(--blue); text-transform: uppercase; font-size: 12px; font-weight: 700; letter-spacing: .08em; }}
    h2 {{ margin: 4px 0 6px; font-size: 25px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }}
    h4 {{ margin: 0 0 8px; font-size: 13px; color: var(--muted); }}
    .sample-id {{ margin: 0; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }}
    .verdict {{ min-width: 74px; text-align: center; border-radius: 6px; color: white; font-weight: 750; padding: 8px 11px; }}
    .verdict.ok {{ background: var(--green); }}
    .verdict.wrong {{ background: var(--red); }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(6, minmax(110px, 1fr)); gap: 10px; margin: 14px 0; }}
    .summary-grid div {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
    .summary-grid span {{ display: block; color: var(--muted); font-size: 12px; }}
    .summary-grid strong {{ display: block; margin-top: 3px; overflow-wrap: anywhere; }}
    .trace-layout {{ display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 14px; align-items: start; }}
    .trace-layout > aside {{ position: sticky; top: 18px; max-height: calc(100vh - 36px); overflow: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .trace-layout > main {{ display: grid; gap: 14px; }}
    .stage1-card, .call-stack {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .stage1-card p, .call-stack p {{ color: var(--muted); margin-top: 0; }}
    dl {{ display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 6px 12px; margin: 10px 0; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    .trace-tree {{ position: relative; display: grid; gap: 6px; padding-left: 8px; }}
    .trace-tree:before {{ content: ""; position: absolute; top: 8px; bottom: 8px; left: 15px; width: 2px; background: var(--line); }}
    .tree-node {{ position: relative; z-index: 1; display: grid; grid-template-columns: 16px 32px 1fr; gap: 8px; align-items: center; min-height: 28px; color: var(--ink); text-decoration: none; border: 1px solid var(--line); background: var(--panel-2); border-radius: 7px; padding: 7px 8px; }}
    .tree-node.root {{ grid-template-columns: 16px 1fr; font-weight: 700; }}
    .tree-node:hover {{ border-color: var(--blue); }}
    .tree-dot {{ width: 10px; height: 10px; border-radius: 50%; background: var(--blue); border: 2px solid white; box-shadow: 0 0 0 1px var(--line-strong); }}
    .tree-node.setup .tree-dot {{ background: #7b61ff; }}
    .tree-node.evidence .tree-dot {{ background: var(--amber); }}
    .tree-node.final .tree-dot {{ background: var(--green); }}
    .tree-index {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .tree-tool {{ font-size: 13px; }}
    .tool-mix {{ margin-top: 16px; display: flex; flex-wrap: wrap; gap: 6px; }}
    .tool-mix h4 {{ width: 100%; margin-top: 0; }}
    .tool-mix span {{ display: inline-block; background: #edf2f7; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; font-size: 12px; color: var(--ink); }}
    .call-card {{ border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; background: var(--panel-2); scroll-margin-top: 16px; }}
    .call-card summary {{ cursor: pointer; list-style: none; display: grid; grid-template-columns: 42px minmax(120px, 180px) minmax(130px, 190px) minmax(0, 1fr); gap: 10px; align-items: baseline; padding: 12px; }}
    .call-card summary::-webkit-details-marker {{ display: none; }}
    .call-card.setup {{ border-left: 4px solid #7b61ff; }}
    .call-card.tool {{ border-left: 4px solid var(--blue); }}
    .call-card.evidence {{ border-left: 4px solid var(--amber); }}
    .call-card.final {{ border-left: 4px solid var(--green); }}
    .step-no {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--muted); font-weight: 700; }}
    .tool-pill {{ background: #e8eef7; border: 1px solid #d2dceb; color: #193b6d; border-radius: 999px; padding: 3px 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .step-title {{ font-weight: 750; }}
    .step-summary {{ color: var(--muted); overflow-wrap: anywhere; }}
    .call-body {{ border-top: 1px solid var(--line); padding: 12px; }}
    .io-grid {{ display: grid; grid-template-columns: minmax(260px, .9fr) minmax(320px, 1.3fr); gap: 12px; }}
    pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 420px; overflow: auto; background: var(--code); color: #eef6ff; border-radius: 7px; padding: 12px; font-size: 12px; line-height: 1.45; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 12px; }}
    .image-card {{ margin: 0; background: white; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .image-card img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 9px; color: var(--muted); font-size: 12px; }}
    figcaption strong, figcaption span, figcaption code {{ display: block; }}
    .empty {{ color: var(--muted); background: #f2f4f7; border: 1px dashed var(--line-strong); border-radius: 7px; padding: 10px; }}
    @media (max-width: 1120px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: relative; height: auto; }}
      .trace-layout {{ grid-template-columns: 1fr; }}
      .trace-layout > aside {{ position: relative; top: 0; max-height: none; }}
      .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .io-grid {{ grid-template-columns: 1fr; }}
      .call-card summary {{ grid-template-columns: 40px 1fr; }}
      .step-summary {{ grid-column: 1 / -1; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <h1>{esc(title)}</h1>
      <p>LangSmith 风格静态查看器。除工具输入/输出原文外，界面说明均为中文。</p>
      <p>Run: <code>{esc(run_id)}</code></p>
      <p>Run commit: <code>{esc(run_commit)}</code> · Report commit: <code>{esc(git_short())}</code></p>
      <p>Artifacts: <code>{esc(run_output)}</code></p>
      <div class="metrics">
        <div class="metric"><span>样本数</span><strong>{esc(n)}</strong></div>
        <div class="metric"><span>Overall</span><strong>{overall * 100:.2f}</strong></div>
        <div class="metric"><span>Easy / Hard</span><strong>{easy * 100:.1f} / {hard * 100:.1f}</strong></div>
        <div class="metric"><span>ViewDep / Indep</span><strong>{view_dep * 100:.1f} / {view_indep * 100:.1f}</strong></div>
      </div>
      <div class="controls">
        <input id="search" type="search" placeholder="搜索 sample id / query / selected / gt">
        <select id="filter">
          <option value="all">全部样本</option>
          <option value="ok">只看正确</option>
          <option value="wrong">只看错误</option>
        </select>
      </div>
      <div class="run-list" id="run-list">{''.join(nav_parts)}</div>
    </aside>
    <section class="content" id="content">{''.join(case_parts)}</section>
  </div>
  <script>
    const buttons = Array.from(document.querySelectorAll('.run-link'));
    const panels = Array.from(document.querySelectorAll('.run-panel'));
    function activate(runId) {{
      buttons.forEach(b => b.classList.toggle('active', b.dataset.run === runId));
      panels.forEach(p => p.classList.toggle('active', p.dataset.run === runId));
      history.replaceState(null, '', '#' + encodeURIComponent(runId));
    }}
    function currentVisibleButton() {{
      return buttons.find(b => b.style.display !== 'none');
    }}
    buttons.forEach(button => button.addEventListener('click', () => activate(button.dataset.run)));
    const wanted = decodeURIComponent(location.hash.replace(/^#/, ''));
    const initial = buttons.find(b => b.dataset.run === wanted) || buttons[0];
    if (initial) activate(initial.dataset.run);
    function applyFilters() {{
      const q = document.getElementById('search').value.toLowerCase();
      const f = document.getElementById('filter').value;
      buttons.forEach(button => {{
        const text = button.innerText.toLowerCase();
        const ok = button.classList.contains('ok');
        const passText = !q || text.includes(q);
        const passFilter = f === 'all' || (f === 'ok' && ok) || (f === 'wrong' && !ok);
        button.style.display = passText && passFilter ? '' : 'none';
      }});
      const active = buttons.find(b => b.classList.contains('active') && b.style.display !== 'none');
      if (!active) {{
        const next = currentVisibleButton();
        if (next) activate(next.dataset.run);
      }}
    }}
    document.getElementById('search').addEventListener('input', applyFilters);
    document.getElementById('filter').addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""


def select_metrics(metrics_payload: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    items = [
        item
        for item in metrics_payload.get("per_sample", [])
        if isinstance(item, dict) and item.get("sample_id")
    ]
    if args.case:
        by_id = {str(item["sample_id"]): item for item in items}
        missing = [sample_id for sample_id in args.case if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample ids missing from metrics: {missing}")
        items = [by_id[sample_id] for sample_id in args.case]
    if args.max_cases is not None:
        items = items[: args.max_cases]
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard-metrics",
        type=Path,
        default=Path("tmp/nr3d_eval_v9_selective_mark_random100_20260514/leaderboard_metrics.json"),
    )
    parser.add_argument(
        "--per-sample-dir",
        type=Path,
        default=Path(
            "tmp/nr3d_eval_v9_selective_mark_random100_20260514/per_sample/pack_nr3d_v8_clean_initial_marked_on_demand"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/nr3d/scannet"))
    parser.add_argument("--pack-name", default="pack_nr3d_v8_clean_initial_marked_on_demand")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark/nr3d/v9_selective_mark_langsmith_trace_20260515.html"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("docs/benchmark/nr3d/assets/v9_selective_mark_langsmith_trace_20260515"),
    )
    parser.add_argument("--title", default="NR3D v9 Selective Mark Agent Trace")
    parser.add_argument("--run-id", default="v9_selective_mark_random100_20260514")
    parser.add_argument("--run-commit", default="4a1fba1-dirty-selective-mark")
    parser.add_argument(
        "--run-output",
        default="tmp/nr3d_eval_v9_selective_mark_random100_20260514/",
    )
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_payload = read_json(args.leaderboard_metrics)
    metrics = select_metrics(metrics_payload, args)
    if not metrics:
        raise ValueError("no metrics selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    html_text = build_html(
        metrics_payload=metrics_payload,
        metrics=metrics,
        data_root=args.data_root,
        per_sample_dir=args.per_sample_dir,
        pack_name=args.pack_name,
        output=args.output,
        assets_dir=args.assets_dir,
        title=args.title,
        run_id=args.run_id,
        run_commit=args.run_commit,
        run_output=args.run_output,
    )
    args.output.write_text(html_text, encoding="utf-8")
    print(f"[nr3d-langsmith-trace] wrote {args.output}")
    print(f"[nr3d-langsmith-trace] assets {args.assets_dir}")
    print(f"[nr3d-langsmith-trace] cases {len(metrics)}")


if __name__ == "__main__":
    main()
