"""Render NR3D v9 catalog-first agent traces as a LangSmith-style static HTML.

This viewer is **v9 catalog-first native**:

- The "Turn 0" panel shows what the agent actually sees on its very first
  HumanMessage — the BEV image, the SceneCatalog category table, the task,
  and a footer reminding the reader that no first-person frames are
  injected** before the agent calls a tool. (v9 build_user_message contract.)
- Tools are grouped by family: `setup`, `catalog`, `selector`, `view`,
  `reason`, `terminal`, `legacy`. Deprecated names from earlier versions still
  get labels so a historical run can still be opened, but they are visually
  flagged as "legacy".
- The viewer is built from persisted benchmark artifacts only:
  - leaderboard_metrics.json for correctness / split labels
  - per-sample checkpoints for the Stage-2 tool trace
  - pack samples for the BEV path / scene_catalog path / camera trajectory
- It never calls the agent or any model; thumbnails are extracted from
  paths referenced inside tool responses.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


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


def load_scene_catalog(sample: dict[str, Any]) -> dict[str, Any] | None:
    catalog_path = sample.get("scene_catalog_path")
    if not catalog_path:
        return None
    p = Path(catalog_path)
    if not p.exists():
        return None
    return read_json(p)


# ---------------------------------------------------------------------------
# Thumbnails + image refs
# ---------------------------------------------------------------------------


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


# Patterns that older traces embed in response_text when a tool produced an image.
_PATH_HINTS = [
    # legacy RGB view mode: raw RGB path
    re.compile(r"frame_id=(?P<fid>\d+) rgb image at (?P<path>[^;\s]+)"),
    # legacy marked / auto mode: produced annotated image
    re.compile(
        r"frame_id=(?P<fid>\d+)\s+(?:filtered\s+)?marked image at (?P<path>[^;\s]+)"
    ),
    # v9.1 mark_frame_with_bbox — same line as legacy marked, different phrase
    re.compile(r"frame_id=(?P<fid>\d+)\s+mark image at (?P<path>[^;\s]+)"),
    # view_bev — re-rendered BEV with highlight
    re.compile(r"bev image at (?P<path>[^;\s]+)"),
    # request_crops legacy
    re.compile(r"crop saved (?:at|to) (?P<path>[^;\s]+)"),
]


def collect_image_refs(
    response_text: str, tool_input: dict[str, Any] | None = None
) -> list[tuple[int | None, str]]:
    """Extract (frame_id, path) tuples for any images this tool produced."""
    refs: list[tuple[int | None, str]] = []

    # Parse JSON responses (list_scene_proposals / list_frame_proposals can
    # embed an annotated_image when filtered).
    parsed = try_json(response_text)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("annotated_image"):
                refs.append((item.get("frame_id"), str(item["annotated_image"])))
    elif isinstance(parsed, dict):
        if parsed.get("annotated_image"):
            refs.append((parsed.get("frame_id"), str(parsed["annotated_image"])))
        # view_bev (json) returns {bev_image_path: ...}
        if parsed.get("bev_image_path"):
            refs.append((None, str(parsed["bev_image_path"])))
        if parsed.get("image_path"):
            refs.append((parsed.get("frame_id"), str(parsed["image_path"])))
        # v9.1 selectors: frames[].image_path (skip null / already_seen)
        frames = parsed.get("frames")
        if isinstance(frames, list):
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                if fr.get("already_seen"):
                    continue
                ip = fr.get("image_path")
                if not ip:
                    continue
                refs.append((fr.get("frame_id"), str(ip)))

    for pat in _PATH_HINTS:
        for m in pat.finditer(response_text):
            d = m.groupdict()
            fid_str = d.get("fid")
            refs.append((int(fid_str) if fid_str else None, d["path"]))

    return refs


# ---------------------------------------------------------------------------
# Tool family / labels / summaries
# ---------------------------------------------------------------------------

_LEGACY_TOOLS = {
    "view_" + "key" + "frame_marked",
    "request_more_views",
    "switch_or_expand_hypothesis",
    "find_proposals_by_category",
    "list_" + "key" + "frames_with_proposals",
    "inspect_stage1_metadata",
}

_FAMILY = {
    # setup / housekeeping
    "list_skills": "setup",
    "load_skill": "setup",
    # catalog reads (text only, no new images)
    "list_scene_proposals": "catalog",
    "list_frame_proposals": "catalog",
    "inspect_proposal": "catalog",
    "retrieve_object_context": "catalog",
    # selector tools (frame discovery; v9.1 returns frames[] + image_path)
    "select_by_text": "selector",
    "select_by_hypothesis": "legacy",
    "select_by_frame_neighbor": "selector",
    "select_by_proposal": "selector",
    "select_by_region": "selector",
    "select_by_coverage": "selector",
    # image-producing view tools
    "view_" + "key" + "frame": "legacy",
    "mark_frame_with_bbox": "view",
    "view_bev": "view",
    "request_crops": "view",  # historically image-producing
    # analytical
    "compare_proposals_spatial": "reason",
    # terminal
    "submit_final": "final",
}

# Legacy aliases keep their pre-v9 labels but family classification.
for _legacy_tool, _family in (
    ("view_" + "key" + "frame_marked", "view"),
    ("request_more_views", "view"),
    ("find_proposals_by_category", "catalog"),
    ("list_" + "key" + "frames_with_proposals", "catalog"),
    ("inspect_stage1_metadata", "catalog"),
    ("switch_or_expand_hypothesis", "setup"),
):
    _FAMILY.setdefault(_legacy_tool, _family)


_LABELS = {
    "list_skills": "列出可用技能",
    "load_skill": "加载 playbook",
    "list_scene_proposals": "查看场景候选目录",
    "list_frame_proposals": "查看某帧候选列表",
    "inspect_proposal": "检查候选元数据",
    "retrieve_object_context": "读取对象上下文",
    "select_by_text": "按文本搜帧",
    "select_by_hypothesis": "按假设搜帧 (legacy)",
    "select_by_frame_neighbor": "按相邻帧搜帧",
    "select_by_proposal": "按候选 id 搜帧",
    "select_by_region": "按 BEV 区域搜帧",
    "select_by_coverage": "按覆盖度搜帧",
    "view_" + "key" + "frame": "查看旧式帧 (legacy)",
    "mark_frame_with_bbox": "带框标注帧 (v9.1)",
    "view_bev": "查看 BEV",
    "request_crops": "请求局部裁剪",
    "compare_proposals_spatial": "几何关系排序",
    "submit_final": "提交最终答案",
    # Legacy
    "view_" + "key" + "frame_marked": "查看带框旧式帧 (legacy)",
    "request_more_views": "请求更多视角 (legacy)",
    "switch_or_expand_hypothesis": "切换/扩展 Stage1 假设 (legacy)",
    "find_proposals_by_category": "按类别搜索候选 (legacy)",
    "list_" + "key" + "frames_with_proposals": "读取初始帧候选清单 (legacy)",
    "inspect_stage1_metadata": "检查 Stage1 元数据 (legacy)",
}


def tool_family(tool: str) -> str:
    return _FAMILY.get(tool, "tool")


def tool_label(tool: str) -> str:
    return _LABELS.get(tool, tool)


def is_legacy(tool: str) -> bool:
    return tool in _LEGACY_TOOLS


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

    # ---- catalog reads ----
    if tool == "list_scene_proposals":
        if isinstance(parsed, dict):
            cats = parsed.get("proposals_by_category") or {}
            n_props = parsed.get("n_proposals") or sum(
                len(v) for v in cats.values() if isinstance(v, list)
            )
            return f"目录读取：{n_props} 候选，{len(cats)} 个类别"
        return "读取场景候选目录"
    if tool == "list_frame_proposals" and isinstance(parsed, dict):
        ltr = parsed.get("left_to_right") or parsed.get("visible_proposal_ids") or []
        suffix = "" if len(ltr) <= 8 else f" ... +{len(ltr) - 8}"
        return (
            f"frame {parsed.get('frame_id')} 可见 {len(ltr)} 个候选："
            + "，".join(map(str, ltr[:8]))
            + suffix
        )
    if tool == "inspect_proposal" and isinstance(parsed, dict):
        frames = parsed.get("frames_appeared") or parsed.get("frames") or []
        return (
            f"候选 #{parsed.get('proposal_id')}（{parsed.get('category')}），"
            f"出现于 {len(frames)} 帧"
        )
    if tool == "retrieve_object_context":
        return "读取对象上下文：" + compact_text(text, max_chars=200)

    # ---- selectors ----
    if tool == "select_by_text":
        q = inp.get("query") if isinstance(inp, dict) else None
        k = inp.get("k") if isinstance(inp, dict) else None
        return f"文本 query=「{q}」 top-{k or '?'}：" + _summarize_selector_response(
            parsed, text
        )
    if tool == "select_by_proposal":
        ids = inp.get("proposal_ids") if isinstance(inp, dict) else None
        return f"按 proposal_ids={ids} 搜帧：" + _summarize_selector_response(
            parsed, text
        )
    if tool == "select_by_hypothesis":
        return "按假设搜帧：" + _summarize_selector_response(parsed, text)
    if tool == "select_by_frame_neighbor":
        anchor = inp.get("frame_id") if isinstance(inp, dict) else None
        return f"以 frame {anchor} 为锚搜邻帧：" + _summarize_selector_response(
            parsed, text
        )
    if tool == "select_by_region":
        return "BEV 区域搜帧：" + _summarize_selector_response(parsed, text)
    if tool == "select_by_coverage":
        return "按覆盖度搜帧：" + _summarize_selector_response(parsed, text)

    # ---- view tools ----
    if tool == ("view_" + "key" + "frame"):
        if isinstance(inp, dict):
            fid = inp.get("frame_id")
            mode = inp.get("mode") or "auto"
            cats = inp.get("categories") or []
            pids = inp.get("proposal_ids") or []
            extra = []
            if cats:
                extra.append(f"categories={cats}")
            if pids:
                extra.append(f"proposal_ids={pids}")
            extra_text = ("；" + " ".join(extra)) if extra else ""
            return f"frame {fid} ({mode}){extra_text}"
        return text[:200]
    if tool == "view_bev":
        if isinstance(inp, dict):
            hl = inp.get("highlight") or []
            return f"BEV 重渲染 highlight={hl}"
        return text[:200]
    if tool == "mark_frame_with_bbox" and isinstance(inp, dict):
        fid = inp.get("frame_id")
        labels = inp.get("labels")
        ids = inp.get("ids")
        img_base = ""
        m = re.search(r"mark image at (?P<path>[^;\s]+)", text)
        if m:
            img_base = Path(m.group("path")).name
        parts: list[str] = [f"frame {fid}"]
        if labels:
            parts.append(f"labels={labels}")
        if ids:
            parts.append(f"ids={ids}")
        if img_base:
            parts.append(f"marked_file={img_base}")
        return "带框标注 · " + " · ".join(parts)
    if tool == "request_crops" and isinstance(inp, dict):
        return (
            f"frame_indices={inp.get('frame_indices')} object_terms={inp.get('object_terms')}："
            + compact_text(text, max_chars=160)
        )

    # ---- legacy ----
    if tool == ("view_" + "key" + "frame_marked"):
        refs = collect_image_refs(text, inp if isinstance(inp, dict) else None)
        frame_id = (
            refs[0][0]
            if refs
            else (inp.get("frame_id") if isinstance(inp, dict) else "?")
        )
        return f"legacy marked frame view(frame={frame_id})"
    if tool == "request_more_views":
        return "legacy request_more_views：" + text[:160]

    # ---- analytical ----
    if tool == "compare_proposals_spatial" and isinstance(parsed, dict):
        return (
            f"relation={parsed.get('relation')} anchor=#{parsed.get('anchor_id')} "
            f"排序={parsed.get('ranked_ids')}"
        )

    # ---- terminal ----
    if tool == "submit_final" and isinstance(inp, dict):
        payload = inp.get("payload")
        rationale = str(inp.get("rationale") or "")
        return f"提交 {payload}；理由：{rationale[:240]}"

    return text[:240] if text else "（无返回文本）"


def _summarize_selector_response(parsed: Any, text: str) -> str:
    if isinstance(parsed, dict):
        n_frames = parsed.get("n_frames")
        frames = parsed.get("frames")
        if isinstance(frames, list) and frames:
            if isinstance(frames[0], dict):
                bits: list[str] = []
                for fr in frames[:8]:
                    if not isinstance(fr, dict):
                        continue
                    fid = fr.get("frame_id")
                    ip = fr.get("image_path")
                    seen = bool(fr.get("already_seen"))
                    if ip and isinstance(ip, str) and not seen:
                        bits.append(f"{fid}:{Path(ip).name}")
                    elif fid is not None:
                        bits.append(f"{fid}{'(seen)' if seen else ''}")
                tail = "" if len(frames) <= 8 else f" ... +{len(frames) - 8}"
                nf = n_frames if n_frames is not None else len(frames)
                if bits:
                    return f"返回 {nf} 帧 [{', '.join(bits)}{tail}]"
                return f"返回 {nf} 帧"
            head = ", ".join(map(str, frames[:8]))
            tail = "" if len(frames) <= 8 else f" ... +{len(frames) - 8}"
            return f"返回 {n_frames or len(frames)} 帧 [{head}{tail}]"
        frame_ids = parsed.get("frame_ids") or []
        if isinstance(frame_ids, list) and frame_ids:
            head = ", ".join(map(str, frame_ids[:8]))
            tail = "" if len(frame_ids) <= 8 else f" ... +{len(frame_ids) - 8}"
            return f"返回 {n_frames or len(frame_ids)} 帧 [{head}{tail}]"
        if n_frames is not None:
            return f"返回 {n_frames} 帧"
    return compact_text(text, max_chars=160)


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------


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
        fid_label = "BEV" if frame_id is None else f"frame {frame_id}"
        cards.append(f"""
            <figure class="image-card">
              <img src="{esc(rel)}" alt="{esc(path)}" loading="lazy">
              <figcaption>
                <strong>{esc(fid_label)}</strong>
                <span>{esc(caption)}</span>
                <code>{esc(path)}</code>
              </figcaption>
            </figure>
            """)
    if not cards:
        return '<div class="empty">这一步没有返回新图像。</div>'
    return '<div class="image-grid">' + "\n".join(cards) + "</div>"


def render_catalog_table(catalog: dict[str, Any] | None) -> str:
    if not catalog:
        return '<div class="empty">未找到 scene_catalog.json</div>'
    by_cat: dict[str, list[int]] = defaultdict(list)
    for prop in catalog.get("proposals", []):
        raw_id = prop.get("id")
        if raw_id is None:
            continue
        try:
            pid = int(raw_id)
        except (TypeError, ValueError):
            continue
        cat = prop.get("category") or "?"
        by_cat[cat].append(pid)
    if not by_cat:
        return '<div class="empty">catalog 为空</div>'
    rows = []
    for cat in sorted(by_cat):
        ids_str = ", ".join(f"#{pid}" for pid in sorted(by_cat[cat]))
        rows.append(f"<tr><td>{esc(cat)}</td><td>{esc(ids_str)}</td></tr>")
    return (
        '<table class="catalog-table"><thead><tr><th>category</th><th>proposal ids</th></tr></thead><tbody>'
        + "\n".join(rows)
        + "</tbody></table>"
    )


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
    family = tool_family(tool)
    legacy_tool_cls = " legacy-tool" if is_legacy(tool) else ""
    view_amber_cls = " view-amber" if tool == "mark_frame_with_bbox" else ""
    deprecated_badge = (
        ' <span class="badge deprecated">DEPRECATED v9.1</span>'
        if family == "legacy"
        else ""
    )
    refs = [
        (frame_id, path, f"#{step_index:02d} {tool_label(tool)}")
        for frame_id, path in collect_image_refs(
            response, inp if isinstance(inp, dict) else None
        )
    ]
    image_html = (
        render_image_cards(
            refs,
            assets_dir=assets_dir,
            html_dir=html_dir,
            prefix=f"{case_prefix}_step{step_index:02d}",
        )
        if refs
        else ""
    )
    return f"""
    <details class="call-card {esc(family)}{legacy_tool_cls}{view_amber_cls}" id="{esc(case_prefix)}-step-{step_index}" open>
      <summary>
        <span class="step-no">{step_index:02d}</span>
        <span class="tool-pill">{esc(tool)}</span>
        <span class="step-title">{esc(tool_label(tool))}{deprecated_badge}</span>
        <span class="step-summary">{esc(summarize_tool(call))}</span>
      </summary>
      <div class="call-body">
        <div class="io-grid">
          <section>
            <h4>模型发起的工具输入</h4>
            <pre>{pretty(inp, max_chars=6_000)}</pre>
          </section>
          <section>
            <h4>工具返回（原文，已截断）</h4>
            <pre>{pretty(compact_text(response, max_chars=12_000), max_chars=13_000)}</pre>
          </section>
        </div>
        {image_html}
      </div>
    </details>
    """


def render_tree(trace: list[dict[str, Any]], *, case_prefix: str) -> str:
    rows = [f"""
        <a class="tree-node turn0" href="#{esc(case_prefix)}-turn0">
          <span class="tree-dot"></span>
          <span class="tree-index">T0</span>
          <span class="tree-tool">初始 BEV + 目录</span>
        </a>
        """]
    for idx, call in enumerate(trace, start=1):
        tool = str(call.get("tool_name") or "")
        v_amb = " view-amber" if tool == "mark_frame_with_bbox" else ""
        rows.append(f"""
            <a class="tree-node {esc(tool_family(tool))}{v_amb}" href="#{esc(case_prefix)}-step-{idx}">
              <span class="tree-dot"></span>
              <span class="tree-index">{idx:02d}</span>
              <span class="tree-tool">{esc(tool_label(tool))}</span>
            </a>
            """)
    return '<div class="trace-tree">' + "\n".join(rows) + "</div>"


def status_label(metric: dict[str, Any]) -> tuple[str, str]:
    if bool(metric.get("is_correct")):
        return "正确", "ok"
    return "错误", "wrong"


# ---------------------------------------------------------------------------
# Per-case + page assembly
# ---------------------------------------------------------------------------


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
    catalog = load_scene_catalog(sample)
    trace = checkpoint.get("tool_trace") or []
    selected_id = checkpoint.get("selected_object_id")
    label_text, label_class = status_label(metric)

    # Tool family stats
    family_counts: Counter[str] = Counter()
    legacy_count = 0
    for call in trace:
        t = str(call.get("tool_name") or "?")
        family_counts[tool_family(t)] += 1
        if is_legacy(t) or tool_family(t) == "legacy":
            legacy_count += 1
    legacy_warning = (
        f'<p class="legacy-warning"><strong>注意</strong>：本 trace 含有 {legacy_count} 次已废弃工具调用，应迁移到 v9 工具表。</p>'
        if legacy_count
        else ""
    )

    # BEV initial image
    bev_path = sample.get("bev_image_path")
    bev_thumb = ""
    if bev_path and Path(bev_path).exists():
        rel = make_thumb(
            Path(bev_path),
            assets_dir=assets_dir,
            html_dir=html_dir,
            prefix=f"{case_prefix}_turn0_bev",
        )
        if rel:
            bev_thumb = f'<figure class="image-card"><img src="{esc(rel)}" alt="initial BEV" loading="lazy"><figcaption><strong>BEV (turn 0)</strong><span>mesh-based top-down render，proposal #id 标在 3D 中心，相机轨迹叠加</span><code>{esc(bev_path)}</code></figcaption></figure>'

    # Per-step cards
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

    split_bits = [
        "Easy" if metric.get("is_easy") else "Hard",
        "View-Dep" if metric.get("is_view_dep") else "View-Indep",
    ]
    selected_label = ""
    target_label = ""
    if catalog:
        for prop in catalog.get("proposals", []):
            raw_id = prop.get("id")
            if raw_id is None:
                continue
            try:
                pid = int(raw_id)
            except (TypeError, ValueError):
                continue
            if pid == int(target_id):
                target_label = str(prop.get("category") or "")
            if selected_id is not None:
                try:
                    sel_pid = int(selected_id)
                except (TypeError, ValueError):
                    sel_pid = None
                if sel_pid is not None and pid == sel_pid:
                    selected_label = str(prop.get("category") or "")

    # Sidebar nav button
    nav = f"""
    <button class="run-link {label_class}" data-run="{esc(sample_id)}">
      <span class="run-title">{esc(sample_id)}</span>
      <span class="run-query">{esc(checkpoint.get('query'))}</span>
      <span class="run-meta">{esc(label_text)} · selected=#{esc(selected_id)} · gt=#{esc(target_id)} · {esc(' / '.join(split_bits))}</span>
    </button>
    """

    family_mix = " ".join(
        f'<span class="chip {esc(fam)}">{esc(fam)}={c}</span>'
        for fam, c in family_counts.most_common()
    )

    catalog_html = render_catalog_table(catalog)
    catalog_size = len(catalog.get("proposals", [])) if catalog else 0
    total_frames = catalog.get("total_frames") if catalog else "?"

    html_case = f"""
    <article class="run-panel" data-run="{esc(sample_id)}">
      <header class="run-header">
        <div>
          <div class="eyebrow">Run {index:03d} · NR3D visual grounding (v9 catalog-first)</div>
          <h2>{esc(checkpoint.get('query'))}</h2>
          <p class="sample-id">{esc(sample_id)}</p>
        </div>
        <div class="verdict {label_class}">{esc(label_text)}</div>
      </header>

      <section class="summary-grid">
        <div><span>GT</span><strong>#{esc(target_id)} {esc(target_label)}</strong></div>
        <div><span>Selected</span><strong>#{esc(selected_id)} {esc(selected_label)}</strong></div>
        <div><span>Confidence</span><strong>{esc(checkpoint.get('confidence'))}</strong></div>
        <div><span>IoU</span><strong>{float(checkpoint.get('iou') or 0):.4f}</strong></div>
        <div><span>Split</span><strong>{esc(' / '.join(split_bits))}</strong></div>
        <div><span>Tool calls</span><strong>{len(trace)}</strong></div>
      </section>

      {legacy_warning}

      <section class="trace-layout">
        <aside>
          <h3>调用链条</h3>
          {render_tree(trace, case_prefix=case_prefix)}
          <div class="tool-mix">
            <h4>工具家族分布</h4>
            <div class="chips">{family_mix}</div>
          </div>
        </aside>
        <main>
          <section class="turn0-card" id="{esc(case_prefix)}-turn0">
            <h3>Turn 0 — Agent 看到的初始上下文（v9 catalog-first）</h3>
            <p>v9 的 <code>build_user_message</code> 只注入 <strong>BEV 图像</strong> 和 <strong>SceneCatalog 文字目录</strong>，<em>不</em> 注入任何第一人称帧。Agent 必须主动调用 selector / <code>mark_frame_with_bbox</code>（v9.1）或历史 trace 中的旧式查看工具才能拿到第一人称证据。</p>
            <div class="turn0-grid">
              <div class="turn0-bev">
                {bev_thumb or '<div class="empty">未找到 BEV 图像路径</div>'}
              </div>
              <div class="turn0-meta">
                <dl>
                  <dt>scene id</dt><dd>{esc(scene_id)}</dd>
                  <dt>total frames</dt><dd>{esc(total_frames)}</dd>
                  <dt>catalog size</dt><dd>{catalog_size} 个 proposal</dd>
                  <dt>scene_catalog_path</dt><dd><code>{esc(sample.get('scene_catalog_path'))}</code></dd>
                  <dt>bev_image_path</dt><dd><code>{esc(bev_path)}</code></dd>
                  <dt>query</dt><dd>{esc(checkpoint.get('query'))}</dd>
                </dl>
              </div>
            </div>
            <details class="catalog-block" open>
              <summary>SceneCatalog (`Proposals by category` 块的展开形式)</summary>
              {catalog_html}
            </details>
          </section>
          <section class="call-stack">
            <h3>Agent 调用栈（每个工具一张卡片）</h3>
            <p>卡片按真实工具调用顺序展开。<code>tool_input</code> 与 <code>response_text</code> 保留原文（已做长度截断），其它说明为中文。工具家族用左侧色条区分：<span class="legend setup">setup</span> <span class="legend catalog">catalog</span> <span class="legend selector">selector</span> <span class="legend view">view</span> <span class="legend view-amber">mark bbox</span> <span class="legend legacy">legacy</span> <span class="legend reason">reason</span> <span class="legend final">final</span>。</p>
            {cards}
          </section>
        </main>
      </section>
    </article>
    """
    return nav, html_case


def select_metrics(
    metrics_payload: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
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
      --purple: #6c3ab0;
      --teal: #0d7188;
      --code: #111827;

      --c-setup: #7b61ff;
      --c-catalog: #2457a6;
      --c-selector: #0d7188;
      --c-view: #b54708;
      --c-reason: #0f7a4d;
      --c-final: #b42318;
      --c-turn0: #4b5563;
      --c-legacy: #94a3b8;
      --c-view-amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", sans-serif; line-height: 1.45; }}
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
    .legacy-warning {{ background: #fff7ed; border: 1px solid #fbd9a4; color: #7a3d09; border-radius: 8px; padding: 10px 14px; margin: 0 0 14px; }}
    .trace-layout {{ display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 14px; align-items: start; }}
    .trace-layout > aside {{ position: sticky; top: 18px; max-height: calc(100vh - 36px); overflow: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .trace-layout > main {{ display: grid; gap: 14px; }}
    .turn0-card, .call-stack {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .turn0-card p, .call-stack p {{ color: var(--muted); margin-top: 0; }}
    .turn0-card .empty {{ color: var(--muted); }}
    .turn0-grid {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(220px, 1fr); gap: 14px; align-items: start; }}
    .turn0-bev figure {{ margin: 0; }}
    .turn0-bev .image-card {{ max-width: 100%; }}
    .turn0-meta dl {{ margin: 0; }}
    .catalog-block {{ margin-top: 12px; border-top: 1px solid var(--line); padding-top: 12px; }}
    .catalog-block summary {{ cursor: pointer; font-weight: 650; padding: 4px 0; }}
    .catalog-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    .catalog-table th, .catalog-table td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; font-size: 13px; vertical-align: top; text-align: left; }}
    .catalog-table th {{ background: #f0f4f8; color: var(--muted); font-weight: 650; }}
    .catalog-table td:first-child {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    dl {{ display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 6px 12px; margin: 10px 0; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    .trace-tree {{ position: relative; display: grid; gap: 6px; padding-left: 8px; }}
    .trace-tree:before {{ content: ""; position: absolute; top: 8px; bottom: 8px; left: 15px; width: 2px; background: var(--line); }}
    .tree-node {{ position: relative; z-index: 1; display: grid; grid-template-columns: 16px 32px 1fr; gap: 8px; align-items: center; min-height: 28px; color: var(--ink); text-decoration: none; border: 1px solid var(--line); background: var(--panel-2); border-radius: 7px; padding: 7px 8px; }}
    .tree-node.turn0 {{ background: #eef2f8; border-color: var(--line-strong); font-weight: 650; }}
    .tree-node:hover {{ border-color: var(--blue); }}
    .tree-dot {{ width: 10px; height: 10px; border-radius: 50%; background: var(--blue); border: 2px solid white; box-shadow: 0 0 0 1px var(--line-strong); }}
    .tree-node.turn0 .tree-dot {{ background: var(--c-turn0); }}
    .tree-node.setup .tree-dot {{ background: var(--c-setup); }}
    .tree-node.catalog .tree-dot {{ background: var(--c-catalog); }}
    .tree-node.selector .tree-dot {{ background: var(--c-selector); }}
    .tree-node.view .tree-dot {{ background: var(--c-view); }}
    .tree-node.view-amber .tree-dot {{ background: var(--c-view-amber); }}
    .tree-node.legacy .tree-dot {{ background: var(--c-legacy); }}
    .tree-node.reason .tree-dot {{ background: var(--c-reason); }}
    .tree-node.final .tree-dot {{ background: var(--c-final); }}
    .tree-index {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .tree-tool {{ font-size: 13px; }}
    .tool-mix {{ margin-top: 16px; }}
    .tool-mix h4 {{ margin-top: 0; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .chip {{ display: inline-block; background: #edf2f7; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; font-size: 12px; color: var(--ink); }}
    .chip.setup {{ background: #efeaff; border-color: #d5c7ff; color: var(--c-setup); }}
    .chip.catalog {{ background: #e6eef9; border-color: #c4d5ed; color: var(--c-catalog); }}
    .chip.selector {{ background: #def0f1; border-color: #b8dde0; color: var(--c-selector); }}
    .chip.view {{ background: #fdecd2; border-color: #f3cea5; color: var(--c-view); }}
    .chip.view-amber {{ background: #fff7eb; border-color: #f8d9a8; color: var(--c-view-amber); }}
    .chip.legacy {{ background: #e8edf3; border-color: #c5ccd6; color: var(--c-legacy); }}
    .chip.reason {{ background: #def1e5; border-color: #b9d7c6; color: var(--c-reason); }}
    .chip.final {{ background: #ffe1dc; border-color: #f3b9b1; color: var(--c-final); }}
    .call-card {{ border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; background: var(--panel-2); scroll-margin-top: 16px; }}
    .call-card summary {{ cursor: pointer; list-style: none; display: grid; grid-template-columns: 42px minmax(120px, 180px) minmax(130px, 220px) minmax(0, 1fr); gap: 10px; align-items: baseline; padding: 12px; }}
    .call-card summary::-webkit-details-marker {{ display: none; }}
    .call-card.setup {{ border-left: 4px solid var(--c-setup); }}
    .call-card.catalog {{ border-left: 4px solid var(--c-catalog); }}
    .call-card.selector {{ border-left: 4px solid var(--c-selector); }}
    .call-card.view {{ border-left: 4px solid var(--c-view); }}
    .call-card.view.view-amber {{ border-left-color: var(--c-view-amber); }}
    .call-card.legacy {{ border-left: 4px solid var(--c-legacy); outline: 2px dashed #f59e0b; outline-offset: -2px; opacity: 0.85; }}
    .call-card.reason {{ border-left: 4px solid var(--c-reason); }}
    .call-card.final {{ border-left: 4px solid var(--c-final); }}
    .call-card.legacy-tool {{ outline: 2px dashed #fbbf24; outline-offset: -2px; }}
    .step-no {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--muted); font-weight: 700; }}
    .tool-pill {{ background: #e8eef7; border: 1px solid #d2dceb; color: #193b6d; border-radius: 999px; padding: 3px 8px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .step-title {{ font-weight: 750; }}
    .badge.deprecated {{
      background: #f59e0b;
      color: #1f2937;
      padding: 2px 6px;
      font-size: 11px;
      border-radius: 4px;
      margin-left: 6px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .step-summary {{ color: var(--muted); overflow-wrap: anywhere; }}
    .call-body {{ border-top: 1px solid var(--line); padding: 12px; }}
    .io-grid {{ display: grid; grid-template-columns: minmax(260px, .9fr) minmax(320px, 1.3fr); gap: 12px; }}
    pre {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 420px; overflow: auto; background: var(--code); color: #eef6ff; border-radius: 7px; padding: 12px; font-size: 12px; line-height: 1.45; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 12px; }}
    .image-card {{ margin: 0; background: white; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .image-card img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 9px; color: var(--muted); font-size: 12px; }}
    figcaption strong, figcaption span, figcaption code {{ display: block; }}
    .empty {{ color: var(--muted); background: #f2f4f7; border: 1px dashed var(--line-strong); border-radius: 7px; padding: 10px; }}
    .legend {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; color: white; margin: 0 2px; }}
    .legend.setup {{ background: var(--c-setup); }}
    .legend.catalog {{ background: var(--c-catalog); }}
    .legend.selector {{ background: var(--c-selector); }}
    .legend.view {{ background: var(--c-view); }}
    .legend.view-amber {{ background: var(--c-view-amber); color: #1f2937; }}
    .legend.legacy {{ background: var(--c-legacy); }}
    .legend.reason {{ background: var(--c-reason); }}
    .legend.final {{ background: var(--c-final); }}
    @media (max-width: 1120px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: relative; height: auto; }}
      .trace-layout {{ grid-template-columns: 1fr; }}
      .trace-layout > aside {{ position: relative; top: 0; max-height: none; }}
      .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
      .io-grid {{ grid-template-columns: 1fr; }}
      .call-card summary {{ grid-template-columns: 40px 1fr; }}
      .step-summary {{ grid-column: 1 / -1; }}
      .turn0-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <h1>{esc(title)}</h1>
      <p>v9 catalog-first 静态查看器。<strong>初始上下文只有 BEV + Cat-B 文字目录</strong>，第一人称帧均由 agent 主动调用 selector、<code>mark_frame_with_bbox</code>（v9.1）或历史旧式查看工具获取。</p>
      <p>Run: <code>{esc(run_id)}</code></p>
      <p>Run commit: <code>{esc(run_commit)}</code> · Report commit: <code>{esc(git_short())}</code></p>
      <p>Artifacts: <code>{esc(run_output)}</code></p>
      <div class="metrics">
        <div class="metric"><span>样本数</span><strong>{esc(n)}</strong></div>
        <div class="metric"><span>Overall</span><strong>{overall * 100:.2f}</strong></div>
        <div class="metric"><span>Easy / Hard</span><strong>{easy * 100:.1f} / {hard * 100:.1f}</strong></div>
        <div class="metric"><span>VDep / VInd</span><strong>{view_dep * 100:.1f} / {view_indep * 100:.1f}</strong></div>
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboard-metrics",
        type=Path,
        default=Path("tmp/nr3d_eval_v9_full_20260515_1401/leaderboard_metrics.json"),
    )
    parser.add_argument(
        "--per-sample-dir",
        type=Path,
        default=Path(
            "tmp/nr3d_eval_v9_full_20260515_1401/per_sample/pack_nr3d_v9_catalog_first"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/nr3d/scannet"))
    parser.add_argument("--pack-name", default="pack_nr3d_v9_catalog_first")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/benchmark/nr3d/v9_catalog_first_langsmith_trace_20260515.html"
        ),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(
            "docs/benchmark/nr3d/assets/v9_catalog_first_langsmith_trace_20260515"
        ),
    )
    parser.add_argument("--title", default="NR3D v9 Catalog-First Agent Trace")
    parser.add_argument("--run-id", default="v9_full_20260515_1401")
    parser.add_argument("--run-commit", default="e0ab061")
    parser.add_argument(
        "--run-output",
        default="tmp/nr3d_eval_v9_full_20260515_1401/",
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
    print(f"[nr3d-trace-v9] wrote {args.output}")
    print(f"[nr3d-trace-v9] assets {args.assets_dir}")
    print(f"[nr3d-trace-v9] cases {len(metrics)}")


if __name__ == "__main__":
    main()
