"""Static HTML renderer for VG agent execution traces.

The renderer consumes benchmark session artifacts produced by the pack-v1 VG
runners. It intentionally treats ``side_by_side.json`` / per-sample JSON files
as the source of truth, so completed runs can be inspected without re-running an
agent or starting the older trace server.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_PATH_RE = re.compile(
    r"(?P<path>(?:/|data/|tmp/|outputs/|docs/)[^\s;'\",)]+?\.(?:png|jpg|jpeg|webp))",
    re.IGNORECASE,
)


@dataclass
class TraceImage:
    """One image referenced by the trace."""

    path: Path
    role: str
    caption: str = ""
    frame_id: int | None = None


@dataclass
class TraceToolCall:
    """One tool call in the agent trace."""

    index: int
    tool_name: str
    tool_input: Any
    response_text: str
    images: list[TraceImage] = field(default_factory=list)


@dataclass
class TraceSample:
    """One visual-grounding sample with normalized trace data."""

    sample_id: str
    query: str
    status: str
    selected_object_id: int | None
    target_id: int | None
    category: str
    confidence: float | None
    iou: float | None
    is_correct: bool | None
    is_easy: bool | None = None
    is_view_dep: bool | None = None
    n_objects: int | None = None
    tokens: list[str] = field(default_factory=list)
    initial_keyframes: list[TraceImage] = field(default_factory=list)
    tool_calls: list[TraceToolCall] = field(default_factory=list)


@dataclass
class TraceSession:
    """A renderable trace session."""

    session_dir: Path
    backend: str
    pack_name: str
    samples: list[TraceSample]


def load_vg_trace_session(
    session_dir: str | Path,
    *,
    backend: str = "pack_v1",
    data_root: str | Path = "data/nr3d/scannet",
    pack_name: str | None = None,
    sample_ids: list[str] | None = None,
    num_correct: int = 2,
    num_failed: int = 2,
    iou_threshold: float = 0.5,
    project_root: str | Path | None = None,
) -> TraceSession:
    """Load a VG eval session into a frontend-friendly trace model.

    Args:
        session_dir: Directory containing ``side_by_side.json`` or
            ``per_sample/**.json``.
        backend: Backend key inside ``side_by_side.json``.
        data_root: Scene data root used to recover sample keyframe metadata.
        pack_name: Optional pack name. If omitted, inferred from
            ``session_dir/per_sample/<pack_name>``.
        sample_ids: Optional explicit sample id list. If omitted, the loader
            selects ``num_correct`` correct and ``num_failed`` failed examples.
        num_correct: Number of correct samples to auto-select.
        num_failed: Number of failed samples to auto-select.
        iou_threshold: Fallback correctness threshold when no classification
            metric or target id is available.
        project_root: Base path for resolving repo-relative image paths.
    """

    session_path = Path(session_dir)
    root = Path(project_root) if project_root is not None else Path.cwd()
    data_path = Path(data_root)
    if not data_path.is_absolute():
        data_path = root / data_path

    inferred_pack = pack_name or _infer_pack_name(session_path)
    rows = _load_side_by_side_rows(session_path, backend)
    if not rows:
        rows = _load_per_sample_rows(session_path, inferred_pack)

    metrics = _load_metrics(session_path)
    sample_files = {
        row.get("sample_id"): _load_sample_json(
            data_path,
            str(row.get("sample_id", "")),
            inferred_pack,
        )
        for row in rows
        if row.get("sample_id")
    }
    normalized = [
        _normalize_sample(
            row=row,
            metric=metrics.get(str(row.get("sample_id", "")), {}),
            sample_meta=sample_files.get(row.get("sample_id")) or {},
            session_dir=session_path,
            project_root=root,
            iou_threshold=iou_threshold,
        )
        for row in rows
    ]

    selected = _select_samples(
        normalized,
        sample_ids=sample_ids,
        num_correct=num_correct,
        num_failed=num_failed,
    )
    return TraceSession(
        session_dir=session_path,
        backend=backend,
        pack_name=inferred_pack,
        samples=selected,
    )


def render_vg_trace_html(
    trace: TraceSession,
    *,
    output_path: str | Path | None = None,
    embed_images: bool = True,
) -> str:
    """Render a VG trace session as a self-contained HTML document."""

    link_base = Path(output_path).parent if output_path is not None else Path.cwd()
    html_text = _build_html(trace, embed_images=embed_images, link_base=link_base)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html_text, encoding="utf-8")
    return html_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a VG eval session as a static LangSmith-style HTML trace."
    )
    parser.add_argument("--session-dir", required=True, help="Eval output/session dir")
    parser.add_argument("--output", help="Output HTML path")
    parser.add_argument("--backend", default="pack_v1", help="side_by_side backend key")
    parser.add_argument(
        "--data-root",
        default="data/nr3d/scannet",
        help="Scene data root used to recover initial keyframes",
    )
    parser.add_argument("--pack-name", help="Pack name, inferred when omitted")
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Explicit sample id to render; can be repeated",
    )
    parser.add_argument("--num-correct", type=int, default=2)
    parser.add_argument("--num-failed", type=int, default=2)
    parser.add_argument(
        "--link-images",
        action="store_true",
        help="Use filesystem image paths instead of embedding base64 images",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    session = Path(args.session_dir)
    output = args.output or str(session / "vg_trace.html")
    trace = load_vg_trace_session(
        session_dir=session,
        backend=args.backend,
        data_root=args.data_root,
        pack_name=args.pack_name,
        sample_ids=args.sample_ids,
        num_correct=args.num_correct,
        num_failed=args.num_failed,
    )
    render_vg_trace_html(trace, output_path=output, embed_images=not args.link_images)
    print(f"Wrote {output}")
    print(f"Rendered {len(trace.samples)} samples from {session}")
    return 0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_pack_name(session_dir: Path) -> str:
    per_sample_dir = session_dir / "per_sample"
    if per_sample_dir.exists():
        children = sorted(p for p in per_sample_dir.iterdir() if p.is_dir())
        if children:
            return children[0].name
    return ""


def _load_side_by_side_rows(session_dir: Path, backend: str) -> list[dict[str, Any]]:
    path = session_dir / "side_by_side.json"
    if not path.exists():
        return []
    payload = _load_json(path)
    if isinstance(payload, dict) and backend in payload:
        backend_payload = payload[backend]
        if isinstance(backend_payload, dict):
            rows = backend_payload.get("per_sample", [])
            return rows if isinstance(rows, list) else []
        if isinstance(backend_payload, list):
            return backend_payload
    if isinstance(payload, dict):
        rows = payload.get("per_sample", [])
        return rows if isinstance(rows, list) else []
    if isinstance(payload, list):
        return payload
    return []


def _load_per_sample_rows(session_dir: Path, pack_name: str) -> list[dict[str, Any]]:
    root = session_dir / "per_sample"
    if not root.exists():
        return []
    search_root = root / pack_name if pack_name else root
    return [_load_json(path) for path in sorted(search_root.rglob("*.json"))]


def _load_metrics(session_dir: Path) -> dict[str, dict[str, Any]]:
    path = session_dir / "leaderboard_metrics.json"
    if not path.exists():
        return {}
    payload = _load_json(path)
    rows = payload.get("per_sample", []) if isinstance(payload, dict) else []
    return {
        str(row.get("sample_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("sample_id")
    }


def _load_sample_json(
    data_root: Path,
    sample_id: str,
    pack_name: str,
) -> dict[str, Any] | None:
    if not sample_id:
        return None
    scene_id = _scene_id_from_sample(sample_id)
    safe_name = _safe_sample_name(sample_id)
    candidates: list[Path] = []
    if scene_id and pack_name:
        candidates.append(data_root / scene_id / pack_name / "samples" / f"{safe_name}.json")
    if scene_id:
        candidates.extend(
            sorted((data_root / scene_id).glob(f"*/samples/{safe_name}.json"))
        )
    for path in candidates:
        if path.exists():
            return _load_json(path)
    return None


def _normalize_sample(
    *,
    row: dict[str, Any],
    metric: dict[str, Any],
    sample_meta: dict[str, Any],
    session_dir: Path,
    project_root: Path,
    iou_threshold: float,
) -> TraceSample:
    sample_id = str(row.get("sample_id") or sample_meta.get("sample_id") or "")
    selected = _coerce_int(
        metric.get("selected_object_id", row.get("selected_object_id"))
    )
    target = _coerce_int(metric.get("target_id", sample_meta.get("target_id")))
    iou = _coerce_float(row.get("iou"))
    is_correct = _coerce_bool(metric.get("is_correct"))
    if is_correct is None and target is not None and selected is not None:
        is_correct = target == selected
    if is_correct is None and iou is not None:
        is_correct = iou >= iou_threshold

    tool_calls = _normalize_tool_trace(
        row.get("tool_trace") or [],
        session_dir=session_dir,
        project_root=project_root,
    )
    return TraceSample(
        sample_id=sample_id,
        query=str(row.get("query") or sample_meta.get("query") or ""),
        status=str(row.get("status") or ""),
        selected_object_id=selected,
        target_id=target,
        category=str(sample_meta.get("category") or metric.get("category") or ""),
        confidence=_coerce_float(row.get("confidence")),
        iou=iou,
        is_correct=is_correct,
        is_easy=_coerce_bool(metric.get("is_easy")),
        is_view_dep=_coerce_bool(metric.get("is_view_dep")),
        n_objects=_coerce_int(metric.get("n_objects")),
        tokens=list(metric.get("tokens") or []),
        initial_keyframes=_normalize_keyframes(
            sample_meta.get("keyframes") or [],
            session_dir=session_dir,
            project_root=project_root,
        ),
        tool_calls=tool_calls,
    )


def _normalize_keyframes(
    keyframes: list[dict[str, Any]],
    *,
    session_dir: Path,
    project_root: Path,
) -> list[TraceImage]:
    images: list[TraceImage] = []
    for idx, keyframe in enumerate(keyframes):
        if not isinstance(keyframe, dict):
            continue
        raw_path = keyframe.get("image_path")
        if not raw_path:
            continue
        frame_id = _coerce_int(keyframe.get("frame_id"))
        path = _resolve_path(str(raw_path), session_dir=session_dir, project_root=project_root)
        caption = f"初始 Keyframe {idx}"
        if frame_id is not None:
            caption += f" / frame_id={frame_id}"
        images.append(
            TraceImage(path=path, role="initial", caption=caption, frame_id=frame_id)
        )
    return images


def _normalize_tool_trace(
    tool_trace: list[dict[str, Any]],
    *,
    session_dir: Path,
    project_root: Path,
) -> list[TraceToolCall]:
    calls: list[TraceToolCall] = []
    for idx, entry in enumerate(tool_trace, start=1):
        if not isinstance(entry, dict):
            continue
        response_text = str(entry.get("response_text") or "")
        calls.append(
            TraceToolCall(
                index=idx,
                tool_name=str(
                    entry.get("tool_name")
                    or entry.get("name")
                    or entry.get("tool")
                    or "unknown_tool"
                ),
                tool_input=entry.get("tool_input", entry.get("input", {})),
                response_text=response_text,
                images=_extract_images_from_text(
                    response_text,
                    session_dir=session_dir,
                    project_root=project_root,
                ),
            )
        )
    return calls


def _extract_images_from_text(
    text: str,
    *,
    session_dir: Path,
    project_root: Path,
) -> list[TraceImage]:
    images: list[TraceImage] = []
    seen: set[Path] = set()
    for match in IMAGE_PATH_RE.finditer(text):
        path = _resolve_path(
            match.group("path"),
            session_dir=session_dir,
            project_root=project_root,
        )
        if path in seen:
            continue
        seen.add(path)
        frame_id = _frame_id_from_path_or_text(path, text)
        role = "marked" if "annotated" in path.parts or "marked image" in text else "image"
        caption = f"工具返回图像: {path.name}"
        if frame_id is not None:
            caption += f" / frame_id={frame_id}"
        images.append(TraceImage(path=path, role=role, caption=caption, frame_id=frame_id))
    return images


def _select_samples(
    samples: list[TraceSample],
    *,
    sample_ids: list[str] | None,
    num_correct: int,
    num_failed: int,
) -> list[TraceSample]:
    if sample_ids:
        by_id = {sample.sample_id: sample for sample in samples}
        selected = []
        missing = []
        for sample_id in sample_ids:
            sample = by_id.get(sample_id)
            if sample is None:
                missing.append(sample_id)
            else:
                selected.append(sample)
        if missing:
            raise ValueError(f"Sample ids not found in session: {missing}")
        return selected

    correct = [sample for sample in samples if sample.is_correct is True]
    failed = [sample for sample in samples if sample.is_correct is False]
    unknown = [sample for sample in samples if sample.is_correct is None]
    return correct[:num_correct] + failed[:num_failed] + unknown[: max(0, num_failed - len(failed))]


def _build_html(trace: TraceSession, *, embed_images: bool, link_base: Path) -> str:
    sample_nav = "\n".join(
        f'<a href="#sample-{idx}" class="nav-item { _status_class(sample) }">'
        f"<span>{idx + 1}. {_escape(_short_id(sample.sample_id))}</span>"
        f"<b>{_result_label(sample)}</b></a>"
        for idx, sample in enumerate(trace.samples)
    )
    samples_html = "\n".join(
        _render_sample(
            idx,
            sample,
            embed_images=embed_images,
            link_base=link_base,
        )
        for idx, sample in enumerate(trace.samples)
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VG Agent Trace 可视化</title>
  {_css()}
</head>
<body>
  <aside>
    <h1>VG Agent Trace 可视化</h1>
    <div class="meta">
      <div><b>Session</b><span>{_escape(str(trace.session_dir))}</span></div>
      <div><b>Backend</b><span>{_escape(trace.backend)}</span></div>
      <div><b>Pack</b><span>{_escape(trace.pack_name or "-")}</span></div>
      <div><b>样本数</b><span>{len(trace.samples)}</span></div>
    </div>
    <nav>{sample_nav}</nav>
  </aside>
  <main>
    <section class="intro">
      <h2>说明</h2>
      <p>本页面从一次已完成的 VG eval session 产物恢复真实 Agent 执行链。中文标题用于导航；真实任务输入、tool input 和 tool output 保持原文。</p>
    </section>
    {samples_html}
  </main>
</body>
</html>
"""


def _render_sample(
    idx: int,
    sample: TraceSample,
    *,
    embed_images: bool,
    link_base: Path,
) -> str:
    chips = [
        f"Status: {_escape(sample.status or '-')}",
        f"GT: #{_display(sample.target_id)}",
        f"预测: #{_display(sample.selected_object_id)}",
        f"IoU: {_display_float(sample.iou)}",
        f"Confidence: {_display_float(sample.confidence)}",
    ]
    if sample.category:
        chips.append(f"GT 类别: {_escape(sample.category)}")
    if sample.is_easy is not None:
        chips.append("Easy" if sample.is_easy else "Hard")
    if sample.is_view_dep is not None:
        chips.append("View-Dep" if sample.is_view_dep else "View-Indep")
    if sample.n_objects is not None:
        chips.append(f"同类候选数: {sample.n_objects}")
    chip_html = "".join(f"<span>{chip}</span>" for chip in chips)
    initial_images = _render_images(
        sample.initial_keyframes,
        embed_images=embed_images,
        link_base=link_base,
    )
    tools = "\n".join(
        _render_tool_call(
            call,
            embed_images=embed_images,
            link_base=link_base,
        )
        for call in sample.tool_calls
    )
    if not tools:
        tools = '<p class="muted">这个样本没有 tool_trace。</p>'
    token_text = " ".join(sample.tokens)
    token_html = (
        f'<div class="subtle-row"><b>NR3D tokens</b><code>{_escape(token_text)}</code></div>'
        if token_text
        else ""
    )
    return f"""<article id="sample-{idx}" class="sample {_status_class(sample)}">
  <header>
    <div>
      <p class="eyebrow">样本 {idx + 1}</p>
      <h2>{_escape(sample.sample_id)}</h2>
    </div>
    <div class="result-pill">{_result_label(sample)}</div>
  </header>
  <section class="facts">
    <h3>任务与结果</h3>
    <div class="query-block">
      <b>真实任务 Query</b>
      <p>{_escape(sample.query or "-")}</p>
    </div>
    {token_html}
    <div class="chips">{chip_html}</div>
  </section>
  <section>
    <h3>初始 Agent 可见图像</h3>
    {initial_images or '<p class="muted">未在 session 中找到初始 keyframe 图像。</p>'}
  </section>
  <section>
    <h3>Agent 执行链</h3>
    <div class="timeline">{tools}</div>
  </section>
</article>"""


def _render_tool_call(
    call: TraceToolCall,
    *,
    embed_images: bool,
    link_base: Path,
) -> str:
    images = _render_images(call.images, embed_images=embed_images, link_base=link_base)
    return f"""<div class="tool">
  <div class="tool-head">
    <span class="step">Step {call.index}</span>
    <h4>{_escape(call.tool_name)}</h4>
  </div>
  {images}
  <details open>
    <summary>真实输入</summary>
    <pre>{_escape(_format_json(call.tool_input))}</pre>
  </details>
  <details>
    <summary>真实输出</summary>
    <pre>{_escape(call.response_text)}</pre>
  </details>
</div>"""


def _render_images(
    images: list[TraceImage],
    *,
    embed_images: bool,
    link_base: Path,
) -> str:
    if not images:
        return ""
    cards = []
    for image in images:
        src = _image_src(image.path, embed_images=embed_images, link_base=link_base)
        if src is None:
            body = f'<div class="missing">图像不存在: {_escape(str(image.path))}</div>'
        else:
            body = f'<img src="{_escape(src)}" alt="{_escape(image.caption or image.path.name)}">'
        cards.append(
            f"""<figure>
  {body}
  <figcaption>{_escape(image.caption or image.path.name)}<br><code>{_escape(str(image.path))}</code></figcaption>
</figure>"""
        )
    return f'<div class="image-grid">{"".join(cards)}</div>'


def _image_src(path: Path, *, embed_images: bool, link_base: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    if not embed_images:
        return os.path.relpath(path, link_base)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _css() -> str:
    return """<style>
* { box-sizing: border-box; }
:root {
  --bg: #f7f7f4;
  --ink: #202124;
  --muted: #667085;
  --line: #d7d7d2;
  --panel: #ffffff;
  --blue: #2454a6;
  --green: #1b7f4b;
  --red: #b42318;
  --amber: #9a6700;
  --code: #101828;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
aside {
  position: fixed;
  inset: 0 auto 0 0;
  width: 320px;
  padding: 20px 16px;
  overflow: auto;
  border-right: 1px solid var(--line);
  background: #ecece7;
}
main {
  margin-left: 320px;
  padding: 24px;
  max-width: 1480px;
}
h1 { font-size: 22px; margin: 0 0 18px; }
h2 { font-size: 20px; margin: 0; overflow-wrap: anywhere; }
h3 { font-size: 16px; margin: 0 0 12px; }
h4 { font-size: 15px; margin: 0; }
.meta {
  display: grid;
  gap: 8px;
  margin-bottom: 18px;
  font-size: 13px;
}
.meta div {
  display: grid;
  gap: 2px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--line);
}
.meta span { color: var(--muted); overflow-wrap: anywhere; }
.nav-item {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 10px 8px;
  margin-bottom: 6px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  text-decoration: none;
}
.nav-item.ok b { color: var(--green); }
.nav-item.bad b { color: var(--red); }
.intro,
.sample {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  margin-bottom: 20px;
}
.sample > header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 18px;
}
.eyebrow {
  margin: 0 0 4px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
}
.result-pill {
  flex: 0 0 auto;
  padding: 6px 10px;
  border-radius: 6px;
  font-weight: 700;
  border: 1px solid var(--line);
}
.sample.ok .result-pill { color: var(--green); border-color: #8bc9a8; }
.sample.bad .result-pill { color: var(--red); border-color: #f0a8a1; }
section {
  padding: 14px 0;
  border-top: 1px solid var(--line);
}
.facts { border-top: 0; padding-top: 0; }
.query-block {
  border-left: 4px solid var(--blue);
  padding: 10px 12px;
  background: #f2f5fb;
  margin-bottom: 12px;
}
.query-block p { margin: 6px 0 0; }
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.chips span {
  display: inline-flex;
  padding: 5px 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fafafa;
  font-size: 13px;
}
.subtle-row {
  display: grid;
  gap: 6px;
  margin: 10px 0 12px;
}
.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  margin: 10px 0;
}
figure {
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: #fbfbfb;
}
figure img {
  display: block;
  width: 100%;
  height: auto;
  max-height: 680px;
  object-fit: contain;
  background: #111;
}
figcaption {
  padding: 8px;
  font-size: 12px;
  color: var(--muted);
  overflow-wrap: anywhere;
}
.timeline {
  display: grid;
  gap: 12px;
}
.tool {
  border-left: 4px solid var(--amber);
  border-top: 1px solid var(--line);
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #fffdf7;
}
.tool-head {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 8px;
}
.step {
  padding: 3px 7px;
  border-radius: 6px;
  color: white;
  background: var(--blue);
  font-size: 12px;
  font-weight: 700;
}
details {
  margin-top: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
}
summary {
  cursor: pointer;
  padding: 8px 10px;
  font-weight: 700;
}
pre, code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
pre {
  margin: 0;
  padding: 10px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: var(--code);
  background: #f5f5f5;
  border-top: 1px solid var(--line);
  max-height: 560px;
  overflow: auto;
}
.missing {
  padding: 18px;
  color: var(--red);
  background: #fff4f2;
  overflow-wrap: anywhere;
}
.muted { color: var(--muted); }
@media (max-width: 900px) {
  aside { position: static; width: auto; border-right: 0; border-bottom: 1px solid var(--line); }
  main { margin-left: 0; padding: 14px; }
  .sample > header { display: grid; }
}
</style>"""


def _format_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        return str(value)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _display(value: Any) -> str:
    return "-" if value is None else str(value)


def _display_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _result_label(sample: TraceSample) -> str:
    if sample.is_correct is True:
        return "正确"
    if sample.is_correct is False:
        return "错误"
    return "未知"


def _status_class(sample: TraceSample) -> str:
    if sample.is_correct is True:
        return "ok"
    if sample.is_correct is False:
        return "bad"
    return "unknown"


def _short_id(sample_id: str) -> str:
    return sample_id.replace("scannet/", "")


def _scene_id_from_sample(sample_id: str) -> str:
    return sample_id.split("::", 1)[0].split("/")[-1] if sample_id else ""


def _safe_sample_name(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def _resolve_path(path_text: str, *, session_dir: Path, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    for base in (project_root, session_dir):
        candidate = base / path
        if candidate.exists():
            return candidate
    return project_root / path


def _frame_id_from_path_or_text(path: Path, text: str) -> int | None:
    match = re.search(r"frame[_-](\d+)", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"frame_id=(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None
