"""Audit `select_by_text` reliability on the NR3D random100 fold.

Question
--------
When we feed the raw NR3D query directly into Stage-1
(`KeyframeSelector.select_keyframes_v2`), how often does the returned
top-K frame set contain at least one frame where the GT target object is
actually visible?

This is the *upper bound* on what `select_by_text` can contribute as the
agent's first move — independent of what Stage-2 does with the frames.

Method
------
Inputs:
  --sample-ids     frozen NR3D fold JSON (each row has
                   sample_id, scene_id, target_id, category)
  --data-root      NR3D scannet root with prepared pack samples
                   (default: data/nr3d/scannet)
  --pack-name      pack used to read the sample query
                   (default: pack_nr3d_v1)
  --k              comma-separated list of K values (default: 1,3,5)
  --max-samples    optional cap (for smoke runs)
  --output         JSON file to write the per-sample + summary report

For each sample:
  1) Read the prepared sample artifact → natural-language query
  2) Read the scene's Phase 8 visibility index (depth-aware) →
     gt_frames(scene, target_id) = {frame_id : target visible}
  3) Build (and cache per scene) a `KeyframeSelector` with stride=1
  4) Call `selector.select_keyframes_v2(query=query, k=max_k)`
  5) For each K in --k, compute:
     - hit@K        : 1 if pred_top_K ∩ gt_frames ≠ ∅
     - recall@K     : |pred_top_K ∩ gt_frames| / K
     - first_hit_rank (1-indexed) or None
     - gt_coverage = |gt_frames| / |valid_frames(scene)|  (difficulty proxy)

The script does call the LLM once per sample (Stage-1 hypothesis parse).
By default it uses gemini-2.5-pro, matching the production wiring.

Outputs
-------
JSON file with shape:

  {
    "config": {...},
    "summary": {
      "n_total": 100, "n_measurable": 97, "n_unmeasurable": 3,
      "hit_at_1": 0.42, "hit_at_3": 0.61, "hit_at_5": 0.69,
      "mean_recall_at_3": 0.31,
      "mean_first_hit_rank_when_hit_at_5": 1.7,
      ...
    },
    "samples": [
      {"sample_id": ..., "query": ..., "target_id": ...,
       "gt_frame_count": ..., "gt_coverage": ...,
       "pred_top_k": [...], "hit_at_K": {...}, "first_hit_rank": ..., ...},
      ...
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

PHASE8_VIS_REL = Path("conceptgraph/indices/visibility_index.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-ids",
        type=Path,
        default=Path(
            "tmp/nr3d_artifacts/v4_agent_guards_fair_views_random100_sample_ids.json"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/nr3d/scannet"),
    )
    parser.add_argument("--pack-name", default="pack_nr3d_v1")
    parser.add_argument(
        "--k",
        default="1,3,5",
        help="Comma-separated K values to evaluate (e.g. '1,3,5').",
    )
    parser.add_argument(
        "--llm-model",
        default="gemini-2.5-pro",
        help="LLM used for query parsing (Stage-1).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON file to write the per-sample + summary report.",
    )
    parser.add_argument(
        "--max-selector-cache-size",
        type=int,
        default=8,
        help="Per-scene KeyframeSelector cache size.",
    )
    parser.add_argument(
        "--use-visual-context",
        action="store_true",
        help="If set, pass use_visual_context=True to select_keyframes_v2 "
        "(generates BEV image for multimodal query parsing). "
        "Default False (text-only parsing, what the agent uses).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If --output already exists, keep its per-sample entries and "
        "only run missing samples.",
    )
    parser.add_argument(
        "--sort-by-scene",
        action="store_true",
        default=True,
        help="Sort samples by scene_id so selector cache hits are maximised. "
        "On by default — pass --no-sort-by-scene to keep input order.",
    )
    parser.add_argument(
        "--no-sort-by-scene",
        dest="sort_by_scene",
        action="store_false",
    )
    return parser.parse_args()


def _safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def load_visibility_index(scene_root: Path) -> dict[int, list[int]]:
    """Return {target_id: [frame_id, ...]} where target is visible (depth-aware)."""
    vis_path = scene_root / PHASE8_VIS_REL
    if not vis_path.exists():
        raise FileNotFoundError(f"Missing visibility index: {vis_path}")
    with open(vis_path, "rb") as f:
        payload = pickle.load(f)
    raw = payload.get("object_to_views") or {}
    out: dict[int, list[int]] = {}
    for key, entries in raw.items():
        out[int(key)] = [int(e[0]) for e in entries]
    return out


def load_valid_frame_ids(scene_dir: Path, pack_name: str) -> list[int]:
    """Read pack visibility.json to learn the scene's valid frame set."""
    vis_path = scene_dir / pack_name / "visibility.json"
    if not vis_path.exists():
        return []
    payload = json.loads(vis_path.read_text(encoding="utf-8"))
    return sorted(int(k) for k in payload.keys())


def load_sample_query(
    data_root: Path,
    scene_id: str,
    sample_id: str,
    pack_name: str,
) -> str:
    sample_path = (
        data_root
        / scene_id
        / pack_name
        / "samples"
        / f"{_safe_sample_id(sample_id)}.json"
    )
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample artifact: {sample_path}")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{sample_id}: missing query field")
    return query


def get_or_build_selector(
    cache: OrderedDict[str, Any],
    cache_limit: int,
    scene_id: str,
    data_root: Path,
    llm_model: str,
) -> Any:
    if scene_id in cache:
        cache.move_to_end(scene_id)
        return cache[scene_id]
    from query_scene import KeyframeSelector

    cg_root = data_root / scene_id / "conceptgraph"
    enriched = cg_root / "enriched_objects.json"
    if not enriched.exists():
        raise FileNotFoundError(
            f"Stage-1 requires enriched object metadata: {enriched}"
        )
    selector = KeyframeSelector.from_scene_path(
        str(cg_root),
        stride=1,
        llm_model=llm_model,
        prefer_lightweight_pcd=True,
        ensure_lightweight_pcd=True,
    )
    cache[scene_id] = selector
    while len(cache) > cache_limit:
        cache.popitem(last=False)
    return selector


def compute_metrics(
    pred_frames: list[int],
    gt_frames: set[int],
    k_values: list[int],
) -> dict[str, Any]:
    """Hit@K, Recall@K, first_hit_rank for one sample."""
    out: dict[str, Any] = {
        "pred_top_k": list(pred_frames),
        "hit_at_K": {},
        "recall_at_K": {},
        "first_hit_rank": None,
        "n_pred_in_gt": 0,
    }
    if not gt_frames:
        out["unmeasurable"] = True
        return out
    out["unmeasurable"] = False
    intersection_total = 0
    first_hit_rank: int | None = None
    for idx, fid in enumerate(pred_frames):
        if fid in gt_frames:
            intersection_total += 1
            if first_hit_rank is None:
                first_hit_rank = idx + 1
    out["n_pred_in_gt"] = intersection_total
    out["first_hit_rank"] = first_hit_rank
    for K in k_values:
        head = pred_frames[:K]
        hits = sum(1 for fid in head if fid in gt_frames)
        out["hit_at_K"][str(K)] = 1 if hits > 0 else 0
        out["recall_at_K"][str(K)] = (hits / K) if K > 0 else 0.0
    return out


def main() -> int:
    args = parse_args()
    k_values = sorted({int(v) for v in args.k.split(",") if v.strip()})
    if not k_values:
        raise ValueError("--k must contain at least one positive integer")
    max_k = max(k_values)

    samples_raw = json.loads(args.sample_ids.read_text(encoding="utf-8"))
    if isinstance(samples_raw, dict):
        samples_raw = samples_raw.get("sample_ids") or samples_raw.get("samples") or []
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ValueError(
            f"--sample-ids JSON must be a non-empty list: {args.sample_ids}"
        )

    if args.max_samples is not None:
        samples_raw = samples_raw[: args.max_samples]

    if args.sort_by_scene:
        samples_raw = sorted(samples_raw, key=lambda s: (s["scene_id"], s["sample_id"]))

    existing_results: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        existing_payload = json.loads(args.output.read_text(encoding="utf-8"))
        for s in existing_payload.get("samples") or []:
            sid = s.get("sample_id")
            if isinstance(sid, str):
                existing_results[sid] = s

    selector_cache: OrderedDict[str, Any] = OrderedDict()
    samples_out: list[dict[str, Any]] = []
    t_start = time.time()
    for idx, row in enumerate(samples_raw, 1):
        sample_id = row["sample_id"]
        scene_id = row["scene_id"]
        target_id = int(row["target_id"])
        category = row.get("category") or ""
        if sample_id in existing_results:
            print(
                f"[{idx}/{len(samples_raw)}] {sample_id}  (skip, resumed)", flush=True
            )
            samples_out.append(existing_results[sample_id])
            continue

        sample_out: dict[str, Any] = {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "target_id": target_id,
            "category": category,
        }
        try:
            query = load_sample_query(
                args.data_root, scene_id, sample_id, args.pack_name
            )
            sample_out["query"] = query

            visibility = load_visibility_index(args.data_root / scene_id)
            gt_frames_list = visibility.get(target_id, [])
            gt_frames: set[int] = {int(f) for f in gt_frames_list}
            valid_frames = load_valid_frame_ids(
                args.data_root / scene_id, args.pack_name
            )
            sample_out["gt_frame_count"] = len(gt_frames)
            sample_out["valid_frame_count"] = len(valid_frames)
            sample_out["gt_coverage"] = (
                len(gt_frames) / len(valid_frames) if valid_frames else None
            )

            selector = get_or_build_selector(
                selector_cache,
                args.max_selector_cache_size,
                scene_id,
                args.data_root,
                args.llm_model,
            )
            t0 = time.time()
            result = selector.select_keyframes_v2(
                query=query,
                k=max_k,
                hidden_categories=[],
                use_visual_context=bool(args.use_visual_context),
            )
            elapsed = time.time() - t0
            pred_frames = [int(v) for v in (result.keyframe_indices or [])]
            sample_out["latency_s"] = round(elapsed, 3)
            sample_out["target_term"] = getattr(result, "target_term", None)
            sample_out["anchor_term"] = getattr(result, "anchor_term", None)
            metadata = result.metadata or {}
            hyp = metadata.get("hypothesis_output") or {}
            sample_out["parse_status"] = metadata.get("status", "ok")
            sample_out["parse_mode"] = hyp.get("parse_mode")
            sample_out["hypothesis_count"] = len(hyp.get("hypotheses") or [])
            metrics = compute_metrics(pred_frames, gt_frames, k_values)
            sample_out.update(metrics)
            print(
                f"[{idx}/{len(samples_raw)}] {sample_id}  query={query!r:.60}  "
                f"gt={len(gt_frames)}/{len(valid_frames)}  "
                f"pred={pred_frames}  "
                f"hit@1={metrics['hit_at_K'].get('1', '-')}"
                f" hit@3={metrics['hit_at_K'].get('3', '-')}"
                f" hit@5={metrics['hit_at_K'].get('5', '-')}"
                f"  t={elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            sample_out["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[{idx}/{len(samples_raw)}] {sample_id}  ERROR: {sample_out['error']}",
                flush=True,
            )
        samples_out.append(sample_out)
        # checkpoint after each sample
        _write_output(args, samples_out, t_start, k_values)

    _write_output(args, samples_out, t_start, k_values, final=True)
    return 0


def _write_output(
    args: argparse.Namespace,
    samples_out: list[dict[str, Any]],
    t_start: float,
    k_values: list[int],
    *,
    final: bool = False,
) -> None:
    summary = summarize(samples_out, k_values)
    payload = {
        "config": {
            "sample_ids": str(args.sample_ids),
            "data_root": str(args.data_root),
            "pack_name": args.pack_name,
            "llm_model": args.llm_model,
            "k_values": k_values,
            "use_visual_context": bool(args.use_visual_context),
        },
        "wall_clock_s": round(time.time() - t_start, 1),
        "summary": summary,
        "samples": samples_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if final:
        print("\n=== summary ===")
        for key, val in summary.items():
            print(f"  {key} = {val}")


def summarize(samples_out: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    n_total = len(samples_out)
    n_error = sum(1 for s in samples_out if "error" in s)
    measurable = [
        s for s in samples_out if not s.get("unmeasurable", True) and "error" not in s
    ]
    n_measurable = len(measurable)
    n_unmeasurable = sum(1 for s in samples_out if s.get("unmeasurable") is True)

    out: dict[str, Any] = {
        "n_total": n_total,
        "n_measurable": n_measurable,
        "n_unmeasurable_no_gt_frames": n_unmeasurable,
        "n_error": n_error,
    }
    if not measurable:
        return out
    for K in k_values:
        hits = sum(s["hit_at_K"][str(K)] for s in measurable)
        recalls = [s["recall_at_K"][str(K)] for s in measurable]
        out[f"hit_at_{K}"] = round(hits / len(measurable), 4)
        out[f"mean_recall_at_{K}"] = round(sum(recalls) / len(recalls), 4)
    hit5 = max(k_values)
    hit5_samples = [s for s in measurable if s["hit_at_K"][str(hit5)] == 1]
    if hit5_samples:
        ranks = [s["first_hit_rank"] for s in hit5_samples if s.get("first_hit_rank")]
        out[f"mean_first_hit_rank_when_hit_at_{hit5}"] = (
            round(sum(ranks) / len(ranks), 3) if ranks else None
        )
    # difficulty stratification
    easy = [s for s in measurable if (s.get("gt_coverage") or 0) >= 0.10]
    hard = [s for s in measurable if (s.get("gt_coverage") or 0) < 0.10]
    if easy:
        for K in k_values:
            hits = sum(s["hit_at_K"][str(K)] for s in easy)
            out[f"hit_at_{K}_easy_cov>=0.10"] = round(hits / len(easy), 4)
        out["n_easy"] = len(easy)
    if hard:
        for K in k_values:
            hits = sum(s["hit_at_K"][str(K)] for s in hard)
            out[f"hit_at_{K}_hard_cov<0.10"] = round(hits / len(hard), 4)
        out["n_hard"] = len(hard)
    # mean GT coverage
    cov = [s["gt_coverage"] for s in measurable if s.get("gt_coverage") is not None]
    if cov:
        out["mean_gt_coverage"] = round(sum(cov) / len(cov), 4)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
