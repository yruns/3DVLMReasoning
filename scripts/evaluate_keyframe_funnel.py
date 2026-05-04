"""ScanRefer keyframe-selection funnel evaluator.

For each sample in a frozen fold (e.g. random100), measures the
calibration funnel of the Stage 1 -> Stage 2 keyframe pipeline:

    100 samples
    └── F1: Phase 8 has target_id visible somewhere
        └── F2: pack-prep initial KFs cover at least one frame where the
                Phase 8 GT target_id is visible
            └── F3: pack-prep initial KFs cover at least one Mask3D
                    candidate whose label fuzzy-matches the GT category
                └── F4: cumulative KFs (initial + view_keyframe_marked +
                        callback-added) cover Phase 8 target_id
                    └── F5: agent submitted a proposal with IoU >= 0.25
                            (Phase 8 GT bbox, the same eval the pack
                             ran under)

The funnel pinpoints where samples fall off:

- F1 fail: scene's Phase 8 visibility never sees the target -> data issue.
- F2 fail: pack-prep KF picker missed the target frame -> Stage 1 KF logic.
- F3 fail: same-category Mask3D candidate is absent from initial KFs ->
  proposal-pool / Mask3D-density / annotated-PNG mismatch.
- F4 fail: Stage 2 callbacks failed to recover -> ReAct loop ineffective.
- F5 fail: the agent saw the right frames but picked the wrong proposal
  -> picking error.

Inputs:
  --sample-ids     frozen fold JSON (random100_sample_ids.json)
  --pack-name      pack used at runtime (pack_scanrefer_v3_iterative,
                   pack_scanrefer_v3p_iterative, ...)
  --data-root      ScanRefer scannet root with the pack samples
  --phase8-data-root NR3D scannet root with Phase 8 visibility indices
  --per-sample-dir directory of Stage 2 per-sample checkpoints with
                   tool_trace, e.g.
                   tmp/v3p3_vertical_spatial_random100_eval/per_sample/<pack>/
  --output         JSON file to write the per-sample + summary report

The script does NOT call any LLM; everything is deterministic from
disk artifacts and the Phase 8 visibility / Mask3D-CG indices.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

VIS_REL = Path("conceptgraph/indices/visibility_index.pkl")
MASK3D_PCD_REL = Path("conceptgraph/pcd_saves/full_pcd_mask3d_axisaligned.pkl.gz")

_NEW_VIEW_IDS_RE = re.compile(r"New view IDs: \[([\d,\s]+)\]")


def _safe_sample_id(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("::", "__")


def _digest8(sample_id: str) -> str:
    """Match how the runner names its checkpoint files."""
    safe = _safe_sample_id(sample_id)
    return hashlib.sha1(safe.encode("utf-8")).hexdigest()[:12]


def _category_tokens(text: str) -> set[str]:
    raw = (text or "").replace("_", " ").lower().strip()
    if not raw:
        return set()
    stop = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "to"}
    return {t for t in raw.split() if t and t not in stop}


def _category_match(label: str, target_categories: Iterable[str]) -> bool:
    label_l = (label or "").replace("_", " ").lower().strip()
    if not label_l:
        return False
    label_toks = _category_tokens(label)
    for cat in target_categories:
        if not cat:
            continue
        cat_l = cat.replace("_", " ").lower().strip()
        if not cat_l or cat_l in {"unknow", "unknown"}:
            continue
        if label_l == cat_l:
            return True
        if label_toks & _category_tokens(cat):
            return True
    return False


def _load_phase8_visibility(scene_root: Path) -> dict[int, list[tuple[int, float]]]:
    p = scene_root / VIS_REL
    if not p.exists():
        raise FileNotFoundError(f"Phase 8 visibility missing: {p}")
    with open(p, "rb") as f:
        payload = pickle.load(f)
    raw = payload.get("object_to_views") or {}
    return {int(k): [(int(e[0]), float(e[1])) for e in v] for k, v in raw.items()}


def _load_mask3d_pool(scene_root: Path) -> tuple[list[dict], dict[int, list[int]]]:
    p = scene_root / MASK3D_PCD_REL
    if not p.exists():
        raise FileNotFoundError(f"Mask3D pkl missing: {p}")
    with gzip.open(p, "rb") as f:
        payload = pickle.load(f)
    raw_objects = payload["objects"]
    proposals: list[dict] = []
    for obj_id, obj in enumerate(raw_objects):
        names = obj.get("class_name") or []
        if not names:
            continue
        label = Counter(names).most_common(1)[0][0]
        proposals.append({"id": obj_id, "label": label})

    vis_p = scene_root / VIS_REL
    if not vis_p.exists():
        raise FileNotFoundError(f"Mask3D visibility missing: {vis_p}")
    return proposals, {}


def _load_mask3d_visibility_view_to_objects(
    scene_root: Path,
) -> dict[int, list[tuple[int, float]]]:
    """The pack-prep writes Mask3D-aligned visibility under
    data/scanrefer/scannet/<scene>/conceptgraph/indices/visibility_index.pkl."""
    p = scene_root / VIS_REL
    if not p.exists():
        raise FileNotFoundError(f"Mask3D visibility missing: {p}")
    with open(p, "rb") as f:
        payload = pickle.load(f)
    raw = payload.get("view_to_objects") or {}
    return {int(k): [(int(e[0]), float(e[1])) for e in v] for k, v in raw.items()}


def _bbox_iou_axis_aligned(a: list[float], b: list[float]) -> float:
    """3D IoU between two 9-DOF bboxes treated as axis-aligned (cx,cy,cz,dx,dy,dz,…)."""
    ac = np.asarray(a[:3], dtype=float)
    ad = np.asarray(a[3:6], dtype=float)
    bc = np.asarray(b[:3], dtype=float)
    bd = np.asarray(b[3:6], dtype=float)
    a_min, a_max = ac - ad / 2.0, ac + ad / 2.0
    b_min, b_max = bc - bd / 2.0, bc + bd / 2.0
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(0.0, inter_max - inter_min).prod()
    vol_a = float(np.prod(ad))
    vol_b = float(np.prod(bd))
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


def _extract_cumulative_frame_ids(
    initial_frame_ids: list[int], tool_trace: list[dict]
) -> tuple[list[int], dict[str, int]]:
    """Return (sorted_unique_frame_ids, breakdown_by_source)."""
    seen: set[int] = set(initial_frame_ids)
    by_source = {
        "initial": len(seen),
        "view_keyframe_marked": 0,
        "request_more_views": 0,
        "switch_or_expand_hypothesis": 0,
    }
    for t in tool_trace or []:
        name = t.get("tool_name")
        if name == "view_keyframe_marked":
            inp = t.get("tool_input") or {}
            fid = inp.get("frame_id")
            if isinstance(fid, int) and fid not in seen:
                seen.add(fid)
                by_source["view_keyframe_marked"] += 1
        elif name in ("request_more_views", "switch_or_expand_hypothesis"):
            resp = t.get("response_text") or ""
            for m in _NEW_VIEW_IDS_RE.finditer(resp):
                ids_str = m.group(1)
                for chunk in ids_str.split(","):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    try:
                        fid = int(chunk)
                    except ValueError:
                        continue
                    if fid not in seen:
                        seen.add(fid)
                        by_source[name] += 1
    return sorted(seen), by_source


def _read_pack_sample(data_root: Path, pack_name: str, sample_id: str) -> dict:
    parts = sample_id.split("/")
    scene_id = parts[-1].split("::")[0]
    safe = _safe_sample_id(sample_id)
    p = data_root / scene_id / pack_name / "samples" / f"{safe}.json"
    if not p.exists():
        raise FileNotFoundError(f"pack sample missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _read_per_sample_checkpoint(
    per_sample_dir: Path, sample_id: str
) -> dict | None:
    safe = _safe_sample_id(sample_id)
    digest = _digest8(sample_id)
    candidate = per_sample_dir / f"{safe}_{digest}.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    # fall back: any file starting with the safe prefix
    matches = sorted(per_sample_dir.glob(f"{safe}_*.json"))
    if matches:
        return json.loads(matches[0].read_text(encoding="utf-8"))
    return None


def evaluate(
    *,
    sample_ids_path: Path,
    pack_name: str,
    data_root: Path,
    phase8_data_root: Path,
    per_sample_dir: Path,
    output_path: Path,
) -> dict:
    fold = json.loads(sample_ids_path.read_text(encoding="utf-8"))
    if not isinstance(fold, list):
        raise ValueError(f"fold JSON must be a list: {sample_ids_path}")

    # cache scene-level data
    phase8_obj_to_views: dict[str, dict[int, list[tuple[int, float]]]] = {}
    mask3d_proposals: dict[str, list[dict]] = {}
    mask3d_view_to_objects: dict[
        str, dict[int, list[tuple[int, float]]]
    ] = {}

    per_sample: list[dict] = []
    for row in fold:
        sample_id = row["sample_id"]
        scene_id = row["scene_id"]
        target_id = int(row["target_id"])
        target_category = str(row.get("category") or "")

        # Phase 8 visibility for the GT target object
        if scene_id not in phase8_obj_to_views:
            phase8_obj_to_views[scene_id] = _load_phase8_visibility(
                phase8_data_root / scene_id
            )
        phase8_views_for_target = phase8_obj_to_views[scene_id].get(target_id, [])
        phase8_visible_frames = {f for f, _ in phase8_views_for_target}
        f1_phase8_has_visible = bool(phase8_visible_frames)

        # Pack-prep sample artifact -> initial keyframes
        pack_sample = _read_pack_sample(data_root, pack_name, sample_id)
        initial_keyframes = pack_sample.get("keyframes") or []
        initial_frame_ids = [int(k["frame_id"]) for k in initial_keyframes]

        # F2: any initial KF visible in Phase 8 for the target id
        f2_initial_kf_covers_phase8_target = bool(
            phase8_visible_frames & set(initial_frame_ids)
        )

        # Mask3D pool / visibility for category-level coverage
        if scene_id not in mask3d_proposals:
            proposals, _ = _load_mask3d_pool(
                data_root / scene_id  # scanrefer scannet root has the Mask3D pkl
            )
            mask3d_proposals[scene_id] = proposals
            mask3d_view_to_objects[scene_id] = (
                _load_mask3d_visibility_view_to_objects(data_root / scene_id)
            )
        proposals = mask3d_proposals[scene_id]
        view_to_objects = mask3d_view_to_objects[scene_id]
        target_cat_tokens = _category_tokens(target_category)
        mask3d_pool_has_match = any(
            _category_match(p["label"], [target_category]) for p in proposals
        )

        # F3: at least one initial KF carries a Mask3D candidate of GT category
        labels_by_id = {p["id"]: p["label"] for p in proposals}
        f3_initial_kf_carries_mask3d_match = False
        for fid in initial_frame_ids:
            entries = view_to_objects.get(fid, [])
            for oid, _w in entries:
                if _category_match(labels_by_id.get(int(oid), ""), [target_category]):
                    f3_initial_kf_carries_mask3d_match = True
                    break
            if f3_initial_kf_carries_mask3d_match:
                break

        # Cumulative keyframes after Stage 2 callbacks
        per_chk = _read_per_sample_checkpoint(per_sample_dir, sample_id)
        tool_trace = (per_chk or {}).get("tool_trace") or []
        cumulative_frame_ids, kf_breakdown = _extract_cumulative_frame_ids(
            initial_frame_ids, tool_trace
        )
        f4_cumulative_covers_phase8_target = bool(
            phase8_visible_frames & set(cumulative_frame_ids)
        )

        # Cumulative same-category Mask3D coverage (whether the agent
        # ever had a same-category mark on screen).
        cumulative_mask3d_match = False
        for fid in cumulative_frame_ids:
            for oid, _w in view_to_objects.get(fid, []):
                if _category_match(labels_by_id.get(int(oid), ""), [target_category]):
                    cumulative_mask3d_match = True
                    break
            if cumulative_mask3d_match:
                break

        # F5: agent picked correctly under whatever evaluator the per-sample
        # was scored with (Phase 8 GT-CG; the same eval the pack used).
        agent_iou = float((per_chk or {}).get("iou", 0.0)) if per_chk else 0.0
        f5_agent_correct_25 = agent_iou >= 0.25
        f5_agent_correct_50 = agent_iou >= 0.50

        per_sample.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "target_id": target_id,
                "target_category": target_category,
                "is_unique": pack_sample.get("is_unique"),
                # raw inputs to the funnel
                "n_phase8_visible_frames": len(phase8_visible_frames),
                "initial_frame_ids": initial_frame_ids,
                "n_cumulative_frame_ids": len(cumulative_frame_ids),
                "kf_breakdown": kf_breakdown,
                "mask3d_pool_size": len(proposals),
                "mask3d_pool_has_category_match": mask3d_pool_has_match,
                "agent_iou": agent_iou,
                "agent_selected_object_id": (
                    (per_chk or {}).get("selected_object_id") if per_chk else None
                ),
                # boolean funnel checkpoints
                "F1_phase8_has_visible_frames": f1_phase8_has_visible,
                "F2_initial_kf_covers_phase8_target": f2_initial_kf_covers_phase8_target,
                "F3_initial_kf_carries_mask3d_category_match": (
                    f3_initial_kf_carries_mask3d_match
                ),
                "F4_cumulative_kf_covers_phase8_target": (
                    f4_cumulative_covers_phase8_target
                ),
                "F4b_cumulative_mask3d_category_match": cumulative_mask3d_match,
                "F5a_agent_iou_ge_025": f5_agent_correct_25,
                "F5b_agent_iou_ge_050": f5_agent_correct_50,
                "checkpoint_present": per_chk is not None,
            }
        )

    n = len(per_sample)
    summary = {
        "n_total": n,
        "n_F1_phase8_has_visible_frames": sum(
            r["F1_phase8_has_visible_frames"] for r in per_sample
        ),
        "n_F2_initial_kf_covers_phase8_target": sum(
            r["F2_initial_kf_covers_phase8_target"] for r in per_sample
        ),
        "n_F3_initial_kf_carries_mask3d_category_match": sum(
            r["F3_initial_kf_carries_mask3d_category_match"] for r in per_sample
        ),
        "n_F4_cumulative_kf_covers_phase8_target": sum(
            r["F4_cumulative_kf_covers_phase8_target"] for r in per_sample
        ),
        "n_F4b_cumulative_mask3d_category_match": sum(
            r["F4b_cumulative_mask3d_category_match"] for r in per_sample
        ),
        "n_F5a_agent_iou_ge_025": sum(r["F5a_agent_iou_ge_025"] for r in per_sample),
        "n_F5b_agent_iou_ge_050": sum(r["F5b_agent_iou_ge_050"] for r in per_sample),
        "n_mask3d_pool_has_category_match": sum(
            r["mask3d_pool_has_category_match"] for r in per_sample
        ),
        "n_checkpoint_present": sum(r["checkpoint_present"] for r in per_sample),
    }
    summary["pct"] = {k: f"{(v / n) * 100:.1f}%" for k, v in summary.items() if k.startswith("n_")}

    output = {
        "fold": str(sample_ids_path),
        "pack_name": pack_name,
        "data_root": str(data_root),
        "phase8_data_root": str(phase8_data_root),
        "per_sample_dir": str(per_sample_dir),
        "summary": summary,
        "per_sample": per_sample,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def _print_summary(report: dict) -> None:
    s = report["summary"]
    n = s["n_total"]
    print(f"\nFunnel report (n={n}, pack={report['pack_name']})")
    print(f"  per_sample_dir: {report['per_sample_dir']}")
    rows = [
        ("F1: Phase 8 has visible frames for target_id", s["n_F1_phase8_has_visible_frames"]),
        ("F2: initial KFs cover Phase 8 target frame", s["n_F2_initial_kf_covers_phase8_target"]),
        ("F3: initial KFs carry Mask3D candidate of GT category", s["n_F3_initial_kf_carries_mask3d_category_match"]),
        ("F4: cumulative KFs cover Phase 8 target frame", s["n_F4_cumulative_kf_covers_phase8_target"]),
        ("F4b: cumulative KFs carry Mask3D candidate of GT category", s["n_F4b_cumulative_mask3d_category_match"]),
        ("F5a: agent IoU >= 0.25 (Phase 8 eval)", s["n_F5a_agent_iou_ge_025"]),
        ("F5b: agent IoU >= 0.50 (Phase 8 eval)", s["n_F5b_agent_iou_ge_050"]),
        ("(ref) Mask3D pool has same-category candidate", s["n_mask3d_pool_has_category_match"]),
        ("(ref) per-sample checkpoint present", s["n_checkpoint_present"]),
    ]
    for label, k in rows:
        print(f"  {label:<60s} {k:3d} / {n:<3d} ({k / n * 100:5.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-ids", required=True, type=Path)
    p.add_argument(
        "--pack-name",
        required=True,
        help="Pack used at runtime, e.g. pack_scanrefer_v3p_iterative.",
    )
    p.add_argument("--data-root", default=Path("data/scanrefer/scannet"), type=Path)
    p.add_argument(
        "--phase8-data-root", default=Path("data/nr3d/scannet"), type=Path
    )
    p.add_argument(
        "--per-sample-dir",
        required=True,
        type=Path,
        help="Directory of per-sample checkpoints, e.g. "
        "tmp/<run>/per_sample/<pack-name>/",
    )
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = evaluate(
        sample_ids_path=args.sample_ids,
        pack_name=args.pack_name,
        data_root=args.data_root,
        phase8_data_root=args.phase8_data_root,
        per_sample_dir=args.per_sample_dir,
        output_path=args.output,
    )
    _print_summary(report)


if __name__ == "__main__":
    main()


__all__ = [
    "evaluate",
    "_extract_cumulative_frame_ids",
    "_category_match",
    "_category_tokens",
]
