#!/usr/bin/env python
"""Analyze VG evaluation failures by category.

Reads a results.json from embodiedscan_vg_pilot and classifies each
IoU=0 prediction into failure categories:
  (a) bbox_3d=None — VLM produced no prediction
  (b) Wrong object — VLM predicted a bbox but IoU=0 (wrong object/location)
  (c) Scene graph missing GT — GT target category not in ConceptGraph
  (d) Keyframe miss — target object not visible in selected keyframes

Usage:
    python -m src.scripts.analyze_vg_failures <results.json> [--data-root data/embodiedscan]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmarks.embodiedscan_eval import compute_oriented_iou_3d
from benchmarks.embodiedscan_loader import EmbodiedScanDataset


def classify_failure(
    pred: dict,
    sample,
    scene_categories: set[str],
) -> str:
    """Classify a single failure into a category."""
    bbox_3d = pred.get("bbox_3d")

    if bbox_3d is None:
        return "no_prediction"

    # Check if it's all zeros
    if all(abs(v) < 1e-6 for v in bbox_3d[:6]):
        return "no_prediction"

    # Has bbox but IoU=0 — check if GT category was in scene graph
    target_lower = sample.target.lower()
    if not any(
        target_lower in cat or cat in target_lower
        for cat in scene_categories
    ):
        return "category_missing"

    return "wrong_object"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VG failures")
    parser.add_argument("results_json", type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "data" / "embodiedscan"
    )
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    predictions = data["predictions"]
    eval_args = data.get("args", {})

    # Reload the same dataset
    ds = EmbodiedScanDataset.from_path(
        args.data_root,
        split=eval_args.get("split", "val"),
        source_filter=eval_args.get("source_filter", "scannet"),
    )
    sample_map = {s.sample_id: s for s in ds}

    # Load scene categories from ConceptGraph
    scene_cat_cache: dict[str, set[str]] = {}

    def get_scene_categories(scene_name: str) -> set[str]:
        if scene_name not in scene_cat_cache:
            from query_scene.keyframe_selector import KeyframeSelector

            cg_path = args.data_root / scene_name / "conceptgraph"
            if cg_path.is_dir():
                try:
                    sel = KeyframeSelector.from_scene_path(
                        str(cg_path), llm_model="gemini-2.5-pro", stride=1
                    )
                    scene_cat_cache[scene_name] = {
                        obj.category.lower()
                        for obj in sel.objects
                        if obj.category
                    }
                except Exception:
                    scene_cat_cache[scene_name] = set()
            else:
                scene_cat_cache[scene_name] = set()
        return scene_cat_cache[scene_name]

    # Classify each prediction
    failure_counts = Counter()
    failure_details: dict[str, list[dict]] = {
        "no_prediction": [],
        "wrong_object": [],
        "category_missing": [],
        "success_025": [],
        "success_050": [],
    }

    total = 0
    for pred in predictions:
        sample_id = pred["sample_id"]
        sample = sample_map.get(sample_id)
        if sample is None:
            continue

        total += 1
        bbox_3d = pred.get("bbox_3d")
        gt_bbox = sample.gt_bbox_3d

        # Compute IoU
        if bbox_3d is not None and gt_bbox is not None:
            try:
                pred_bbox = [float(v) for v in bbox_3d[:9]]
                gt_list = [float(v) for v in gt_bbox[:9]]
                while len(pred_bbox) < 9:
                    pred_bbox.append(0.0)
                while len(gt_list) < 9:
                    gt_list.append(0.0)
                iou = compute_oriented_iou_3d(pred_bbox, gt_list)
            except (TypeError, ValueError):
                iou = 0.0
        else:
            iou = 0.0

        scene_name = sample.scan_id.split("/")[-1]
        detail = {
            "sample_id": sample_id,
            "scene": scene_name,
            "target": sample.target,
            "query": sample.query[:80],
            "iou": iou,
            "selected_id": pred.get("selected_object_id"),
            "confidence": pred.get("confidence"),
        }

        if iou >= 0.50:
            failure_counts["success_050"] += 1
            failure_details["success_050"].append(detail)
        elif iou >= 0.25:
            failure_counts["success_025"] += 1
            failure_details["success_025"].append(detail)
        else:
            scene_cats = get_scene_categories(scene_name)
            category = classify_failure(pred, sample, scene_cats)
            failure_counts[category] += 1
            failure_details[category].append(detail)

    # Print report
    print("=" * 70)
    print("EmbodiedScan VG Failure Analysis")
    print("=" * 70)
    print(f"Total samples: {total}")
    print(f"Success@0.50:      {failure_counts['success_050']:3d} ({failure_counts['success_050']/total*100:.1f}%)")
    print(f"Success@0.25:      {failure_counts['success_025']:3d} ({failure_counts['success_025']/total*100:.1f}%)")
    print()
    print("Failure breakdown (IoU < 0.25):")
    fail_total = sum(
        failure_counts[k]
        for k in ["no_prediction", "wrong_object", "category_missing"]
    )
    for cat, label in [
        ("no_prediction", "(a) No prediction (bbox_3d=None/zeros)"),
        ("wrong_object", "(b) Wrong object (bbox mismatch)"),
        ("category_missing", "(c) GT category missing from scene graph"),
    ]:
        count = failure_counts[cat]
        pct_total = count / total * 100
        pct_fail = count / max(fail_total, 1) * 100
        print(f"  {label}: {count:3d} ({pct_total:.1f}% of total, {pct_fail:.1f}% of failures)")

    # Sample details
    print()
    for cat in ["no_prediction", "wrong_object", "category_missing"]:
        details = failure_details[cat][:5]
        if details:
            print(f"\nExample {cat} failures:")
            for d in details:
                print(
                    f"  {d['sample_id']}: target={d['target']}, "
                    f"scene={d['scene']}, conf={d['confidence']}, "
                    f"query={d['query']}"
                )

    # Success examples
    successes = failure_details["success_050"] + failure_details["success_025"]
    if successes:
        print(f"\nSuccess examples ({len(successes)}):")
        for d in successes[:5]:
            print(
                f"  {d['sample_id']}: target={d['target']}, "
                f"iou={d['iou']:.3f}, obj={d['selected_id']}, "
                f"conf={d['confidence']}"
            )


if __name__ == "__main__":
    main()
