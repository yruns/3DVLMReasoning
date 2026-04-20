#!/usr/bin/env python3
"""v15 vs v14 apples-to-apples MNAS analysis.

Usage:
    python scripts/analyze_v15_results.py                     # table + categories
    python scripts/analyze_v15_results.py --low-score CFG     # dump N=1 failures of CFG
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL_ROOTS = {
    "v14_rerun":   REPO / "tmp/openeqa_eval_v14_rerun",
    "v15_s1_l1":   REPO / "tmp/openeqa_eval_v15_s1_l1",
    "v15_s1_l2":   REPO / "tmp/openeqa_eval_v15_s1_l2",
    "v15_s1s2_l1": REPO / "tmp/openeqa_eval_v15_s1s2_l1",
    "v15_s1s2_l2": REPO / "tmp/openeqa_eval_v15_s1s2_l2",
}
FROZEN_QUESTIONS = Path("/tmp/v15_frozen_questions.json")


def load_metrics(root: Path, which: str = "stage2") -> dict[str, int]:
    fp = root / f"official_predictions_{which}-metrics.json"
    if not fp.exists():
        return {}
    return json.load(open(fp))


def load_selected() -> list[dict]:
    return json.load(open(FROZEN_QUESTIONS)) if FROZEN_QUESTIONS.exists() else []


def mnas(scores: list[int]) -> float:
    if not scores:
        return 0.0
    arr = np.array(scores)
    return float(np.mean(100.0 * (np.clip(arr, 1, 5) - 1) / 4))


def aggregate(which: str = "stage2") -> dict[str, dict[str, int]]:
    return {name: load_metrics(root, which) for name, root in EVAL_ROOTS.items()}


def categories_map() -> dict[str, str]:
    samples = load_selected()
    return {s["question_id"]: s.get("category", "?") for s in samples}


def table_overall():
    all_s2 = aggregate("stage2")
    all_e2e = aggregate("e2e")
    print(f"\n{'column':<14} {'n_s2':>5} {'MNAS_s2':>8} {'n_e2e':>6} {'MNAS_e2e':>9}")
    print("-" * 50)
    for name in EVAL_ROOTS:
        s2 = all_s2.get(name, {}); e2e = all_e2e.get(name, {})
        print(f"{name:<14} {len(s2):>5} {mnas(list(s2.values())):>8.2f} {len(e2e):>6} {mnas(list(e2e.values())):>9.2f}")


def apples_to_apples(a: str, b: str, which: str = "stage2"):
    ma = load_metrics(EVAL_ROOTS[a], which)
    mb = load_metrics(EVAL_ROOTS[b], which)
    if not ma or not mb:
        print(f"skip {a} vs {b}: one is empty")
        return
    common = set(ma.keys()) & set(mb.keys())
    aa = np.array([ma[q] for q in common])
    bb = np.array([mb[q] for q in common])
    print(f"\n=== {a} vs {b} ({which}) on {len(common)} common ===")
    print(f"  MNAS {a}:   {mnas(list(aa)):.2f}")
    print(f"  MNAS {b}:   {mnas(list(bb)):.2f}")
    print(f"  Δ        :   {mnas(list(bb)) - mnas(list(aa)):+.2f}")
    # score distribution
    ca, cb = Counter(aa), Counter(bb)
    print(f"  score:        1     2     3     4     5")
    print(f"  {a[:8]:<8}   {ca[1]:4d}  {ca[2]:4d}  {ca[3]:4d}  {ca[4]:4d}  {ca[5]:4d}")
    print(f"  {b[:8]:<8}   {cb[1]:4d}  {cb[2]:4d}  {cb[3]:4d}  {cb[4]:4d}  {cb[5]:4d}")
    print(f"  delta:     {cb[1]-ca[1]:+4d}  {cb[2]-ca[2]:+4d}  {cb[3]-ca[3]:+4d}  {cb[4]-ca[4]:+4d}  {cb[5]-ca[5]:+4d}")


def per_category(a: str, b: str, which: str = "stage2"):
    ma = load_metrics(EVAL_ROOTS[a], which)
    mb = load_metrics(EVAL_ROOTS[b], which)
    if not ma or not mb:
        return
    cats = categories_map()
    common = set(ma.keys()) & set(mb.keys())
    by_cat_a = defaultdict(list); by_cat_b = defaultdict(list)
    for q in common:
        c = cats.get(q, "?")
        by_cat_a[c].append(ma[q]); by_cat_b[c].append(mb[q])
    print(f"\n=== Per-category MNAS {a} vs {b} ({which}) ===")
    print(f"  {'category':<26} {'n':>4} {'MNAS_'+a[:7]:>12} {'MNAS_'+b[:7]:>12} {'Δ':>8}")
    for cat in sorted(by_cat_a.keys()):
        n = len(by_cat_a[cat])
        mA = mnas(by_cat_a[cat]); mB = mnas(by_cat_b[cat])
        print(f"  {cat:<26} {n:>4} {mA:>12.2f} {mB:>12.2f} {mB-mA:>+8.2f}")


def low_score_dump(name: str, top: int = 10):
    metrics = load_metrics(EVAL_ROOTS[name], "stage2")
    low = sorted([(qid, sc) for qid, sc in metrics.items() if sc <= 2], key=lambda x: x[1])[:top]
    cats = categories_map()
    sels = {s["question_id"]: s for s in load_selected()}
    print(f"\n=== {name} low-score (<=2) sample, first {top} ===")
    for qid, sc in low:
        cat = cats.get(qid, "?")
        q = sels.get(qid, {}).get("question", "?")[:80]
        print(f"  score={sc}  cat={cat:<24}  qid={qid[:8]}  q='{q}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-score", help="name of config to dump low-score samples")
    ap.add_argument("--pairs", action="store_true", help="all pairwise apples-to-apples")
    args = ap.parse_args()

    table_overall()

    # Pairwise analysis (only those with metrics)
    pairs = [
        ("v14_rerun", "v15_s1_l1"),
        ("v14_rerun", "v15_s1_l2"),
        ("v15_s1_l1", "v15_s1_l2"),
        ("v14_rerun", "v15_s1s2_l1"),
        ("v14_rerun", "v15_s1s2_l2"),
        ("v15_s1_l1", "v15_s1s2_l1"),
        ("v15_s1_l2", "v15_s1s2_l2"),
    ]
    for a, b in pairs:
        apples_to_apples(a, b, "stage2")
        per_category(a, b, "stage2")

    if args.low_score:
        low_score_dump(args.low_score)


if __name__ == "__main__":
    main()
