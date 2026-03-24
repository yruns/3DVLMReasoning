#!/usr/bin/env python3
"""Compare RAM+GDINO (v1) vs Florence-2 (v2) merged scene graphs."""

from __future__ import annotations

import gzip
import pickle
from collections import Counter
from pathlib import Path

OPENEQA_ROOT = Path.home() / "Datasets/OpenEQA/scannet"

SCENES = [
    "002-scannet-scene0709_00",
    "003-scannet-scene0762_00",
    "012-scannet-scene0785_00",
    "013-scannet-scene0720_00",
    "014-scannet-scene0714_00",
]

JUNK_WORDS = {
    "sit", "stand", "walk", "run", "hang", "connect", "open", "close",
    "lead to", "place", "lean", "lay", "fold", "stack", "attach", "wrap",
    "cover", "rest", "play", "cook", "eat", "read", "write", "sleep",
    "white", "black", "red", "blue", "green", "yellow", "brown", "gray",
    "grey", "pink", "orange", "purple", "dark", "light", "bright",
    "tight", "small", "large", "big", "tall", "short", "long", "round",
    "flat", "clean", "dirty", "modern", "old", "new",
    "other item", "item", "other", "none",
    "house", "room", "apartment", "building", "space", "area",
}


def is_junk(label: str) -> bool:
    return label.lower().strip() in JUNK_WORDS


def load_categories(pkl_path: Path) -> Counter:
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    cats = Counter()
    for obj in data["objects"]:
        names = obj.get("class_name", [])
        if names:
            cat = Counter(names).most_common(1)[0][0]
        else:
            cat = "unknown"
        cats[cat] += 1
    return cats


def find_post_pkl(scene_dir: Path) -> Path | None:
    pcd_dir = scene_dir / "pcd_saves"
    if not pcd_dir.exists():
        return None
    candidates = sorted(pcd_dir.glob("*_post.pkl.gz"))
    return candidates[0] if candidates else None


def main():
    all_v1 = Counter()
    all_v2 = Counter()

    for scene_name in SCENES:
        scene = OPENEQA_ROOT / scene_name
        v1_pkl = find_post_pkl(scene / "conceptgraph_v1")
        v2_pkl = find_post_pkl(scene / "conceptgraph")

        print("=" * 80)
        print(f"  {scene_name}")
        print("=" * 80)

        for label, pkl, accum in [
            ("RAM+GDINO (v1)", v1_pkl, all_v1),
            ("Florence-2 (v2)", v2_pkl, all_v2),
        ]:
            if pkl is None:
                print(f"  [{label}] No scene graph found")
                continue
            cats = load_categories(pkl)
            accum.update(cats)
            n_obj = sum(cats.values())
            n_junk = sum(v for k, v in cats.items() if is_junk(k))
            junk_labels = sorted(k for k in cats if is_junk(k))
            print(f"\n  [{label}] {n_obj} objects, {len(cats)} categories")
            if junk_labels:
                print(f"    JUNK ({n_junk} objects): {junk_labels}")
            for cat, cnt in cats.most_common(15):
                tag = " ← JUNK" if is_junk(cat) else ""
                print(f"      {cat:<35} ×{cnt:>3}{tag}")
            if len(cats) > 15:
                print(f"      … +{len(cats) - 15} more")

    # ── Aggregate ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  AGGREGATE (5 scenes)")
    print("=" * 80)

    for label, counter in [("RAM+GDINO (v1)", all_v1), ("Florence-2 (v2)", all_v2)]:
        n_obj = sum(counter.values())
        n_cats = len(counter)
        junk_cats = {k for k in counter if is_junk(k)}
        junk_objs = sum(counter[k] for k in junk_cats)
        clean_cats = n_cats - len(junk_cats)
        print(f"\n  [{label}]")
        print(f"    Objects:    {n_obj}")
        print(f"    Categories: {n_cats} ({clean_cats} clean, {len(junk_cats)} junk)")
        print(f"    Junk objs:  {junk_objs}/{n_obj} ({junk_objs/max(n_obj,1)*100:.1f}%)")
        if junk_cats:
            print(f"    Junk list:  {sorted(junk_cats)}")

    # Side-by-side top categories
    print(f"\n  {'RAM+GDINO top-20':<42} {'Florence-2 top-20'}")
    print(f"  {'─' * 40}   {'─' * 40}")
    v1_top = all_v1.most_common(20)
    v2_top = all_v2.most_common(20)
    for i in range(max(len(v1_top), len(v2_top))):
        left = ""
        if i < len(v1_top):
            k, v = v1_top[i]
            j = " *" if is_junk(k) else ""
            left = f"{k} ×{v}{j}"
        right = ""
        if i < len(v2_top):
            k, v = v2_top[i]
            j = " *" if is_junk(k) else ""
            right = f"{k} ×{v}{j}"
        print(f"  {left:<42} {right}")


if __name__ == "__main__":
    main()
