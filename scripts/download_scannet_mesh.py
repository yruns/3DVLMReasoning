#!/usr/bin/env python3
"""Download ScanNet v2 mesh and segmentation files for OpenEQA scenes.

Downloads only the files needed for BEV rendering and semantic analysis:
  - <scanId>_vh_clean_2.ply          (reconstructed mesh)
  - <scanId>_vh_clean_2.0.010000.segs.json (over-segmentation)
  - <scanId>_vh_clean.segs.json      (segmentation)
  - <scanId>.aggregation.json        (instance aggregation)
  - <scanId>_vh_clean.aggregation.json (alt instance aggregation)

Usage:
    # Download all 89 OpenEQA scenes
    python scripts/download_scannet_mesh.py

    # Download specific scenes
    python scripts/download_scannet_mesh.py --scenes scene0709_00 scene0762_00

    # Download only mesh PLY files
    python scripts/download_scannet_mesh.py --types _vh_clean_2.ply

    # Dry run (show what would be downloaded)
    python scripts/download_scannet_mesh.py --dry-run

    # Use custom output directory
    python scripts/download_scannet_mesh.py -o /path/to/output

Prerequisites:
    You must have agreed to the ScanNet Terms of Use:
    https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf

    Set the environment variable SCANNET_BASE_URL to the download base URL
    provided after your TOS agreement is approved.
    Example: export SCANNET_BASE_URL="https://kaldir.vc.in.tum.de/scannet/v2/scans"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# File suffixes to download
DEFAULT_SUFFIXES = [
    "_vh_clean_2.ply",
    "_vh_clean_2.0.010000.segs.json",
    "_vh_clean.segs.json",
    ".aggregation.json",
    "_vh_clean.aggregation.json",
]

# Default base URL (requires TOS agreement)
DEFAULT_BASE_URL = os.environ.get(
    "SCANNET_BASE_URL",
    "https://kaldir.vc.in.tum.de/scannet/v2/scans",
)


def get_openeqa_scene_ids(data_root: Path) -> list[str]:
    """Extract ScanNet scene IDs from local OpenEQA prepared scenes."""
    openeqa_dir = data_root / "OpenEQA" / "scannet"
    if not openeqa_dir.exists():
        return []

    scene_ids = []
    for clip_dir in sorted(openeqa_dir.iterdir()):
        if not clip_dir.is_dir():
            continue
        # e.g. "002-scannet-scene0709_00" -> "scene0709_00"
        parts = clip_dir.name.split("-", 2)
        if len(parts) >= 3 and parts[2].startswith("scene"):
            scene_ids.append(parts[2])
    return scene_ids


def read_scene_file(path: Path) -> list[str]:
    """Read ScanNet scene ids from a text file.

    Lines may be plain scene ids, scan ids like ``scannet/scene0415_00``, blank,
    or comments starting with ``#``.
    """
    scene_ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "/" in line:
            line = line.split("/")[-1]
        scene_ids.append(line)
    return scene_ids


def download_file(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Download a file with progress indication.

    Returns True if file was downloaded or already exists.
    """
    if dest.exists():
        print(f"  [skip] {dest.name} (already exists, {dest.stat().st_size / 1e6:.1f} MB)")
        return True

    if dry_run:
        print(f"  [dry-run] would download: {url}")
        print(f"            -> {dest}")
        return True

    print(f"  [download] {dest.name} ...", end="", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        tmp_path.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f" OK ({size_mb:.1f} MB)")
        return True
    except urllib.error.HTTPError as e:
        print(f" FAILED ({e.code} {e.reason})")
        tmp_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f" ERROR ({e})")
        tmp_path.unlink(missing_ok=True)
        return False


def download_scene(
    scan_id: str,
    output_dir: Path,
    base_url: str,
    suffixes: list[str],
    dry_run: bool = False,
) -> dict[str, bool]:
    """Download all requested files for one scene.

    Returns dict mapping suffix -> success boolean.
    """
    scene_dir = output_dir / scan_id
    results = {}

    for suffix in suffixes:
        filename = f"{scan_id}{suffix}"
        url = f"{base_url}/{scan_id}/{filename}"
        dest = scene_dir / filename
        results[suffix] = download_file(url, dest, dry_run=dry_run)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ScanNet v2 mesh/segmentation files for OpenEQA scenes."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "scannetv2",
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Project data root (for auto-detecting OpenEQA scenes).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Specific scene IDs to download (e.g. scene0709_00). "
             "Default: auto-detect from OpenEQA data.",
    )
    parser.add_argument(
        "--scene-file",
        type=Path,
        default=None,
        help="Text file containing scene ids, one per line.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Specific file suffixes to download. "
             f"Default: {DEFAULT_SUFFIXES}",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="ScanNet download base URL (or set SCANNET_BASE_URL env var).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limit number of scenes to download (for testing).",
    )
    args = parser.parse_args()

    # Determine scene list
    if args.scene_file:
        scene_ids = read_scene_file(args.scene_file)
    elif args.scenes:
        scene_ids = args.scenes
    else:
        scene_ids = get_openeqa_scene_ids(args.data_root)
        if not scene_ids:
            print("ERROR: No OpenEQA scenes found. Specify --scenes manually.")
            sys.exit(1)

    if args.max_scenes:
        scene_ids = scene_ids[: args.max_scenes]

    suffixes = args.types or DEFAULT_SUFFIXES

    print(f"ScanNet v2 Downloader")
    print(f"  Output dir:  {args.output}")
    print(f"  Base URL:    {args.base_url}")
    print(f"  Scenes:      {len(scene_ids)}")
    print(f"  File types:  {suffixes}")
    print(f"  Dry run:     {args.dry_run}")
    print()

    if "kaldir" in args.base_url and not args.dry_run:
        print("NOTE: You must have agreed to the ScanNet Terms of Use.")
        print("      https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf")
        print()

    # Download
    total_ok, total_fail, total_skip = 0, 0, 0

    for i, scan_id in enumerate(scene_ids):
        print(f"[{i+1}/{len(scene_ids)}] {scan_id}")
        results = download_scene(
            scan_id=scan_id,
            output_dir=args.output,
            base_url=args.base_url,
            suffixes=suffixes,
            dry_run=args.dry_run,
        )
        for suffix, ok in results.items():
            if ok:
                dest = args.output / scan_id / f"{scan_id}{suffix}"
                if dest.exists():
                    total_skip += 1
                else:
                    total_ok += 1
            else:
                total_fail += 1

    print()
    print(f"Done. Downloaded: {total_ok}  Skipped: {total_skip}  Failed: {total_fail}")

    # Create symlink mapping from OpenEQA clip IDs to ScanNet scene IDs
    openeqa_dir = args.data_root / "OpenEQA" / "scannet"
    if openeqa_dir.exists() and not args.dry_run:
        mapping_path = args.output / "openeqa_scene_mapping.json"
        import json

        mapping = {}
        for clip_dir in sorted(openeqa_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            parts = clip_dir.name.split("-", 2)
            if len(parts) >= 3 and parts[2].startswith("scene"):
                mapping[clip_dir.name] = parts[2]

        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
        print(f"Saved OpenEQA -> ScanNet ID mapping to {mapping_path}")


if __name__ == "__main__":
    main()
