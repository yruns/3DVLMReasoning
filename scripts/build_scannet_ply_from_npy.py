#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

import numpy as np

DEFAULT_RUN_ID = "v3_projectable_2k_adaptive"
DEFAULT_DB_PATH = Path("docs/benchmark/embodiedscan/runs.sqlite")
DEFAULT_SOURCE_ROOT = Path("/data1/dyj_dataset/scannet/train")
DEFAULT_OUTPUT_ROOT = Path("data/scannet_meshes_v3")


class SceneMaterializationError(RuntimeError):
    """Raised for one scene's expected, reportable materialization failure."""


class MaterializationResult(NamedTuple):
    scene_id: str
    status: str
    vertex_count: int
    output_path: Path
    reason: str | None = None


def load_scene_ids_from_db(db_path: str | Path, run_id: str) -> list[str]:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark DB: {path}")
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            "SELECT rowid, sample_id, scene_id FROM samples WHERE run_id=? ORDER BY rowid",
            (run_id,),
        ).fetchall()
    if not rows:
        raise ValueError(f"No samples found for run_id={run_id!r} in {path}")

    seen: set[str] = set()
    scene_ids: list[str] = []
    for _rowid, sample_id, scene_id in rows:
        resolved = scene_id or _scene_id_from_sample_id(sample_id)
        if resolved not in seen:
            seen.add(resolved)
            scene_ids.append(resolved)
    return scene_ids


def load_scene_ids_from_json(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get("sample_ids")
        if values is None:
            values = list(payload.values())
    else:
        values = payload
    if not isinstance(values, list):
        raise ValueError(f"Sample-id JSON must contain a list: {path}")

    seen: set[str] = set()
    scene_ids: list[str] = []
    for sample_id in values:
        scene_id = _scene_id_from_sample_id(str(sample_id))
        if scene_id not in seen:
            seen.add(scene_id)
            scene_ids.append(scene_id)
    return scene_ids


def build_vertex_array(
    coord: np.ndarray,
    color: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    coord_arr = np.asarray(coord)
    color_arr = np.asarray(color)
    normal_arr = np.asarray(normal)
    _validate_scene_arrays(coord_arr, color_arr, normal_arr)

    vertex = np.empty(
        len(coord_arr),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
        ],
    )
    vertex["x"] = coord_arr[:, 0].astype(np.float32, copy=False)
    vertex["y"] = coord_arr[:, 1].astype(np.float32, copy=False)
    vertex["z"] = coord_arr[:, 2].astype(np.float32, copy=False)
    vertex["red"] = color_arr[:, 0]
    vertex["green"] = color_arr[:, 1]
    vertex["blue"] = color_arr[:, 2]
    vertex["nx"] = normal_arr[:, 0].astype(np.float32, copy=False)
    vertex["ny"] = normal_arr[:, 1].astype(np.float32, copy=False)
    vertex["nz"] = normal_arr[:, 2].astype(np.float32, copy=False)
    return vertex


def materialize_scene(
    scene_id: str,
    *,
    source_root: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
) -> MaterializationResult:
    source_scene = Path(source_root) / scene_id
    output_path = _output_ply_path(output_root, scene_id)
    coord, color, normal = load_scene_arrays(source_scene)
    vertex = build_vertex_array(coord, color, normal)
    vertex_count = int(len(vertex))

    if output_path.exists() and not overwrite:
        existing_count = read_ply_vertex_count(output_path)
        if existing_count == vertex_count:
            return MaterializationResult(
                scene_id=scene_id,
                status="skipped",
                vertex_count=vertex_count,
                output_path=output_path,
            )

    write_ply(output_path, vertex)
    return MaterializationResult(
        scene_id=scene_id,
        status="written",
        vertex_count=vertex_count,
        output_path=output_path,
    )


def load_scene_arrays(source_scene: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(source_scene)
    paths = {
        "coord": root / "coord.npy",
        "color": root / "color.npy",
        "normal": root / "normal.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SceneMaterializationError(
            f"Missing required npy files for {root.name}: {', '.join(missing)}"
        )
    return (
        np.load(paths["coord"]),
        np.load(paths["color"]),
        np.load(paths["normal"]),
    )


def read_ply_vertex_count(path: str | Path) -> int:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError(
            "plyfile is required to read/write ScanNet PLY files; run this script in the vdetr env"
        ) from exc

    ply = PlyData.read(str(path))
    return int(len(ply["vertex"].data))


def write_ply(path: str | Path, vertex: np.ndarray) -> None:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise ImportError(
            "plyfile is required to read/write ScanNet PLY files; run this script in the vdetr env"
        ) from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    element = PlyElement.describe(vertex, "vertex")
    PlyData([element], text=False).write(str(output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize ScanNet-style vertex PLYs from coord/color/normal npy arrays."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--sample-ids-json", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--min-success", type=int, default=215)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scene_ids = (
        load_scene_ids_from_json(args.sample_ids_json)
        if args.sample_ids_json is not None
        else load_scene_ids_from_db(args.db, args.run_id)
    )
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise ValueError("--max-scenes must be positive")
        scene_ids = scene_ids[: args.max_scenes]

    results: list[MaterializationResult] = []
    failures: list[dict[str, str]] = []
    for scene_id in scene_ids:
        try:
            result = materialize_scene(
                scene_id,
                source_root=args.source_root,
                output_root=args.output_root,
                overwrite=args.overwrite,
            )
            results.append(result)
            print(
                f"{result.status}: {scene_id} "
                f"vertices={result.vertex_count} path={result.output_path}"
            )
        except SceneMaterializationError as exc:
            failures.append({"scene_id": scene_id, "reason": str(exc)})
            print(f"failed: {scene_id} reason={exc}")

    success_count = len(results)
    summary = {
        "source": (
            str(args.sample_ids_json)
            if args.sample_ids_json is not None
            else f"{args.db}:samples run_id={args.run_id}"
        ),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "requested_scene_count": len(scene_ids),
        "success_count": success_count,
        "written_count": sum(1 for result in results if result.status == "written"),
        "skipped_count": sum(1 for result in results if result.status == "skipped"),
        "failure_count": len(failures),
        "failures": failures,
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if success_count < args.min_success:
        raise RuntimeError(
            f"Materialized {success_count}/{len(scene_ids)} scenes; "
            f"required at least {args.min_success}"
        )


def _validate_scene_arrays(
    coord: np.ndarray,
    color: np.ndarray,
    normal: np.ndarray,
) -> None:
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise SceneMaterializationError(f"coord.npy must have shape (N, 3), got {coord.shape}")
    if color.ndim != 2 or color.shape[1] != 3:
        raise SceneMaterializationError(f"color.npy must have shape (N, 3), got {color.shape}")
    if normal.ndim != 2 or normal.shape[1] != 3:
        raise SceneMaterializationError(f"normal.npy must have shape (N, 3), got {normal.shape}")
    if len(coord) != len(color) or len(coord) != len(normal):
        raise SceneMaterializationError(
            "coord.npy, color.npy, and normal.npy must have the same vertex count"
        )
    if color.dtype != np.uint8:
        raise SceneMaterializationError(f"color.npy must be uint8, got {color.dtype}")
    if not np.isfinite(coord).all():
        raise SceneMaterializationError("coord.npy contains non-finite values")
    if not np.isfinite(normal).all():
        raise SceneMaterializationError("normal.npy contains non-finite values")


def _scene_id_from_sample_id(sample_id: str) -> str:
    if "::" in sample_id:
        return sample_id.split("::", 1)[0]
    if "__" in sample_id:
        return sample_id.split("__", 1)[0]
    raise ValueError(f"Cannot derive scene_id from sample_id={sample_id!r}")


def _output_ply_path(output_root: str | Path, scene_id: str) -> Path:
    return Path(output_root) / "scans" / scene_id / f"{scene_id}_vh_clean_2.ply"


if __name__ == "__main__":
    main()
