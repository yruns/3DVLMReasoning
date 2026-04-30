"""Ingest an NR3D VG side-by-side run into a SQLite database.

The runner writes ``side_by_side.json`` with aggregate metrics and one row
per sample. This ingester makes the run queryable under
``docs/benchmark/nr3d/runs.sqlite``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    branch          TEXT,
    commit_hash     TEXT,
    output_dir      TEXT NOT NULL,
    backend         TEXT NOT NULL,
    n               INTEGER,
    mean_iou        REAL,
    acc25           REAL,
    acc50           REAL,
    judge_model     TEXT,
    started_at      REAL,
    ingested_at     REAL NOT NULL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    run_id                  TEXT NOT NULL,
    sample_id               TEXT NOT NULL,
    scene_id                TEXT,
    target_id               INTEGER,
    query                   TEXT,
    status                  TEXT,
    selected_object_id      INTEGER,
    confidence              REAL,
    iou                     REAL,
    acc25                   INTEGER,
    acc50                   INTEGER,
    predicted_bbox_3d_9dof  TEXT,
    gt_bbox_3d_9dof         TEXT,
    PRIMARY KEY (run_id, sample_id)
);

CREATE INDEX IF NOT EXISTS idx_nr3d_samples_run
    ON samples(run_id);
CREATE INDEX IF NOT EXISTS idx_nr3d_samples_scene
    ON samples(scene_id);
CREATE INDEX IF NOT EXISTS idx_nr3d_samples_iou
    ON samples(iou);

CREATE TABLE IF NOT EXISTS tool_calls (
    run_id          TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    turn_idx        INTEGER NOT NULL,
    tool_name       TEXT NOT NULL,
    tool_input      TEXT,
    response_text   TEXT
);

CREATE INDEX IF NOT EXISTS idx_nr3d_tools_run_qid
    ON tool_calls(run_id, question_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    run_id              TEXT NOT NULL,
    ts                  REAL,
    model               TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    cached_tokens       INTEGER,
    session_id          TEXT,
    question_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_nr3d_llm_run
    ON llm_calls(run_id);
"""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_metric(metrics: dict[str, Any], key: str, backend: str) -> Any:
    if key not in metrics:
        raise ValueError(f"side_by_side.json[{backend}] missing required key: {key}")
    return metrics[key]


def _parse_sample_id(sample_id: str) -> tuple[str, int]:
    parts = sample_id.split("::")
    if len(parts) < 2:
        raise ValueError(
            f"sample_id {sample_id!r} is malformed: expected "
            f"'<scan_id>::<target_id>::<assignment_id>'"
        )
    try:
        target_id = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"sample_id {sample_id!r} target_id segment is not an int: {parts[1]!r}"
        ) from exc
    return parts[0], target_id


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def ingest(
    *,
    db_path: Path,
    output_dir: Path,
    run_id: str,
    branch: str | None,
    commit_hash: str | None,
    backend: str = "pack_v1",
    judge_model: str | None = None,
    notes: str | None = None,
) -> None:
    side_by_side_path = output_dir / "side_by_side.json"
    if not side_by_side_path.exists():
        raise FileNotFoundError(f"Missing side_by_side.json: {side_by_side_path}")
    payload = _read_json(side_by_side_path)
    metrics = payload.get(backend)
    if not isinstance(metrics, dict):
        raise ValueError(f"side_by_side.json missing backend={backend!r}")
    per_sample = _require_metric(metrics, "per_sample", backend)
    if not isinstance(per_sample, list):
        raise ValueError(f"side_by_side.json[{backend}].per_sample must be a list")
    if not per_sample:
        raise ValueError(f"side_by_side.json[{backend}].per_sample must be non-empty")
    n = int(_require_metric(metrics, "n", backend))
    if n != len(per_sample):
        raise ValueError(
            f"side_by_side.json[{backend}] n={n} but per_sample has "
            f"{len(per_sample)} rows"
        )
    mean_iou = float(_require_metric(metrics, "mean_iou", backend))
    acc25 = float(_require_metric(metrics, "Acc@0.25", backend))
    acc50 = float(_require_metric(metrics, "Acc@0.50", backend))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                branch,
                commit_hash,
                str(output_dir),
                backend,
                n,
                mean_iou,
                acc25,
                acc50,
                judge_model,
                None,
                time.time(),
                notes,
            ),
        )
        cur.execute("DELETE FROM samples WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM tool_calls WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM llm_calls WHERE run_id=?", (run_id,))

        for item in per_sample:
            if not isinstance(item, dict):
                raise ValueError(f"per_sample entry must be an object: {item!r}")
            sample_id = str(item.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"per_sample entry missing sample_id: {item!r}")
            scene_id, target_id = _parse_sample_id(sample_id)
            iou = item.get("iou")
            iou_float = float(iou) if iou is not None else None
            cur.execute(
                """INSERT INTO samples VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    sample_id,
                    scene_id,
                    target_id,
                    item.get("query"),
                    item.get("status"),
                    item.get("selected_object_id"),
                    item.get("confidence"),
                    iou_float,
                    int(iou_float is not None and iou_float >= 0.25),
                    int(iou_float is not None and iou_float >= 0.50),
                    _json_or_none(item.get("predicted_bbox_3d_9dof")),
                    _json_or_none(item.get("gt_bbox_3d_9dof")),
                ),
            )

        conn.commit()
    finally:
        conn.close()

    print(f"[ingest] run_id={run_id} samples={len(per_sample)} db={db_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--commit", default=None, dest="commit_hash")
    parser.add_argument("--backend", default="pack_v1")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--db", default=Path("docs/benchmark/nr3d/runs.sqlite"), type=Path)
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()
    ingest(
        db_path=args.db,
        output_dir=args.output_dir,
        run_id=args.run_id,
        branch=args.branch,
        commit_hash=args.commit_hash,
        backend=args.backend,
        judge_model=args.judge_model,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
