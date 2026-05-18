"""Ingest a ScanRefer VG side-by-side run into a SQLite database.

Mirrors scripts/ingest_nr3d_run.py with Unique/Multiple slicing columns
instead of NR3D's Easy/Hard/View-Dep/View-Indep.
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
    n_total         INTEGER,
    n_unique        INTEGER,
    n_multiple      INTEGER,
    acc25_overall   REAL,
    acc50_overall   REAL,
    acc25_unique    REAL,
    acc50_unique    REAL,
    acc25_multiple  REAL,
    acc50_multiple  REAL,
    mean_iou_overall REAL,
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
    ann_id                  TEXT,
    description             TEXT,
    status                  TEXT,
    selected_object_id      INTEGER,
    confidence              REAL,
    iou                     REAL,
    acc25                   INTEGER,
    acc50                   INTEGER,
    is_unique               INTEGER,
    predicted_bbox_3d_9dof  TEXT,
    gt_bbox_3d_9dof         TEXT,
    PRIMARY KEY (run_id, sample_id)
);

CREATE INDEX IF NOT EXISTS idx_scanrefer_samples_run ON samples(run_id);
CREATE INDEX IF NOT EXISTS idx_scanrefer_samples_scene ON samples(scene_id);
CREATE INDEX IF NOT EXISTS idx_scanrefer_samples_unique ON samples(is_unique);

CREATE TABLE IF NOT EXISTS tool_calls (
    run_id          TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    turn_idx        INTEGER NOT NULL,
    tool_name       TEXT NOT NULL,
    tool_input      TEXT,
    response_text   TEXT
);
CREATE INDEX IF NOT EXISTS idx_scanrefer_tools_run_qid
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
CREATE INDEX IF NOT EXISTS idx_scanrefer_llm_run ON llm_calls(run_id);
"""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_sample_id(sample_id: str) -> tuple[str, int, str]:
    parts = sample_id.split("::")
    if len(parts) < 3:
        raise ValueError(
            f"sample_id {sample_id!r} malformed (need <scan_id>::<target_id>::<ann_id>)"
        )
    try:
        target_id = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"sample_id {sample_id!r} target_id segment not int") from exc
    return parts[0], target_id, parts[2]


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
    leaderboard_metrics_path: Path | None = None,
) -> None:
    side_by_side_path = output_dir / "side_by_side.json"
    if not side_by_side_path.exists():
        raise FileNotFoundError(f"Missing side_by_side.json: {side_by_side_path}")
    payload = _read_json(side_by_side_path)
    metrics = payload.get(backend)
    if not isinstance(metrics, dict):
        raise ValueError(f"side_by_side.json missing backend={backend!r}")
    per_sample = metrics.get("per_sample")
    if not isinstance(per_sample, list) or not per_sample:
        raise ValueError(
            f"side_by_side.json[{backend}].per_sample must be a non-empty list"
        )

    leaderboard: dict[str, Any] | None = None
    leaderboard_per_sample: dict[str, dict[str, Any]] = {}
    if leaderboard_metrics_path is not None:
        leaderboard = _read_json(Path(leaderboard_metrics_path))
        for entry in leaderboard.get("per_sample", []):
            sid = entry.get("sample_id")
            if isinstance(sid, str):
                leaderboard_per_sample[sid] = entry

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        cur = conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO runs (
                run_id, branch, commit_hash, output_dir, backend,
                n_total, n_unique, n_multiple,
                acc25_overall, acc50_overall,
                acc25_unique, acc50_unique,
                acc25_multiple, acc50_multiple,
                mean_iou_overall,
                judge_model, started_at, ingested_at, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                branch,
                commit_hash,
                str(output_dir),
                backend,
                leaderboard.get("n_total") if leaderboard else None,
                leaderboard.get("n_unique") if leaderboard else None,
                leaderboard.get("n_multiple") if leaderboard else None,
                leaderboard.get("acc25_overall") if leaderboard else None,
                leaderboard.get("acc50_overall") if leaderboard else None,
                leaderboard.get("acc25_unique") if leaderboard else None,
                leaderboard.get("acc50_unique") if leaderboard else None,
                leaderboard.get("acc25_multiple") if leaderboard else None,
                leaderboard.get("acc50_multiple") if leaderboard else None,
                leaderboard.get("mean_iou_overall") if leaderboard else None,
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
            sample_id = str(item.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"per_sample missing sample_id: {item!r}")
            scene_id, target_id, ann_id = _parse_sample_id(sample_id)
            iou_raw = item.get("iou")
            iou = float(iou_raw) if iou_raw is not None else None
            extra = leaderboard_per_sample.get(sample_id)
            cur.execute(
                """INSERT INTO samples (
                    run_id, sample_id, scene_id, target_id, ann_id,
                    description, status, selected_object_id, confidence,
                    iou, acc25, acc50, is_unique,
                    predicted_bbox_3d_9dof, gt_bbox_3d_9dof
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    sample_id,
                    scene_id,
                    target_id,
                    ann_id,
                    item.get("query"),
                    item.get("status"),
                    item.get("selected_object_id"),
                    item.get("confidence"),
                    iou,
                    int(iou is not None and iou >= 0.25),
                    int(iou is not None and iou >= 0.50),
                    (
                        int(extra["is_unique"])
                        if extra and extra.get("is_unique") is not None
                        else None
                    ),
                    _json_or_none(item.get("predicted_bbox_3d_9dof")),
                    _json_or_none(item.get("gt_bbox_3d_9dof")),
                ),
            )
            for turn_idx, tool_call in enumerate(item.get("tool_trace") or []):
                if not isinstance(tool_call, dict):
                    continue
                cur.execute(
                    """INSERT INTO tool_calls (
                        run_id, question_id, turn_idx, tool_name,
                        tool_input, response_text
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        sample_id,
                        turn_idx,
                        str(tool_call.get("tool_name") or ""),
                        _json_or_none(tool_call.get("tool_input")),
                        str(tool_call.get("response_text") or ""),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    print(f"[ingest] run_id={run_id} samples={len(per_sample)} db={db_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--branch", default=None)
    p.add_argument("--commit", default=None, dest="commit_hash")
    p.add_argument("--backend", default="pack_v1")
    p.add_argument("--judge-model", default=None)
    p.add_argument(
        "--db", default=Path("docs/benchmark/scanrefer/runs.sqlite"), type=Path
    )
    p.add_argument("--notes", default=None)
    p.add_argument(
        "--leaderboard-metrics",
        default=None,
        type=Path,
        help="optional path to leaderboard_metrics.json",
    )
    args = p.parse_args()
    ingest(
        db_path=args.db,
        output_dir=args.output_dir,
        run_id=args.run_id,
        branch=args.branch,
        commit_hash=args.commit_hash,
        backend=args.backend,
        judge_model=args.judge_model,
        notes=args.notes,
        leaderboard_metrics_path=args.leaderboard_metrics,
    )


if __name__ == "__main__":
    main()
