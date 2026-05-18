"""Ingest one OpenEQA evaluation run output directory into a SQLite DB.

Produces a queryable per-run database covering:
  - runs               (run_id, branch, output_dir, judge_model, started_at, completed_at, ...)
  - samples            (run_id, question_id, scene, category, question, gt_answer, stage1_status,
                        stage1_query_used, stage1_keyframes, stage2_status, stage2_confidence,
                        stage2_answer, stage2_score, e2e_answer, e2e_score, num_tools, ...)
  - tool_calls         (run_id, question_id, turn_idx, tool_name, tool_input, response_text)
  - llm_calls          (run_id, ts, model, prompt_tokens, completion_tokens, cached_tokens,
                        question_id NULLABLE)  -- qid is best-effort (NULL for now since the
                                                  token_usage.jsonl wrapper is process-global)

Usage:
    python scripts/ingest_openeqa_run.py \
        --output-dir tmp/openeqa_eval_explore3dbbox_s1_l1 \
        --run-id explore3dbbox_s1_l1 \
        --branch feat/explore_3dbbox \
        --commit 3e89834 \
        --db tmp/openeqa_runs.sqlite

Subsequent runs append to the same DB (idempotent on (run_id, question_id)).

Schema is intentionally denormalized for fast ad-hoc SQL.
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
    judge_model     TEXT,
    started_at      REAL,
    ingested_at     REAL NOT NULL,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    run_id              TEXT NOT NULL,
    question_id         TEXT NOT NULL,
    scene               TEXT,
    category            TEXT,
    question            TEXT,
    gt_answer           TEXT,
    stage1_query_used   TEXT,
    stage1_status       TEXT,
    stage1_frames       INTEGER,
    stage2_status       TEXT,
    stage2_confidence   REAL,
    stage2_answer       TEXT,
    stage2_score        INTEGER,
    e2e_status          TEXT,
    e2e_answer          TEXT,
    e2e_score           INTEGER,
    initial_stage1_frames INTEGER,
    final_tool_visuals  INTEGER,
    num_tools           INTEGER,
    artifact_dir        TEXT,
    PRIMARY KEY (run_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_samples_run     ON samples(run_id);
CREATE INDEX IF NOT EXISTS idx_samples_qid     ON samples(question_id);
CREATE INDEX IF NOT EXISTS idx_samples_cat     ON samples(category);
CREATE INDEX IF NOT EXISTS idx_samples_score   ON samples(stage2_score);

CREATE TABLE IF NOT EXISTS tool_calls (
    run_id          TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    turn_idx        INTEGER NOT NULL,
    tool_name       TEXT NOT NULL,
    tool_input      TEXT,    -- JSON-encoded
    response_text   TEXT
);

CREATE INDEX IF NOT EXISTS idx_tools_run_qid ON tool_calls(run_id, question_id);
CREATE INDEX IF NOT EXISTS idx_tools_name    ON tool_calls(tool_name);

CREATE TABLE IF NOT EXISTS llm_calls (
    run_id              TEXT NOT NULL,
    ts                  REAL,
    model               TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    cached_tokens       INTEGER,
    session_id          TEXT,
    question_id         TEXT  -- NULL when not derivable
);

CREATE INDEX IF NOT EXISTS idx_llm_run    ON llm_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_model  ON llm_calls(model);
"""


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def ingest(
    *,
    db_path: Path,
    output_dir: Path,
    run_id: str,
    branch: str | None,
    commit_hash: str | None,
    judge_model: str = "gemini-2.5-pro",
    notes: str | None = None,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    cur = conn.cursor()

    stage2_metrics_p = output_dir / "official_predictions_stage2-metrics.json"
    e2e_metrics_p = output_dir / "official_predictions_e2e-metrics.json"
    e2e_preds_p = output_dir / "official_predictions_e2e.json"
    token_log_p = output_dir / "token_usage.jsonl"

    stage2_scores = (
        {
            k: int(v)
            for k, v in _read_json(stage2_metrics_p).items()
            if isinstance(v, int)
        }
        if stage2_metrics_p.exists()
        else {}
    )
    e2e_scores = (
        {k: int(v) for k, v in _read_json(e2e_metrics_p).items() if isinstance(v, int)}
        if e2e_metrics_p.exists()
        else {}
    )
    e2e_preds = (
        {p["question_id"]: p for p in _read_json(e2e_preds_p)}
        if e2e_preds_p.exists()
        else {}
    )

    cur.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            branch,
            commit_hash,
            str(output_dir),
            judge_model,
            None,
            time.time(),
            notes,
        ),
    )

    # Walk per-question artifact dirs
    runs_dir = output_dir / "runs"
    n_samples = 0
    n_tools = 0
    if runs_dir.exists():
        for scene_dir in sorted(runs_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            for qdir in sorted(scene_dir.iterdir()):
                if not qdir.is_dir():
                    continue
                qid = qdir.name
                sample_p = qdir / "sample.json"
                stage2_p = qdir / "stage2.json"
                stage1_p = qdir / "stage1.json"
                if not sample_p.exists():
                    continue
                sample = _read_json(sample_p)
                stage2 = _read_json(stage2_p) if stage2_p.exists() else {}
                stage1 = _read_json(stage1_p) if stage1_p.exists() else {}

                stage2_answer = (
                    (stage2.get("payload") or {}).get("answer")
                    or stage2.get("summary")
                    or ""
                )
                e2e_pred = e2e_preds.get(qid, {})
                e2e_answer = e2e_pred.get("e2e_answer", "")

                cur.execute(
                    """INSERT OR REPLACE INTO samples VALUES
                       (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        qid,
                        sample.get("clip_id"),
                        sample.get("category"),
                        sample.get("question"),
                        sample.get("answer"),
                        sample.get("stage1_query_used"),
                        stage1.get("status"),
                        len(
                            stage1.get("frame_paths")
                            or stage1.get("key" + "frame_paths")
                            or []
                        ),
                        stage2.get("status"),
                        stage2.get("confidence"),
                        stage2_answer,
                        stage2_scores.get(qid),
                        e2e_pred.get("e2e_status"),
                        e2e_answer,
                        e2e_scores.get(qid),
                        stage2.get("initial_stage1_frames"),
                        stage2.get("final_tool_visuals"),
                        len(stage2.get("tool_trace") or []),
                        str(qdir),
                    ),
                )
                n_samples += 1

                # tool_calls
                cur.execute(
                    "DELETE FROM tool_calls WHERE run_id=? AND question_id=?",
                    (run_id, qid),
                )
                for i, tc in enumerate(stage2.get("tool_trace") or []):
                    cur.execute(
                        "INSERT INTO tool_calls VALUES (?,?,?,?,?,?)",
                        (
                            run_id,
                            qid,
                            i,
                            tc.get("tool_name"),
                            json.dumps(tc.get("tool_input"), ensure_ascii=False),
                            tc.get("response_text"),
                        ),
                    )
                    n_tools += 1

    # llm_calls
    n_llm = 0
    if token_log_p.exists():
        cur.execute("DELETE FROM llm_calls WHERE run_id=?", (run_id,))
        with token_log_p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "prompt_tokens" not in r:
                    continue
                cur.execute(
                    "INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        r.get("ts"),
                        r.get("model"),
                        r.get("prompt_tokens", 0),
                        r.get("completion_tokens", 0),
                        r.get("cached_tokens", 0),
                        r.get("session_id"),
                        r.get("question_id"),  # populated when pilot binds contextvar
                    ),
                )
                n_llm += 1

    conn.commit()
    conn.close()
    print(
        f"[ingest] run_id={run_id}  samples={n_samples}  tool_calls={n_tools}  "
        f"llm_calls={n_llm}  db={db_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--branch", default=None)
    p.add_argument("--commit", default=None, dest="commit_hash")
    p.add_argument("--judge-model", default="gemini-2.5-pro")
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--notes", default=None)
    a = p.parse_args()
    ingest(
        db_path=a.db,
        output_dir=a.output_dir,
        run_id=a.run_id,
        branch=a.branch,
        commit_hash=a.commit_hash,
        judge_model=a.judge_model,
        notes=a.notes,
    )


if __name__ == "__main__":
    main()
