import json
import sqlite3
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Tiny SQLite store for benchmark artifacts and run reports."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_reports (
                    run_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def put_artifact(self, artifact_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (id, kind, payload)
                VALUES (?, ?, ?)
                """,
                (artifact_id, kind, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM artifacts WHERE id = ?",
                (artifact_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def put_report(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_reports (run_id, payload)
                VALUES (?, ?)
                """,
                (run_id, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )

    def get_report(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM run_reports WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None
