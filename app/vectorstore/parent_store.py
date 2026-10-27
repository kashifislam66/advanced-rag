import json
import logging
import sqlite3
import threading
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ParentStore:
    """Small SQLite key-value store: parent_id -> parent Document. Thread-safe."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parents (
                    parent_id TEXT PRIMARY KEY,
                    source    TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    metadata  TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_parents_source ON parents(source)"
            )
            self._conn.commit()

    def upsert(self, parents: list[Document]) -> int:
        rows = [
            (
                p.metadata["parent_id"],
                str(p.metadata.get("source", "")),
                p.page_content,
                json.dumps(p.metadata, ensure_ascii=False, default=str),
            )
            for p in parents
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO parents (parent_id, source, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
        logger.info("Stored %d parents", len(rows))
        return len(rows)

    def get_many(self, parent_ids: list[str]) -> dict[str, Document]:
        if not parent_ids:
            return {}
        placeholders = ",".join("?" * len(parent_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT parent_id, content, metadata FROM parents "
                f"WHERE parent_id IN ({placeholders})",
                parent_ids,
            ).fetchall()
        return {
            pid: Document(page_content=content, metadata=json.loads(metadata))
            for pid, content, metadata in rows
        }

    def delete_by_source(self, source: str) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM parents WHERE source = ?", (source,))
            deleted = cursor.rowcount
        logger.info("Deleted %d parents for source=%s", deleted, source)
        return deleted

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]

    def list_sources(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT source FROM parents").fetchall()
        return {row[0] for row in rows}

    def sample(self, count: int) -> list[Document]:
        if count <= 0:
            raise ValueError("count must be positive")
        with self._lock:
            rows = self._conn.execute(
                "SELECT content, metadata FROM parents ORDER BY RANDOM() LIMIT ?", (count,)
            ).fetchall()
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()