import json
import logging
import sqlite3
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ParentStore:
    """Small SQLite key-value store: parent_id -> parent Document."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
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
        """Insert parents, replacing any with the same parent_id."""
        rows = [
            (
                p.metadata["parent_id"],
                str(p.metadata.get("source", "")),
                p.page_content,
                json.dumps(p.metadata, ensure_ascii=False, default=str),
            )
            for p in parents
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO parents (parent_id, source, content, metadata) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
        logger.info("Stored %d parents", len(rows))
        return len(rows)

    def get_many(self, parent_ids: list[str]) -> dict[str, Document]:
        """Fetch parents by ID. Missing IDs are simply absent from the result."""
        if not parent_ids:
            return {}
        placeholders = ",".join("?" * len(parent_ids))
        cursor = self._conn.execute(
            f"SELECT parent_id, content, metadata FROM parents "
            f"WHERE parent_id IN ({placeholders})",
            parent_ids,
        )
        return {
            pid: Document(page_content=content, metadata=json.loads(metadata))
            for pid, content, metadata in cursor
        }

    def delete_by_source(self, source: str) -> int:
        """Delete every parent that came from `source`."""
        with self._conn:
            cursor = self._conn.execute("DELETE FROM parents WHERE source = ?", (source,))
        logger.info("Deleted %d parents for source=%s", cursor.rowcount, source)
        return cursor.rowcount

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]

    def close(self) -> None:
        self._conn.close()