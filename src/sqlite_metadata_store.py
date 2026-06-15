from __future__ import annotations

import sqlite3
from pathlib import Path

from src.metadata_store import MetadataStore


class SQLiteMetadataStore(MetadataStore):
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    file_path TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    last_modified REAL NOT NULL,
                    last_ingested_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    title_path TEXT,
                    content TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(file_path) REFERENCES documents(file_path)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
                """
            )

    def get_document_by_path(self, file_path: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_path = ?",
                (file_path,),
            ).fetchone()
        return dict(row) if row else None

    def upsert_document(self, document: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    file_path, file_name, content_hash, last_modified,
                    last_ingested_at, status, chunk_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_name = excluded.file_name,
                    content_hash = excluded.content_hash,
                    last_modified = excluded.last_modified,
                    last_ingested_at = excluded.last_ingested_at,
                    status = excluded.status,
                    chunk_count = excluded.chunk_count
                """,
                (
                    document["file_path"],
                    document["file_name"],
                    document["content_hash"],
                    document["last_modified"],
                    document["last_ingested_at"],
                    document["status"],
                    document["chunk_count"],
                ),
            )

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY file_path").fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, file_path: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))

    def upsert_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, file_path, title_path, content,
                    char_count, chunk_index, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    title_path = excluded.title_path,
                    content = excluded.content,
                    char_count = excluded.char_count,
                    chunk_index = excluded.chunk_index,
                    created_at = excluded.created_at
                """,
                [
                    (
                        chunk["chunk_id"],
                        chunk["file_path"],
                        chunk.get("title_path", ""),
                        chunk["content"],
                        chunk["char_count"],
                        chunk["chunk_index"],
                        chunk["created_at"],
                    )
                    for chunk in chunks
                ],
            )

    def delete_chunks_by_file_path(self, file_path: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []

        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()

        chunk_map = {row["chunk_id"]: dict(row) for row in rows}
        return [chunk_map[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_map]

    def search_chunks_by_keywords(self, keywords: list[str], limit: int = 20) -> list[dict]:
        keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        if not keywords:
            return []

        clauses: list[str] = []
        params: list[str] = []
        for keyword in keywords:
            clauses.append("(content LIKE ? OR title_path LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])

        sql = f"""
            SELECT * FROM chunks
            WHERE {" OR ".join(clauses)}
            ORDER BY file_path, chunk_index
            LIMIT ?
        """
        params.append(str(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]
