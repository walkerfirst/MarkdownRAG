from pathlib import Path

from src.sqlite_metadata_store import SQLiteMetadataStore


def test_metadata_store_crud(tmp_path: Path) -> None:
    store = SQLiteMetadataStore(str(tmp_path / "metadata.sqlite"))
    store.init_schema()

    store.upsert_document(
        {
            "file_path": "notes/a.md",
            "file_name": "a.md",
            "content_hash": "hash1",
            "last_modified": 1.0,
            "last_ingested_at": "2024-01-01T00:00:00+00:00",
            "status": "indexed",
            "chunk_count": 1,
        }
    )
    assert store.get_document_by_path("notes/a.md")["content_hash"] == "hash1"

    store.upsert_document(
        {
            "file_path": "notes/a.md",
            "file_name": "a.md",
            "content_hash": "hash2",
            "last_modified": 2.0,
            "last_ingested_at": "2024-01-02T00:00:00+00:00",
            "status": "indexed",
            "chunk_count": 2,
        }
    )
    assert store.get_document_by_path("notes/a.md")["content_hash"] == "hash2"

    store.upsert_chunks(
        [
            {
                "chunk_id": "a.md::0001",
                "file_path": "notes/a.md",
                "title_path": "公司分析",
                "content": "内容1",
                "char_count": 3,
                "chunk_index": 1,
                "created_at": "2024-01-01T00:00:00+00:00",
            },
            {
                "chunk_id": "a.md::0002",
                "file_path": "notes/a.md",
                "title_path": "公司分析 > 渠道",
                "content": "内容2",
                "char_count": 3,
                "chunk_index": 2,
                "created_at": "2024-01-01T00:00:00+00:00",
            },
        ]
    )
    chunks = store.get_chunks_by_ids(["a.md::0001", "a.md::0002"])
    assert len(chunks) == 2

    store.delete_chunks_by_file_path("notes/a.md")
    assert store.get_chunks_by_ids(["a.md::0001", "a.md::0002"]) == []
