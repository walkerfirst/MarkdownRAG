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
            "domain": "stock",
            "type": "companies",
            "evidence_level": "Primary",
            "freshness": "Stable",
            "last_updated": "2026-05-02",
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
            "domain": "stock",
            "type": "companies",
            "evidence_level": "Primary",
            "freshness": "Stable",
            "last_updated": "2026-05-02",
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
                "domain": "stock",
                "type": "companies",
                "evidence_level": "",
                "freshness": "",
            },
            {
                "chunk_id": "a.md::0002",
                "file_path": "notes/a.md",
                "title_path": "公司分析 > 渠道",
                "content": "内容2",
                "char_count": 3,
                "chunk_index": 2,
                "created_at": "2024-01-01T00:00:00+00:00",
                "domain": "stock",
                "type": "companies",
                "evidence_level": "",
                "freshness": "",
            },
        ]
    )
    chunks = store.get_chunks_by_ids(["a.md::0001", "a.md::0002"])
    assert len(chunks) == 2

    store.delete_chunks_by_file_path("notes/a.md")
    assert store.get_chunks_by_ids(["a.md::0001", "a.md::0002"]) == []


def test_metadata_store_filters_keywords_by_domain_and_type(tmp_path: Path) -> None:
    store = SQLiteMetadataStore(str(tmp_path / "m.sqlite"))
    store.init_schema()
    store.upsert_chunks(
        [
            {
                "chunk_id": "stock/companies/a.md::0001",
                "file_path": "stock/companies/a.md",
                "title_path": "A",
                "content": "牧原 屠宰 护城河",
                "char_count": 8,
                "chunk_index": 1,
                "created_at": "2024-01-01T00:00:00+00:00",
                "domain": "stock",
                "type": "companies",
                "evidence_level": "Primary source | User view",
                "freshness": "Stable",
            },
            {
                "chunk_id": "study/concepts/b.md::0001",
                "file_path": "study/concepts/b.md",
                "title_path": "B",
                "content": "牧原 屠宰 另一处",
                "char_count": 8,
                "chunk_index": 1,
                "created_at": "2024-01-01T00:00:00+00:00",
                "domain": "study",
                "type": "concepts",
                "evidence_level": "",
                "freshness": "",
            },
        ]
    )

    hits = store.search_chunks_by_keywords(["牧原"], domain="stock")
    assert [row["file_path"] for row in hits] == ["stock/companies/a.md"]
    assert hits[0]["evidence_level"] == "Primary source | User view"

    typed = store.search_chunks_by_keywords(["牧原"], note_type="concepts")
    assert [row["file_path"] for row in typed] == ["study/concepts/b.md"]
