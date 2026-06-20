from pathlib import Path

from src.vector_store import ChromaVectorStore


def _chunk(cid: str, domain: str, note_type: str) -> dict:
    return {
        "chunk_id": cid,
        "file_path": f"{domain}/{note_type}/x.md",
        "title_path": "T",
        "content": "内容",
        "char_count": 2,
        "chunk_index": 1,
        "domain": domain,
        "type": note_type,
        "evidence_level": "Primary source | User view",
        "freshness": "Stable",
    }


def test_vector_store_where_filters_by_domain(tmp_path: Path) -> None:
    store = ChromaVectorStore(str(tmp_path / "chroma"))
    store.upsert_chunks(
        [_chunk("stock/c/x.md::0001", "stock", "companies"),
         _chunk("study/c/x.md::0001", "study", "concepts")],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    results = store.search([1.0, 0.0], top_k=5, where={"domain": "stock"})

    assert [r["file_path"] for r in results] == ["stock/companies/x.md"]
    assert results[0]["domain"] == "stock"
    assert results[0]["evidence_level"] == "Primary source | User view"
