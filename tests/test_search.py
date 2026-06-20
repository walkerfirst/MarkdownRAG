from src.search import search_chunks


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def search(self, query_embedding: list[float], top_k: int = 5, where=None) -> list[dict]:
        return [
            {
                "chunk_id": "a.md::0001",
                "file_path": "notes/a.md",
                "title_path": "A",
                "content": "高相关",
                "score": 0.72,
                "domain": "stock",
                "type": "companies",
                "evidence_level": "Primary source",
            },
            {
                "chunk_id": "b.md::0001",
                "file_path": "notes/b.md",
                "title_path": "B",
                "content": "低相关",
                "score": 0.42,
                "domain": "study",
                "type": "concepts",
                "evidence_level": "Unverified",
            },
        ]


class FakeMetadataStore:
    def search_chunks_by_keywords(
        self, keywords: list[str], limit: int = 20, domain=None, note_type=None
    ) -> list[dict]:
        return []


def test_search_chunks_filters_by_min_score() -> None:
    results = search_chunks(
        query="测试",
        min_score=0.5,
        relative_score_ratio=0.0,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        metadata_store=FakeMetadataStore(),
    )
    assert [result["file_path"] for result in results] == ["notes/a.md"]


def test_search_chunks_uses_keyword_matches() -> None:
    class KeywordMetadataStore:
        def search_chunks_by_keywords(
            self, keywords: list[str], limit: int = 20, domain=None, note_type=None
        ) -> list[dict]:
            return [
                {
                    "chunk_id": "tesla.md::0001",
                    "file_path": "notes/tesla.md",
                    "title_path": "特斯拉",
                    "content": "特斯拉和马斯克相关内容",
                    "char_count": 12,
                    "domain": "stock",
                    "type": "companies",
                    "evidence_level": "Secondary source",
                    "freshness": "Stable",
                }
            ]

    results = search_chunks(
        query="特斯拉 马斯克",
        min_score=0.0,
        relative_score_ratio=0.7,
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        metadata_store=KeywordMetadataStore(),
    )
    assert results[0]["file_path"] == "notes/tesla.md"


class FilterVectorStore:
    def search(self, query_embedding, top_k=5, where=None):
        rows = [
            {
                "chunk_id": "stock/c/a.md::0001",
                "file_path": "stock/c/a.md",
                "title_path": "A",
                "content": "高相关",
                "score": 0.72,
                "domain": "stock",
                "type": "companies",
                "evidence_level": "Primary source | User view",
                "freshness": "Stable",
            },
            {
                "chunk_id": "study/c/b.md::0001",
                "file_path": "study/c/b.md",
                "title_path": "B",
                "content": "次相关",
                "score": 0.6,
                "domain": "study",
                "type": "concepts",
                "evidence_level": "Unverified",
                "freshness": "Stale risk",
            },
        ]
        if where and "domain" in where:
            rows = [r for r in rows if r["domain"] == where["domain"]]
        return rows


class EmptyKeywordStore:
    def search_chunks_by_keywords(self, keywords, limit=20, domain=None, note_type=None):
        return []


def test_search_chunks_filters_by_domain() -> None:
    results = search_chunks(
        query="测试",
        min_score=0.0,
        relative_score_ratio=0.0,
        domain="stock",
        embedder=FakeEmbedder(),
        vector_store=FilterVectorStore(),
        metadata_store=EmptyKeywordStore(),
    )
    assert [r["file_path"] for r in results] == ["stock/c/a.md"]


def test_search_chunks_filters_by_evidence_substring() -> None:
    results = search_chunks(
        query="测试",
        min_score=0.0,
        relative_score_ratio=0.0,
        evidence="Primary",
        embedder=FakeEmbedder(),
        vector_store=FilterVectorStore(),
        metadata_store=EmptyKeywordStore(),
    )
    assert [r["file_path"] for r in results] == ["stock/c/a.md"]
