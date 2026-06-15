from src.search import search_chunks


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeVectorStore:
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        return [
            {
                "chunk_id": "a.md::0001",
                "file_path": "notes/a.md",
                "title_path": "A",
                "content": "高相关",
                "score": 0.72,
            },
            {
                "chunk_id": "b.md::0001",
                "file_path": "notes/b.md",
                "title_path": "B",
                "content": "低相关",
                "score": 0.42,
            },
        ]


class FakeMetadataStore:
    def search_chunks_by_keywords(self, keywords: list[str], limit: int = 20) -> list[dict]:
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
        def search_chunks_by_keywords(self, keywords: list[str], limit: int = 20) -> list[dict]:
            return [
                {
                    "chunk_id": "tesla.md::0001",
                    "file_path": "notes/tesla.md",
                    "title_path": "特斯拉",
                    "content": "特斯拉和马斯克相关内容",
                    "char_count": 12,
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
