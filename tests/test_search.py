from pathlib import Path

import yaml

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


class FakeReranker:
    """把候选倒序并重新赋分,证明 reranker 输出顺序确实流到最终结果。"""

    def rerank(self, query, candidates):
        reversed_candidates = list(reversed(candidates))
        for index, candidate in enumerate(reversed_candidates):
            candidate["score"] = 1.0 - index * 0.1
        return reversed_candidates


def _write_reranker_config(tmp_path: Path) -> Path:
    source_dir = tmp_path / "wiki"
    source_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "project": {"name": "t"},
        "paths": {
            "chroma_dir": "data/chroma",
            "processed_dir": "data/processed",
            "metadata_db": "data/metadata.sqlite",
            "eval_queries": "eval/queries.yaml",
        },
        "sources": [{"path": str(source_dir), "domain": "stock"}],
        "embedding": {"provider": "fake", "model_name": "fake"},
        "chunking": {"target_chars": 80, "max_chars": 160, "min_chars": 40, "overlap_chars": 20},
        "search": {"top_k": 5, "min_score": 0.0, "relative_score_ratio": 0.0, "keyword_top_k": 20},
        "reranker": {"enabled": True, "model_name": "x", "candidate_k": 30},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


def test_search_applies_reranker_when_enabled(tmp_path: Path) -> None:
    config_path = _write_reranker_config(tmp_path)
    # 不开重排时 stock/c/a.md(0.72)排第一;FakeReranker 倒序后应换成 study/c/b.md
    results = search_chunks(
        query="测试",
        config_path=str(config_path),
        embedder=FakeEmbedder(),
        vector_store=FilterVectorStore(),
        metadata_store=EmptyKeywordStore(),
        reranker=FakeReranker(),
    )
    assert results[0]["file_path"] == "study/c/b.md"
