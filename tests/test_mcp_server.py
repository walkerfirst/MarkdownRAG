import src.mcp_server as mcp_server


def _inject_loaded(monkeypatch):
    # 注入非 None 占位,让 _load() 直接返回、不拉真模型
    monkeypatch.setattr(mcp_server, "_embedder", object())
    monkeypatch.setattr(mcp_server, "_vector_store", object())
    monkeypatch.setattr(mcp_server, "_metadata_store", object())


def test_search_notes_returns_structured_hits(monkeypatch):
    _inject_loaded(monkeypatch)

    def fake_search_chunks(**kwargs):
        return [
            {
                "title_path": "牧原股份财务记录 (002714) > 业务结构",
                "file_path": "stock/companies/002714-financials.md",
                "score": 0.987654,
                "content": "牧原屠宰业务...",
                "domain": "stock",
            }
        ]

    monkeypatch.setattr(mcp_server, "search_chunks", fake_search_chunks)

    out = mcp_server.search_notes("牧原 屠宰")
    assert out == [
        {
            "title": "牧原股份财务记录 (002714) > 业务结构",
            "file_path": "stock/companies/002714-financials.md",
            "score": 0.9877,
            "content": "牧原屠宰业务...",
        }
    ]


def test_search_notes_passes_domain_and_disables_reranker(monkeypatch):
    _inject_loaded(monkeypatch)
    captured = {}

    def fake_search_chunks(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(mcp_server, "search_chunks", fake_search_chunks)

    result = mcp_server.search_notes("价值投资", domain="study", top_k=8)
    assert result == []
    assert captured["domain"] == "study"
    assert captured["top_k"] == 8
    assert captured["reranker"] is None
