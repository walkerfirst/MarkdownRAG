from src.chunker import chunk_markdown


def test_chunker_creates_chunks_with_titles() -> None:
    text = (
        "# 公司分析\n\n"
        "## 业务结构\n\n"
        "这里是第一段。" * 20
        + "\n\n### 渠道\n\n"
        + "这里是第二段。" * 20
    )
    chunks = chunk_markdown(text, "notes/demo.md", target_chars=80, max_chars=160, min_chars=40)
    assert chunks
    assert all(chunk["chunk_id"] for chunk in chunks)
    assert all(chunk["title_path"] for chunk in chunks)
    assert all(chunk["char_count"] <= 160 for chunk in chunks)


def test_chunker_returns_empty_for_blank_text() -> None:
    assert chunk_markdown("", "notes/demo.md") == []
