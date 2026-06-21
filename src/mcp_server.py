from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.config import load_config, resolve_path
from src.embedder import LocalEmbedder
from src.search import search_chunks
from src.sqlite_metadata_store import SQLiteMetadataStore
from src.vector_store import ChromaVectorStore

mcp = FastMCP("markdown-rag")

# 惰性常驻:首次查询或 __main__ 启动时加载一次,之后每轮查询不再重载。
# 用惰性而非模块顶层加载,是为了让单元测试能 monkeypatch 这三个组件、不拉真模型。
_embedder = None
_vector_store = None
_metadata_store = None


def _load() -> None:
    global _embedder, _vector_store, _metadata_store
    if _embedder is not None:
        return
    config = load_config("config.yaml")
    _embedder = LocalEmbedder(
        config["embedding"]["model_name"],
        query_instruction=config["embedding"].get("query_instruction", ""),
    )
    _vector_store = ChromaVectorStore(
        str(resolve_path(config["root_dir"], config["paths"]["chroma_dir"]))
    )
    _metadata_store = SQLiteMetadataStore(
        str(resolve_path(config["root_dir"], config["paths"]["metadata_db"]))
    )


@mcp.tool()
def search_notes(query: str, domain: str | None = None, top_k: int = 5) -> list[dict]:
    """检索本地投资/学习笔记 wiki。domain 可选:投资问题传 "stock",学习问题传 "study",
    不确定就不传(全库)。返回命中页的标题、路径、相关分、正文片段,供你综合分析。"""
    _load()
    results = search_chunks(
        query=query,
        config_path="config.yaml",
        top_k=top_k,
        domain=domain,
        embedder=_embedder,
        vector_store=_vector_store,
        metadata_store=_metadata_store,
        reranker=None,  # 多轮场景不开 reranker(每轮 ~28s 不可接受),走快的混合检索
    )
    return [
        {
            "title": r["title_path"],
            "file_path": r["file_path"],
            "score": round(r["score"], 4),
            "content": r["content"],
        }
        for r in results
    ]


if __name__ == "__main__":
    _load()  # 启动即加载,缺模型/索引立刻 fail-fast
    mcp.run()  # 默认 stdio
