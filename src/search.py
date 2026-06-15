from __future__ import annotations

import re
import sys

import typer
from rich.console import Console
from rich.panel import Panel

from src.config import load_config, resolve_path
from src.embedder import LocalEmbedder
from src.metadata_store import MetadataStore
from src.sqlite_metadata_store import SQLiteMetadataStore
from src.vector_store import ChromaVectorStore

app = typer.Typer(add_completion=False)
console = Console()
QUERY_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def extract_query_keywords(query: str) -> list[str]:
    keywords = [token.strip() for token in re.split(r"\s+", query) if token.strip()]
    if keywords:
        return keywords
    return [token for token in QUERY_TOKEN_RE.findall(query) if len(token) >= 2]


def _keyword_score(chunk: dict, keywords: list[str]) -> float:
    text = f"{chunk.get('title_path', '')}\n{chunk.get('content', '')}".lower()
    if not keywords:
        return 0.0

    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    if hits == 0:
        return 0.0
    return 0.7 + 0.3 * (hits / len(keywords))


def _keyword_results(chunks: list[dict], keywords: list[str]) -> list[dict]:
    results: list[dict] = []
    for chunk in chunks:
        score = _keyword_score(chunk, keywords)
        if score <= 0:
            continue
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "file_path": chunk["file_path"],
                "title_path": chunk.get("title_path", ""),
                "char_count": chunk.get("char_count", len(chunk.get("content", ""))),
                "content": chunk["content"],
                "score": score,
            }
        )
    return results


def _merge_results(vector_results: list[dict], keyword_results: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for result in vector_results + keyword_results:
        chunk_id = result["chunk_id"]
        if chunk_id not in merged or result["score"] > merged[chunk_id]["score"]:
            merged[chunk_id] = result
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)


def _filter_results(
    results: list[dict],
    min_score: float,
    relative_score_ratio: float,
    top_k: int,
) -> list[dict]:
    if not results:
        return []

    filtered = [result for result in results if result["score"] >= min_score]
    if not filtered:
        return []

    best_score = filtered[0]["score"]
    if relative_score_ratio > 0:
        filtered = [
            result
            for result in filtered
            if best_score == 0 or result["score"] >= best_score * relative_score_ratio
        ]
    return filtered[:top_k]


def search_chunks(
    query: str,
    config_path: str = "config.yaml",
    top_k: int | None = None,
    min_score: float | None = None,
    relative_score_ratio: float | None = None,
    embedder: LocalEmbedder | None = None,
    vector_store: ChromaVectorStore | None = None,
    metadata_store: MetadataStore | None = None,
) -> list[dict]:
    config = load_config(config_path)
    top_k = top_k or config["search"]["top_k"]
    min_score = config["search"].get("min_score", 0.0) if min_score is None else min_score
    relative_score_ratio = (
        config["search"].get("relative_score_ratio", 0.0)
        if relative_score_ratio is None
        else relative_score_ratio
    )
    chroma_dir = resolve_path(config["root_dir"], config["paths"]["chroma_dir"])
    metadata_db = resolve_path(config["root_dir"], config["paths"]["metadata_db"])

    embedder = embedder or LocalEmbedder(config["embedding"]["model_name"])
    vector_store = vector_store or ChromaVectorStore(str(chroma_dir))
    metadata_store = metadata_store or SQLiteMetadataStore(str(metadata_db))

    query_embedding = embedder.embed_query(query)
    vector_results = vector_store.search(query_embedding, top_k=max(top_k * 3, top_k))

    keywords = extract_query_keywords(query)
    keyword_chunks = metadata_store.search_chunks_by_keywords(
        keywords,
        limit=config["search"].get("keyword_top_k", top_k * 4),
    )
    merged = _merge_results(vector_results, _keyword_results(keyword_chunks, keywords))
    return _filter_results(
        merged,
        min_score=min_score,
        relative_score_ratio=relative_score_ratio,
        top_k=top_k,
    )


def render_results(query: str, results: list[dict], preview_chars: int = 500) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    console.print(f"Query: {query}\n")

    if not results:
        console.print("[yellow]没有检索到结果，请先运行 ingest 或调整查询词。[/yellow]")
        return

    for index, result in enumerate(results, start=1):
        preview = result["content"][:preview_chars].strip().replace("\xa0", " ")
        body = (
            f"source: {result['file_path']}\n"
            f"title: {result['title_path']}\n"
            f"score: {result['score']:.4f}\n\n"
            f"{preview}"
        )
        console.print(Panel(body, title=f"[{index}]"))


@app.command()
def main(
    query: str = typer.Argument(..., help="检索问题"),
    top_k: int = typer.Option(None, "--top-k", help="返回结果数量"),
    min_score: float | None = typer.Option(None, "--min-score", help="最低相似度分数"),
) -> None:
    try:
        config = load_config()
        results = search_chunks(query=query, top_k=top_k, min_score=min_score)
        render_results(query, results, preview_chars=config["search"]["preview_chars"])
    except Exception as exc:  # pragma: no cover - CLI 错误出口
        console.print(f"[red]search 失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
