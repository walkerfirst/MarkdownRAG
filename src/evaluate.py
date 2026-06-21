from __future__ import annotations

import typer
import yaml
from rich.console import Console

from src.config import load_config, resolve_path
from src.embedder import LocalEmbedder
from src.reranker import LocalReranker
from src.search import search_chunks
from src.sqlite_metadata_store import SQLiteMetadataStore
from src.vector_store import ChromaVectorStore

app = typer.Typer(add_completion=False)
console = Console()


def run_evaluation(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    eval_file = resolve_path(config["root_dir"], config["paths"]["eval_queries"])
    payload = yaml.safe_load(eval_file.read_text(encoding="utf-8")) or {}
    queries = payload.get("queries", [])

    # 重量级组件只加载一次再复用(尤其 2.2GB reranker),避免每条 query 重载
    embedder = LocalEmbedder(
        config["embedding"]["model_name"],
        query_instruction=config["embedding"].get("query_instruction", ""),
    )
    vector_store = ChromaVectorStore(str(resolve_path(config["root_dir"], config["paths"]["chroma_dir"])))
    metadata_store = SQLiteMetadataStore(str(resolve_path(config["root_dir"], config["paths"]["metadata_db"])))
    reranker_cfg = config.get("reranker", {})
    reranker = (
        LocalReranker(
            reranker_cfg["model_name"],
            max_passage_chars=reranker_cfg.get("max_passage_chars", 0),
        )
        if reranker_cfg.get("enabled")
        else None
    )

    passed = 0
    details: list[dict] = []

    for item in queries:
        # domain 是真实使用时已知的"选库"维度(投资走 stock、学习走 study),
        # 故评测按问题 domain 过滤;显式 domain 字段优先,否则从 expected_files 前缀推断。
        expected_files = item.get("expected_files", [])
        domain = item.get("domain") or (
            expected_files[0].split("/")[0] if expected_files else None
        )
        results = search_chunks(
            query=item["query"],
            config_path=config_path,
            top_k=config["search"]["top_k"],
            domain=domain,
            embedder=embedder,
            vector_store=vector_store,
            metadata_store=metadata_store,
            reranker=reranker,
        )
        files = {result["file_path"] for result in results}
        combined_text = "\n".join(result["content"] for result in results)
        file_ok = any(path in files for path in item.get("expected_files", []))
        keyword_ok = all(keyword in combined_text for keyword in item.get("expected_keywords", []))
        success = file_ok and keyword_ok
        if success:
            passed += 1
        details.append(
            {
                "query": item["query"],
                "passed": success,
                "file_ok": file_ok,
                "keyword_ok": keyword_ok,
            }
        )

    total = len(queries)
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "details": details,
    }


@app.command()
def main() -> None:
    try:
        summary = run_evaluation()
    except Exception as exc:  # pragma: no cover - CLI 错误出口
        console.print(f"[red]evaluate 失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print("Evaluation result:")
    for detail in summary["details"]:
        mark = "[green]PASS[/green]" if detail["passed"] else "[red]FAIL[/red]"
        flags = f"file={'ok' if detail['file_ok'] else 'x'} kw={'ok' if detail['keyword_ok'] else 'x'}"
        console.print(f"  {mark} ({flags}) {detail['query']}")
    console.print(f"Total queries: {summary['total']}")
    console.print(f"Passed: {summary['passed']}")
    console.print(f"Failed: {summary['failed']}")
    console.print(f"Pass rate: {summary['pass_rate']:.1f}%")

    if summary["pass_rate"] < 80:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
