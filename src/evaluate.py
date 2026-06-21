from __future__ import annotations

import typer
import yaml
from rich.console import Console

from src.config import load_config, resolve_path
from src.search import search_chunks

app = typer.Typer(add_completion=False)
console = Console()


def run_evaluation(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    eval_file = resolve_path(config["root_dir"], config["paths"]["eval_queries"])
    payload = yaml.safe_load(eval_file.read_text(encoding="utf-8")) or {}
    queries = payload.get("queries", [])

    passed = 0
    details: list[dict] = []

    for item in queries:
        results = search_chunks(
            query=item["query"],
            config_path=config_path,
            top_k=config["search"]["top_k"],
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
