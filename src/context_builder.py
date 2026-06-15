from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer

from src.search import search_chunks

app = typer.Typer(add_completion=False)
DEFAULT_CONTEXT_FILE = "context.md"
RESEARCH_OUTPUTS_DIR = "research_outputs"


def build_context(results: list[dict], query: str, max_chars: int = 6000) -> str:
    if not results:
        text = f"用户问题：{query}\n\n没有检索到相关资料。请先运行 ingest，或换一组更准确的关键词。"
        return text[:max_chars]

    parts = [
        "以下是从本地知识库检索到的相关资料：",
        "",
        f"用户问题：{query}",
        "",
    ]
    current_length = len("\n".join(parts))

    for index, result in enumerate(results, start=1):
        block = (
            f"[资料{index}]\n"
            f"来源：{result['file_path']}\n"
            f"标题路径：{result['title_path']}\n"
            f"相似度：{result['score']:.4f}\n"
            f"内容：\n{result['content'].strip()}\n"
        )
        if current_length + len(block) <= max_chars:
            parts.append(block)
            parts.append("")
            current_length = len("\n".join(parts))
            continue

        remaining = max_chars - current_length - 40
        if remaining > 0:
            trimmed_content = result["content"][:remaining].strip()
            parts.append(
                f"[资料{index}]\n"
                f"来源：{result['file_path']}\n"
                f"标题路径：{result['title_path']}\n"
                f"相似度：{result['score']:.4f}\n"
                f"内容：\n{trimmed_content}\n"
            )
        break

    return "\n".join(parts)[:max_chars]


def write_context(context: str, output: str = DEFAULT_CONTEXT_FILE) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(context, encoding="utf-8")
    return output_path


def save_analysis(context: str, output_dir: str = RESEARCH_OUTPUTS_DIR) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_path = output_path / f"analysis_{timestamp}.md"
    analysis_path.write_text(context, encoding="utf-8")
    return analysis_path


@app.command()
def main(
    query: str = typer.Argument(..., help="检索问题"),
    top_k: int = typer.Option(None, "--top-k", help="返回结果数量"),
    min_score: float | None = typer.Option(None, "--min-score", help="最低相似度分数"),
    relative_score_ratio: float | None = typer.Option(None, "--relative-score-ratio", help="相对最高分的保留比例"),
    max_chars: int = typer.Option(6000, "--max-chars", help="最大上下文字数"),
    output: str = typer.Option(DEFAULT_CONTEXT_FILE, "--output", help="临时上下文文件，默认覆盖 context.md"),
    stdout: bool = typer.Option(False, "--stdout", help="输出到 stdout，用于管道传递"),
    save_history: bool = typer.Option(False, "--save-analysis", help="另存一份到 research_outputs/"),
) -> None:
    try:
        results = search_chunks(
            query=query,
            top_k=top_k,
            min_score=min_score,
            relative_score_ratio=relative_score_ratio,
        )
        context = build_context(results=results, query=query, max_chars=max_chars)
        if stdout:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(context)
        else:
            path = write_context(context, output=output)
            typer.echo(f"Context written to: {path}")
        if save_history:
            analysis_path = save_analysis(context)
            typer.echo(f"Analysis saved to: {analysis_path}")
    except Exception as exc:  # pragma: no cover - CLI 错误出口
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
