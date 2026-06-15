from pathlib import Path

from src.context_builder import build_context, save_analysis, write_context


def test_context_builder_contains_required_fields() -> None:
    results = [
        {
            "file_path": "notes/a.md",
            "title_path": "公司分析 > 渠道",
            "score": 0.91,
            "content": "这是一个测试内容。",
        }
    ]
    context = build_context(results, query="测试问题", max_chars=500)
    assert "用户问题：测试问题" in context
    assert "来源：notes/a.md" in context
    assert "标题路径：公司分析 > 渠道" in context


def test_context_builder_respects_max_chars() -> None:
    results = [
        {
            "file_path": "notes/a.md",
            "title_path": "标题",
            "score": 0.9,
            "content": "内容" * 200,
        }
    ]
    context = build_context(results, query="测试问题", max_chars=120)
    assert len(context) <= 120


def test_context_builder_handles_empty_results() -> None:
    context = build_context([], query="测试问题", max_chars=100)
    assert "没有检索到相关资料" in context


def test_write_context_overwrites_temp_file(tmp_path: Path) -> None:
    output = tmp_path / "context.md"
    write_context("旧内容", output=str(output))
    write_context("新内容", output=str(output))
    assert output.read_text(encoding="utf-8") == "新内容"


def test_save_analysis_writes_research_outputs_file(tmp_path: Path) -> None:
    analysis_path = save_analysis("分析内容", output_dir=str(tmp_path / "research_outputs"))
    assert analysis_path.parent.name == "research_outputs"
    assert analysis_path.name.startswith("analysis_")
    assert analysis_path.read_text(encoding="utf-8") == "分析内容"
