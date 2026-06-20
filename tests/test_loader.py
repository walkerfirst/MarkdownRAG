from pathlib import Path

from src.loader import load_markdown_files


def _make_wiki(root: Path) -> None:
    (root / "companies").mkdir(parents=True)
    (root / "templates").mkdir(parents=True)
    (root / "companies" / "600519.md").write_text(
        "# 600519 贵州茅台\n\n"
        "**Summary**: 高端白酒龙头,护城河来自品牌与渠道。\n"
        "**Sources**: `raw/brokerage/2026-05-02__x.xls`\n"
        "**Last updated**: 2026-05-02\n"
        "**Freshness**: Stable\n"
        "**Evidence level**: Primary source | User view\n\n"
        "---\n\n"
        "## 业务\n\n白酒主业稳定。\n",
        encoding="utf-8",
    )
    (root / "index.md").write_text("# 索引\n\n[[companies/]]\n", encoding="utf-8")
    (root / "templates" / "company.md").write_text("# 模板\n\n占位。\n", encoding="utf-8")


def test_loader_parses_header_and_classifies(tmp_path: Path) -> None:
    root = tmp_path / "investing" / "wiki"
    _make_wiki(root)

    files = load_markdown_files(
        [{"path": str(root), "domain": "stock"}],
        exclude_names=["index.md", "log.md"],
        exclude_dirs=["templates"],
    )

    assert len(files) == 1
    item = files[0]
    assert item["file_path"] == "stock/companies/600519.md"
    assert item["domain"] == "stock"
    assert item["type"] == "companies"
    assert item["evidence_level"] == "Primary source | User view"
    assert item["freshness"] == "Stable"
    assert item["last_updated"] == "2026-05-02"
    # 正文:保留 H1 标题 + Summary,丢弃 Sources/bold-key 行
    assert "600519 贵州茅台" in item["content"]
    assert "护城河来自品牌与渠道" in item["content"]
    assert "白酒主业稳定" in item["content"]
    assert "raw/brokerage" not in item["content"]
    assert "**Evidence level**" not in item["content"]


def test_loader_no_header_keeps_whole_body(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    (root / "journal").mkdir(parents=True)
    (root / "journal" / "x.md").write_text("# 随记\n\n第一段。\n\n第二段。\n", encoding="utf-8")

    files = load_markdown_files([{"path": str(root), "domain": "stock"}])

    assert len(files) == 1
    assert files[0]["type"] == "journal"
    assert files[0]["evidence_level"] == ""
    assert "第一段" in files[0]["content"] and "第二段" in files[0]["content"]
