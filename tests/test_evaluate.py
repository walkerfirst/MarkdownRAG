from pathlib import Path

import yaml


def test_wiki_queries_schema() -> None:
    """真实评测集结构自检:不依赖 corpus/模型,只校验 yaml 格式与路径约定。"""
    payload = yaml.safe_load(Path("eval/queries.wiki.yaml").read_text(encoding="utf-8"))
    queries = payload["queries"]

    assert len(queries) >= 10
    for item in queries:
        assert item["query"].strip()
        assert item["expected_files"]
        assert all(path.startswith(("stock/", "study/")) for path in item["expected_files"])
        assert item["expected_keywords"]
