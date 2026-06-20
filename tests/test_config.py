from pathlib import Path

import pytest
import yaml

from src.config import load_config


def _write(root: Path, source_dir: Path) -> Path:
    config = {
        "project": {"name": "t"},
        "paths": {
            "chroma_dir": "data/chroma",
            "processed_dir": "data/processed",
            "metadata_db": "data/metadata.sqlite",
            "eval_queries": "eval/queries.yaml",
        },
        "sources": [{"path": str(source_dir), "domain": "stock"}],
        "embedding": {"provider": "fake", "model_name": "fake"},
        "chunking": {"target_chars": 80, "max_chars": 160, "min_chars": 40, "overlap_chars": 20},
        "search": {"top_k": 5, "preview_chars": 120},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


def test_load_config_accepts_sources(tmp_path: Path) -> None:
    source_dir = tmp_path / "wiki"
    source_dir.mkdir()
    config_path = _write(tmp_path, source_dir)

    config = load_config(str(config_path))

    assert config["sources"][0]["domain"] == "stock"
    assert config["exclude_names"] == []
    assert config["exclude_dirs"] == []


def test_load_config_rejects_missing_source(tmp_path: Path) -> None:
    config_path = _write(tmp_path, tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        load_config(str(config_path))
