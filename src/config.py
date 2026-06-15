from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def resolve_path(root_dir: str | Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(root_dir) / path


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config["root_dir"] = str(config_file.parent)
    paths = config.setdefault("paths", {})
    required_dirs = ("notes_dir", "chroma_dir", "processed_dir")

    for key in required_dirs:
        path = resolve_path(config["root_dir"], paths[key])
        path.mkdir(parents=True, exist_ok=True)

    metadata_db = resolve_path(config["root_dir"], paths["metadata_db"])
    metadata_db.parent.mkdir(parents=True, exist_ok=True)

    eval_queries = resolve_path(config["root_dir"], paths["eval_queries"])
    eval_queries.parent.mkdir(parents=True, exist_ok=True)

    return config
