from __future__ import annotations

from pathlib import Path


def load_markdown_files(notes_dir: str) -> list[dict]:
    notes_path = Path(notes_dir).resolve()
    if not notes_path.exists():
        return []

    files: list[dict] = []
    for file_path in sorted(notes_path.rglob("*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"文件编码不是 UTF-8: {file_path}") from exc
        except OSError as exc:
            raise OSError(f"读取 Markdown 文件失败: {file_path}") from exc

        if not content.strip():
            continue

        relative_path = file_path.relative_to(notes_path.parent).as_posix()
        files.append(
            {
                "file_path": relative_path,
                "file_name": file_path.name,
                "content": content,
                "last_modified": file_path.stat().st_mtime,
            }
        )

    return files
