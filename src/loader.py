from __future__ import annotations

import re
from pathlib import Path


HEADING_RE = re.compile(r"^#{1,6}\s+\S")
BOLD_KEY_RE = re.compile(r"^\*\*([^*]+)\*\*\s*[:：]\s*(.+?)\s*$")


def _split_header(content: str) -> tuple[dict[str, str], str]:
    """按第一个独立 '---' 行切分页头。

    仅当 '---' 之前存在 bold-key 行时才视为元数据头并剥离;
    否则整篇当正文(避免误伤把 '---' 当内容分隔线的页面)。
    重建正文 = 保留的标题行 + Summary 值 + '---' 之后正文。
    """
    lines = content.splitlines()
    sep_idx = next((i for i, line in enumerate(lines) if line.strip() == "---"), None)
    if sep_idx is None:
        return {}, content.strip()

    header_block = lines[:sep_idx]
    fields: dict[str, str] = {}
    kept_headings: list[str] = []
    for line in header_block:
        match = BOLD_KEY_RE.match(line.strip())
        if match:
            fields[match.group(1).strip().lower()] = match.group(2).strip()
        elif HEADING_RE.match(line):
            kept_headings.append(line)

    if not fields:
        return {}, content.strip()

    after = "\n".join(lines[sep_idx + 1:]).strip()
    summary = fields.get("summary", "")
    parts = [*kept_headings]
    if summary:
        parts.append(summary)
    if after:
        parts.append(after)
    body = "\n\n".join(part for part in parts if part.strip()).strip()
    return fields, body


def load_markdown_files(
    sources: list[dict],
    exclude_names: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
) -> list[dict]:
    exclude_name_set = set(exclude_names or [])
    exclude_dir_set = set(exclude_dirs or [])

    files: list[dict] = []
    for source in sources:
        root = Path(source["path"]).resolve()
        domain = source["domain"]
        if not root.exists():
            continue

        for file_path in sorted(root.rglob("*.md")):
            if file_path.name in exclude_name_set:
                continue
            rel = file_path.relative_to(root)
            if any(part in exclude_dir_set for part in rel.parts[:-1]):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"文件编码不是 UTF-8: {file_path}") from exc
            except OSError as exc:
                raise OSError(f"读取 Markdown 文件失败: {file_path}") from exc

            if not content.strip():
                continue

            fields, body = _split_header(content)
            if not body.strip():
                continue

            note_type = rel.parts[0] if len(rel.parts) > 1 else ""
            files.append(
                {
                    "file_path": f"{domain}/{rel.as_posix()}",
                    "file_name": file_path.name,
                    "content": body,
                    "domain": domain,
                    "type": note_type,
                    "evidence_level": fields.get("evidence level", ""),
                    "freshness": fields.get("freshness", ""),
                    "last_updated": fields.get("last updated", ""),
                    "last_modified": file_path.stat().st_mtime,
                }
            )

    return files
