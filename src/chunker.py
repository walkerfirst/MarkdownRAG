from __future__ import annotations

import re
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_sections(text: str) -> list[tuple[list[str], str]]:
    lines = text.splitlines()
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[str] = []
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((heading_stack.copy(), body))

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)
            body_lines = []
            continue
        body_lines.append(line)

    flush()
    return sections


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_long_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    sentences = re.split(r"(?<=[。！？；.!?])", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), limit):
                chunks.append(sentence[start : start + limit].strip())
            continue
        candidate = f"{current}{sentence}"
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())
    return chunks


def _build_body_chunks(
    paragraphs: list[str],
    target_chars: int,
    max_chars: int,
    min_chars: int,
    overlap_chars: int,
) -> list[str]:
    body_limit = max(80, max_chars - 80)
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(_split_long_text(paragraph, body_limit))

    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if len(candidate) <= max_chars and (len(current) < target_chars or len(unit) < min_chars):
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
        overlap = current[-overlap_chars:].strip() if current and overlap_chars > 0 else ""
        current = f"{overlap}\n\n{unit}".strip() if overlap else unit

    if current:
        chunks.append(current.strip())

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars and len(merged[-1]) + len(chunk) + 2 <= max_chars:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)
    return merged


def chunk_markdown(
    text: str,
    file_path: str,
    target_chars: int = 800,
    max_chars: int = 1200,
    min_chars: int = 200,
    overlap_chars: int = 100,
) -> list[dict]:
    if not text or not text.strip():
        return []

    sections = _split_sections(text)
    if not sections:
        sections = [([], text.strip())]

    results: list[dict] = []
    chunk_index = 1
    file_name = Path(file_path).name

    for titles, body in sections:
        title_path = " > ".join(titles) if titles else Path(file_name).stem
        prefix = f"标题路径：{title_path}\n\n"
        body_chunks = _build_body_chunks(
            paragraphs=_split_paragraphs(body),
            target_chars=max(120, target_chars - len(prefix)),
            max_chars=max(160, max_chars - len(prefix)),
            min_chars=max(80, min_chars - min(len(prefix), 80)),
            overlap_chars=overlap_chars,
        )

        for body_chunk in body_chunks:
            content = f"{prefix}{body_chunk}".strip()
            if len(content) > max_chars:
                for part in _split_long_text(content, max_chars):
                    results.append(
                        {
                            "chunk_id": f"{file_path}::{chunk_index:04d}",
                            "file_path": file_path,
                            "title_path": title_path,
                            "content": part.strip(),
                            "char_count": len(part.strip()),
                            "chunk_index": chunk_index,
                        }
                    )
                    chunk_index += 1
                continue

            results.append(
                {
                    "chunk_id": f"{file_path}::{chunk_index:04d}",
                    "file_path": file_path,
                    "title_path": title_path,
                    "content": content,
                    "char_count": len(content),
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    return results
