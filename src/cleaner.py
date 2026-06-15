from __future__ import annotations

import re


INVALID_HTML_TAG_RE = re.compile(
    r"</?(?:div|span|section|article|main|header|footer|aside|nav|font)[^>]*>",
    re.IGNORECASE,
)
BREAK_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\((?:https?://|www\.)[^)\s]+(?:\s+\"[^\"]*\")?\)")
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\((?:https?://|www\.)[^)\s]+(?:\s+\"[^\"]*\")?\)")
AUTO_LINK_RE = re.compile(r"<(?:https?://|www\.)[^>\s]+>")
BARE_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+")
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(?:https?://|www\.)\S+.*$", re.MULTILINE)


def strip_links(text: str) -> str:
    """去掉 URL 噪声，保留 Markdown 链接中真正可读的文字。"""
    cleaned = REFERENCE_LINK_RE.sub("", text)
    cleaned = IMAGE_LINK_RE.sub(r"\1", cleaned)
    cleaned = MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = AUTO_LINK_RE.sub("", cleaned)
    cleaned = BARE_URL_RE.sub("", cleaned)
    return cleaned


def clean_markdown(text: str) -> str:
    if not text or not text.strip():
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = BREAK_TAG_RE.sub("\n", cleaned)
    cleaned = INVALID_HTML_TAG_RE.sub("", cleaned)
    cleaned = strip_links(cleaned)
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
