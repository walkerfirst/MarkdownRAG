# T6 多源 wiki + 分类与元数据过滤检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把数据源从单一 `notes_dir` 升级为多个 wiki 源,按路径自动分 `domain`/`type`、抽取页头 `evidence_level`/`freshness`,并让检索可按这些维度过滤。

**Architecture:** `loader` 遍历 `sources` 列表、解析页头 bold-key、按 `domain/` 前缀生成唯一 `file_path` 并产出分类字段;字段冗余落到每个 chunk(SQLite + Chroma metadata);检索时 `domain`/`type` 推入检索层(Chroma `where` + SQL),`evidence` 做结果后过滤。

**Tech Stack:** Python 3 / pytest / Typer CLI / ChromaDB / SQLite / sentence-transformers(bge-small-zh-v1.5)。

## Global Constraints

- 本机不可用 `uv sync` / `uv run`(会按锁从 pytorch 源重装 torch,国内必断)。装依赖用 `uv pip install`;跑测试/脚本用 `.venv/bin/python -m ...`。
- HF 模型走本地目录 + `HF_HUB_OFFLINE=1`。
- 所有回复、注释、文档用中文。
- 匹配现有代码风格(`from __future__ import annotations`、Typer CLI、pytest + Fake 注入)。
- 真实数据源(本机):`/home/neo/Documents/obsidian/investing/wiki/`(domain=stock)、`/home/neo/Documents/obsidian/learning/wiki/`(domain=study)。
- 排除:文件名 `index.md`、`log.md`;目录名 `templates`。
- 当前在 `master` 分支:执行前先开特性分支 `git checkout -b t6-multi-source-wiki`。
- schema/字段变更后,真实库需 `--reset` 重建一次(最终验证任务做)。

---

### Task 1: cleaner 清洗 `[[wikilink]]`

**Files:**
- Modify: `src/cleaner.py`
- Test: `tests/test_cleaner.py`

**Interfaces:**
- Consumes: 无。
- Produces: `clean_markdown(text: str) -> str` 行为扩展——`[[page]]`/`[[path/page]]`/`[[path/page|alias]]` 被清成可读文字。

- [ ] **Step 1: 写失败测试**

在 `tests/test_cleaner.py` 末尾追加:

```python
def test_cleaner_strips_wikilinks_to_text() -> None:
    text = (
        "见 [[companies/600519|贵州茅台]] 与 [[industries/白酒]] "
        "和 [[牧原股份]] 以及 [[sources/]]。"
    )
    cleaned = clean_markdown(text)
    assert "贵州茅台" in cleaned
    assert "白酒" in cleaned
    assert "牧原股份" in cleaned
    assert "sources" in cleaned
    assert "[[" not in cleaned
    assert "companies/600519" not in cleaned
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_cleaner.py::test_cleaner_strips_wikilinks_to_text -v`
Expected: FAIL(`[[` 仍在,断言失败)。

- [ ] **Step 3: 实现**

在 `src/cleaner.py` 顶部正则区追加:

```python
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
```

新增函数(放在 `strip_links` 之后):

```python
def _wikilink_text(match: re.Match) -> str:
    inner = match.group(1).strip()
    if "|" in inner:
        return inner.split("|")[-1].strip()
    target = inner.split("#")[0].rstrip("/")
    return target.split("/")[-1].strip()


def strip_wikilinks(text: str) -> str:
    """把 Obsidian [[链接]] 清成纯文字(保留别名或末段页名)。"""
    return WIKILINK_RE.sub(_wikilink_text, text)
```

在 `clean_markdown` 里,`strip_links` 之后插入一行:

```python
    cleaned = strip_links(cleaned)
    cleaned = strip_wikilinks(cleaned)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_cleaner.py -v`
Expected: PASS(全部 cleaner 测试)。

- [ ] **Step 5: 提交**

```bash
git add src/cleaner.py tests/test_cleaner.py
git commit -m "feat(cleaner): 清洗 Obsidian [[wikilink]] 为纯文字"
```

---

### Task 2: config.py 支持 `sources` 与排除项

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`(新建)

**Interfaces:**
- Consumes: 无。
- Produces: `load_config(config_path) -> dict`,新增:校验顶层 `sources`(列表,每项 `{path, domain}`,path 必须存在);`config["exclude_names"]` / `config["exclude_dirs"]` 缺省为 `[]`;不再要求 `paths.notes_dir`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_config.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL(当前 `load_config` 要求 `paths.notes_dir`,KeyError 或不校验 sources)。

- [ ] **Step 3: 实现**

把 `src/config.py` 的 `load_config` 改为(替换 `required_dirs` 一段及其后):

```python
    config["root_dir"] = str(config_file.parent)
    paths = config.setdefault("paths", {})

    for key in ("chroma_dir", "processed_dir"):
        path = resolve_path(config["root_dir"], paths[key])
        path.mkdir(parents=True, exist_ok=True)

    metadata_db = resolve_path(config["root_dir"], paths["metadata_db"])
    metadata_db.parent.mkdir(parents=True, exist_ok=True)

    eval_queries = resolve_path(config["root_dir"], paths["eval_queries"])
    eval_queries.parent.mkdir(parents=True, exist_ok=True)

    sources = config.get("sources") or []
    if not sources:
        raise ValueError("config 缺少 sources(至少一个 {path, domain})")
    for source in sources:
        source_path = Path(source["path"])
        if not source_path.exists():
            raise FileNotFoundError(f"source 路径不存在: {source_path}")

    config.setdefault("exclude_names", [])
    config.setdefault("exclude_dirs", [])

    return config
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): notes_dir 改为多源 sources 列表并校验路径"
```

---

### Task 3: loader 多源遍历 + 页头解析 + 排除

**Files:**
- Rewrite: `src/loader.py`
- Test: `tests/test_loader.py`(新建)

**Interfaces:**
- Consumes: 无(纯函数,参数显式传入,不读 config)。
- Produces:
  `load_markdown_files(sources: list[dict], exclude_names: list[str] | None = None, exclude_dirs: list[str] | None = None) -> list[dict]`
  每个 dict 字段:`file_path`(=`"{domain}/{相对 source 根的 posix 路径}"`)、`file_name`、`content`(页头剥离 + Summary 前置后的正文)、`domain`、`type`(domain 后第一层子目录名,无则 `""`)、`evidence_level`、`freshness`、`last_updated`、`last_modified`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_loader.py`:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_loader.py -v`
Expected: FAIL(旧 `load_markdown_files` 只接受 `notes_dir` 字符串)。

- [ ] **Step 3: 实现 — 重写 `src/loader.py`**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_loader.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/loader.py tests/test_loader.py
git commit -m "feat(loader): 多源遍历 + 页头元数据解析 + domain/type 分类"
```

---

### Task 4: SQLite store 增列与关键词过滤

**Files:**
- Modify: `src/sqlite_metadata_store.py`
- Test: `tests/test_metadata_store.py`

**Interfaces:**
- Consumes: chunk dict 现含 `domain`/`type`/`evidence_level`/`freshness`;document dict 现含 `domain`/`type`/`evidence_level`/`freshness`/`last_updated`。
- Produces:
  - `chunks` 表增列 `domain`/`type`/`evidence_level`/`freshness`;`documents` 表增列 `domain`/`type`/`evidence_level`/`freshness`/`last_updated`。
  - `upsert_document` / `upsert_chunks` 写入新列。
  - `search_chunks_by_keywords(keywords, limit=20, domain=None, note_type=None) -> list[dict]`,返回行含新列;`domain`/`note_type` 为精确等值过滤。

- [ ] **Step 1: 写失败测试**

在 `tests/test_metadata_store.py` 末尾追加(并在用到的 `upsert_*` 里补字段):

```python
def test_metadata_store_filters_keywords_by_domain_and_type(tmp_path: Path) -> None:
    store = SQLiteMetadataStore(str(tmp_path / "m.sqlite"))
    store.init_schema()
    store.upsert_chunks(
        [
            {
                "chunk_id": "stock/companies/a.md::0001",
                "file_path": "stock/companies/a.md",
                "title_path": "A",
                "content": "牧原 屠宰 护城河",
                "char_count": 8,
                "chunk_index": 1,
                "created_at": "2024-01-01T00:00:00+00:00",
                "domain": "stock",
                "type": "companies",
                "evidence_level": "Primary source | User view",
                "freshness": "Stable",
            },
            {
                "chunk_id": "study/concepts/b.md::0001",
                "file_path": "study/concepts/b.md",
                "title_path": "B",
                "content": "牧原 屠宰 另一处",
                "char_count": 8,
                "chunk_index": 1,
                "created_at": "2024-01-01T00:00:00+00:00",
                "domain": "study",
                "type": "concepts",
                "evidence_level": "",
                "freshness": "",
            },
        ]
    )

    hits = store.search_chunks_by_keywords(["牧原"], domain="stock")
    assert [row["file_path"] for row in hits] == ["stock/companies/a.md"]
    assert hits[0]["evidence_level"] == "Primary source | User view"

    typed = store.search_chunks_by_keywords(["牧原"], note_type="concepts")
    assert [row["file_path"] for row in typed] == ["study/concepts/b.md"]
```

并把该文件已有的 `test_metadata_store_crud` 里 `upsert_document` 的两个 dict 各补:`"domain": "stock", "type": "companies", "evidence_level": "Primary", "freshness": "Stable", "last_updated": "2026-05-02"`;`upsert_chunks` 的两个 dict 各补:`"domain": "stock", "type": "companies", "evidence_level": "", "freshness": ""`。

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_metadata_store.py -v`
Expected: FAIL(无新列 / `search_chunks_by_keywords` 不接受 `domain`)。

- [ ] **Step 3: 实现**

`init_schema` 的两张表改为:

```python
                CREATE TABLE IF NOT EXISTS documents (
                    file_path TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    last_modified REAL NOT NULL,
                    last_ingested_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    domain TEXT NOT NULL DEFAULT '',
                    type TEXT NOT NULL DEFAULT '',
                    evidence_level TEXT NOT NULL DEFAULT '',
                    freshness TEXT NOT NULL DEFAULT '',
                    last_updated TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    title_path TEXT,
                    content TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT '',
                    type TEXT NOT NULL DEFAULT '',
                    evidence_level TEXT NOT NULL DEFAULT '',
                    freshness TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(file_path) REFERENCES documents(file_path)
                );
```

`upsert_document` 改为写全列:

```python
    def upsert_document(self, document: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    file_path, file_name, content_hash, last_modified,
                    last_ingested_at, status, chunk_count,
                    domain, type, evidence_level, freshness, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_name = excluded.file_name,
                    content_hash = excluded.content_hash,
                    last_modified = excluded.last_modified,
                    last_ingested_at = excluded.last_ingested_at,
                    status = excluded.status,
                    chunk_count = excluded.chunk_count,
                    domain = excluded.domain,
                    type = excluded.type,
                    evidence_level = excluded.evidence_level,
                    freshness = excluded.freshness,
                    last_updated = excluded.last_updated
                """,
                (
                    document["file_path"],
                    document["file_name"],
                    document["content_hash"],
                    document["last_modified"],
                    document["last_ingested_at"],
                    document["status"],
                    document["chunk_count"],
                    document.get("domain", ""),
                    document.get("type", ""),
                    document.get("evidence_level", ""),
                    document.get("freshness", ""),
                    document.get("last_updated", ""),
                ),
            )
```

`upsert_chunks` 的 INSERT 改为写全列:

```python
                INSERT INTO chunks (
                    chunk_id, file_path, title_path, content,
                    char_count, chunk_index, created_at,
                    domain, type, evidence_level, freshness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    title_path = excluded.title_path,
                    content = excluded.content,
                    char_count = excluded.char_count,
                    chunk_index = excluded.chunk_index,
                    created_at = excluded.created_at,
                    domain = excluded.domain,
                    type = excluded.type,
                    evidence_level = excluded.evidence_level,
                    freshness = excluded.freshness
```

参数行改为:

```python
                [
                    (
                        chunk["chunk_id"],
                        chunk["file_path"],
                        chunk.get("title_path", ""),
                        chunk["content"],
                        chunk["char_count"],
                        chunk["chunk_index"],
                        chunk["created_at"],
                        chunk.get("domain", ""),
                        chunk.get("type", ""),
                        chunk.get("evidence_level", ""),
                        chunk.get("freshness", ""),
                    )
                    for chunk in chunks
                ],
```

`search_chunks_by_keywords` 改为支持过滤:

```python
    def search_chunks_by_keywords(
        self,
        keywords: list[str],
        limit: int = 20,
        domain: str | None = None,
        note_type: str | None = None,
    ) -> list[dict]:
        keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
        if not keywords:
            return []

        kw_clauses: list[str] = []
        params: list[str] = []
        for keyword in keywords:
            kw_clauses.append("(content LIKE ? OR title_path LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])

        where = f"({' OR '.join(kw_clauses)})"
        if domain:
            where += " AND domain = ?"
            params.append(domain)
        if note_type:
            where += " AND type = ?"
            params.append(note_type)

        sql = f"""
            SELECT * FROM chunks
            WHERE {where}
            ORDER BY file_path, chunk_index
            LIMIT ?
        """
        params.append(str(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [dict(row) for row in rows]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_metadata_store.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/sqlite_metadata_store.py tests/test_metadata_store.py
git commit -m "feat(store): chunks/documents 增加 domain/type/evidence/freshness 列与关键词过滤"
```

---

### Task 5: Chroma vector store 元数据与 where 过滤

**Files:**
- Modify: `src/vector_store.py`
- Test: `tests/test_vector_store.py`(新建)

**Interfaces:**
- Consumes: chunk dict 现含 `domain`/`type`/`evidence_level`/`freshness`。
- Produces:
  - `upsert_chunks` 的 Chroma metadata 增加这 4 个键。
  - `search(query_embedding, top_k=5, where: dict | None = None) -> list[dict]`,把 `where` 透传给 `collection.query`;返回 dict 含 `domain`/`type`/`evidence_level`/`freshness`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_vector_store.py`:

```python
from pathlib import Path

from src.vector_store import ChromaVectorStore


def _chunk(cid: str, domain: str, note_type: str) -> dict:
    return {
        "chunk_id": cid,
        "file_path": f"{domain}/{note_type}/x.md",
        "title_path": "T",
        "content": "内容",
        "char_count": 2,
        "chunk_index": 1,
        "domain": domain,
        "type": note_type,
        "evidence_level": "Primary source | User view",
        "freshness": "Stable",
    }


def test_vector_store_where_filters_by_domain(tmp_path: Path) -> None:
    store = ChromaVectorStore(str(tmp_path / "chroma"))
    store.upsert_chunks(
        [_chunk("stock/c/x.md::0001", "stock", "companies"),
         _chunk("study/c/x.md::0001", "study", "concepts")],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    results = store.search([1.0, 0.0], top_k=5, where={"domain": "stock"})

    assert [r["file_path"] for r in results] == ["stock/companies/x.md"]
    assert results[0]["domain"] == "stock"
    assert results[0]["evidence_level"] == "Primary source | User view"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_vector_store.py -v`
Expected: FAIL(`search` 不接受 `where` / 返回无 `domain`)。

- [ ] **Step 3: 实现**

`upsert_chunks` 的 metadatas 改为:

```python
        metadatas = [
            {
                "file_path": chunk["file_path"],
                "title_path": chunk["title_path"],
                "char_count": int(chunk["char_count"]),
                "chunk_index": int(chunk["chunk_index"]),
                "domain": chunk.get("domain", ""),
                "type": chunk.get("type", ""),
                "evidence_level": chunk.get("evidence_level", ""),
                "freshness": chunk.get("freshness", ""),
            }
            for chunk in chunks
        ]
```

`search` 改为:

```python
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[dict] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance or 0.0))
            results.append(
                {
                    "chunk_id": chunk_id,
                    "file_path": metadata.get("file_path", ""),
                    "title_path": metadata.get("title_path", ""),
                    "char_count": metadata.get("char_count", 0),
                    "domain": metadata.get("domain", ""),
                    "type": metadata.get("type", ""),
                    "evidence_level": metadata.get("evidence_level", ""),
                    "freshness": metadata.get("freshness", ""),
                    "content": content,
                    "score": score,
                }
            )
        return results
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_vector_store.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat(vector): Chroma metadata 增加分类字段并支持 where 过滤"
```

---

### Task 6: ingest 接入多源 + 注入字段 + config.yaml

**Files:**
- Modify: `src/ingest.py`
- Modify: `config.yaml`
- Test: `tests/test_incremental_ingest.py`

**Interfaces:**
- Consumes: `load_markdown_files(sources, exclude_names, exclude_dirs)`(Task 3);store 新列(Task 4)。
- Produces: `ingest_project` 用 `config["sources"]` 建库;每个 chunk 注入 `domain`/`type`/`evidence_level`/`freshness`;`upsert_document` 带这些字段 + `last_updated`。

- [ ] **Step 1: 改 config.yaml(非测试,先改配置)**

把 `config.yaml` 的 `paths.notes_dir` 行删掉,并在顶层新增:

```yaml
sources:
  - path: /home/neo/Documents/obsidian/investing/wiki/
    domain: stock
  - path: /home/neo/Documents/obsidian/learning/wiki/
    domain: study
exclude_names: [index.md, log.md]
exclude_dirs: [templates]
```

- [ ] **Step 2: 写失败测试 — 改 `tests/test_incremental_ingest.py`**

把 `write_config` 里 `paths` 的 `notes_dir` 删除,改为在 config 顶层加(注意 source 指向 tmp 目录):

```python
def write_config(root: Path) -> Path:
    source_dir = root / "wiki"
    source_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "project": {"name": "test-rag"},
        "paths": {
            "chroma_dir": "data/chroma",
            "processed_dir": "data/processed",
            "metadata_db": "data/metadata.sqlite",
            "eval_queries": "eval/queries.yaml",
        },
        "sources": [{"path": str(source_dir), "domain": "stock"}],
        "exclude_names": ["index.md", "log.md"],
        "exclude_dirs": ["templates"],
        "embedding": {"provider": "fake", "model_name": "fake"},
        "chunking": {"target_chars": 80, "max_chars": 160, "min_chars": 40, "overlap_chars": 20},
        "search": {"top_k": 5, "preview_chars": 120},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path
```

该文件原有用例里向 `notes/` 写测试 .md 的地方,改为写到 `root / "wiki" / "companies"`(确保在子目录下,带 type);并把断言里期望的 `file_path` 由 `notes/...` 改为 `stock/companies/...`。新增一条断言验证注入字段:

```python
def test_incremental_ingest_injects_domain_type(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    wiki = tmp_path / "wiki" / "companies"
    wiki.mkdir(parents=True, exist_ok=True)
    (wiki / "a.md").write_text(
        "# A 公司\n\n**Evidence level**: Primary\n**Freshness**: Stable\n\n---\n\n正文内容这里很长。" * 1,
        encoding="utf-8",
    )
    vector_store = FakeVectorStore()
    metadata_store_path = tmp_path / "data" / "metadata.sqlite"
    metadata_store_path.parent.mkdir(parents=True, exist_ok=True)

    from src.sqlite_metadata_store import SQLiteMetadataStore

    store = SQLiteMetadataStore(str(metadata_store_path))
    ingest_project(
        config_path=str(config_path),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        metadata_store=store,
    )

    rows = store.search_chunks_by_keywords(["正文"], domain="stock")
    assert rows
    assert rows[0]["type"] == "companies"
    assert rows[0]["evidence_level"] == "Primary"
```

> 注:`FakeVectorStore` 的 `upsert_chunks` 已存 chunk 字段;无需改。若原用例断言 `scanned_files`/`new_files` 计数,按新结构(文件落在 `wiki/companies/` 下)核对更新。

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_incremental_ingest.py -v`
Expected: FAIL(`ingest_project` 仍读 `paths.notes_dir`)。

- [ ] **Step 4: 实现 — 改 `src/ingest.py`**

把 `ingest_project` 里读取 `notes_dir` 与 `load_markdown_files` 调用改为:

```python
    chroma_dir = resolve_path(config["root_dir"], config["paths"]["chroma_dir"])
    processed_dir = resolve_path(config["root_dir"], config["paths"]["processed_dir"])
    metadata_db = resolve_path(config["root_dir"], config["paths"]["metadata_db"])

    metadata_store = metadata_store or SQLiteMetadataStore(str(metadata_db))
    metadata_store.init_schema()
    vector_store = vector_store or ChromaVectorStore(str(chroma_dir))
    lazy_embedder = embedder

    files = load_markdown_files(
        config["sources"],
        exclude_names=config.get("exclude_names", []),
        exclude_dirs=config.get("exclude_dirs", []),
    )
```

(删除原 `notes_dir = resolve_path(...)` 行。)

在 chunk 生成后、upsert 前注入字段。把:

```python
        for chunk in chunks:
            chunk["created_at"] = _utc_now()
```

改为:

```python
        for chunk in chunks:
            chunk["created_at"] = _utc_now()
            chunk["domain"] = item["domain"]
            chunk["type"] = item["type"]
            chunk["evidence_level"] = item["evidence_level"]
            chunk["freshness"] = item["freshness"]
```

`upsert_document` 的 dict 增加字段:

```python
        metadata_store.upsert_document(
            {
                "file_path": item["file_path"],
                "file_name": item["file_name"],
                "content_hash": content_hash,
                "last_modified": item["last_modified"],
                "last_ingested_at": _utc_now(),
                "status": "indexed" if chunks else "empty",
                "chunk_count": len(chunks),
                "domain": item["domain"],
                "type": item["type"],
                "evidence_level": item["evidence_level"],
                "freshness": item["freshness"],
                "last_updated": item["last_updated"],
            }
        )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_incremental_ingest.py -v`
Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/ingest.py config.yaml tests/test_incremental_ingest.py
git commit -m "feat(ingest): 接入多源 sources 并把分类字段注入 chunk/document"
```

---

### Task 7: search 过滤检索 + CLI

**Files:**
- Modify: `src/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `vector_store.search(..., where=)`(Task 5);`metadata_store.search_chunks_by_keywords(..., domain=, note_type=)`(Task 4)。
- Produces: `search_chunks(query, ..., domain=None, type=None, evidence=None, ...)`;`domain`/`type` 推入检索层,`evidence` 子串后过滤。CLI 加 `--domain`/`--type`/`--evidence`。

- [ ] **Step 1: 写失败测试 — 改 `tests/test_search.py`**

`FakeVectorStore.search` 签名补 `where`,并按 `where` 过滤;`FakeMetadataStore.search_chunks_by_keywords` 补 `domain`/`note_type`。在两个 Fake 的返回 dict 里加 `domain`/`type`/`evidence_level` 字段。新增过滤用例:

```python
class FilterVectorStore:
    def search(self, query_embedding, top_k=5, where=None):
        rows = [
            {"chunk_id": "stock/c/a.md::0001", "file_path": "stock/c/a.md", "title_path": "A",
             "content": "高相关", "score": 0.72, "domain": "stock", "type": "companies",
             "evidence_level": "Primary source | User view", "freshness": "Stable"},
            {"chunk_id": "study/c/b.md::0001", "file_path": "study/c/b.md", "title_path": "B",
             "content": "次相关", "score": 0.6, "domain": "study", "type": "concepts",
             "evidence_level": "Unverified", "freshness": "Stale risk"},
        ]
        if where and "domain" in where:
            rows = [r for r in rows if r["domain"] == where["domain"]]
        return rows


class EmptyKeywordStore:
    def search_chunks_by_keywords(self, keywords, limit=20, domain=None, note_type=None):
        return []


def test_search_chunks_filters_by_domain() -> None:
    results = search_chunks(
        query="测试", min_score=0.0, relative_score_ratio=0.0, domain="stock",
        embedder=FakeEmbedder(), vector_store=FilterVectorStore(), metadata_store=EmptyKeywordStore(),
    )
    assert [r["file_path"] for r in results] == ["stock/c/a.md"]


def test_search_chunks_filters_by_evidence_substring() -> None:
    results = search_chunks(
        query="测试", min_score=0.0, relative_score_ratio=0.0, evidence="Primary",
        embedder=FakeEmbedder(), vector_store=FilterVectorStore(), metadata_store=EmptyKeywordStore(),
    )
    assert [r["file_path"] for r in results] == ["stock/c/a.md"]
```

并把已有的 `FakeVectorStore.search` 改成 `def search(self, query_embedding, top_k=5, where=None)`,`FakeMetadataStore` / `KeywordMetadataStore` 的方法签名补 `domain=None, note_type=None`。

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: FAIL(`search_chunks` 无 `domain`/`evidence` 形参)。

- [ ] **Step 3: 实现 — 改 `src/search.py`**

`_keyword_results` 的结果 dict 补字段(从 chunk 透传):

```python
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "file_path": chunk["file_path"],
                "title_path": chunk.get("title_path", ""),
                "char_count": chunk.get("char_count", len(chunk.get("content", ""))),
                "domain": chunk.get("domain", ""),
                "type": chunk.get("type", ""),
                "evidence_level": chunk.get("evidence_level", ""),
                "freshness": chunk.get("freshness", ""),
                "content": chunk["content"],
                "score": score,
            }
        )
```

`search_chunks` 签名与逻辑:

```python
def search_chunks(
    query: str,
    config_path: str = "config.yaml",
    top_k: int | None = None,
    min_score: float | None = None,
    relative_score_ratio: float | None = None,
    domain: str | None = None,
    type: str | None = None,
    evidence: str | None = None,
    embedder: LocalEmbedder | None = None,
    vector_store: ChromaVectorStore | None = None,
    metadata_store: MetadataStore | None = None,
) -> list[dict]:
    config = load_config(config_path)
    top_k = top_k or config["search"]["top_k"]
    min_score = config["search"].get("min_score", 0.0) if min_score is None else min_score
    relative_score_ratio = (
        config["search"].get("relative_score_ratio", 0.0)
        if relative_score_ratio is None
        else relative_score_ratio
    )
    chroma_dir = resolve_path(config["root_dir"], config["paths"]["chroma_dir"])
    metadata_db = resolve_path(config["root_dir"], config["paths"]["metadata_db"])

    embedder = embedder or LocalEmbedder(
        config["embedding"]["model_name"],
        query_instruction=config["embedding"].get("query_instruction", ""),
    )
    vector_store = vector_store or ChromaVectorStore(str(chroma_dir))
    metadata_store = metadata_store or SQLiteMetadataStore(str(metadata_db))

    conditions = []
    if domain:
        conditions.append({"domain": domain})
    if type:
        conditions.append({"type": type})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}
    else:
        where = None

    query_embedding = embedder.embed_query(query)
    vector_results = vector_store.search(query_embedding, top_k=max(top_k * 3, top_k), where=where)

    keywords = extract_query_keywords(query)
    keyword_chunks = metadata_store.search_chunks_by_keywords(
        keywords,
        limit=config["search"].get("keyword_top_k", top_k * 4),
        domain=domain,
        note_type=type,
    )
    merged = _merge_results(vector_results, _keyword_results(keyword_chunks, keywords))

    if evidence:
        needle = evidence.lower()
        merged = [r for r in merged if needle in (r.get("evidence_level") or "").lower()]

    return _filter_results(
        merged,
        min_score=min_score,
        relative_score_ratio=relative_score_ratio,
        top_k=top_k,
    )
```

CLI `main` 增加选项并透传:

```python
@app.command()
def main(
    query: str = typer.Argument(..., help="检索问题"),
    top_k: int = typer.Option(None, "--top-k", help="返回结果数量"),
    min_score: float | None = typer.Option(None, "--min-score", help="最低相似度分数"),
    domain: str | None = typer.Option(None, "--domain", help="按 domain 过滤(stock/study)"),
    type: str | None = typer.Option(None, "--type", help="按 type 过滤(companies/industries…)"),
    evidence: str | None = typer.Option(None, "--evidence", help="按证据级别子串过滤(Primary/Secondary…)"),
) -> None:
    try:
        config = load_config()
        results = search_chunks(
            query=query, top_k=top_k, min_score=min_score,
            domain=domain, type=type, evidence=evidence,
        )
        render_results(query, results, preview_chars=config["search"]["preview_chars"])
    except Exception as exc:  # pragma: no cover - CLI 错误出口
        console.print(f"[red]search 失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_search.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/search.py tests/test_search.py
git commit -m "feat(search): domain/type 检索层过滤 + evidence 子串后过滤 + CLI 选项"
```

---

### Task 8: context_builder CLI 透传过滤

**Files:**
- Modify: `src/context_builder.py`
- Test: `tests/test_context_builder.py`

**Interfaces:**
- Consumes: `search_chunks(..., domain=, type=, evidence=)`(Task 7)。
- Produces: `src.context_builder` CLI 加 `--domain`/`--type`/`--evidence`,透传给 `search_chunks`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_context_builder.py` 末尾追加(用 monkeypatch 拦截 `search_chunks`,断言过滤参数透传)。先看该文件已有导入风格,采用同款。示例:

```python
def test_context_cli_passes_filters(monkeypatch, tmp_path) -> None:
    import src.context_builder as cb

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cb, "search_chunks", fake_search)
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        cb.app,
        ["测试", "--domain", "stock", "--type", "companies", "--evidence", "Primary",
         "--stdout"],
    )
    assert result.exit_code == 0
    assert captured["domain"] == "stock"
    assert captured["type"] == "companies"
    assert captured["evidence"] == "Primary"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_context_builder.py::test_context_cli_passes_filters -v`
Expected: FAIL(CLI 无 `--domain` 选项)。

- [ ] **Step 3: 实现 — 改 `src/context_builder.py` 的 `main`**

在 `main` 选项里加:

```python
    domain: str | None = typer.Option(None, "--domain", help="按 domain 过滤"),
    type: str | None = typer.Option(None, "--type", help="按 type 过滤"),
    evidence: str | None = typer.Option(None, "--evidence", help="按证据级别子串过滤"),
```

`search_chunks(...)` 调用补:

```python
        results = search_chunks(
            query=query,
            top_k=top_k,
            min_score=min_score,
            relative_score_ratio=relative_score_ratio,
            domain=domain,
            type=type,
            evidence=evidence,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_context_builder.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/context_builder.py tests/test_context_builder.py
git commit -m "feat(context): CLI 透传 domain/type/evidence 过滤"
```

---

### Task 9: 全量验证(真实 wiki 重建 + 全测试)

**Files:**
- 无代码改动(验证 + 记录)。
- Modify: `RAG增强方案(待审阅).md`(把 T6 标记为已完成并记录结果)。

**Interfaces:**
- Consumes: 前 8 个任务。

- [ ] **Step 1: 全测试绿**

Run: `.venv/bin/python -m pytest -q`
Expected: 全部 PASS。

- [ ] **Step 2: 真实库重建**

Run: `HF_HUB_OFFLINE=1 .venv/bin/python -m src.ingest --reset`
Expected: 扫描计数符合预期(约 51 stock + 2 study 内容页;`index.md`/`log.md`/`templates/` 不计入);`deleted/new` 计数合理,无报错。

- [ ] **Step 3: 抽样过滤检索**

Run:
```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m src.search "牧原 屠宰 护城河" --domain stock --type companies
HF_HUB_OFFLINE=1 .venv/bin/python -m src.search "牧原" --evidence Primary
```
Expected: 第一条只返回 `stock/companies/...`;第二条结果的 `evidence_level` 均含 `Primary`;Top 结果主观相关。

- [ ] **Step 4: 验证 Summary 进向量、Sources 不进正文**

Run: 任取一个含 `**Summary**` 的页,用其 Summary 里的短语检索,确认能命中;并确认结果正文不含 `**Sources**` 的 raw 路径。

- [ ] **Step 5: 记录并提交**

更新 `RAG增强方案(待审阅).md`:T6 标 `已完成`,记录实测计数与抽样结果。

```bash
git add RAG增强方案\(待审阅\).md
git commit -m "docs: T6 完成,记录多源 wiki 重建与过滤检索验证结果"
```

---

## Self-Review(已对照 spec)

- **Spec §1 配置形态** → Task 2(config 校验)+ Task 6(config.yaml)。
- **Spec §2 file_path 唯一性 / type** → Task 3(loader 前缀 + type)。
- **Spec §3 页头解析 / Summary 保留 / 兜底** → Task 3(`_split_header`)。
- **Spec §4 [[链接]] 清洗** → Task 1。
- **Spec §5 schema denormalize** → Task 4(SQLite)+ Task 5(Chroma)+ Task 6(ingest 注入)。
- **Spec §6 过滤检索(domain/type 入检索层,evidence 后过滤)** → Task 7 + Task 8(CLI)。
- **Spec §7 增量同步跨源** → Task 6(`load_markdown_files` 返回并集,既有对账逻辑复用)。
- **Spec 验证标准** → Task 9。

类型一致性核对:`search_chunks_by_keywords(..., domain=, note_type=)`、`vector_store.search(..., where=)`、`search_chunks(..., domain=, type=, evidence=)`、chunk/document 注入字段名,跨 Task 4/5/6/7 一致。

> 命名注意:CLI/检索层外部参数用 `type`(与 config/Chroma 元数据键一致);SQLite 层方法形参用 `note_type`(避免与内建 `type` 混淆),列名仍为 `type`。
