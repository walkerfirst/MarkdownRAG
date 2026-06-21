# MCP Server (T3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把本地 RAG 检索暴露成 MCP 工具,让 Claude Code 自主多轮调用 `search_notes` 检索投资/学习笔记并综合分析。

**Architecture:** 新增单文件 `src/mcp_server.py`,用官方 `mcp` SDK 的 FastMCP 暴露唯一工具 `search_notes`,复用现有 `search_chunks`,模型惰性常驻加载,stdio 传输。不改动任何现有模块。

**Tech Stack:** Python 3.13、`mcp`(FastMCP)、复用 chromadb / sentence-transformers / sqlite。

## Global Constraints

- 环境:不用 `uv sync` / `uv run`;跑命令用 `.venv/bin/python`,装依赖用 `uv pip install`(走清华源)。
- 涉及向量/模型的命令加 `HF_HUB_OFFLINE=1`(用本地模型副本,避免连 HuggingFace)。
- 只新增 `src/mcp_server.py`、`tests/test_mcp_server.py`,改 `pyproject.toml` / `README.md` / `USAGE.md` / `RAG增强方案(待审阅).md`;**不改其他现有模块**。
- KISS、不过度防御;注释用简体中文,匹配现有风格。
- `search_chunks` 返回字段为 `title_path` / `file_path` / `score` / `content`(非 `title`)。

---

### Task 1: 实现 `search_notes` MCP 工具与单元测试

**Files:**
- Modify: `pyproject.toml`(dependencies 加 `mcp`)
- Create: `src/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `src.search.search_chunks(query, config_path, top_k, domain, embedder, vector_store, metadata_store, reranker)` → `list[dict]`,每条含 `title_path/file_path/score/content`。
- Produces:
  - `src.mcp_server.search_notes(query: str, domain: str | None = None, top_k: int = 5) -> list[dict]`,每条返回 `{"title": str, "file_path": str, "score": float, "content": str}`。
  - `src.mcp_server._load() -> None`(惰性加载三件套)、模块级 `_embedder/_vector_store/_metadata_store`(测试可 monkeypatch)。

- [ ] **Step 1: 装 mcp 依赖并写入 pyproject**

Run:
```fish
cd /home/neo/project/MarkdownRAG; and .venv/bin/python -m pip install mcp
```
(若 `pip` 不可用,用 `uv pip install mcp`。)然后在 `pyproject.toml` 的 `dependencies` 列表按字母位置加一行:
```toml
    "markdown-it-py",
    "mcp",
    "pyyaml",
```
验证 import 可用:
```fish
.venv/bin/python -c "from mcp.server.fastmcp import FastMCP; print('ok')"
```
Expected: 打印 `ok`。

- [ ] **Step 2: 写失败测试**

Create `tests/test_mcp_server.py`:
```python
import src.mcp_server as mcp_server


def _inject_loaded(monkeypatch):
    # 注入非 None 占位,让 _load() 直接返回、不拉真模型
    monkeypatch.setattr(mcp_server, "_embedder", object())
    monkeypatch.setattr(mcp_server, "_vector_store", object())
    monkeypatch.setattr(mcp_server, "_metadata_store", object())


def test_search_notes_returns_structured_hits(monkeypatch):
    _inject_loaded(monkeypatch)

    def fake_search_chunks(**kwargs):
        return [
            {
                "title_path": "牧原股份财务记录 (002714) > 业务结构",
                "file_path": "stock/companies/002714-financials.md",
                "score": 0.987654,
                "content": "牧原屠宰业务...",
                "domain": "stock",
            }
        ]

    monkeypatch.setattr(mcp_server, "search_chunks", fake_search_chunks)

    out = mcp_server.search_notes("牧原 屠宰")
    assert out == [
        {
            "title": "牧原股份财务记录 (002714) > 业务结构",
            "file_path": "stock/companies/002714-financials.md",
            "score": 0.9877,
            "content": "牧原屠宰业务...",
        }
    ]


def test_search_notes_passes_domain_and_disables_reranker(monkeypatch):
    _inject_loaded(monkeypatch)
    captured = {}

    def fake_search_chunks(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(mcp_server, "search_chunks", fake_search_chunks)

    result = mcp_server.search_notes("价值投资", domain="study", top_k=8)
    assert result == []
    assert captured["domain"] == "study"
    assert captured["top_k"] == 8
    assert captured["reranker"] is None
```

- [ ] **Step 3: 运行测试确认失败**

Run:
```fish
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_mcp_server.py -q
```
Expected: FAIL —— `ModuleNotFoundError: No module named 'src.mcp_server'`。

- [ ] **Step 4: 实现 `src/mcp_server.py`**

Create:
```python
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.config import load_config, resolve_path
from src.embedder import LocalEmbedder
from src.search import search_chunks
from src.sqlite_metadata_store import SQLiteMetadataStore
from src.vector_store import ChromaVectorStore

mcp = FastMCP("markdown-rag")

# 惰性常驻:首次查询或 __main__ 启动时加载一次,之后每轮查询不再重载。
# 用惰性而非模块顶层加载,是为了让单元测试能 monkeypatch 这三个组件、不拉真模型。
_embedder = None
_vector_store = None
_metadata_store = None


def _load() -> None:
    global _embedder, _vector_store, _metadata_store
    if _embedder is not None:
        return
    config = load_config("config.yaml")
    _embedder = LocalEmbedder(
        config["embedding"]["model_name"],
        query_instruction=config["embedding"].get("query_instruction", ""),
    )
    _vector_store = ChromaVectorStore(
        str(resolve_path(config["root_dir"], config["paths"]["chroma_dir"]))
    )
    _metadata_store = SQLiteMetadataStore(
        str(resolve_path(config["root_dir"], config["paths"]["metadata_db"]))
    )


@mcp.tool()
def search_notes(query: str, domain: str | None = None, top_k: int = 5) -> list[dict]:
    """检索本地投资/学习笔记 wiki。domain 可选:投资问题传 "stock",学习问题传 "study",
    不确定就不传(全库)。返回命中页的标题、路径、相关分、正文片段,供你综合分析。"""
    _load()
    results = search_chunks(
        query=query,
        config_path="config.yaml",
        top_k=top_k,
        domain=domain,
        embedder=_embedder,
        vector_store=_vector_store,
        metadata_store=_metadata_store,
        reranker=None,  # 多轮场景不开 reranker(每轮 ~28s 不可接受),走快的混合检索
    )
    return [
        {
            "title": r["title_path"],
            "file_path": r["file_path"],
            "score": round(r["score"], 4),
            "content": r["content"],
        }
        for r in results
    ]


if __name__ == "__main__":
    _load()  # 启动即加载,缺模型/索引立刻 fail-fast
    mcp.run()  # 默认 stdio
```

注意:`@mcp.tool()` 在官方 mcp SDK 中返回原函数,故测试可直接调用 `mcp_server.search_notes(...)`。若该版本装饰器返回包装对象导致测试调用失败,改为在测试里调用其底层可调用(如 `.fn`),并在本步骤记录实际行为。

- [ ] **Step 5: 运行测试确认通过**

Run:
```fish
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests/test_mcp_server.py -q
```
Expected: PASS,2 passed。

- [ ] **Step 6: 跑全量测试确认无回归**

Run:
```fish
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
```
Expected: 全绿(原 31 + 新 2 = 33 passed)。

- [ ] **Step 7: 提交**

```fish
git add pyproject.toml src/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): T3 暴露 search_notes MCP 工具(FastMCP/stdio,复用 search_chunks)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Claude Code 接入文档与实测验收

**Files:**
- Modify: `README.md`(新增「与 Claude Code 对接(MCP)」小节)
- Modify: `USAGE.md`(新增 MCP 接入小节)
- Modify: `RAG增强方案(待审阅).md`(T3 标记完成)

**Interfaces:**
- Consumes: `src.mcp_server`(Task 1 产出),通过 `python -m src.mcp_server` 以 stdio 启动。
- Produces: 无代码接口;产出文档与一条验证过的 `claude mcp add` 命令。

- [ ] **Step 1: README 加 MCP 接入小节**

在 `README.md` 的「与 Codex CLI 对接」小节之后插入:
```markdown
## 与 Claude Code 对接（MCP，多轮自主检索）

把检索暴露成 MCP 工具，Claude Code 作为 client 可自主多轮调用 `search_notes`，
你只需用自然语言提问，AI 自己决定查几轮、查什么、综合分析。

接入（`env -C` 让 server 在项目目录启动，解决 `src` 包与 `config.yaml` 的相对路径）：

```fish
claude mcp add markdown-rag -- \
  env -C /home/neo/project/MarkdownRAG HF_HUB_OFFLINE=1 \
  /home/neo/project/MarkdownRAG/.venv/bin/python -m src.mcp_server
```

之后在 Claude Code 里直接问，例如「分析牧原的护城河和疫病风险」，AI 会自动调用
`search_notes` 多轮检索后给出分析。MCP 路径默认走快的混合检索（秒回），不开 reranker。
```

- [ ] **Step 2: USAGE 加 MCP 接入小节**

在 `USAGE.md` 的「### 3. 生成 LLM 上下文」小节之后插入:
```markdown
### 3.5 MCP 接入 Claude Code（多轮自主检索）
把检索暴露成 MCP 工具，Claude Code 自主多轮调用，自然语言提问即可：
```fish
claude mcp add markdown-rag -- \
  env -C /home/neo/project/MarkdownRAG HF_HUB_OFFLINE=1 \
  /home/neo/project/MarkdownRAG/.venv/bin/python -m src.mcp_server
claude mcp list                 # 确认 markdown-rag 连接成功
```
MCP 路径默认混合检索秒回、不开 reranker(多轮每轮 28s 不可接受)。
```

- [ ] **Step 3: 实测验收(本机,我执行)**

注册并确认 server 能启动握手:
```fish
claude mcp add markdown-rag -- env -C /home/neo/project/MarkdownRAG HF_HUB_OFFLINE=1 /home/neo/project/MarkdownRAG/.venv/bin/python -m src.mcp_server
claude mcp list
```
Expected: `markdown-rag` 显示已连接(✓ Connected)。
再非交互实跑一轮多轮检索分析:
```fish
claude -p "用 search_notes 检索并分析牧原的护城河和疫病风险" 2>&1 | tail -30
```
Expected: 输出里能看到对 `search_notes` 的调用,且基于命中内容给出分析(非空、非瞎编)。
若 `claude mcp add` 子命令语法在当前版本不同,以 `claude mcp --help` 为准调整,并把可用命令回填到 Step 1/2 文档。

- [ ] **Step 4: RAG 方案 T3 段标记完成**

在 `RAG增强方案(待审阅).md` 中:
- 把标题 `## T3. 暴露为 MCP server(对接 Codex/Claude 等)— 暂缓` 改为 `## T3. 暴露为 MCP server(对接 Claude Code)— 已完成(2026-06-21)`。
- 在该段补一行:`实现:单文件 src/mcp_server.py,FastMCP/stdio,单 search_notes 工具,详见 docs/superpowers/specs/2026-06-21-mcp-server-design.md。`
- 更新顶部状态行(第 5 行)的「后续:T3」为 T3 已完成。

- [ ] **Step 5: 提交**

```fish
git add README.md USAGE.md "RAG增强方案(待审阅).md"
git commit -m "docs(mcp): T3 Claude Code 接入文档与方案收尾

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage**:
- 选型 FastMCP/stdio/新依赖 mcp → Task 1 Step 1+4。✓
- 单文件 `src/mcp_server.py`、惰性常驻 `_load`、单 `search_notes` 工具 → Task 1 Step 4。✓
- 工具签名与返回字段(title/file_path/score/content,值取 title_path) → Task 1 Step 2+4。✓
- reranker=None → Task 1 Step 2(断言)+ Step 4。✓
- fail-fast 启动 / 冒泡查询错误 / 空结果返空 → Step 4(`__main__` 调 `_load`;异常未捕获自然冒泡;`search_notes` 对空 results 返回 `[]`)+ Step 2 第二个测试验证空返回。✓
- 测试两条(结构 + domain 透传) → Task 1 Step 2。✓
- Claude Code 接入(env -C) → Task 2 Step 1-3。✓
- 验收(pytest 全绿 + 实跑多轮) → Task 1 Step 6 + Task 2 Step 3。✓
- 不改现有模块 → 全程仅新增 + 文档/依赖,无现有模块改动。✓

**2. Placeholder scan**: 无 TBD/TODO;错误处理、测试均给出具体代码;`@mcp.tool()` 可调用性风险已注明应对。✓

**3. Type consistency**: `search_notes(query, domain, top_k)` 签名、返回 key `title/file_path/score/content`、内部取 `title_path`、`_load`/`_embedder`/`_vector_store`/`_metadata_store` 命名在 Task 1/2 间一致。✓
