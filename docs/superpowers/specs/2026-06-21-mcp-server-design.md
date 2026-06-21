# T3. MCP Server 设计(对接 Claude Code)

> 状态:设计已批准(2026-06-21),待写实施计划。
> 关联:`RAG增强方案(待审阅).md` T3 段。

## 目标

把本地 RAG 检索暴露成 MCP 工具,让 Claude Code 作为 client **自主多轮**调用:
用户问一句自然语言,AI 自己决定查几轮、每轮查什么、传不传 domain,边查边综合分析——
替代当前「手动想查询词 → `context_builder --stdout` 单轮管道」的方式。

## 背景与动机

- 现状路径 A(管道/文件)是**单轮**:用户手动定查询词,一次性拿上下文喂给 AI。
- MCP 的增量价值是 **agentic 多轮**:检索成为 AI 可反复调用的工具,AI 按需追问。
- MarkdownRAG 本身不含 LLM,只做检索;AI 推理来自 Claude Code。MCP 把「检索」这个工具接到 AI agent 上。

## 选型(已定)

- **SDK**:官方 `mcp` Python SDK 内置的 **FastMCP**(`from mcp.server.fastmcp import FastMCP`)。
  装饰器风格,server 可压到几十行。不用第三方 `fastmcp` 2.x(企业功能本地多余),不用 low-level Server(啰嗦)。
- **传输**:stdio(本地 CLI 标准,Claude Code `claude mcp add` 直接接)。
- **新依赖**:`mcp`(`uv pip install mcp`,走清华源)。

## 架构与组件

新增**一个文件** `src/mcp_server.py`,复用现有 `search_chunks`,**不改动任何现有模块**。

```python
# src/mcp_server.py
from mcp.server.fastmcp import FastMCP
from src.config import load_config, resolve_path
from src.embedder import LocalEmbedder
from src.search import search_chunks
from src.sqlite_metadata_store import SQLiteMetadataStore
from src.vector_store import ChromaVectorStore

mcp = FastMCP("markdown-rag")

# 惰性常驻:首次查询或 __main__ 启动时加载一次,之后每轮查询不再重载。
# 用惰性而非模块顶层加载,是为了让单元测试能 monkeypatch 这三个组件、不拉真模型。
_embedder = _vector_store = _metadata_store = None

def _load() -> None:
    global _embedder, _vector_store, _metadata_store
    if _embedder is not None:
        return
    config = load_config("config.yaml")
    _embedder = LocalEmbedder(
        config["embedding"]["model_name"],
        query_instruction=config["embedding"].get("query_instruction", ""),
    )
    _vector_store = ChromaVectorStore(str(resolve_path(config["root_dir"], config["paths"]["chroma_dir"])))
    _metadata_store = SQLiteMetadataStore(str(resolve_path(config["root_dir"], config["paths"]["metadata_db"])))

@mcp.tool()
def search_notes(query: str, domain: str | None = None, top_k: int = 5) -> list[dict]:
    """检索本地投资/学习笔记 wiki。domain 可选:投资问题传 "stock",学习问题传 "study",
    不确定就不传(全库)。返回命中页的标题、路径、相关分、正文片段,供你综合分析。"""
    _load()
    results = search_chunks(
        query=query, config_path="config.yaml", top_k=top_k, domain=domain,
        embedder=_embedder, vector_store=_vector_store, metadata_store=_metadata_store,
        reranker=None,  # 多轮场景不开 reranker(每轮 ~28s 不可接受),走快的混合检索
    )
    return [
        {"title": r["title_path"], "file_path": r["file_path"],
         "score": round(r["score"], 4), "content": r["content"]}
        for r in results
    ]

if __name__ == "__main__":
    _load()       # 启动即加载,缺模型/索引立刻 fail-fast
    mcp.run()     # 默认 stdio
```

**唯一工具** `search_notes(query, domain?, top_k?)`:
- 工具 docstring 是 AI 选库(domain)的唯一依据,务必写清。
- 返回结构化 list,每条 `title/file_path/score/content`;AI 读 `content` 自己综合。
- `reranker=None`、模型常驻 → 秒回,支撑多轮。

## 数据流(agentic 多轮)

```
用户在 Claude Code:"分析牧原的护城河和疫病风险"
  └─ claude 作为 MCP host,启动 mcp_server 子进程(stdio,常驻)
       └─ AI 自主第 1 轮:search_notes("牧原 屠宰 渠道 护城河", domain="stock")
            └─ search_chunks → 向量(Chroma)+关键词(SQLite) 合并+尾部过滤 → 结构化命中
       └─ AI 读 content,信息不够,自主第 2 轮:search_notes("牧原 疫病 生猪价格 风险", domain="stock")
       └─ AI 综合两轮 → 输出分析
```

查几轮、每轮查什么、传不传 domain,全由 AI 决定。

## 错误处理(按项目"不过度防御"风格)

| 时机 | 情况 | 处理 |
|---|---|---|
| 启动期(`_load`) | config/模型/索引缺失 | **fail-fast**:抛错退出,Claude Code 显示「MCP 连接失败」。不静默兜底,否则 AI 拿坏结果不自知 |
| 查询期 | `search_chunks` 内部异常 | 让异常冒泡,FastMCP 自动转成 MCP tool error 返回给 AI,AI 看得到、可重试 |
| 查询期 | 命中为空 | 返回 `[]`(正常返回,非错误),AI 自行换词重查或告知用户没找到 |

## 测试

`tests/test_mcp_server.py`,仿现有 `test_search` 的 fake 风格,**不真起 stdio**:
- `test_search_notes_returns_structured_hits`:monkeypatch 三个模块级组件为 fake(非 None,`_load` 自动跳过真加载),调 `search_notes`,断言每条含 `title/file_path/score/content`、`score` 已 round。
- `test_search_notes_passes_domain`:fake 记录收到的 `domain`,验证参数透传。

## Claude Code 接入

复用 `ragsearch` 的 `env -C` 经验(MCP server 启动同样有 CWD 坑:`-m src.mcp_server` 找包、`config.yaml` 相对路径都要项目根):

```fish
claude mcp add markdown-rag -- \
  env -C /home/neo/project/MarkdownRAG HF_HUB_OFFLINE=1 \
  /home/neo/project/MarkdownRAG/.venv/bin/python -m src.mcp_server
```

文档:更新 README / USAGE 给出此接入命令与一句多轮查询示例。

## 验收标准

- [ ] `pytest` 全绿(含新增 `test_mcp_server.py`)。
- [ ] `claude mcp add` 后在 Claude Code 实跑一句多轮查询,确认 AI 自主多轮调 `search_notes` 并综合输出。
- [ ] 不改动任何现有模块(只新增 `src/mcp_server.py`、`tests/test_mcp_server.py`、依赖与文档)。

## 非目标(YAGNI)

- 不暴露 `context_builder`/`read_note` 等额外工具(单 search 已够 agentic)。
- 不做 SSE/HTTP 传输、认证、多用户(本地单用户 stdio 足矣)。
- MCP 路径不接 reranker(太慢);需要高准确率仍走 CLI + config 开关。
- 不动 wiki 生成端的 4 个遗留失败页(属另一条线)。
