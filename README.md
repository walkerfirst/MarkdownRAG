# Local Markdown RAG

一个面向个人投资研究的本地 Markdown RAG 项目。它会扫描 `notes/` 中的笔记，切块、向量化并写入本地 ChromaDB，同时用 SQLite 维护元数据，支持增量同步、命令行搜索和上下文导出。

## 设计原则

- `notes/` 是源数据，长期保留原始 Markdown。
- `data/` 是缓存，删掉后可以从 `notes/` 完整重建。
- `python -m src.ingest` 做的是同步，不是重复全量导入。

## 安装

```bash
uv venv
uv sync
```

如需兼容传统 pip，也可以继续使用：

```bash
pip install -r requirements.txt
```

## 目录

```text
MarkdownRAG/
├── README.md
├── config.yaml
├── eval/
│   └── queries.yaml
├── notes/
│   ├── sample_maotai.md
│   └── sample_muyuan.md
├── data/
│   ├── chroma/
│   └── processed/
├── src/
│   ├── cleaner.py
│   ├── chunker.py
│   ├── config.py
│   ├── context_builder.py
│   ├── embedder.py
│   ├── evaluate.py
│   ├── hash_utils.py
│   ├── ingest.py
│   ├── loader.py
│   ├── metadata_store.py
│   ├── search.py
│   ├── sqlite_metadata_store.py
│   └── vector_store.py
└── tests/
```

## 导入 Markdown

把你的投资笔记、行业研究或公司分析文章放进 [notes/](/C:/Users/Administrator/project/stock_note/notes)。

清洗阶段会去掉 URL 噪声：`[研报标题](https://...)` 会保留为 `研报标题`，裸 URL、自动链接和引用式链接定义会被移除。

## 构建与增量同步

首次构建：

```bash
uv run python -m src.ingest --reset
```

后续同步：

```bash
uv run python -m src.ingest
```

同步规则：

- 新文件：新增切块、向量和元数据
- 未变化文件：直接跳过，不重复 embedding
- 已修改文件：删除旧 chunk 后重建
- 已删除文件：同步删除 SQLite 和 ChromaDB 中的缓存

## 搜索

```bash
uv run python -m src.search "牧原股份 屠宰业务 渠道"
uv run python -m src.search "贵州茅台 经销商 i茅台" --top-k 8
```

默认使用向量检索和关键词检索合并排序，并按 `search.relative_score_ratio` 过滤尾部低相关结果，避免上下文混入明显无关的笔记。

### 全局别名（fish，任意目录可用）

`python -m src.search` 依赖工作目录：`src` 包和 `config.yaml` 都按当前目录解析，必须在项目根下执行。把下面这个 fish function 存为 `~/.config/fish/functions/ragsearch.fish`（属于本机 shell 个人配置，不在仓库内），即可在任意目录调用：

```fish
function ragsearch --description 'MarkdownRAG 检索:任意目录可用,透传查询词与参数'
    # env -C 在项目目录里执行(src 包与 config.yaml 都是相对 CWD),但不改当前 shell 的 cwd
    env -C /home/neo/project/MarkdownRAG HF_HUB_OFFLINE=1 \
        /home/neo/project/MarkdownRAG/.venv/bin/python -m src.search $argv
end
```

fish 会自动加载，存好即生效。用法（`$argv` 原样透传查询词与参数）：

```fish
ragsearch "牧原股份 屠宰业务 渠道"
ragsearch "贵州茅台 经销商 i茅台" --top-k 8
```

## 生成 LLM 上下文

输出到终端：

```bash
uv run python -m src.context_builder "牧原股份 屠宰业务 护城河"
```

输出到文件：

```bash
uv run python -m src.context_builder "贵州茅台 经销商 i茅台" --output context.md
```

## 与 Codex CLI 对接

标准输出管道方式（必须加 `--stdout`，否则默认写文件、管道收到空输入）：

```bash
uv run python -m src.context_builder "牧原股份 屠宰业务 护城河" --stdout | codex exec "基于输入资料，做价值投资分析"
```

文件方式：

```bash
uv run python -m src.context_builder "贵州茅台 经销商 i茅台" --output context.md
```

## 检索验收

```bash
uv run python -m src.evaluate
uv run pytest
```

## 后续扩展

- PostgreSQL 元数据存储
- Qdrant 向量库
- 混合检索与 reranker
- PDF / Notion / 网页导入

## 大模型调用（以codex为例）
方案 1：管道直传给 Codex CLI
让你的 RAG 输出上下文，然后通过 stdin 交给 Codex（管道必须加 `--stdout`）：
``` python -m src.context_builder "牧原股份 屠宰业务 护城河" --stdout \
  | codex exec "基于输入资料，做一份价值投资分析，重点分析护城河、竞争格局、风险和估值影响。"```
Codex 官方支持这种 prompt + stdin 模式，适合把命令输出作为上下文传给 Codex。

方案 2：生成 context.md，让 Codex 读取
你的脚本输出文件：
python -m src.context_builder "贵州茅台 经销商 i茅台" --output context.md
然后在 Codex CLI 或桌面版 Codex 里说：“请读取 context.md，并基于其中资料分析：i茅台对茅台经销商体系的长期影响。”
Codex CLI 可以读取本地目录中的代码和文件；Codex 桌面版也是面向本地项目和多线程工作的桌面体验。
命令参数说明：
不带参数时 context_builder 会写入并覆盖 context.md；
需要管道时用 --stdout；需要留档时加 --save-analysis，它会额外写入 research_outputs/analysis_时间戳.md

方案 3：做成本地 MCP 工具
这是进阶方案：把你的 RAG 项目暴露成一个 MCP server，让 Codex 直接调用：
search_local_notes(query, top_k)
build_investment_context(query)
然后你在 Codex 里说：
请调用本地 RAG 工具，检索“牧原股份 屠宰业务 护城河”，再基于结果做分析。
Codex 支持配置 MCP server，可用 codex mcp add 添加，也可以写入 ~/.codex/config.toml 或项目级 .codex/config.toml；CLI 和 IDE 扩展共享配置。
示例配置：

[mcp_servers.local_rag]
command = "python"
args = ["-m", "src.mcp_server"]
cwd = "/你的路径/local-md-rag"
同时在项目根目录加 AGENTS.md：
当用户要求分析投资笔记、公司研究、行业资料时，优先调用 local_rag MCP 工具检索本地知识库，再基于检索结果回答。
官方也建议用 AGENTS.md 指导 Codex 何时使用 MCP 工具。
