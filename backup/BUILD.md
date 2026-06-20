# Local Markdown RAG Project Build Spec

## 1. 项目目标

建立一个本地 Markdown 知识库 RAG 项目，用于管理和分析个人投资笔记、行业研究资料、公司分析文章等内容。

项目第一阶段只处理本地 `.md` 文件，不接入 Notion、PDF、网页爬虫等复杂数据源。

核心目标：

1. 扫描本地 `notes/` 目录下的 Markdown 文件
2. 对 Markdown 内容进行清洗
3. 按标题层级和段落语义进行切块
4. 为每个 chunk 生成 embedding
5. 将向量和元数据存入本地向量数据库
6. 支持命令行检索
7. 支持把检索结果整理成可直接交给 LLM 分析的上下文

---

## 2. 技术栈

### 编程语言

- Python 3.11+

### 推荐依赖

- `chromadb`：本地向量数据库
- `sentence-transformers`：本地 embedding 模型，第一阶段优先使用
- `markdown-it-py`：Markdown 解析
- `pyyaml`：读取配置文件
- `rich`：美化命令行输出
- `typer`：构建命令行工具
- `tqdm`：显示处理进度

### 第一阶段不使用

- Notion API
- OpenAI API
- PDF 解析
- Web 爬虫
- 前端界面
- 复杂 Agent 框架

---

## 3. 项目目录结构

请按以下结构创建项目：

```text
local-md-rag/
├── BUILD.md
├── README.md
├── requirements.txt
├── config.yaml
├── notes/
│   ├── sample_muyuan.md
│   └── sample_maotai.md
├── data/
│   ├── chroma/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── ingest.py
│   ├── search.py
│   └── context_builder.py
└── tests/
    ├── test_cleaner.py
    └── test_chunker.py
```

---

## 4. 配置文件要求

创建 `config.yaml`：

```yaml
project:
  name: local-md-rag

paths:
  notes_dir: notes
  chroma_dir: data/chroma
  processed_dir: data/processed

embedding:
  provider: sentence_transformers
  model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

chunking:
  target_chars: 800
  max_chars: 1200
  min_chars: 200
  overlap_chars: 100

search:
  top_k: 5
```

说明：

- 第一阶段默认使用本地 embedding 模型
- 中文笔记可以使用 multilingual 模型
- 后续可以扩展 OpenAI embedding，但不要在第一阶段实现

---

## 5. 功能模块要求

### 5.1 config.py

职责：

- 读取 `config.yaml`
- 提供统一配置对象
- 检查必要目录是否存在
- 如果目录不存在则自动创建

需要实现：

```python
load_config(config_path: str = "config.yaml") -> dict
```

---

### 5.2 loader.py

职责：

- 扫描 `notes/` 目录
- 读取所有 `.md` 文件
- 返回文件路径和正文内容

需要实现：

```python
load_markdown_files(notes_dir: str) -> list[dict]
```

返回格式：

```python
[
    {
        "file_path": "notes/sample_muyuan.md",
        "file_name": "sample_muyuan.md",
        "content": "...markdown text..."
    }
]
```

要求：

- 递归扫描子目录
- 只读取 `.md` 文件
- 使用 UTF-8 编码
- 忽略空文件
- 对读取失败的文件给出清晰错误提示

---

### 5.3 cleaner.py

职责：

- 清洗 Markdown 文本
- 减少无效 token

需要实现：

```python
clean_markdown(text: str) -> str
```

清洗规则：

1. 统一换行符
2. 去除连续 3 个以上空行
3. 去除行尾空格
4. 去除明显无效的 HTML 标签
5. 保留 Markdown 标题
6. 保留列表结构
7. 不要过度清洗，不要破坏正文语义

---

### 5.4 chunker.py

职责：

- 将 Markdown 文本切成适合检索的 chunk
- 尽量保留标题路径

需要实现：

```python
chunk_markdown(
    text: str,
    file_path: str,
    target_chars: int = 800,
    max_chars: int = 1200,
    min_chars: int = 200,
    overlap_chars: int = 100
) -> list[dict]
```

返回格式：

```python
[
    {
        "chunk_id": "sample_muyuan.md::0001",
        "file_path": "notes/sample_muyuan.md",
        "title_path": "牧原股份 > 屠宰业务 > 渠道挑战",
        "content": "...chunk text...",
        "char_count": 756
    }
]
```

切块原则：

1. 优先按 Markdown 标题切分
2. 同一标题下内容过长时，按段落继续切分
3. 每块尽量控制在 `target_chars` 附近
4. 不超过 `max_chars`
5. 过短内容可以与相邻段落合并
6. chunk 中应包含必要标题上下文
7. 每个 chunk 需要保存 `title_path`

---

### 5.5 embedder.py

职责：

- 加载 embedding 模型
- 将文本转成向量

需要实现：

```python
class LocalEmbedder:
    def __init__(self, model_name: str):
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...
```

要求：

- 使用 `sentence-transformers`
- 支持批量 embedding
- 对空文本做防御处理
- 模型只加载一次，避免重复加载造成性能浪费

---

### 5.6 vector_store.py

职责：

- 管理 ChromaDB
- 写入 chunk
- 执行相似度检索

需要实现：

```python
class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "markdown_chunks"):
        ...

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        ...

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        ...

    def reset(self) -> None:
        ...
```

要求：

- 每个 chunk 使用稳定 `chunk_id`
- metadata 至少包括：
  - `file_path`
  - `title_path`
  - `char_count`
- Chroma 文档内容存 `chunk["content"]`
- 支持重复 ingest 时更新已有 chunk，而不是重复插入

---

### 5.7 ingest.py

职责：

- 执行完整入库流程

命令：

```bash
python -m src.ingest
```

流程：

1. 读取配置
2. 扫描 Markdown 文件
3. 清洗正文
4. 切块
5. 生成 embedding
6. 写入 ChromaDB
7. 输出处理统计

输出示例：

```text
Loaded files: 12
Generated chunks: 148
Embedded chunks: 148
Saved to: data/chroma
Done.
```

可选参数：

```bash
python -m src.ingest --reset
```

`--reset` 表示清空原有向量库后重新构建。

---

### 5.8 search.py

职责：

- 执行命令行检索

命令：

```bash
python -m src.search "牧原股份 屠宰业务 护城河"
```

流程：

1. 读取配置
2. 加载 embedding 模型
3. 将 query 转成 embedding
4. 从 ChromaDB 检索 top_k 个 chunk
5. 在终端输出结果

输出格式：

```text
Query: 牧原股份 屠宰业务 护城河

[1] score: 0.83
source: notes/sample_muyuan.md
title: 牧原股份 > 屠宰业务 > 渠道挑战

正文摘要：
牧原在屠宰业务上的核心挑战不是产能，而是渠道、品牌、终端客户关系...
```

要求：

- 使用 `rich` 美化输出
- 显示 source、title_path、score、正文片段
- 默认显示前 500 字
- 支持 `--top-k`

---

### 5.9 context_builder.py

职责：

- 把检索结果整理成适合 LLM 使用的上下文

需要实现：

```python
build_context(results: list[dict], max_chars: int = 6000) -> str
```

输出格式：

```text
以下是从本地知识库检索到的相关资料：

[资料1]
来源：notes/sample_muyuan.md
标题路径：牧原股份 > 屠宰业务 > 渠道挑战
内容：
...

[资料2]
来源：notes/sample_maotai.md
标题路径：贵州茅台 > 渠道改革
内容：
...
```

要求：

- 控制最大字符数
- 保留来源信息
- 方便复制给 ChatGPT / Codex / Claude 等模型继续分析

---

## 6. 命令行使用方式

### 6.1 安装依赖

```bash
pip install -r requirements.txt
```

### 6.2 准备笔记

把 Markdown 文件放入：

```text
notes/
```

### 6.3 构建向量库

```bash
python -m src.ingest
```

或者重建：

```bash
python -m src.ingest --reset
```

### 6.4 搜索

```bash
python -m src.search "贵州茅台 i茅台 经销商体系"
```

### 6.5 指定返回数量

```bash
python -m src.search "牧原股份 屠宰业务 渠道建设" --top-k 8
```

---

## 7. requirements.txt

请创建：

```text
chromadb
sentence-transformers
markdown-it-py
pyyaml
rich
typer
tqdm
pytest
```

---

## 8. README.md 要求

README 需要包含：

1. 项目简介
2. 安装方法
3. 目录结构
4. 如何导入 Markdown
5. 如何构建向量库
6. 如何搜索
7. 后续扩展计划

---

## 9. 测试要求

至少实现以下测试：

### test_cleaner.py

测试：

1. 能去除多余空行
2. 能保留 Markdown 标题
3. 能保留列表
4. 不会把正文清空

### test_chunker.py

测试：

1. 能按标题生成 chunk
2. 每个 chunk 有 `chunk_id`
3. 每个 chunk 有 `title_path`
4. chunk 不超过 `max_chars`
5. 空文本返回空列表

---

## 10. 样例笔记

请在 `notes/sample_muyuan.md` 中创建：

```markdown
# 牧原股份分析

## 生猪养殖主业

牧原股份的核心能力在于自繁自养、一体化管理和成本控制。

## 屠宰业务

屠宰业务的核心挑战不是产能建设，而是渠道建设、客户结构和品牌能力。

### 渠道挑战

面对双汇等传统屠宰企业时，牧原最大的难点在于分销网络、终端客户关系和品牌认知。
```

请在 `notes/sample_maotai.md` 中创建：

```markdown
# 贵州茅台分析

## 经销商体系

茅台长期依赖经销商体系完成渠道渗透、市场维护和价格预期管理。

## i茅台

i茅台提高了公司直控能力，但也对公司治理、消费者运营和价格管理提出了更高要求。
```

---

## 11. 代码质量要求

1. 每个模块职责单一
2. 函数命名清晰
3. 关键逻辑加注释
4. 对异常情况有处理
5. 命令行错误提示要清楚
6. 不要把所有代码写进一个文件
7. 不要引入不必要的大型框架
8. 第一阶段不要过度设计

---

## 12. 第一阶段验收标准

当以下命令可以成功运行，即视为第一阶段完成：

```bash
python -m src.ingest --reset
python -m src.search "牧原股份 屠宰业务 渠道"
python -m src.search "贵州茅台 经销商 i茅台"
pytest
```

搜索结果中应能看到：

- 来源文件
- 标题路径
- 相似度分数
- 相关正文片段

---

## 13. 后续扩展方向

第一阶段完成后，再逐步增加：

1. Notion 导入
2. PDF 导入
3. 网页剪藏
4. OpenAI embedding
5. 混合检索：关键词 + 向量
6. SQLite 元数据管理
7. 增量更新
8. 笔记去重
9. 简单 Web UI
10. 投资研究专用 prompt 模板

---

## 14. 给 Codex 的执行要求

请 Codex 根据本文件完成项目代码。

执行顺序：

1. 创建完整项目目录
2. 创建 `requirements.txt`
3. 创建 `config.yaml`
4. 创建样例 Markdown 文件
5. 实现 `src/config.py`
6. 实现 `src/loader.py`
7. 实现 `src/cleaner.py`
8. 实现 `src/chunker.py`
9. 实现 `src/embedder.py`
10. 实现 `src/vector_store.py`
11. 实现 `src/ingest.py`
12. 实现 `src/search.py`
13. 实现 `src/context_builder.py`
14. 创建测试文件
15. 创建 README
16. 运行测试并修复错误

注意：

- 不要跳过测试
- 不要使用 Notion API
- 不要使用 OpenAI API
- 不要创建前端
- 第一阶段只做本地 Markdown RAG
- 代码必须能在本地直接运行


---

# 15. 方案一增强要求：增量同步、接口抽象与验收测试

本项目第一阶段采用：

```text
SQLite + ChromaDB + 本地 sentence-transformers embedding
```

但代码结构必须为未来升级预留空间，避免业务逻辑与具体数据库强绑定。

---

## 15.1 关键设计原则

### 15.1.1 notes/ 是唯一源数据

```text
notes/ = source of truth
data/  = 可重建缓存
```

要求：

1. 原始 Markdown 文件长期保留在 `notes/`
2. 不要因为文件已完成向量化就移动或删除源文件
3. `data/` 目录中的 SQLite、ChromaDB 数据均视为缓存
4. 删除 `data/` 后，只要 `notes/` 存在，就应能完整重建知识库

---

### 15.1.2 ingest 是同步，不是重复导入

每次执行：

```bash
python -m src.ingest
```

系统应执行“同步”逻辑：

```text
扫描 notes/
↓
计算 content_hash
↓
对比 SQLite 元数据
↓
新增 / 更新 / 跳过 / 删除
↓
同步 ChromaDB
```

---

## 15.2 新增 SQLite 元数据模块

新增文件：

```text
src/metadata_store.py
src/sqlite_metadata_store.py
```

### 15.2.1 metadata_store.py

定义抽象接口，不直接写 SQLite 逻辑。

需要实现：

```python
from abc import ABC, abstractmethod

class MetadataStore(ABC):
    @abstractmethod
    def init_schema(self) -> None:
        ...

    @abstractmethod
    def get_document_by_path(self, file_path: str) -> dict | None:
        ...

    @abstractmethod
    def upsert_document(self, document: dict) -> None:
        ...

    @abstractmethod
    def list_documents(self) -> list[dict]:
        ...

    @abstractmethod
    def delete_document(self, file_path: str) -> None:
        ...

    @abstractmethod
    def upsert_chunks(self, chunks: list[dict]) -> None:
        ...

    @abstractmethod
    def delete_chunks_by_file_path(self, file_path: str) -> None:
        ...

    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        ...
```

要求：

- `ingest.py` 不应直接调用 `sqlite3`
- `search.py` 不应直接调用 `sqlite3`
- 未来可以新增 `postgres_metadata_store.py` 替换实现

---

### 15.2.2 sqlite_metadata_store.py

使用 SQLite 实现 `MetadataStore`。

数据库路径：

```text
data/metadata.sqlite
```

建议表结构：

```sql
CREATE TABLE IF NOT EXISTS documents (
    file_path TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_modified REAL NOT NULL,
    last_ingested_at TEXT NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    title_path TEXT,
    content TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(file_path) REFERENCES documents(file_path)
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
```

---

## 15.3 content_hash 要求

新增工具函数，可放在：

```text
src/hash_utils.py
```

需要实现：

```python
compute_sha256(text: str) -> str
```

要求：

1. 使用 SHA256
2. 对清洗前原始 Markdown 内容计算 hash
3. 文件内容不变则 hash 不变
4. 文件路径变化时，应视为新文件

---

## 15.4 ingest 增量更新机制

`ingest.py` 必须支持以下逻辑：

### 15.4.1 新文件

条件：

```text
file_path 不存在于 documents 表
```

动作：

1. 清洗
2. 切块
3. embedding
4. 写入 ChromaDB
5. 写入 SQLite documents 和 chunks

---

### 15.4.2 未变化文件

条件：

```text
file_path 存在
且 content_hash 一致
```

动作：

```text
跳过，不重新 embedding
```

---

### 15.4.3 已修改文件

条件：

```text
file_path 存在
但 content_hash 不一致
```

动作：

1. 删除该文件旧 chunks
2. 删除 ChromaDB 中该文件对应向量
3. 重新清洗
4. 重新切块
5. 重新 embedding
6. 写入新 chunks 和向量
7. 更新 documents 记录

---

### 15.4.4 已删除文件

条件：

```text
SQLite 中存在 file_path
但 notes/ 中已经不存在
```

动作：

1. 删除 SQLite 中对应 document 和 chunks
2. 删除 ChromaDB 中对应向量

---

### 15.4.5 ingest 输出统计

每次运行必须输出：

```text
Scanned files:
New files:
Updated files:
Skipped files:
Deleted files:
Generated chunks:
Embedded chunks:
```

连续两次运行 `python -m src.ingest` 时，第二次应显示：

```text
New files: 0
Updated files: 0
Deleted files: 0
Skipped files: 全部文件数量
Generated chunks: 0
Embedded chunks: 0
```

---

## 15.5 ChromaVectorStore 接口补充

`vector_store.py` 需要补充删除能力。

新增方法：

```python
def delete_by_file_path(self, file_path: str) -> None:
    ...
```

要求：

1. 根据 metadata 中的 `file_path` 删除对应向量
2. 修改源文件重新 ingest 时，不允许旧向量残留
3. 删除源文件后，不允许搜索结果继续返回已删除文件内容

---

## 15.6 context_builder.py 接口要求

`context_builder.py` 既要能作为模块调用，也要能作为命令行工具使用。

### 15.6.1 函数接口

```python
def build_context(results: list[dict], query: str, max_chars: int = 6000) -> str:
    ...
```

输出必须包含：

```text
用户问题
资料来源
标题路径
chunk 内容
```

### 15.6.2 命令行接口

支持：

```bash
python -m src.context_builder "牧原股份 屠宰业务 护城河"
```

支持输出到文件：

```bash
python -m src.context_builder "牧原股份 屠宰业务 护城河" --output context.md
```

支持控制上下文长度：

```bash
python -m src.context_builder "牧原股份 屠宰业务 护城河" --max-chars 8000
```

---

## 15.7 Codex CLI 对接要求

第一阶段不实现 MCP，但必须支持 stdout 和文件两种对接方式。

### 15.7.1 stdout 管道方式

项目完成后，应能运行：

```bash
python -m src.context_builder "牧原股份 屠宰业务 护城河" | codex exec "基于输入资料，做价值投资分析"
```

### 15.7.2 context.md 文件方式

项目完成后，应能运行：

```bash
python -m src.context_builder "贵州茅台 经销商 i茅台" --output context.md
```

然后可让 Codex / ChatGPT 读取：

```text
请读取 context.md，并基于其中资料进行分析。
```

---

## 15.8 测试补充

新增测试文件：

```text
tests/test_metadata_store.py
tests/test_incremental_ingest.py
tests/test_context_builder.py
```

---

### 15.8.1 test_metadata_store.py

至少测试：

1. 能初始化 SQLite schema
2. 能插入 document
3. 能查询 document
4. 能更新 document hash
5. 能插入 chunks
6. 能根据 file_path 删除 chunks
7. 能根据 chunk_ids 查询 chunks

---

### 15.8.2 test_incremental_ingest.py

使用临时目录构造测试数据。

至少测试：

#### 测试 1：首次 ingest

输入：

```text
notes/a.md
notes/b.md
```

期望：

```text
new_files = 2
updated_files = 0
skipped_files = 0
deleted_files = 0
generated_chunks > 0
```

#### 测试 2：第二次 ingest 跳过

连续运行两次 ingest。

第二次期望：

```text
new_files = 0
updated_files = 0
skipped_files = 2
deleted_files = 0
generated_chunks = 0
embedded_chunks = 0
```

#### 测试 3：修改文件后只更新该文件

修改 `notes/a.md`。

期望：

```text
updated_files = 1
skipped_files = 1
```

并且旧 chunk 不应残留。

#### 测试 4：删除文件后同步删除

删除 `notes/b.md`。

期望：

```text
deleted_files = 1
```

搜索结果中不应再出现 `notes/b.md`。

---

### 15.8.3 test_context_builder.py

至少测试：

1. 输出包含用户问题
2. 输出包含来源文件
3. 输出包含标题路径
4. 输出不超过 `max_chars`
5. 空结果时返回明确提示，而不是报错

---

## 15.9 检索质量验收集

新增文件：

```text
eval/queries.yaml
```

内容示例：

```yaml
queries:
  - query: "牧原股份 屠宰业务 渠道"
    expected_files:
      - "notes/sample_muyuan.md"
    expected_keywords:
      - "渠道"
      - "屠宰"
      - "双汇"

  - query: "贵州茅台 经销商 i茅台"
    expected_files:
      - "notes/sample_maotai.md"
    expected_keywords:
      - "经销商"
      - "i茅台"
      - "直控"
```

新增脚本：

```text
src/evaluate.py
```

命令：

```bash
python -m src.evaluate
```

验收逻辑：

1. 读取 `eval/queries.yaml`
2. 对每个 query 执行 search
3. 检查 top_k 结果是否包含 expected_files
4. 检查结果正文是否包含 expected_keywords
5. 输出通过率

输出示例：

```text
Evaluation result:
Total queries: 2
Passed: 2
Failed: 0
Pass rate: 100%
```

第一阶段合格标准：

```text
Pass rate >= 80%
```

---

## 15.10 第一阶段最终验收命令

项目完成后，必须通过以下命令：

```bash
python -m src.ingest --reset
python -m src.ingest
python -m src.search "牧原股份 屠宰业务 渠道"
python -m src.context_builder "贵州茅台 经销商 i茅台" --output context.md
python -m src.evaluate
pytest
```

合格标准：

1. 所有命令无异常
2. 第二次 ingest 不重复 embedding
3. 搜索结果包含来源、标题路径、正文片段、score
4. `context.md` 可直接交给 Codex / ChatGPT 使用
5. `pytest` 全部通过
6. `evaluate` 通过率不低于 80%

---

## 15.11 第一阶段暂不实现内容

为了防止过度工程化，第一阶段明确不实现：

1. PostgreSQL
2. Qdrant
3. Notion API
4. PDF 解析
5. Web UI
6. FastAPI
7. Next.js
8. Temporal
9. MCP Server
10. OpenAI / Voyage / Gemini API 调用

但代码结构必须允许未来增加：

```text
postgres_metadata_store.py
qdrant_vector_store.py
reranker.py
mcp_server.py
```

---

## 15.12 给 Codex 的额外执行要求

Codex 生成代码时必须遵守：

1. 先实现基础功能，再实现增量同步
2. 所有数据库访问集中在 `sqlite_metadata_store.py`
3. 业务模块通过接口调用 metadata store
4. 不允许在 `ingest.py` 中写 SQL 语句
5. 不允许每次 ingest 都全量重建，除非显式传入 `--reset`
6. `--reset` 应删除 ChromaDB 和 SQLite 后重新构建
7. 所有路径必须从 `config.yaml` 读取
8. 测试中必须使用临时目录，不污染真实 `data/` 和 `notes/`
9. 命令行输出必须清晰显示统计信息
10. README 必须说明：
    - notes/ 是源数据
    - data/ 是缓存
    - 如何增量更新
    - 如何生成 context.md
    - 如何与 Codex CLI 对接
