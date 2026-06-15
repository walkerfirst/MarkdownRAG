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


请为 ingest 增加增量更新机制：

1. 为每个源文件计算 SHA256 content_hash
2. 使用 SQLite 记录 file_path、content_hash、last_modified、last_ingested_at、status
3. 每次 ingest 时：
   - 新文件：处理并入库
   - hash 未变化：跳过
   - hash 已变化：删除旧 chunks 和旧向量后重新入库
   - notes/ 中已不存在的文件：从 metadata 和 vector store 中删除
4. 不允许重复插入相同 chunk
5. 输出统计：
   - scanned files
   - new files
   - updated files
   - skipped files
   - deleted files
   - generated chunks