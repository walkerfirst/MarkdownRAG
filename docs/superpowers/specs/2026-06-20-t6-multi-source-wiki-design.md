# T6 设计:多源 wiki + 分类与元数据过滤检索

> 状态:已确认设计,待生成实施计划。
> 关联:`RAG增强方案(待审阅).md` 的 T6;接 T5(chunk_id 防撞)与 T1(bge-small embedding)。

## 目标

把数据源从单一 `notes_dir` 升级为多个 `wiki/` 根目录;按路径自动分类(`domain` 手填、`type` 取子目录),并从 wiki 页头抽取 bold-key 元数据(`evidence_level` / `freshness`),让检索可按这些维度过滤。**无需多个 Chroma collection**:路径 + 页头分类同时解决「多 corpus 分离」与「分类过滤」。

## 现状事实(已核验)

- 真实数据源存在:`/home/neo/Documents/obsidian/investing/wiki/`(51 个内容页)、`/home/neo/Documents/obsidian/learning/wiki/`(目前基本是空的:`templates/` 4 个 + `index.md`,真正内容页约 2~3)。
- investing 子目录分布:companies 32、portfolio 8、playbooks 3、journal 3、industries 2、checklists 2、decisions 1、events 0,另加根下 `index.md`、`log.md`。
- 页头格式一致:`**Summary**` / `**Sources**` / `**Last updated**` / `**Freshness**` / `**Evidence level**`,后接独立 `---` 行,再正文。
- `[[链接]]` 普遍:51 个内容页里 49 个含 `[[ ]]`。
- 撞名风险确认:当前 `loader` 用 `file_path = relative_to(notes_path.parent)`,两个源都叫 `wiki/` 必然撞名 → 必须加 `domain/` 前缀。

## 已定决策

1. **数据源范围**:两源都配,`investing/wiki`(domain=stock)+ `learning/wiki`(domain=study);learning 现在没内容也先占位,以后写了自动进库。investing 是真实评测库。
2. **排除规则**:排除 `index.md`、`log.md`(文件名级)与 `templates/`(目录名级)——都是 TOC / 操作日志 / 空白格式模板,非知识内容。
3. **清洗与元数据**:本任务一起做 `[[链接]]` 清洗 + 页头元数据抽取。
4. **Summary 处理**:页头整块不进向量,**唯独保留 `**Summary**` 的值拼回正文开头**作为高信号语义锚;其余 bold-key 行 + `---` 不进正文。

## 设计

### 1. 配置形态(`config.yaml` + `src/config.py`)

`paths.notes_dir` → 顶层 `sources` 列表:

```yaml
sources:
  - path: /home/neo/Documents/obsidian/investing/wiki/
    domain: stock
  - path: /home/neo/Documents/obsidian/learning/wiki/
    domain: study
exclude_names: [index.md, log.md]   # 文件名级排除
exclude_dirs: [templates]            # 目录名级排除
```

`src/config.py`:把 `notes_dir` 相关逻辑改为遍历 `sources`,**校验每个 `path` 存在**(外部只读,不 `mkdir`);`chroma_dir` / `processed_dir` / `metadata_db` / `eval_queries` 的处理不变。`exclude_names` / `exclude_dirs` 缺省给空列表。

### 2. file_path 唯一性(接 T5)

`file_path = "{domain}/{相对 source 根的 posix 路径}"`,例:`stock/companies/600519.md`、`study/concepts/xxx.md`。

- 跨源唯一,撞名彻底消除;`chunk_id = f"{file_path}::{idx:04d}"`(T5 已改)天然继续唯一。
- **`type` = `domain` 之后第一层子目录名**(`stock/companies/x.md` → `companies`)。因为 index/log/templates 已排除,剩余文件必在子目录下,`type` 永远有值,无需处理「根下裸文件」特例。

### 3. 页头元数据解析(`src/loader.py`)

按**第一个独立 `---` 行**切分:之前为 header 块,之后为正文。header 块内逐行解析 `**Key**: value`:

| 页头字段 | 去向 |
|---|---|
| `**Evidence level**` | → `evidence_level` 字段(过滤用) |
| `**Freshness**` | → `freshness` 字段(过滤用) |
| `**Last updated**` | → `last_updated` 字段(展示用,先存着) |
| `**Summary**` | **保留**,作为正文第一段进向量(高信号语义锚) |
| `**Sources**` | **丢弃**(一堆 raw 文件路径,纯噪声) |

正文重建:`Summary 值 + "\n\n" + (---之后的正文)`。

**兜底**:页面无独立 `---` 或无 bold-key 时,整篇当正文;`evidence_level` / `freshness` / `last_updated` 留空;`type` 仍来自路径。

`loader.load_markdown_files` 改为遍历所有 `sources`,每个文件返回 dict 增加 `domain` / `type` / `evidence_level` / `freshness` / `last_updated` 字段,`content` 为重建后的正文,`file_path` 带 `domain/` 前缀。应用 `exclude_names` / `exclude_dirs` 过滤。

### 4. `[[链接]]` 清洗(`src/cleaner.py`)

新增规则,接在现有 `strip_links` 流程里:

- `[[path/page|alias]]` → `alias`
- `[[path/page]]` → `page`(取末段、去 `/` 与 `#anchor`)
- `[[page]]` → `page`

### 5. schema 与存储(denormalize 到 chunk)

`domain` / `type` / `evidence_level` / `freshness` **冗余写到每个 chunk**(过滤需落到 chunk 粒度):

- `src/sqlite_metadata_store.py`:`chunks` 表与 `documents` 表各加 `domain` / `type` / `evidence_level` / `freshness` 列(`documents` 另加 `last_updated`)。`init_schema` / `upsert_*` 同步。
- `src/vector_store.py`:`upsert_chunks` 的 Chroma metadata 加这 4 个键;`search` 返回结果带上。
- 无需多 collection。

`src/chunker.py` / `src/ingest.py`:chunk dict 透传这 4 个字段(从 file dict 注入到每个 chunk),其余切块逻辑不变。

### 6. 过滤检索(`src/search.py` / `src/context_builder.py` / CLI)

`search_chunks` 增加 `domain` / `type` / `evidence` 参数。`domain` / `type` 是干净的标量,`evidence_level` 是复合值(如 `Primary source | User view`),两者过滤方式不同:

- **`domain` / `type`(精确等值)**:推到检索层。向量侧 Chroma `collection.query(where={...})`(多条件 `$and`);关键词侧 `search_chunks_by_keywords` 加 WHERE 等值条件。
- **`evidence`(子串、大小写不敏感)**:因复合值无法走 Chroma 精确 `where`,改为**对合并后的结果做 Python 后过滤**(`evidence.lower() in chunk["evidence_level"].lower()`)。evidence 是粗粒度选择器,后过滤可接受;若日后过滤过于损召回,再把 `evidence_level` 规范化成前导质量 token 推入检索层。
- CLI:`src.search` 与 `src.context_builder` 加 `--domain` / `--type` / `--evidence` 选项,透传给 `search_chunks`。

### 7. 增量同步(`src/ingest.py`)

`load_markdown_files` 遍历所有 sources、带 `domain/` 前缀返回并集。现有「按 `file_path` 集合对账删除」逻辑天然支持跨源删除(`set(existing) - current_paths`),无需大改。`upsert_document` 带上新字段。schema 变更后须 `--reset` 重建一次。

## 验证标准

- [ ] `--reset` 在真实 wiki 上重建成功,扫描/入库计数符合预期(约 51 stock + 2 study 内容页;index/log/templates 不计入)。
- [ ] 构造跨源同名页(`stock/.../note.md` 与 `study/.../note.md`),两者 chunk 均存在、互不覆盖。
- [ ] 抽样过滤生效:`src.search "牧原 屠宰" --domain stock --type companies`、`--evidence Primary` 各跑一遍,结果符合过滤条件。
- [ ] 一个含 `**Summary**` 的页,其 Summary 文本能被检索命中(确认进了向量);`**Sources**` 的 raw 路径不出现在 chunk 正文。
- [ ] `[[牧原股份]]` 类链接在入库正文里被清成纯文字。
- [ ] `uv run pytest` 全绿(更新 loader / cleaner / chunker / store 相关断言,新增页头解析与链接清洗测试)。

## 不在本任务

- T7 真实 wiki 评测集(下一个任务)。
- reranker(T2)、MCP(T3)。
- raw / daily-notes 纳入(将来加 `stage` 字段)。
- 细粒度专题标签。

## 环境备忘(本机,沿用本轮)

- 不用 `uv sync` / `uv run`(会按锁从 pytorch 源重装 torch,国内必断);用 `uv pip install` 装依赖,跑脚本用 `.venv/bin/python ...`。
- HF 模型走本地目录 + `HF_HUB_OFFLINE=1`。
