# RAG 增强方案(待审阅)

> 用途:列出后续要做的更新,采用「任务 + 验收」格式,供审阅与讨论。
> 状态约定:`待审阅` = 方案待你确认;`已确认` = 可动手;`已完成` = 做完并通过验收。
> **本轮 T0 + T5 + T1 + 前置配置 已完成并验证(2026-06-15)。T6 + T7 已完成并验证(2026-06-20)。T2 已完成并测量(2026-06-21,reranker 全文 80%/+25pp,默认关 opt-in)。T3 已完成(2026-06-21,MCP server 对接 Claude Code)。原 T4 已并入 T6。**
> **数据源路径已迁移(2026-06-20):`/home/neo/project/notes/{investment,learning}/wiki/`。**

## 执行记录与环境备忘(本轮已完成)

**完成项与验收结果**
- **T0 依赖**:torch 2.12.0+cpu、`cuda False`、`uv.lock` 无 nvidia;`requirements.txt` 已删、统一 uv。
- **T5 chunk_id**:改用 `file_path`,实测 `notes/sample_maotai.md::0001`;同名不同目录不再冲突。
- **T1 embedding**:`bge-small-zh-v1.5` 本地副本,backend=sentence-transformers,query/passage 维度均 512,query 带指令前缀、passage 归一化。
- **前置配置**:`notes_dir = /home/neo/project/notes/`,外部建库成功(28 文件 / 419 块),抽样搜索命中正常。
- 样例笔记验证:`evaluate` 2/2(100%)、`pytest` 15 passed。

**环境踩坑备忘(本机国内网络,重要)**
- **不能用 `uv sync` / `uv run`**:它们会强制按锁从 pytorch 源重装 torch,而国内拉 `download.pytorch.org` 大文件必断 → 整个 sync 回滚。
  - 正确姿势:`uv pip install` 装依赖;跑脚本用 `.venv/bin/python ...`(或 `uv run --no-sync`)。
- **torch CPU 轮子**:SJTU 镜像只镜像列表、二进制会跳回官方源(无效);**Aliyun 扁平目录真托管**:`https://mirrors.aliyun.com/pytorch-wheels/cpu/<wheel>`,curl 下来 `uv pip install` 本地轮子。
- **PyPI**:全局 `~/.config/uv/uv.toml` 已配清华源。
- **HF 模型**:`huggingface_hub` 客户端连 hf-mirror 也不稳;改为 curl 全量下载到 `models/bge-small-zh-v1.5/`,config 指向本地路径,跑时加 `HF_HUB_OFFLINE=1`。`models/` 已加入 `.gitignore`。

**待你确认**
- `/home/neo/project/notes/` 当前是 **28 篇技术/系统类笔记(非投资笔记)**。本项目定位是投资笔记 RAG,请确认是否就用这批做知识库;若投资笔记在别处,告诉我新路径。
- 注意:`eval/queries.yaml` 绑定的是样例投资笔记,corpus 切到外部技术笔记后 `evaluate` 不再适配(属预期)。


---

## 背景与原则

- 现有架构已正确:ChromaDB(向量) + SQLite(元数据/增量) + sentence-transformers + 向量/关键词混合检索 + `context_builder` 导出给 LLM。本方案是**增强,不是重写**。
- 改动遵循最小影响面:能靠改 `config.yaml` 完成的不动代码;新增能力优先做成独立文件,不破坏现有增量同步逻辑。
- 每个任务独立可回退。本轮已完成 **T0 → T5 → T1**;后续顺序 **T6 → T7 → T2 → T3**(原 T4 已并入 T6),理由见「决策记录」。

### 数据源原则:RAG 索引 wiki 蒸馏层

投研笔记采用 Karpathy LLM Wiki 模式(见 `investing/AGENTS.md`),是一条**加工流水线,raw 与 wiki 是不同的层、不是同一内容的两种语法**:

```
raw/(一手原文:年报/研报/新闻/券商,不可变,只读)
daily-notes/(用户每日思考,User view)
        │ AI 蒸馏
        ▼
wiki/(结构化、互链、带证据级别/citation 的知识页)
```

- **RAG 主索引 `wiki/`**(蒸馏层):高信号、已成稿、正对"定点查询",契合 AGENTS.md 的 QA 工作流(读 index→读页→引用 wiki)。
- `daily-notes/` 可选纳入;`raw/` 默认不进(体量大、噪声多、已蒸馏进 wiki;仅需回溯原始证据时再单列)。
- **角色分工**:wiki 给人浏览/关联(广度),RAG 给机器查询/命中(精度),互补。
- wiki 页里的 `[[链接]]` 嵌入前清成文字(小卫生);链接图反哺 RAG 做图检索是可选的深层对接,暂缓。

### 为什么自建,而不直接用 Google NotebookLM

定位不同:**本地方案是私有知识库引擎(沉淀资产),NotebookLM 是一次性多模态加工厂(出花活)**。按个人投研画像,主力选本地,理由集中在四点本地占优、NotebookLM 是结构性短板:

- **隐私**:投资笔记含持仓/未公开判断,本地数据不出机器;NotebookLM 需把全部源文档上传 Google 云。
- **成本**:本地零订阅(LLM 复用已有 Codex/Claude);NotebookLM 要 Gemini 3 全量需 Pro $19.99/月起。
- **可控**:切块/embedding/重排/prompt 全可调,中文投资语料的检索精度可超过通用黑盒;NotebookLM 内部不可干预。
- **工作流**:T3 做成 MCP 工具后可嵌入 agentic 链路、脚本化、自动化;NotebookLM 是封闭 Web 应用。

NotebookLM 的不可替代优势(作为补充):**音频/视频概览、幻灯片/报告一键生成**,以及 PDF/网页/视频多源开箱即用、零维护。
**用法**:敏感数据只走本地;非敏感的公开研报临时拿 NotebookLM 做音频概览/快速摸底。

### 与开源框架的关系(RAGFlow / PrivateGPT / LlamaIndex / LangChain)

自建核心**保留**,不被框架替换。原因同上:轻量、可控、可调试、中文适配、与 Codex/Claude 的 agentic 集成,这几点正是框架的短板或与 KISS 冲突。框架"省去从零搭 RAG"的价值对你已是沉没收益。

- **LangChain / PrivateGPT — 不采用**:前者重抽象难调试且 agent 与已有链路重叠;后者与自建能力重叠、丢控制权。
- **RAGFlow — 不整体采用**:重型服务产品(Docker+ES),仅当复杂表格 PDF 研报成为主力来源时,当独立解析服务备选。
- **LlamaIndex — 仅按库借一块**:做 PDF/网页导入时,**优先评估 LlamaParse / node parsers** 只用其解析切块组件,产出仍写回现有 Chroma+SQLite,不引入其 index/query engine 抽象。

> 唯一真实能力缺口 = 复杂文档解析,届时按需借用,其余一律不引入。

---

## 前置配置:笔记存放路径(已定:项目外绝对路径)

**决策**:笔记放**项目外**,在 `config.yaml` 用**绝对路径**指定(方案 2a),不走子文件夹、不走软链接。

**改动点**
- `config.yaml` 的 `paths.notes_dir` 改为绝对路径(已定):`notes_dir: /home/neo/project/notes/`。
- `src/config.py` 的 `resolve_path` 已支持绝对路径([config.py:9](/home/neo/project/MarkdownRAG/src/config.py:9)),无需改代码。
- 项目内原 `notes/` 仅留 sample 或清空;真实笔记不入本仓库 git。

**验收标准**
- [ ] `config.yaml` 指向外部绝对路径后,`uv run python -m src.ingest --reset` 能扫描到外部笔记并建库。
- [ ] 外部目录下新增/修改/删除 .md 后,`src.ingest` 增量同步正确(新增/更新/删除计数对得上)。

**注意**
- 绝对路径是本机相关,换机器需改 `config.yaml`(可接受)。
- 笔记目录身份决定 `file_path`;首次切到新路径需 `--reset` 重建一次。

---

## T0. 依赖修复:torch 改用 CPU 版(本轮,前置)

**目标**:AMD 平台不装 NVIDIA/CUDA 包,`uv sync` 能跑通。

**改动点**
- `pyproject.toml` 增加 uv 源配置,把 `torch` 指向 PyTorch CPU 源:
  ```toml
  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }

  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  ```
- 同步检查 `requirements.txt` 是否需要相应说明(传统 pip 路径)。

**验收标准**
- [ ] `uv sync` 成功,无 `nvidia-*` 包出现在 `uv.lock`(`grep nvidia uv.lock` 无结果)。
- [ ] `uv run python -c "import torch; print(torch.__version__)"` 正常,`torch.cuda.is_available()` 返回 `False` 不报错。
- [ ] `uv run pytest` 原有测试全绿(确认依赖切换没破坏现状)。

**待讨论**
- 是否同时保留 `requirements.txt` 的 pip 路径?还是统一只用 uv?

---

## T1. 升级 embedding 模型:MiniLM → bge-small-zh-v1.5(本轮,速度优先)

**目标**:中文检索提升 + 轻量快速。`paraphrase-multilingual-MiniLM-L12-v2`(2021,384维)换成 `BAAI/bge-small-zh-v1.5`(512维,约 95MB,CPU 快;中文专用)。质量优先的 bge-m3 本轮不选,留作后续按 evaluate 数据再评估。

**改动点**
- `config.yaml` 的 `embedding.model_name` 改为 `BAAI/bge-small-zh-v1.5`。
- `src/embedder.py`:
  - `encode` 启用 `normalize_embeddings=True`(bge 系列建议归一化走余弦)。
  - **非对称检索**:  建议给 query 加指令前缀「为这个句子生成表示以用于检索相关文章:」,passage 不加。故在 `embed_query` 路径加前缀,`embed_texts`(建库 passage)不加;前缀在 config 配置、留空即关闭。
- **维度变化(384→512)**:Chroma collection 维度固定,换模型后必须 `--reset` 全量重建(与 T5 合并一次)。

**验收标准**
- [ ] `embedder.backend == "sentence-transformers"`(确认不是哈希兜底)。
- [ ] `--reset` 重建成功,向量维度为 512。
- [ ] `uv run python -m src.evaluate` 的命中指标 ≥ 换模型前(用同一份 `eval/queries.yaml` 跑前后对比)。
- [ ] 抽查 2~3 个真实 query(如「牧原 屠宰业务 护城河」),Top-3 结果主观相关性不劣于现状。

**待讨论 / 后续**
- 若 evaluate 显示小模型不够,再评估升级 `BAAI/bge-m3`(1024维,~2.2GB,更慢更强)。

**bge-m3 vs bge-small-zh-v1.5 对比(2026-06-20 讨论)**

| 维度 | bge-small-zh-v1.5(现状) | bge-m3 |
|---|---|---|
| 参数/体积 | ~24M / ~95MB | ~560M / ~2.2GB |
| 向量维度 | 512 | 1024 |
| **最大输入长度** | **512 token(超出截断)** | **8192 token** |
| 语言 | 中文专用 | 多语种(中英混排强) |
| CPU 速度 | 快(已验证) | 慢约 1 个数量级 |
| 内置能力 | 纯 dense | dense + sparse + ColBERT 多向量 |
| query 指令前缀 | 需要 | 不需要 |
| 中文检索质量(C-MTEB) | 中上 | 明显更强(长/复杂 query) |

- **决定性差异是「输入长度」而非「质量分」**:wiki 蒸馏页偏长,bge-small 512 token 上限会**静默截断**长 chunk;当前 `target_chars=800` 大体在限内,但调大 chunk 或含长表格/清单时会丢尾部。bge-m3 的 8192 上限免疫此问题。
- **m3 的 sparse 与自建混合检索重叠**:统一链路需改 `embedder`/`search` 融合逻辑并引入 `FlagEmbedding`,与 KISS 冲突,本轮不为 sparse 上 m3。
- **CPU 代价真实**:560M 模型单条 query 编码从几十 ms 升到数百 ms 量级,建库变慢;2.2GB 模型还要走国内下载老路。

**结论(锚定 T7,不凭感觉)**:本轮维持 bge-small。顺序仍为 **T6→T7→用同一评测集对比 small / m3 / +reranker**。经验上「small + reranker(T2)」通常优于「直接换 m3 无重排」,故 m3 优先级排在 T2 之后。唯一提前考虑 m3 的信号:T6 后发现 wiki 页普遍 >512 token 且须整页/大块检索——此时是「长度需求」逼上 m3,可只换 m3 的 dense(经 sentence-transformers 加载,改动小),先不碰 sparse/ColBERT。

---

## T8. 清行内 `(source: ...)` 引用噪声 — 已验证无效,放弃(2026-06-20)

**假设**:wiki 正文行内 `(source: 长文件名.pdf)`(财务页 ~23% 行)是 embedding 噪声,清掉可提升兄弟页区分度。曾实现"向量用清洗版、存储留原版"(嵌入/存储分离,保溯源)。
**结果(实测)**:T7 域内命中率 **55%→55% 零变化**;失败页名次几乎纹丝不动(22→22/22→23/21→21…),个别略降。
**结论**:source 噪声**不是瓶颈**。它在所有兄弟页里均匀分布,清掉只降绝对噪声、不改**相对**排序;真正的瓶颈是同公司 7+ 兄弟页**语义高度重叠**——这是 reranker(T2)与 wiki 生成端重构的territory,不是清噪声能解的。**已撤回实现**,记此教训以免重犯。

> wiki 生成端规约(你审后另行处理):表格关键事实补散文(针对 `当前持仓` 类召回缺失)、中文小标题、标题/Summary 带公司名、Summary 写明与兄弟页的区分点。

---

## T2. 加入本地 reranker(进一步提准)— 已完成并测量,默认关/opt-in(2026-06-21)

**目标**:在混合检索召回后,用交叉编码器对候选重排,提升 Top-K 精度。

**已定设计(2026-06-21)**
- **模型**:`bge-reranker-v2-m3`(质量优先);本地 `models/bge-reranker-v2-m3/`,`HF_HUB_OFFLINE=1`。
- **分数融合**:**直接用重排分替换**(reranker 即权威);重排分经 sigmoid 归一 0~1,与 cosine 同量纲,下游 `_filter_results` 不变。
- **candidate_k = 30**:T7 诊断显示失败页埋在第 21~27 名,窗口必须够深;**开重排时向量取 candidate_k**(否则深埋页进不了候选),keyword limit≥candidate_k。

**改动点(已实现)**
- ✅ `config.yaml` 新增 `reranker` 段:`enabled: false`(默认关、可回退)、`model_name`、`candidate_k: 30`。
- ✅ 新增 `src/reranker.py`:`LocalReranker`,`CrossEncoder` 加载,"模型不可用则跳过"兜底(同 embedder 风格)。
- ✅ `src/search.py` 的 `search_chunks` 加 `reranker` 参数:开启时按 candidate_k 取候选 → merge → 对前 candidate_k 个 `rerank` 替换分 → `_filter_results`。
- ✅ 测试:`test_reranker`(重排序/兜底跳过)、`test_search` 集成(enabled 时顺序随重排)。`pytest` 31 passed,默认关行为不变。

**验收标准**
- [x] `enabled=false` 行为与现状一致(可回退)。
- [x] 模型加载失败自动跳过、不中断(打 warn)。
- [x] `enabled=true` 在 T7(域内,基线 55%)上测量:**全文重排命中率 80%(16/20,+25pp)**,救回 5/9 失败页。

**测量结果(T7 域内评测,2026-06-21)**

| 方案 | 命中率 | 单 query CPU 耗时 |
|---|---|---|
| 关 reranker(基线) | 55% (11/20) | 秒回 |
| reranker 全文 | **80% (16/20)** | ~28s |
| reranker 截 256 字 | 75% (15/20) | ~7s |

- **延迟根因**:cross-encoder 对 candidate_k=30 个约 800 字 passage 逐对前向,纯 CPU 无 GPU,~28s/query(整套 20 query 评测 9m13s 即由此而来)。
- **提速杠杆**:打分 passage 截到 256 字可 28s→7s(快 4 倍),代价命中率 80%→75%(截断有损:只看前 256 字判相关性,后段被忽略,打乱同文件 chunk 排序)。已在 `LocalReranker(max_passage_chars=...)` 实现,config 默认不启用。
- **遗留 4 个失败页**(reranker 解不了,属 wiki 生成端):当前持仓(recall miss >30,表格锁住实体→需散文化)、牧原风险/牧原决策(同公司兄弟页语义过近→需区分性摘要)、Samba(页内容稀疏→需补充)。

**默认决策(2026-06-21,用户定)**:**默认关 + opt-in**。日常检索秒回(55%);需要高准确率时手动 `enabled: true`,用全文吃满 80%、接受 ~28s。28s 是用户主动选择换来的精度,值;不接受日常每查都等。若后续不满意再改默认开。

---

## T3. 暴露为 MCP server(对接 Claude Code)— 已完成(2026-06-21)

**目标**:把检索做成 MCP 工具,Claude Code 作为 client 自主多轮调用,替代手动管道/文件方式。

**实现(已完成)**
- 新增单文件 `src/mcp_server.py`:FastMCP/stdio,暴露唯一工具 `search_notes(query, domain?, top_k?)`,复用 `search_chunks`,惰性常驻加载,`reranker=None` 走快的混合检索支撑多轮。
- `pyproject.toml` 加 `mcp` 依赖;新增 `tests/test_mcp_server.py`(2 条),全量 33 passed。
- 接入:`claude mcp add` + `env -C`,`claude mcp list` 实测 ✔ Connected;端到端真实查询返回牧原命中。README/USAGE 已补接入示例。
- 设计与计划:`docs/superpowers/specs/2026-06-21-mcp-server-design.md`、`docs/superpowers/plans/2026-06-21-mcp-server.md`。

**验收标准**
- [ ] `uv run python -m src.mcp_server` 能启动并被 MCP 客户端发现两个工具。
- [ ] 在 Codex 中调用 `search_local_notes`,返回结果与 CLI `src.search` 一致。
- [ ] `build_investment_context` 输出与 `src.context_builder` 一致。

**待讨论**
- MCP 框架选型:官方 `mcp` SDK 还是 `fastmcp`?
- 传输方式:stdio(本地最简) 还是 HTTP/SSE?
- 工具是否要加 `top_k`、`rerank` 等参数透传?

---

## T5. chunk_id 防撞修复(基础修复,本轮,先于 T1)

**目标**:消除不同文件夹同名文件的 chunk_id 冲突。当前 `chunk_id = f"{file_name}::{idx}"`([chunker.py:148](/home/neo/project/MarkdownRAG/src/chunker.py:148)),`茅台/note.md` 与 `牧原/note.md` 会生成相同 id 互相覆盖,导致数据丢失。你将把真实笔记放进外部目录(很可能含子文件夹),同名风险真实存在,故本轮先修。
> 已核验改动安全:`chunk_id` 仅用作 Chroma `ids` 与 SQLite 主键(字符串,见 [vector_store.py:33](/home/neo/project/MarkdownRAG/src/vector_store.py:33));`delete_by_file_path` 走 `file_path` 不受影响;`test_chunker` 未断言 chunk_id 格式。

**改动点**
- `src/chunker.py`:`chunk_id` 改用 `file_path`(相对路径)而非仅 `file_name` 作前缀,例 `f"{file_path}::{idx:04d}"`,保证全局唯一。
- 确认 `vector_store` / `metadata_store` 对新格式 chunk_id 无长度或字符假设(SQLite TEXT 主键无碍;Chroma id 为字符串亦可)。
- 因 chunk_id 是身份,修改后须 `uv run python -m src.ingest --reset` 重建(与 T1 合并一次重建)。

**验收标准**
- [ ] 构造两个不同子目录下的同名 .md,`--reset` 后两者 chunk 均存在、互不覆盖。
- [ ] `uv run pytest` 中 `test_chunker` 等相关测试通过(同步更新断言)。
- [ ] 重建后 `src.evaluate` 指标不劣于修复前。

**待讨论**
- chunk_id 用完整相对路径会变长,是否接受?(备选:对 file_path 取短 hash 前缀)

---

## T6. 多源 wiki + 分类与元数据过滤检索(已完成 2026-06-20;原 T4 并入)

> **完成记录(2026-06-20)**:设计/计划见 `docs/superpowers/specs/2026-06-20-t6-multi-source-wiki-design.md` 与 `docs/superpowers/plans/2026-06-20-t6-multi-source-wiki.md`。
> - `notes_dir` → `sources` 列表;`file_path` 加 `domain/` 前缀防跨源撞名;`type` 自动取子目录;页头 bold-key 抽 `evidence_level`/`freshness`/`last_updated`,Summary 保留进向量、Sources 丢弃;`[[链接]]` 清成文字。
> - **排除策略(2026-06-20 修订)**:仅排除 `log.md`(操作日志)与 `templates/`(空模板);`index.md` **改为纳入**——带注释的 index 是全库地图,概览/定位类查询里常是唯一能一页答全的命中(根级无子目录,`type=""`)。
> - 字段冗余落到 chunk(SQLite + Chroma metadata),无需多 collection。检索 `--domain`/`--type` 入检索层(Chroma where + SQL),`--evidence` 子串后过滤。
> - **实测**:`--reset` 重建,扫描 89 → 入库 79(stock 49 + study 30;index/log/templates 全部排除生效)。`pytest` 26 passed。抽样:`--domain stock --type companies` 只返回该类;`--evidence Primary` 12→10 剔除非 Primary;Summary 文本可检索命中、chunk 正文无 `**Sources**`/`**Evidence level**` 残留。

**目标**:支持多个 `wiki/` 根目录(取代单一 `notes_dir`);按路径自动分类(domain 手填、type 自动取子目录)+ 抽取 wiki 页头部元数据,检索可按这些维度过滤。
> **原 T4 已并入**:T4 的"解析 YAML frontmatter tags"方向**作废**——wiki 页用的是 **bold-key 元数据**(`**Evidence level**:`、`**Freshness**:`、`**Sources**:`),不是 YAML;改为抽取这些字段。

**配置形态(拟)**
```yaml
sources:
  - path: /home/neo/Documents/obsidian/investing/wiki/
    domain: stock
  - path: /home/neo/Documents/obsidian/learning/wiki/
    domain: study
# type 自动取 wiki 下一级子目录名:companies / industries / journal / decisions / portfolio / playbooks / events / checklists
```

**分类与元数据**
- `domain`:每个 source 手填(stock / study)。
- `type`:自动取该 source 下**第一层子目录名**;`wiki/` 根下无子目录的文件见"待讨论"。
- `evidence_level` / `freshness`:从 wiki 页头部 bold-key 行抽取(Primary/Secondary/Market view/User view/Unverified;Stable/Time-sensitive/Stale risk)。

**改动点**
- `src/config.py`:`notes_dir` → `sources` 列表(path + domain),逐个校验。
- `src/loader.py`:遍历多个 root;每文件带 `domain`+`type`;file_path 以 `domain/` 前缀保证唯一(接 T5);解析头部 bold-key 元数据成字段,`---` 分隔线之前的元数据块**不进正文向量**。
- `src/sqlite_metadata_store.py` / `src/vector_store.py`:`documents`/`chunks` + Chroma metadata 增加 `domain` / `type` / `evidence_level` / `freshness`。
- `src/search.py` / `src/context_builder.py` / CLI:加 `--domain` / `--type` / `--evidence` 过滤(Chroma `where` + SQL)。
- `src/ingest.py`:增量同步按"全部 sources 并集"对账(删除判断跨源)。

**定位**:domain/type/evidence 全部**自动**从路径与页头得到(免手写,用来"选库/选类/挑证据级")。若将来还要细粒度专题标签,再单列,不在本任务。

**可行性结论**:可行且干净。路径+页头分类同时解决"多 corpus 分离"与"分类过滤",**无需多个 Chroma collection**。

**待讨论**
- `wiki/` 根下 `index.md`、`log.md` 是否索引?(log 是操作日志、index 是 TOC,建议**排除**)
- 是否一并扩展 `cleaner.py` 把 `[[链接]]` 清成纯文字(小卫生,可同任务做)。
- raw / daily-notes 暂不纳入(见"数据源原则");将来纳入再加 `stage` 字段。

---

## T7. 真实 wiki 评测集(已完成 2026-06-20,前置于 T2)

> **完成记录(2026-06-20)**:`eval/queries.wiki.yaml` 共 **20 条**真实问题(stock 9 / study 11,贴合语料 50:62);`config.yaml` 的 `eval_queries` 指向它;`evaluate.py` 补每条 pass/fail 明细;`tests/test_evaluate.py` 做 schema 自检。
> - **按 domain 过滤评测**:真实使用时 domain 是已知的"选库"维度(投资走 stock、学习走 study),故 `evaluate` 对每条问题按其 domain 过滤(从 expected_files 前缀推断),不裸跑全库——这才贴合真实用法。
> - **域内基线命中率 55%(11/20)**,作为 T2 reranker 前后对比的锚。
> - 通过集中在单主题独立页(财报/概念/人物/工具 how-to);**9 条失败 = 同库内的真检索弱点**:精确页淹没在同公司兄弟页、短决策页排不上、synthesis/行业页输给其构成页、entity 输给概念碎片(如 `马斯克` 输给 `技术奇点/AGI`)、个别页域内也排不上(如 `Samba`)。
> - `expected_files` 放宽纪律:仅当被召回页本身是该问题的好答案(年报答营收、Sing-box 页答 sing-box 搭建)才纳入;只答局部的不放宽,保留为真失败。
> - 旧 `eval/queries.yaml`(绑已不存在的 sample 笔记)弃用、留档不动。

**目标**:`eval/queries.yaml` 现绑定玩具样例笔记;切到 wiki corpus 后无法量化检索质量,T2 reranker 的验收("evaluate ≥ T1")失去依据。先建一份真实评测集做锚。

**改动点**
- 新增 `eval/queries.wiki.yaml`:10~20 条真实投研问题,每条标 `expected_files`(期望命中的 wiki 页)+ `expected_keywords`。
- 覆盖不同 `type`(companies / industries / journal …)与不同 `evidence_level`,顺带验证 T6 的过滤。
- `src/evaluate.py` 已通用,指向新评测集即可;必要时补按 domain/type 分组的命中率。

**验收标准**
- [ ] wiki corpus 上 `evaluate` 跑出基线命中率(作为 T2 前后对比的锚)。
- [ ] 评测覆盖至少 3 种 type、2 种 evidence_level。

**待讨论**
- 评测问题手写,还是先从 wiki 页标题/小结自动生成草稿再人工筛?

---

## 决策记录(已定)

- **范围**:本轮已完成 **T0 + T5 + T1 + 前置配置**;后续按下方顺序推进。
- **embedding**:`BAAI/bge-small-zh-v1.5`(速度优先);bge-m3 留作后续备选。
- **笔记路径**:`/home/neo/project/notes/`(外部绝对路径,本轮验证用)。
- **依赖**:已统一 uv、删除 `requirements.txt`(✅ 完成)。
- **任务整合**:原 T4 并入 T6 并重定方向(wiki bold-key 元数据,非 YAML tags);新增 T7(真实评测集)。

**后续执行顺序:T6 → T7 → T2 → T3**(理由链)
- **T6 打地基**:决定索引什么(wiki/)+ domain/type/evidence 元数据 schema;并把数据源从技术笔记切到真正的投研 wiki。
- **T7 评测集**:有真实 corpus 才能建;是 T2 能被量化验证的前提。
- **T2 reranker**:小 embedding 模型下是质量最大杠杆,但须在 T7 评测集上验证才值得做。
- **T3 MCP 最后**:检索/过滤定型后再做出口,工具签名含 `--domain/--type/--evidence/rerank` 透传,不返工。
