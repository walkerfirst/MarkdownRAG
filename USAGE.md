# 使用说明(当前环境)

> 本机(Kubuntu / 国内网络 / AMD)实测可用的命令。与 README 里旧的 `uv run` 写法不同,**以本文为准**。

## ⚠️ 两条铁律
1. **不要用 `uv run` / `uv sync`**:它们会按锁从 pytorch 源重装 torch,国内必断、整个环境回滚。一律用 **`.venv/bin/python`**。
2. **涉及向量的命令都加 `HF_HUB_OFFLINE=1`**:embedding 模型是本地副本(`models/bge-small-zh-v1.5`),离线标志避免它去连 HuggingFace 失败回退到哈希兜底。

## 当前配置(`config.yaml`)
| 项 | 值 |
|---|---|
| 知识库 `sources`(多源) | `notes/investment/wiki/`→`stock`、`notes/learning/wiki/`→`study`(均外部) |
| 模型 `model_name` | `models/bge-small-zh-v1.5`(本地,512 维) |
| query 指令前缀 | `为这个句子生成表示以用于检索相关文章：` |
| 向量库 | `data/chroma/` |
| 元数据 | `data/metadata.sqlite` |
| 评测集 `eval_queries` | `eval/queries.wiki.yaml`(真实 wiki,域内基线 55%) |
| reranker | 默认关;`enabled: true` 开则全文重排 55%→80%,但单 query CPU ~28s |

## 别名

### 临时泛用别名(fish,仅当前会话;需在项目根)
```fish
alias rag 'env HF_HUB_OFFLINE=1 /home/neo/project/MarkdownRAG/.venv/bin/python -m'
```
设完即可把下文的 `env HF_HUB_OFFLINE=1 .venv/bin/python -m` 简写为 `rag`(如 `rag src.search "..."`)。**仍须在项目根目录执行**(否则找不到 `src` 包和 `config.yaml`)。

### 持久化检索别名(fish,任意目录可用)
检索是最高频操作,单独做一个全局别名。存为 `~/.config/fish/functions/ragsearch.fish`(本机 shell 个人配置,不在仓库):
```fish
function ragsearch --description 'MarkdownRAG 检索:任意目录可用,透传查询词与参数'
    # env -C 在项目目录里执行(src 包与 config.yaml 都是相对 CWD),但不改当前 shell 的 cwd
    env -C /home/neo/project/MarkdownRAG HF_HUB_OFFLINE=1 \
        /home/neo/project/MarkdownRAG/.venv/bin/python -m src.search $argv
end
```
fish 自动加载,存好即生效,任意目录直接:
```fish
ragsearch "牧原 屠宰 渠道"
ragsearch "贵州茅台 经销商 i茅台" --top-k 8
```

## 日常操作(均在项目根目录)

### 1. 建库 / 同步(笔记增删改后)
```fish
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.ingest          # 增量同步
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.ingest --reset  # 清空重建(换模型/换路径后)
```
同步规则:新文件→新增;未变→跳过;改了→重建该文件块;删了→同步删缓存。

### 2. 检索
```fish
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.search "贵州茅台 经销商 i茅台"
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.search "牧原 屠宰 渠道" --top-k 8
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.search "价值投资 安全边际" --domain study  # 限定库
```
默认走「向量+关键词」混合检索,秒回。CLI 默认全库检索;加 `--domain stock/study`(或 `--type`、`--evidence`)按库/类型过滤,贴合「投资走 stock、学习走 study」的真实用法。需要更高准确率时,把 `config.yaml` 的 `reranker.enabled` 改 `true` 开交叉编码器重排(域内命中 55%→80%),代价是单 query CPU **~28s**;查完记得改回 `false`。

### 3. 生成 LLM 上下文(喂给 Codex/Claude)
```fish
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.context_builder "牧原股份 护城河"   # 写到 context.md
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.context_builder "贵州茅台 i茅台" | codex exec "基于资料做价值投资分析"
```

### 4. 自检
```fish
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.evaluate   # 命中率(真实 wiki 评测集;域内基线 55%,开 reranker 80%)
env HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q       # 单元测试
```

## 装新依赖
```fish
uv pip install <包名>     # ✅ 走清华源(全局 ~/.config/uv/uv.toml 已配)
# uv sync / uv run        # ❌ 不要用,会因 torch 源重装而崩
```

## 注意点
- `evaluate` 已切到真实 wiki 评测集(`eval/queries.wiki.yaml`,20 题、stock 9/study 11);按问题 `domain` 过滤后域内基线命中 55%,开 reranker 80%。
- 选库:CLI 默认全库;日常按用途加 `--domain stock`(投资)或 `--domain study`(学习)避免跨域混排。`evaluate` 则自动按每题 `domain` 过滤。
- 换知识库路径 / 换模型后,记得 `--reset` 重建一次。
- 详细方案与待办见 `RAG增强方案(待审阅).md`。

## 环境怎么搭起来的(备查)
- torch CPU 轮子:从 `https://mirrors.aliyun.com/pytorch-wheels/cpu/` 用 curl 下载,`uv pip install` 本地轮子(SJTU 镜像会跳回官方源、无效)。
- 其余依赖:`uv pip install`(清华源)。
- embedding 模型:curl 从 `hf-mirror.com` 全量下到 `models/bge-small-zh-v1.5/`,config 指向本地路径。
