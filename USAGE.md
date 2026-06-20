# 使用说明(当前环境)

> 本机(Kubuntu / 国内网络 / AMD)实测可用的命令。与 README 里旧的 `uv run` 写法不同,**以本文为准**。

## ⚠️ 两条铁律
1. **不要用 `uv run` / `uv sync`**:它们会按锁从 pytorch 源重装 torch,国内必断、整个环境回滚。一律用 **`.venv/bin/python`**。
2. **涉及向量的命令都加 `HF_HUB_OFFLINE=1`**:embedding 模型是本地副本(`models/bge-small-zh-v1.5`),离线标志避免它去连 HuggingFace 失败回退到哈希兜底。

## 当前配置(`config.yaml`)
| 项 | 值 |
|---|---|
| 知识库目录 `notes_dir` | `/home/neo/project/notes/`(外部) |
| 模型 `model_name` | `models/bge-small-zh-v1.5`(本地,512 维) |
| query 指令前缀 | `为这个句子生成表示以用于检索相关文章：` |
| 向量库 | `data/chroma/` |
| 元数据 | `data/metadata.sqlite` |

## 建议别名(省去每次敲长串;fish,当前会话有效)
```fish
alias rag 'env HF_HUB_OFFLINE=1 /home/neo/project/MarkdownRAG/.venv/bin/python -m'
```
设完即可把下文的 `env HF_HUB_OFFLINE=1 .venv/bin/python -m` 简写为 `rag`。

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
```

### 3. 生成 LLM 上下文(喂给 Codex/Claude)
```fish
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.context_builder "牧原股份 护城河"   # 写到 context.md
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.context_builder "贵州茅台 i茅台" | codex exec "基于资料做价值投资分析"
```

### 4. 自检
```fish
env HF_HUB_OFFLINE=1 .venv/bin/python -m src.evaluate   # 命中率
env HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q       # 单元测试
```

## 装新依赖
```fish
uv pip install <包名>     # ✅ 走清华源(全局 ~/.config/uv/uv.toml 已配)
# uv sync / uv run        # ❌ 不要用,会因 torch 源重装而崩
```

## 注意点
- `evaluate` 的题(`eval/queries.yaml`)绑定的是**样例投资笔记**;当前库是技术笔记,故 evaluate 不适配(预期;待 T7 建真实评测集)。
- 换知识库路径 / 换模型后,记得 `--reset` 重建一次。
- 详细方案与待办见 `RAG增强方案(待审阅).md`。

## 环境怎么搭起来的(备查)
- torch CPU 轮子:从 `https://mirrors.aliyun.com/pytorch-wheels/cpu/` 用 curl 下载,`uv pip install` 本地轮子(SJTU 镜像会跳回官方源、无效)。
- 其余依赖:`uv pip install`(清华源)。
- embedding 模型:curl 从 `hf-mirror.com` 全量下到 `models/bge-small-zh-v1.5/`,config 指向本地路径。
