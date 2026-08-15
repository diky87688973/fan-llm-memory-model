# fan-llm-memory-model

**让 LLM 拥有"内化记忆"：把个人/会话事实训练进模型参数，回忆是生成，不是检索。**

核心思路：对话经历 → 提纯为结构化记忆（raw + QA）→ 训练进模型权重（LoRA）→ 回忆时模型直接生成，而非外部数据库检索。当前为 **v0.2**，早期版本见 git tag `v0.1`。

## 目录结构

```
├── core/      核心逻辑
│   ├── session_to_memory.py    对话 → 记忆提纯（raw/QA 生成）
│   ├── memory_pipeline_core.py 记忆流水线核心（多轮 history 在进程内）
│   ├── memory_utils.py         通用工具
│   └── memory_session_id.py    会话 ID 生成/校验
├── serve/
│   └── memory_pipeline_server.py  记忆对话 HTTP 服务（SSE 流式）
├── cli/       命令行入口
│   ├── memory_pipeline_cli.py    流水线 REPL（本机直连模式）
│   ├── memory_extract_cli.py     记忆提取 CLI
│   └── memory_pipeline_client.py 连接 Server 的 REPL（IDE 右键运行）
├── tools/     独立工具
│   ├── memory_session_raw_merge.py  原始会话语料合并
│   └── session_corpus_duplicate.py  语料去重
├── web/       chat_ui.html（浏览器对话页）
├── docs/      方案设计文档
└── requirements.txt
```

## 快速开始

依赖 `transformers`、`torch`、`peft` 等（见 `requirements.txt`），需自备 HF 基座模型（如 Qwen 系列）。

```bash
pip install -r requirements.txt

# 记忆提取（终端交互录入对话）
python cli/memory_extract_cli.py

# 启动记忆对话服务（浏览器访问 http://127.0.0.1:端口）
python serve/memory_pipeline_server.py
```

跨目录引用已内置 sys.path 引导（`core`/`serve`/`cli`），IDE 右键运行任意入口脚本即可。

## 版本历史

| Tag | 内容 |
|---|---|
| `v0.1` | LoRA 训练（train_memory/replay）、记忆提取、API 服务、用户画像 |
| `v0.2` | 会话 ID 管理、原始语料合并/去重、记忆提取与对话管线重构 |
