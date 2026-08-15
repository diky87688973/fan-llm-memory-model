# fan-llm-memory-model

**让 LLM 拥有"内化记忆"：把个人/会话事实训练进模型参数（LoRA），回忆是生成，不是检索。**

本项目实现了一套完整的「记忆模型」：对话经历经过提纯（蒸馏）成为训练数据，用 QLoRA 微调写入模型权重；回答时模型从参数中"想起"相关记忆。与 RAG（向量库检索 + 拼上下文）是两条不同的技术路线——本方案强调**记忆即模型本身**，回忆靠条件生成，而非外部检索。

## 核心思想

- **记忆 = 模型参数**：经历通过 `session_to_memory` 提纯成 `raw`（带时间戳的事实行）/ `qa`（问答对）/ `画像`（用户/角色稳定画像），再经 `train_memory` QLoRA 训练进 LoRA 适配器。
- **回忆 = 生成**：`memory_api_server` 加载「基座 + LoRA」，对自然语言 query 直接生成记忆相关陈述；系统提示 `SYSTEM_EXTRACT` 约束模型"只答明确记录过的内容，未记录答「未记录」"。
- **巩固 = 海马体回放**：`train_memory_replay` 模拟人脑睡眠巩固——当日记忆全量 + 历史池 5% 抽样，增量训练，对抗灾难性遗忘。
- **时间轴在数据里**：`raw.txt` 每行事实带 `[YYYY-MM-DD HH:MM:SS]` 前缀，QA 带 `t` 字段，支持"那天聊过什么"类时间检索。
- **防幻觉在编码时**：提纯阶段二阶段事实核对（摘录事实 + 原文支持 → 一致性过滤），而非回忆时补救。

## 架构

```
在线：用户输入 → 路由 LLM（判断 need_memory + 生成 N 条问句）→ memory_api_server（POST /memory/extract × N）→ 终答 LLM → 回答
离线：对话 → session_to_memory（提纯）→ session/raw·qa·memory.json → train_memory / train_memory_replay → outputs/LoRA 适配器 → 加载进服务
```

- **1.0**：完整管线——提纯 + 训练 + 回放 + 记忆服务 + 编排；含 `train_memory.py`、`train_memory_replay.py`、`download_hf_tokenizer.py`。
- **2.0**：编排与服务演进——路由（DeepSeek Tool Calls / Ollama JSON）、记忆服务、终答（SSE 流式）、浏览器单页 `chat_ui.html`（对话/今日会话/历史会话/记忆库侧栏）；训练与 tokenizer 离线准备见 1.0。

## 快速开始

```bash
# 1. 离线准备：tokenizer（1.0 目录）
python 1.0/download_hf_tokenizer.py          # 或设置 MEMORY_TOKENIZER_PATH

# 2. 采集会话（放到 session_staging/ 下 cli_*.txt）→ 提纯
python 1.0/session_to_memory.py              # 生成 session/.../*.memory.json · *.raw.txt · *.qa.jsonl

# 3. 训练（需要 CUDA）
python 1.0/train_memory.py                    # 首次全量 QLoRA
python 1.0/train_memory_replay.py             # 每日回放巩固（当日全量 + 历史 5%）

# 4. 启动记忆服务（加载基座 + LoRA）
python 1.0/memory_api_server.py --adapter_dir outputs/memory_lora_v2

# 5. 对话编排（本机 REPL 或浏览器）
python 1.0/memory_pipeline_cli.py             # REPL
python 1.0/memory_pipeline_server.py          # 浏览器 http://localhost:8890（需 1.0/2.0 各自 chat_ui.html）
```

## 目录结构

```
1.0/
  session_to_memory.py        会话提纯（判断入库 → 二阶段事实核对 → JSON 记忆 → raw/qa/画像）
  train_memory.py             QLoRA 全量训练（Qwen2.5-7B 基座，早停 + 随机 QA 记忆探针）
  train_memory_replay.py      海马体回放：当日全量 + 历史池 5% 增量训练（从已有 LoRA 继续）
  memory_api_server.py        记忆抽取服务（FastAPI，单进程持有基座 + LoRA）
  memory_pipeline_core.py     编排核心：路由（JSON 强制）→ N 次 extract → 终答
  memory_pipeline_server.py   流水线 HTTP 服务（会话/侧栏/SSE 流式，默认 :: 双栈 8890）
  memory_pipeline_cli.py      本机 REPL 编排
  memory_utils.py             公共库：模板、tokenizer、生成清洗、SYSTEM_EXTRACT
  记忆模型方案设计文档.md      完整技术方案（架构/选型/提示词/踩坑/多用户/与 RAG 对照）
2.0/
  memory_pipeline_*.py        编排与服务（路由/终答/SSE/会话管理）
  memory_session_raw_merge.py session_raw 按 stem 合并训练语料
  memory_session_id.py        会话 ID 生成
  chat_ui.html                浏览器单页
  记忆模型方案设计文档.md      2.0 版方案（不含训练脚本说明见 1.0）
记忆模型里程碑计划.md          v1-v4 路线图（主线 + 探索里程碑）
```

## 注意事项

- **训练/推理一致性**：`[记忆提取]` 前缀、`SYSTEM_EXTRACT`、`MEMORY_LORA_DIRNAME`、`adapter_config.json` 的 base model 必须一致，否则记忆"胡说"。
- **API Key**：`session_to_memory.py` / `memory_pipeline_core.py` 顶部内置了 DeepSeek Key 默认值，**生产环境务必改为环境变量并轮换**，切勿提交真实密钥。
- **GPU**：7B 基座 + QLoRA 需 16GB 级显卡；OOM 时降 `BUILTIN_MAX_SEQ_LENGTH` / `BUILTIN_LORA_R`。
- **本仓库不含**：`outputs/`（LoRA 产物）、`session/` `session_raw/`（记忆数据）、`hf_tokenizer/`（tokenizer 缓存）——请按 `.gitignore` 排除，训练产物自行保管。
