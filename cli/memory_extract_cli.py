# -*- coding: utf-8 -*-
"""
记忆抽取：右键运行，终端输入问题；送入模型前 user 侧会自动加 `[记忆提取]` 前缀（与 1.0 train_memory 中 QA 训练格式一致），新 API 实现时请保持同一格式。默认与调试超参见文首 **HF 与同段内置**；需要完整提示词与原始输出时将 ``BUILTIN_VERBOSE_DEBUG`` 设为 True。

每轮对话会追加写入 `session_staging/cli_{时间}_{sessionId}.txt`（每次启动 CLI 生成一个 sessionId），便于后续整理为 `session_raw/*.raw.txt`、抽取 `*.qa.jsonl` 再训练。
每轮以 `[轮次时间: YYYY-MM-DD HH:MM:SS]` 记录**用户提交该轮问题时的本地绝对时间**（日期+时分秒，在模型推理前打点），供提纯与 raw「记忆时间」锚点使用。

退出：输入 quit 或 exit（与 GPT-4.1 生成脚本一致）。
"""

from __future__ import annotations

import sys
from pathlib import Path
for _sub in ("core", "train", "serve"):
    _p = str(Path(__file__).resolve().parent.parent / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# HF：必须在 import memory_utils 之前（与 1.0 memory_api_server / train_memory 一致）
# ---------------------------------------------------------------------------
BUILTIN_HF_MIRROR = True
BUILTIN_HF_ENDPOINT = ""
BUILTIN_HF_DOWNLOAD_TIMEOUT_SEC = "1800"
BUILTIN_HF_DISABLE_XET = True
BUILTIN_HF_DISABLE_SYMLINKS_WARNING = True

# ---------------------------------------------------------------------------
# 路径与模型内置（与上方 HF 同区；不 import memory_utils）
# 未设 ``MEMORY_LORA_DIRNAME`` 时与 2.0 memory_utils 默认「memory_lora_v4」一致
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
BUILTIN_LORA_DIRNAME = (os.environ.get("MEMORY_LORA_DIRNAME", "").strip() or "memory_lora_v4")
BUILTIN_HUB_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BUILTIN_BASE_MODEL = BUILTIN_HUB_BASE_MODEL
BUILTIN_ADAPTER_DIR = str(_script_dir / "outputs" / BUILTIN_LORA_DIRNAME)
BUILTIN_NO_4BIT = False
BUILTIN_MAX_NEW_TOKENS = 512
BUILTIN_VERBOSE_DEBUG = False
BUILTIN_SAVE_SESSION_STAGING = True
BUILTIN_SESSION_STAGING_DIR = str(_script_dir / "session_staging")
# ---------------------------------------------------------------------------


def _apply_hf_runtime_early() -> None:
    ep = BUILTIN_HF_ENDPOINT.strip().rstrip("/")
    if ep:
        os.environ["HF_ENDPOINT"] = ep
    elif BUILTIN_HF_MIRROR:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", BUILTIN_HF_DOWNLOAD_TIMEOUT_SEC)
    if BUILTIN_HF_DISABLE_XET:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if BUILTIN_HF_DISABLE_SYMLINKS_WARNING:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


_apply_hf_runtime_early()

import warnings

import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

from peft import PeftModel

from memory_session_id import next_session_id
from memory_utils import (
    build_extract_messages,
    clean_memory_generation_output,
    generate_extract_completion,
    load_base_model_causal_lm,
    load_tokenizer_for_inference,
    memory_extract_gate_by_query,
    sanitize_memory_extract_output,
)

_model = None
_tokenizer = None
_device = None


def _resolve_local_tokenizer_dir() -> str | None:
    """2.0 不内置 hf_tokenizer 目录；需要离线 tokenizer 时设置环境变量 MEMORY_TOKENIZER_PATH。"""
    env = os.environ.get("MEMORY_TOKENIZER_PATH", "").strip()
    return env or None


def _load_model(base_model_id: str, adapter_dir: Path, use_4bit: bool) -> None:
    global _model, _tokenizer, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    # 必须用完整基座 tokenizer；outputs 下 LoRA 目录内 save 的 tokenizer 若损坏会导致中文 0 token、input_ids≈4。
    _tokenizer = load_tokenizer_for_inference(
        base_model_id,
        local_tokenizer_dir=_resolve_local_tokenizer_dir(),
    )
    base = load_base_model_causal_lm(base_model_id, use_4bit=use_4bit)
    _model = PeftModel.from_pretrained(base, str(adapter_dir))
    _model.eval()
    if _device == "cpu" and not use_4bit:
        _model = _model.to(_device)


def _build_prompt_and_inputs(query: str) -> tuple[str, dict]:
    if _model is None or _tokenizer is None:
        raise RuntimeError("模型未加载")
    messages = build_extract_messages(query.strip())
    if getattr(_tokenizer, "chat_template", None) is None:
        prompt = f"[SYSTEM]\n{messages[0]['content']}\n[USER]\n{messages[1]['content']}\n[ASSISTANT]\n"
    else:
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # apply_chat_template 已含 BOS/角色标记；再 add_special_tokens=True 会重复或编码异常（input_ids 极短）
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )
    if _device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    return prompt, inputs


def _append_session_staging_turn(
    log_path: Path,
    user_query: str,
    assistant_cleaned: str,
    *,
    turn_time: datetime,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = turn_time.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    block = f"[轮次时间: {ts}]\nUser: {user_query}\nAssistant: {assistant_cleaned}\n\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(block)


def run_memory_extract_once(query: str) -> tuple[str, str, str, dict]:
    """
    执行一轮记忆提取（调用方须已 _load_model）。
    返回 (清洗后文本, 模型原始输出, prompt 字符串, tokenizer inputs)。
    """
    prompt, inputs = _build_prompt_and_inputs(query)
    raw = generate_extract_completion(_model, _tokenizer, inputs, BUILTIN_MAX_NEW_TOKENS)
    cleaned = sanitize_memory_extract_output(clean_memory_generation_output(raw))
    cleaned = memory_extract_gate_by_query(cleaned, query.strip())
    return cleaned, raw, prompt, inputs


def print_extract_round(query: str) -> str:
    """生成回复；默认只打印清洗后文本。返回清洗后的助手文本（供写入 session_staging）。"""
    cleaned, raw, prompt, inputs = run_memory_extract_once(query)

    if BUILTIN_VERBOSE_DEBUG:
        n_tok = int(inputs["input_ids"].shape[1])
        print("\n" + "=" * 72, flush=True)
        print("【提示词】送入模型的完整字符串（含 chat 模板）", flush=True)
        print(f"（字符数={len(prompt)}，input_ids 长度={n_tok}）", flush=True)
        print("-" * 72, flush=True)
        print(prompt, flush=True)
        print("=" * 72, flush=True)
        print("\n【模型原始输出】", flush=True)
        print("-" * 72, flush=True)
        print(raw, flush=True)
        print("-" * 72, flush=True)
        print("\n【清洗后】", flush=True)
        print(cleaned, flush=True)
    else:
        print(cleaned, flush=True)

    return cleaned


def main() -> None:
    adapter_dir = Path(BUILTIN_ADAPTER_DIR)
    if not adapter_dir.is_dir():
        print(f"错误：适配器目录不存在: {adapter_dir}", flush=True)
        print("请将 BUILTIN_ADAPTER_DIR 指向已训练的 LoRA 目录（训练脚本在 记忆模型/1.0/train_memory.py）。", flush=True)
        sys.exit(1)

    print("正在加载基座 + LoRA 适配器（首次可能较慢）…", flush=True)
    _load_model(BUILTIN_BASE_MODEL, adapter_dir, use_4bit=not BUILTIN_NO_4BIT)
    print("记忆抽取已就绪。输入 quit 或 exit 退出。", flush=True)

    staging_log: Path | None = None
    if BUILTIN_SAVE_SESSION_STAGING:
        session_id = next_session_id()
        staging_log = Path(BUILTIN_SESSION_STAGING_DIR) / f"cli_{session_id}.txt"
        staging_log.parent.mkdir(parents=True, exist_ok=True)
        print(f"sessionId={session_id}  会话记录: {staging_log.name}", flush=True)
        if BUILTIN_VERBOSE_DEBUG:
            print(f"完整路径: {staging_log}", flush=True)

    while True:
        try:
            user = input("\n请输入: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。", flush=True)
            break
        if user.lower() in ("quit", "exit"):
            print("再见。", flush=True)
            break
        if not user:
            print("输入不能为空。", flush=True)
            continue
        try:
            turn_time = datetime.now().replace(microsecond=0)
            cleaned = print_extract_round(user)
            if staging_log is not None:
                _append_session_staging_turn(staging_log, user, cleaned, turn_time=turn_time)
        except Exception as e:
            print(f"生成出错: {e}", flush=True)


if __name__ == "__main__":
    main()
