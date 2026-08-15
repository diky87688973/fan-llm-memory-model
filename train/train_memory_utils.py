# -*- coding: utf-8 -*-
"""训练专用工具（自 1.0 迁移，对齐 2.0 管线）。

2.0 的 ``core/memory_utils`` 聚焦推理/提取；训练相关符号（qa/raw 训练消息、
LoRA 输出目录、保存、token 统计、本地 HF 开关）集中在本模块。
公共符号（SYSTEM_RAW/SYSTEM_EXTRACT/load_tokenizer 等）复用 ``core.memory_utils``。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

for _sub in ("core", "train", "serve"):
    _p = str(Path(__file__).resolve().parent.parent / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memory_utils import (  # noinspection PyUnresolvedReferences
    SYSTEM_RAW,
    SYSTEM_EXTRACT,
    user_content_with_memory_extract_prefix,
    get_memory_lora_dirname,
)

# 本地 HF 训练开关（2.0 core.memory_utils 已移除该全局，保留在训练侧；
# 本地 tokenizer 仍可通过 MEMORY_TOKENIZER_PATH / load_tokenizer_for_inference 使用）
TRAIN_USE_LOCAL_HF = False

_ISO_DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_iso_date_dir_name(name: str) -> bool:
    """判断目录名是否为 ``YYYY-MM-DD`` 日期目录（session_raw 下按日归档）。"""
    return bool(_ISO_DATE_DIR.match((name or "").strip()))


def sanitize_path_user_id(raw: str, *, max_len: int = 80) -> str:
    """目录名用的 user_id：仅字母数字下划线连字符；空则 fallback。"""
    s = (raw or "").strip()
    t = re.sub(r"[^0-9A-Za-z_-]", "", s)
    if not t:
        return "user"
    return t[:max_len]


def qa_training_messages(
    q: str,
    a: str,
    *,
    memory_time_iso: str | None = None,
) -> list[dict]:
    """与 format_qa_sample 相同对话结构，供 TRL messages 列使用。"""
    q = (q or "").strip()
    mt = (memory_time_iso or "").strip()
    if mt:
        q = f"【记忆时刻】{mt}\n{q}"
    u = user_content_with_memory_extract_prefix(q)
    return [
        {"role": "system", "content": SYSTEM_EXTRACT},
        {"role": "user", "content": u},
        {"role": "assistant", "content": a},
    ]


def raw_training_messages(raw: str) -> list[dict]:
    """与 format_raw_sample 相同对话结构。"""
    return [
        {"role": "system", "content": SYSTEM_RAW},
        {"role": "user", "content": raw},
        {"role": "assistant", "content": "好的，我记住了。"},
    ]


def apply_memory_train_template(tokenizer, messages: list[dict]) -> str:
    """将训练用 messages 展成与旧版单串 text 训练相同的字符串（估算 token 或与探针对齐）。"""
    if getattr(tokenizer, "chat_template", None) is None:
        system = messages[0]["content"]
        user = messages[1]["content"]
        asst = messages[2]["content"]
        return f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n{asst}"
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def memory_training_messages_token_len(tokenizer, messages: list[dict]) -> int:
    """与历史回放脚本中按串计 token 的方式一致。"""
    text = apply_memory_train_template(tokenizer, messages)
    return len(tokenizer.encode(text, add_special_tokens=False))


def per_user_lora_output_dir(script_dir: Path, user_id: str) -> Path:
    """``outputs/<sanitize(user_id)>/<MEMORY_LORA_DIRNAME 或默认>/``，与 train_memory 约定一致。"""
    uid = sanitize_path_user_id(user_id)
    return script_dir / "outputs" / uid / get_memory_lora_dirname()


def save_trainer_model_and_tokenizer(trainer, output_dir: str, tokenizer) -> None:
    """训练结束写入目录。保存期间临时去掉 HF_ENDPOINT，避免 Trainer / tokenizer.save_pretrained
    仍向失效镜像或 Hub 发起请求（与 list_repo_templates 同类超时）。"""
    ep = os.environ.pop("HF_ENDPOINT", None)
    try:
        trainer.save_model(str(output_dir))
        if hasattr(tokenizer, "name_or_path"):
            try:
                tokenizer.name_or_path = str(Path(output_dir).resolve())
            except Exception:
                pass
        tokenizer.save_pretrained(str(output_dir), push_to_hub=False)
    finally:
        if ep is not None:
            os.environ["HF_ENDPOINT"] = ep
    print(
        "[train_memory_utils] trainer.save_model 与 tokenizer.save_pretrained 已完成"
        + ("（已恢复 HF_ENDPOINT）。" if ep else "。"),
        flush=True,
    )


def apply_peft_save_pretrained_embedding_layers_explicit(model) -> None:
    """LoRA 的 target_modules 若包含 embed_tokens / lm_head 等嵌入相关模块，PEFT 在
    save_pretrained(..., save_embedding_layers="auto")（Trainer 默认）时会自动改为保存嵌入侧权重；
    保存前显式传入 save_embedding_layers=True 与自动行为一致，不再依赖 auto 推断。"""
    try:
        from peft import PeftModel
    except ImportError:
        return
    if not isinstance(model, PeftModel):
        return
    _orig = model.save_pretrained

    def _wrapped(*args, **kwargs):
        kwargs.setdefault("save_embedding_layers", True)
        return _orig(*args, **kwargs)

    model.save_pretrained = _wrapped  # type: ignore[method-assign]
