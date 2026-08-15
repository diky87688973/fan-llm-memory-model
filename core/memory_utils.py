# -*- coding: utf-8 -*-
"""本地 HF 基座 + LoRA 的加载与对话模板（训练与 API 共用）。"""

from __future__ import annotations

import logging
import os
import re
import warnings
from difflib import SequenceMatcher
from pathlib import Path

# 须在 import transformers 之前：降低「generation flags not valid」等提示级别
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerFast

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

SYSTEM_RAW = "请完整记忆以下会话内容，后续问答将依赖这些材料。"
# 记忆抽取（CLI / memory_api_server）与 QA 训练、训练探针共用同一段，避免 system 分布不一致。
SYSTEM_EXTRACT = (
    "注意：你是一个记忆模型，只能输出用户对话历史中明确写过的内容。如果历史中未提及，请回答「未记录」。"
    "你已内化用户会话记忆。请用简短自然语言直接回答当前问题（不要输出思考过程、不要重复系统提示）；"
    "若没有与问题相关的可引用记忆，请回答「未记录」。"
    "请用中文作答，不要堆砌英文单词、无关专名或代码片段（除非问题明确要求）。"
    "禁止输出与问题无关的元话语（脱离用户问题的自述、评价或规则说明）；"
    "若问的是用户偏好或事实，只答记忆中与问题直接相关的具体内容，勿用空话敷衍。"
    "严格按问题所问的**维度**作答：回答须与该维度一致，**禁止**用其它维度的事实偷换；除非问题明确问及的正是该维度。"
    "禁止只输出残缺占位片段；若确实无对应记忆，请回答「未记录」。"
    "若没有任何可引用的记忆，请回答「未记录」；禁止用问句复述检索问题（勿以问句冒充答案）。"
)

# 训练产物默认子目录名（相对本脚本所在目录下的 `outputs/`，即 `记忆模型/1.0/outputs/`），与 train_memory / CLI / API 一致。
# 环境变量 MEMORY_LORA_DIRNAME 可覆盖（重训换目录、或指回旧适配器如 memory_lora）。
DEFAULT_MEMORY_LORA_DIRNAME = "memory_lora_v2"
# 记忆抽取单次续写 max_new_tokens 硬上限（2 的幂；与调用方传入值取 min，例如 API 传 512 时实际为 256）
MEMORY_EXTRACT_MAX_NEW_TOKENS = 256


def get_memory_lora_dirname() -> str:
    v = os.environ.get("MEMORY_LORA_DIRNAME", "").strip()
    return v if v else DEFAULT_MEMORY_LORA_DIRNAME


def per_user_lora_output_dir(script_dir: Path, user_id: str) -> Path:
    """``outputs/<sanitize(user_id)>/<MEMORY_LORA_DIRNAME 或默认>/``，与 train_memory / memory_api_server 约定一致。"""
    from memory_user_paths import sanitize_path_user_id

    uid = sanitize_path_user_id(user_id)
    return script_dir / "outputs" / uid / get_memory_lora_dirname()

# 记忆提取：user 固定以此前缀开头，训练与推理一致，便于与「非记忆」对话区分。
MEMORY_EXTRACT_USER_PREFIX = "[记忆提取]\n"


def user_content_with_memory_extract_prefix(user_text: str) -> str:
    """训练/抽取共用：为 user 内容加上 [记忆提取] 前缀；已带此前缀则不重复添加。"""
    t = (user_text or "").strip()
    if t.startswith("[记忆提取]"):
        return t
    return f"{MEMORY_EXTRACT_USER_PREFIX}{t}" if t else MEMORY_EXTRACT_USER_PREFIX.rstrip()


def build_bits_and_bytes_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _resolve_memory_tokenizer_local_dir() -> str:
    """与 memory_extract_cli / memory_api_server 一致：环境变量 MEMORY_TOKENIZER_PATH 或同目录下 hf_tokenizer。"""
    env = os.environ.get("MEMORY_TOKENIZER_PATH", "").strip()
    if env:
        return env
    return str(Path(__file__).resolve().parent / "hf_tokenizer")


def qwen_local_tokenizer_dir() -> str:
    """Qwen 基座 tokenizer 持久化目录（与 hf_tokenizer 分离，避免误放 Llama/DeepSeek 词表时被优先读到）。"""
    return str(Path(__file__).resolve().parent / "qwen_tokenizer")


def _ordered_local_tokenizer_dirs(model_id: str) -> list[str]:
    """本地尝试顺序：显式 MEMORY_TOKENIZER_PATH →（Qwen 时）qwen_tokenizer → hf_tokenizer。"""
    out: list[str] = []
    env = os.environ.get("MEMORY_TOKENIZER_PATH", "").strip()
    if env:
        out.append(env)
    if "qwen" in (model_id or "").lower():
        qd = qwen_local_tokenizer_dir()
        if qd not in out:
            out.append(qd)
    hf_tok = str(Path(__file__).resolve().parent / "hf_tokenizer")
    if hf_tok not in out:
        out.append(hf_tok)
    return out


def _save_tokenizer_pretrained_clear_hf_ep(tok, directory: str) -> None:
    """写入 tokenizer 目录；保存期间临时去掉 HF_ENDPOINT，减少保存阶段仍请求镜像。"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    ep = os.environ.pop("HF_ENDPOINT", None)
    try:
        tok.save_pretrained(directory, push_to_hub=False)
    finally:
        if ep is not None:
            os.environ["HF_ENDPOINT"] = ep


def _cache_qwen_tokenizer_after_hub_load(model_id: str, tok) -> None:
    """从 Hub 成功得到 Qwen tokenizer 后写入 qwen_tokenizer，失败仅打印不影响返回。"""
    if "qwen" not in (model_id or "").lower():
        return
    try:
        _save_tokenizer_pretrained_clear_hf_ep(tok, qwen_local_tokenizer_dir())
        print(
            f"[memory_utils] 已将 Qwen tokenizer 缓存到 {qwen_local_tokenizer_dir()!r}，下次优先本地。",
            flush=True,
        )
    except BaseException as e:
        print(f"[memory_utils] 写入 qwen_tokenizer 失败（不影响本次加载）: {e!r}", flush=True)


def memory_local_base_model_dir() -> str:
    """训练用本地基座目录：环境变量 MEMORY_BASE_MODEL_PATH，否则为「本文件所在目录/hf_model」。"""
    env = os.environ.get("MEMORY_BASE_MODEL_PATH", "").strip()
    if env:
        return env
    return str(Path(__file__).resolve().parent / "hf_model")


# train_memory / train_memory_replay 在 import 后赋值：True 表示仅从本地 hf_tokenizer + 本地基座训练。
TRAIN_USE_LOCAL_HF: bool = False


def apply_peft_save_pretrained_embedding_layers_explicit(model: torch.nn.Module) -> None:
    """
    LoRA 的 ``target_modules`` 若包含 ``embed_tokens`` / ``lm_head`` 等嵌入相关模块，PEFT 在
    ``save_pretrained(..., save_embedding_layers=\"auto\")``（Trainer 默认如此）时会自动改为保存嵌入侧权重，
    并打出 ``Setting save_embedding_layers to True...`` 的 UserWarning。

    在保存前显式传入 ``save_embedding_layers=True`` 与上述自动行为一致，且符合 PEFT 文档用法，不再依赖 auto 推断。
    **与训练数据中的 input_ids / assistant_masks 无关**（该告警来自保存阶段，非 token 掩码）。
    """
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


def save_trainer_model_and_tokenizer(trainer, output_dir: str, tokenizer) -> None:
    """
    训练结束写入目录。保存期间临时去掉 HF_ENDPOINT，避免 Trainer / tokenizer.save_pretrained
    仍向失效镜像或 Hub 发起请求（与 list_repo_templates 同类超时）。
    """
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
        "[memory_utils] trainer.save_model 与 tokenizer.save_pretrained 已完成"
        + ("（已恢复 HF_ENDPOINT）。" if ep else "。"),
        flush=True,
    )


def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    ep = os.environ.get("HF_ENDPOINT", "").strip()
    print(
        f"[memory_utils] HF_ENDPOINT={ep or '(未设置，使用官方 huggingface.co)'}",
        flush=True,
    )
    for local_dir in _ordered_local_tokenizer_dirs(model_id):
        tok_local = _try_local_full_tokenizer(model_id, local_dir, trust_remote_code)
        if tok_local is not None:
            print(
                "[memory_utils] 已从本地目录加载 tokenizer（无需 Hub list_repo_templates）。",
                flush=True,
            )
            return tok_local
    if TRAIN_USE_LOCAL_HF:
        qhint = f"  或准备 Qwen 专用目录：{qwen_local_tokenizer_dir()!r}\n" if "qwen" in (model_id or "").lower() else ""
        raise RuntimeError(
            "当前为本地训练模式（BUILTIN_TRAIN_USE_LOCAL_HF=True）：未找到可用 tokenizer。\n"
            + qhint
            + "  可设置 MEMORY_TOKENIZER_PATH，或使用默认 hf_tokenizer / qwen_tokenizer（见 _ordered_local_tokenizer_dirs）。"
        )
    print(
        f"[memory_utils] AutoTokenizer.from_pretrained({model_id!r}) 开始 …\n"
        f"  （首次运行需联网；成功后 Qwen 将写入 {qwen_local_tokenizer_dir()!r} 以便下次离线。）",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    print("[memory_utils] AutoTokenizer 加载完成。", flush=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    if "qwen" in (model_id or "").lower():
        _cache_qwen_tokenizer_after_hub_load(model_id, tok)
    return tok


def ensure_chat_template_for_trl_assistant_masks(tokenizer, model_id: str = "") -> bool:
    """
    TRL ``SFTTrainer(assistant_only_loss=True)`` 需要 ``apply_chat_template(..., return_assistant_tokens_mask=True)``
    能标出 assistant 段；模板须含 ``{% generation %}`` / ``{% endgeneration %}``。
    官方 Qwen2  tokenizer 常缺该关键字，会导致 ``assistant_masks`` 全 0 并在 tokenize 阶段报错。
    若检测到为 Qwen 且无 generation，则替换为与 ``qa_training_messages`` / ``raw_training_messages``
    相同结构（system / user / assistant）的兼容模板。
    """
    ct = getattr(tokenizer, "chat_template", None) or ""
    if re.search(r"\{\%-?\s*generation\s*-?\%\}", ct):
        return True
    if "qwen" not in (model_id or "").lower():
        return False
    im_end = getattr(tokenizer, "eos_token", None) or ""
    if not im_end:
        return False
    tokenizer.chat_template = (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\\n' + message['content'] + '"
        + im_end
        + "\\n' }}"
        "{%- elif message['role'] == 'user' %}"
        "{{ '<|im_start|>user\\n' + message['content'] + '"
        + im_end
        + "\\n' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% generation %}"
        "{{ message['content'] }}"
        "{% endgeneration %}"
        "{{ '"
        + im_end
        + "\\n' }}"
        "{%- endif %}"
        "{%- endfor %}"
    )
    print(
        "[memory_utils] 已为 TRL assistant_only_loss 注入含 {% generation %} 的 Qwen chat_template。",
        flush=True,
    )
    return True


def _tokenizer_encodes_cjk(tok) -> bool:
    """
    损坏或不完整的 tokenizer.json 会导致中文 encode 长度为 0。
    须同时尝试 encode 默认参数、add_special_tokens 与 tokenize（仅用 False 可能误判）。
    """
    for text in ("记忆测试", "你"):
        try:
            if len(tok.encode(text)) >= 1:
                return True
        except Exception:
            pass
        for add_sp in (False, True):
            try:
                if len(tok.encode(text, add_special_tokens=add_sp)) >= 1:
                    return True
            except Exception:
                pass
        try:
            if len(tok.tokenize(text)) >= 1:
                return True
        except Exception:
            pass
    return False


def _print_tokenizer_probe_debug(tok) -> None:
    for s in ("a", "hi", "记忆", "记忆测试"):
        for add_sp in (None, False, True):
            try:
                if add_sp is None:
                    n = len(tok.encode(s))
                    label = "encode默认"
                else:
                    n = len(tok.encode(s, add_special_tokens=add_sp))
                    label = f"add_special_tokens={add_sp}"
                print(f"[memory_utils] 探针调试 {label} {s!r} -> {n} ids", flush=True)
            except Exception as e:
                print(f"[memory_utils] 探针调试 encode({s!r}) 异常: {e!r}", flush=True)
        try:
            print(f"[memory_utils] 探针调试 tokenize({s!r}) -> {len(tok.tokenize(s))} 片", flush=True)
        except Exception as e:
            print(f"[memory_utils] 探针调试 tokenize 异常: {e!r}", flush=True)


def _from_pretrained_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool,
    use_fast: bool,
    force_download: bool,
    use_official_hf_endpoint: bool,
):
    """
    use_official_hf_endpoint=True 时临时去掉 HF_ENDPOINT，让 huggingface_hub 走官方 huggingface.co
    （镜像上的 tokenizer 常损坏：中文 encode 为 0 token，英文正常）。
    """
    saved_ep = None
    if use_official_hf_endpoint:
        saved_ep = os.environ.pop("HF_ENDPOINT", None)
        print(
            "[memory_utils] 临时清除 HF_ENDPOINT，从官方 huggingface.co 请求 tokenizer "
            f"（恢复前原值={saved_ep!r}）",
            flush=True,
        )
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
            force_download=force_download,
        )
    finally:
        if use_official_hf_endpoint and saved_ep is not None:
            os.environ["HF_ENDPOINT"] = saved_ep


def _local_tokenizer_matches_base(lp: Path, model_id: str) -> bool:
    """本地 hf_tokenizer 若为 DeepSeek/Llama 词表，不可与 Qwen 基座混用。"""
    cfg = lp / "tokenizer_config.json"
    if not cfg.is_file():
        return True
    try:
        t = cfg.read_text(encoding="utf-8")
    except OSError:
        return True
    deepseek_style = "redacted_begin" in t or "LlamaTokenizer" in t
    if "qwen" in model_id.lower() and deepseek_style:
        return False
    return True


def _try_local_full_tokenizer(
    model_id: str, local_tokenizer_dir: str, trust_remote_code: bool
) -> PreTrainedTokenizerFast | None:
    """本地完整 tokenizer（local_files_only）：避免 transformers 对 Hub 调用 list_repo_templates 导致超时。"""
    lp = Path(local_tokenizer_dir)
    has_cfg = (lp / "tokenizer_config.json").is_file()
    has_json = (lp / "tokenizer.json").is_file()
    if not lp.is_dir() or not (has_cfg or has_json):
        return None
    if not _local_tokenizer_matches_base(lp, model_id):
        print(
            "[memory_utils] 跳过本地 tokenizer 目录：词表与当前基座不匹配（勿混用 DeepSeek 词表 + Qwen 权重）。",
            flush=True,
        )
        return None
    print(f"[memory_utils] 优先从本地目录加载 tokenizer: {lp}", flush=True)
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(
            str(lp),
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        if _tokenizer_encodes_cjk(tok):
            print("[memory_utils] 本地 tokenizer 中文探针通过（PreTrainedTokenizerFast）。", flush=True)
            return tok
        print("[memory_utils] 本地 tokenizer 未通过中文探针，继续尝试在线…", flush=True)
    except BaseException as e:
        print(f"[memory_utils] 本地 tokenizer 加载失败: {e!r}", flush=True)
    return None


def load_tokenizer_for_inference(
    model_id: str,
    trust_remote_code: bool = True,
    *,
    local_tokenizer_dir: str | None = None,
):
    """
    推理专用：必须从「完整基座」加载 tokenizer（不要用 outputs/memory_lora 里 save 的那份做分词）。
    若 HF 缓存里的 tokenizer 损坏，encode 中文会得到 0 token，表现为 input_ids 长度≈4、生成立刻 EOS。

    local_tokenizer_dir：无代理、镜像 tokenizer 损坏时，可指向已放入完整 tokenizer 文件的本地目录
    （见 memory_extract_cli 顶部说明）。
    """
    if "qwen" in (model_id or "").lower():
        tok_q = _try_local_full_tokenizer(
            model_id, qwen_local_tokenizer_dir(), trust_remote_code
        )
        if tok_q is not None:
            return tok_q
    if local_tokenizer_dir:
        tok = _try_local_full_tokenizer(model_id, local_tokenizer_dir, trust_remote_code)
        if tok is not None:
            return tok

    ep = os.environ.get("HF_ENDPOINT", "").strip()
    print(
        f"[memory_utils] load_tokenizer_for_inference({model_id!r}) HF_ENDPOINT={ep or '(官方)'}",
        flush=True,
    )
    last_err: BaseException | None = None
    last_tok = None
    for use_fast in (True, False):
        label = "fast" if use_fast else "slow"
        print(f"[memory_utils] 尝试 {label} tokenizer …", flush=True)
        try:
            tok = _from_pretrained_tokenizer(
                model_id,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast,
                force_download=False,
                use_official_hf_endpoint=False,
            )
        except BaseException as e:
            last_err = e
            continue
        last_tok = tok
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        if _tokenizer_encodes_cjk(tok):
            print(f"[memory_utils] tokenizer 中文探针通过（{label}）。", flush=True)
            _cache_qwen_tokenizer_after_hub_load(model_id, tok)
            return tok
        print(f"[memory_utils] 警告：{label} tokenizer 未通过中文探针，换下一个实现。", flush=True)
    print(
        "[memory_utils] 尝试 PreTrainedTokenizerFast（绕开 LlamaTokenizerFast 对中文 encode 为 0）…",
        flush=True,
    )
    try:
        tok_pt = PreTrainedTokenizerFast.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if tok_pt.pad_token is None and tok_pt.eos_token is not None:
            tok_pt.pad_token = tok_pt.eos_token
        if _tokenizer_encodes_cjk(tok_pt):
            print("[memory_utils] tokenizer 中文探针通过（PreTrainedTokenizerFast + hub）。", flush=True)
            _cache_qwen_tokenizer_after_hub_load(model_id, tok_pt)
            return tok_pt
        last_tok = tok_pt
    except BaseException as e:
        last_err = e
        print(f"[memory_utils] PreTrainedTokenizerFast 加载失败: {e!r}", flush=True)
    print(
        "[memory_utils] 尝试 force_download=True 重新拉取 tokenizer（修复不完整镜像缓存，可能较慢）…",
        flush=True,
    )
    try:
        tok_fd = _from_pretrained_tokenizer(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=True,
            force_download=True,
            use_official_hf_endpoint=False,
        )
        if tok_fd.pad_token is None and tok_fd.eos_token is not None:
            tok_fd.pad_token = tok_fd.eos_token
        if _tokenizer_encodes_cjk(tok_fd):
            print("[memory_utils] tokenizer 中文探针通过（fast + force_download）。", flush=True)
            _cache_qwen_tokenizer_after_hub_load(model_id, tok_fd)
            return tok_fd
        last_tok = tok_fd
    except BaseException as e:
        last_err = e
        print(f"[memory_utils] force_download 加载失败: {e!r}", flush=True)
    print(
        "[memory_utils] 镜像 tokenizer 中文仍为 0 token：尝试官方 huggingface.co + force_download …",
        flush=True,
    )
    for use_fast in (True, False):
        label = "fast" if use_fast else "slow"
        try:
            tok_of = _from_pretrained_tokenizer(
                model_id,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast,
                force_download=True,
                use_official_hf_endpoint=True,
            )
        except BaseException as e:
            last_err = e
            print(f"[memory_utils] 官方 endpoint + {label} 加载失败: {e!r}", flush=True)
            continue
        last_tok = tok_of
        if tok_of.pad_token is None and tok_of.eos_token is not None:
            tok_of.pad_token = tok_of.eos_token
        if _tokenizer_encodes_cjk(tok_of):
            print(
                f"[memory_utils] tokenizer 中文探针通过（官方 huggingface.co + {label} + force_download）。",
                flush=True,
            )
            _cache_qwen_tokenizer_after_hub_load(model_id, tok_of)
            return tok_of
        print(
            f"[memory_utils] 警告：官方 endpoint + {label} 仍未通过中文探针。",
            flush=True,
        )
    msg = (
        f"Tokenizer 无法编码中文（model_id={model_id!r}），推理会退化为仅若干 special token。\n"
        "无 VPN 时：在「记忆模型/1.0」目录下建 hf_tokenizer 文件夹，从 hf-mirror 上 **与当前基座同名** 的仓库 "
        "（默认 Qwen/Qwen2.5-7B-Instruct；勿把 DeepSeek 词表与 Qwen 权重混用）逐个下载 "
        "tokenizer.json、tokenizer.model、tokenizer_config.json、special_tokens_map.json、"
        "chat_template.jinja 等全部文件放入该目录；或设置环境变量 MEMORY_TOKENIZER_PATH 指向该目录。\n"
        "其他：1) 删除损坏的 HF 缓存目录（models--<组织>--<模型名>）后重试；"
        "2) 有代理时可临时取消 HF_ENDPOINT 走官方；3) 见下方探针调试。"
    )
    if last_err is not None:
        msg += f"\n最近一次加载异常: {last_err!r}"
    if last_tok is not None:
        print("[memory_utils] 探针失败，对已加载 tokenizer 打印调试：", flush=True)
        _print_tokenizer_probe_debug(last_tok)
    raise RuntimeError(msg)


def load_base_model_causal_lm(
    model_id: str,
    *,
    trust_remote_code: bool = True,
    use_4bit: bool = True,
    device_map: str | dict = "auto",
):
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    if use_4bit:
        kwargs["quantization_config"] = build_bits_and_bytes_config()
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ep = os.environ.get("HF_ENDPOINT", "").strip()
    print(f"[memory_utils] HF_ENDPOINT={ep or '(未设置，使用官方 huggingface.co)'}", flush=True)
    model_path = Path(model_id)
    is_local = model_path.is_dir() and (model_path / "config.json").is_file()
    if is_local:
        kwargs["local_files_only"] = True
        print(
            f"[memory_utils] AutoModelForCausalLM.from_pretrained({model_id!r}) 开始（local_files_only=True）…",
            flush=True,
        )
    else:
        print(
            f"[memory_utils] AutoModelForCausalLM.from_pretrained({model_id!r}) 开始 …\n"
            "  （首次运行需下载完整权重，体积极大、耗时很长，请耐心等待；请勿当作卡死直接结束。）",
            flush=True,
        )
    m = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    print("[memory_utils] AutoModelForCausalLM 加载完成。", flush=True)
    return m


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
    """将训练用 messages 展成与旧版单串 text 训练相同的字符串（用于估算 token 或与探针对齐）。"""
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


def format_qa_sample(
    tokenizer,
    q: str,
    a: str,
    *,
    memory_time_iso: str | None = None,
) -> str:
    return apply_memory_train_template(
        tokenizer, qa_training_messages(q, a, memory_time_iso=memory_time_iso)
    )


def format_raw_sample(tokenizer, raw: str) -> str:
    return apply_memory_train_template(tokenizer, raw_training_messages(raw))


def build_extract_messages(user_query: str):
    return [
        {"role": "system", "content": SYSTEM_EXTRACT},
        {"role": "user", "content": user_content_with_memory_extract_prefix(user_query)},
    ]


def _decode_gen_ids(tokenizer, gen_ids: torch.Tensor) -> str:
    """R1 系：skip_special_tokens=True 时可能把可见内容一并剥成空串，故先 False 再兜底。"""
    if gen_ids.numel() == 0:
        return ""
    raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    t = (raw or "").strip()
    for tok in (getattr(tokenizer, "eos_token", None), getattr(tokenizer, "bos_token", None), getattr(tokenizer, "pad_token", None)):
        if tok:
            t = t.replace(tok, "")
    t = t.strip()
    if not t:
        raw2 = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if isinstance(raw2, list):
            raw2 = raw2[0] if raw2 else ""
        t = (raw2 or "").strip()
    return t


def generate_extract_completion(
    model,
    tokenizer,
    inputs: dict,
    max_new_tokens: int,
) -> str:
    """
    使用显式 generation 参数，避免 **inputs 展开 + generation_config 合并** 时带上模型里旧的 temperature。

    **与 train_memory 中记忆探针 `_generate_probe_answer` 一致：默认贪心（do_sample=False）**，
    避免与训练早停/探针通过条件不一致，并减少采样带来的基座式闲聊、反问句。
    若解码为空则再贪心一次（同上参数）。
    """
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    gc = getattr(model, "generation_config", None)
    eos_id = getattr(gc, "eos_token_id", None) if gc is not None else None
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    # 记忆抽取宜短；过长易拖出训练里常见的英文碎片
    max_new_tokens = min(max_new_tokens, MEMORY_EXTRACT_MAX_NEW_TOKENS)

    loggers = [
        logging.getLogger("transformers.generation"),
        logging.getLogger("transformers.generation.utils"),
    ]
    olds = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
                do_sample=False,
            )
            if attn is not None:
                kwargs["attention_mask"] = attn
            with torch.no_grad():
                out = model.generate(**kwargs)
            gen = out[0][input_ids.shape[1] :]
            text = _decode_gen_ids(tokenizer, gen)
            if not text.strip():
                with torch.no_grad():
                    out = model.generate(**kwargs)
                gen = out[0][input_ids.shape[1] :]
                text = _decode_gen_ids(tokenizer, gen)
    finally:
        for lg, old in olds:
            lg.setLevel(old)
    return text


def build_memory_extract_messages_and_prompts_for_queries(
    user_queries: list[str],
    tokenizer,
) -> list[tuple[list[dict], str]]:
    """
    与 ``generate_extract_completions_batched`` / 单条记忆抽取 使用同一套 messages 与展平后 ``prompt``，
    供调用方对拍日志或在外部先 ``tokenizer`` 再传入 ``prebuilt_enc``，避免与推理不一致。
    """
    rows: list[tuple[list[dict], str]] = []
    for q in user_queries:
        messages = build_extract_messages((q or "").strip())
        if getattr(tokenizer, "chat_template", None) is None:
            prompt = (
                f"[SYSTEM]\n{messages[0]['content']}\n[USER]\n{messages[1]['content']}\n[ASSISTANT]\n"
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        rows.append((messages, prompt))
    return rows


def generate_extract_completions_batched(
    model,
    tokenizer,
    user_queries: list[str],
    max_new_tokens: int,
    *,
    prebuilt_enc: dict | None = None,
) -> list[str]:
    """
    同一路由下多条检索问句合并为一次 ``model.generate``（左 padding），降低多轮串行 RT。
    与 ``generate_extract_completion`` 使用相同 generation 参数；单条对话仍与单条路径一致。

    若由调用方已按左 padding 得到 ``input_ids`` / ``attention_mask``（与内部一致），可传 ``prebuilt_enc`` 避免重复分词。
    """
    if prebuilt_enc is not None:
        if prebuilt_enc["input_ids"].shape[0] == 0:
            return []
    elif not user_queries:
        return []
    dev = next(model.parameters()).device
    if prebuilt_enc is not None:
        enc = {k: v.to(dev) for k, v in prebuilt_enc.items()}
    else:
        rows = build_memory_extract_messages_and_prompts_for_queries(
            user_queries, tokenizer
        )
        prompts = [p for _, p in rows]
        pad_side = getattr(tokenizer, "padding_side", "right")
        if pad_side != "left":
            tokenizer.padding_side = "left"
        try:
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
        finally:
            tokenizer.padding_side = pad_side
        enc = {k: v.to(dev) for k, v in enc.items()}
    max_new_tokens = min(int(max_new_tokens), MEMORY_EXTRACT_MAX_NEW_TOKENS)
    gc = getattr(model, "generation_config", None)
    eos_id = getattr(gc, "eos_token_id", None) if gc is not None else None
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    loggers = [
        logging.getLogger("transformers.generation"),
        logging.getLogger("transformers.generation.utils"),
    ]
    olds = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            kwargs = dict(
                input_ids=enc["input_ids"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
                do_sample=False,
            )
            if enc.get("attention_mask") is not None:
                kwargs["attention_mask"] = enc["attention_mask"]
            in_w = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**kwargs)
            new_part = out[:, in_w:]
            out_texts: list[str] = []
            for i in range(new_part.size(0)):
                # 须为一维 id 序列；二维 [1,L] 时部分 tokenizer 的 decode 会返回 list 而非 str
                text = _decode_gen_ids(tokenizer, new_part[i].contiguous())
                if not text.strip():
                    one_msg = build_extract_messages((user_queries[i] or "").strip())
                    if getattr(tokenizer, "chat_template", None) is None:
                        pone = f"[SYSTEM]\n{one_msg[0]['content']}\n[USER]\n{one_msg[1]['content']}\n[ASSISTANT]\n"
                    else:
                        pone = tokenizer.apply_chat_template(
                            one_msg, tokenize=False, add_generation_prompt=True
                        )
                    one_inp = tokenizer(
                        pone,
                        return_tensors="pt",
                        truncation=False,
                        add_special_tokens=False,
                    )
                    one_inp = {k: v.to(dev) for k, v in one_inp.items()}
                    text = generate_extract_completion(model, tokenizer, one_inp, 512)
                out_texts.append(text)
    finally:
        for lg, old in olds:
            lg.setLevel(old)
    return out_texts


def clean_memory_generation_output(text: str) -> str:
    """去掉 DeepSeek-R1 等模型常见的思维链标记与解码残留显示。"""
    t = text
    # 角括号标签内包含 redacted_think（含 </redacted_thinking> 单独成行）
    t = re.sub(r"<[^>]*redacted_think[^>]*>", "", t, flags=re.IGNORECASE)
    for s in (
        "<｜end of sentence｜>",
        "<|end of sentence|>",
        "<｜end▁of▁sentence｜>",
        # Qwen ChatML / 部分模板解码后残留在可见文本中的片段（与记忆正文无关）
        "<|redacted_im_end|>",
        "<|redacted_im_start|>",
        "<|im_start|>",
        "<|endoftext|>",
    ):
        t = t.replace(s, "")
    t = t.replace("Ċ", "\n").replace("Ġ", " ")
    # 训练时在 user 侧注入的「记忆时刻」行，不得出现在对用户展示的助手回复中（防 LoRA 复述训练格式）
    t = re.sub(r"^\s*【记忆时刻】[^\r\n]*(\r?\n)?", "", t, flags=re.MULTILINE)
    t = re.sub(r"【记忆时刻】[^\r\n]*", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


# 记忆提取 API/CLI 在清洗后额外丢弃的已知误生成或训练污染整句（与有效事实无关）
_KNOWN_GARBAGE_MEMORY_EXACT = frozenset(
    {
        "本段会话",
        "本段会话。",
        "本段会话可读性优先于严格拼写正确率。",
    }
)


def sanitize_memory_extract_output(text: str) -> str:
    """在 clean_memory_generation_output 之后调用：去掉已知的无效记忆生成内容；无效则返回空串。"""
    t = (text or "").strip()
    if not t:
        return ""
    if t in _KNOWN_GARBAGE_MEMORY_EXACT:
        return ""
    if re.match(r"^本段会话[。]?\s*$", t):
        return ""
    if "可读性" in t and "拼写" in t:
        return ""
    return t


def _normalize_for_memory_echo_compare(s: str) -> str:
    """弱化标点与人称差异，便于与检索 query 比对是否同义复述。"""
    t = (s or "").strip()
    for a, b in (("你", "用户"), ("您", "用户")):
        t = t.replace(a, b)
    t = re.sub(r"[\s　]+", "", t)
    t = re.sub(r"[。！？!?；;,.，、…]", "", t)
    return t


def memory_extract_gate_by_query(memory: str, query: str) -> str:
    """
    在 sanitize 之后调用：若模型以问句复述 query、或与 query 高度同义、或仅表示「不知道」，
    则返回空串（API/CLI 应视为 has_memory=false）。
    """
    t = (memory or "").strip()
    q = (query or "").strip()
    if not t:
        return ""
    if re.match(
        r"^(不确定|不知道|暂无|没有相关|无相关|不清楚|想不起来|记不得|无)[。！…]?$",
        t,
    ):
        return ""
    if t.endswith("？") or t.endswith("?"):
        return ""
    qa = _normalize_for_memory_echo_compare(q)
    ta = _normalize_for_memory_echo_compare(t)
    if qa and ta and len(qa) >= 6 and len(ta) >= 6:
        if SequenceMatcher(None, qa, ta).ratio() >= 0.58:
            return ""
    return t
