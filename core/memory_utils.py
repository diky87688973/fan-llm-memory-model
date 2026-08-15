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

SYSTEM_RAW = "你是 AI 助手。以下是**你方记忆库**中的会话内化材料（记录者用「我」书写；**用户的事**须写成「用户告诉我…」「我了解到用户…」，**禁止**把用户偏好写成「我喜欢/我不喜欢…」），后续检索与问答都当作**你的记忆**使用。"
# 记忆抽取（CLI / memory_api_server）与 QA 训练、训练探针共用同一段，避免 system 分布不一致。
SYSTEM_EXTRACT = (
    "注意：你是 **AI 记忆模型**；库里的句子是**你作为记录者**记下的内容（第一人称），其中关于**用户**的事实与偏好应表述为「用户告诉我…」「我了解到用户…」，**禁止**把用户偏好说成**模型自己的**「我喜欢…」「我习惯于…」。"
    "你只能根据对话历史中明确写过的内容作答；若未提及，请回答「未记录」。"
    "回答时须与记忆中**主语归属**一致：用户侧用上述句式，**禁止**中性第三人称或把用户事说成「我」的亲身经历。"
    "请用简短中文直接作答（不要输出思考过程、不要重复系统提示）；若没有可引用记忆，请回答「未记录」。"
    "不要堆砌英文单词、无关专名或代码片段（除非问题明确要求）。"
    "禁止输出与问题无关的元话语；若问的是用户偏好或事实，只答记忆中与问题直接相关的具体内容。"
    "严格按问题所问的**维度**作答，**禁止**用其它维度的事实偷换。"
    "禁止只输出残缺占位片段；若确实无对应记忆，请回答「未记录」。"
    "禁止用问句复述检索问题（勿以问句冒充答案）。"
)

# 训练产物默认子目录名（相对本目录下 `outputs/`）。2.0 不含训练脚本，训练见 记忆模型/1.0；推理侧目录名与 1.0 / memory_extract_cli 一致。
# 环境变量 MEMORY_LORA_DIRNAME 可覆盖（重训换目录、或指回旧适配器如 memory_lora）。
DEFAULT_MEMORY_LORA_DIRNAME = "memory_lora_v4"


def get_memory_lora_dirname() -> str:
    v = os.environ.get("MEMORY_LORA_DIRNAME", "").strip()
    return v if v else DEFAULT_MEMORY_LORA_DIRNAME

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


def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    ep = os.environ.get("HF_ENDPOINT", "").strip()
    print(
        f"[memory_utils] HF_ENDPOINT={ep or '(未设置，使用官方 huggingface.co)'}",
        flush=True,
    )
    print(
        f"[memory_utils] AutoTokenizer.from_pretrained({model_id!r}) 开始 …\n"
        "  （首次运行需联网下载 tokenizer 相关文件；控制台若暂时无新输出，多半仍在下载或校验。）",
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
    return tok


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
    （2.0 通过环境变量 MEMORY_TOKENIZER_PATH 传入；不再使用仓库内固定 hf_tokenizer 目录）。
    """
    if local_tokenizer_dir:
        lp = Path(local_tokenizer_dir)
        has_cfg = (lp / "tokenizer_config.json").is_file()
        has_json = (lp / "tokenizer.json").is_file()
        if lp.is_dir() and (has_cfg or has_json):
            if not _local_tokenizer_matches_base(lp, model_id):
                print(
                    "[memory_utils] 跳过本地 hf_tokenizer：词表与当前基座不匹配（勿混用 DeepSeek 词表 + Qwen 权重）。",
                    flush=True,
                )
            else:
                print(f"[memory_utils] 优先从本地目录加载 tokenizer: {lp}", flush=True)
                try:
                    # tokenizer_config 声明 LlamaTokenizerFast 时，AutoTokenizer 会走该类，会把中文 encode 成 0
                    # id；底层 tokenizer.json 正常。须用 PreTrainedTokenizerFast 直接加载。
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
            return tok_of
        print(
            f"[memory_utils] 警告：官方 endpoint + {label} 仍未通过中文探针。",
            flush=True,
        )
    msg = (
        f"Tokenizer 无法编码中文（model_id={model_id!r}），推理会退化为仅若干 special token。\n"
        "无 VPN 时：在任意目录准备与 **当前基座同名** 仓库的完整 tokenizer 文件（勿把 DeepSeek 词表与 Qwen 权重混用），"
        "并设置环境变量 MEMORY_TOKENIZER_PATH 指向该目录；或参考「记忆模型/1.0」侧做法。\n"
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
    print(
        f"[memory_utils] AutoModelForCausalLM.from_pretrained({model_id!r}) 开始 …\n"
        "  （首次运行需下载完整权重，体积极大、耗时很长，请耐心等待；请勿当作卡死直接结束。）",
        flush=True,
    )
    m = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    print("[memory_utils] AutoModelForCausalLM 加载完成。", flush=True)
    return m


def format_qa_sample(
    tokenizer,
    q: str,
    a: str,
    *,
    memory_time_iso: str | None = None,
) -> str:
    q = (q or "").strip()
    mt = (memory_time_iso or "").strip()
    if mt:
        q = f"【记忆时刻】{mt}\n{q}"
    u = user_content_with_memory_extract_prefix(q)
    messages = [
        {"role": "system", "content": SYSTEM_EXTRACT},
        {"role": "user", "content": u},
        {"role": "assistant", "content": a},
    ]
    if getattr(tokenizer, "chat_template", None) is None:
        return f"[SYSTEM]\n{SYSTEM_EXTRACT}\n[USER]\n{u}\n[ASSISTANT]\n{a}"
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def format_raw_sample(tokenizer, raw: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_RAW},
        {"role": "user", "content": raw},
        {"role": "assistant", "content": "好的，我记住了。"},
    ]
    if getattr(tokenizer, "chat_template", None) is None:
        return f"[SYSTEM]\n{SYSTEM_RAW}\n[USER]\n{raw}\n[ASSISTANT]\n好的，我记住了。"
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def build_extract_messages(user_query: str):
    return [
        {"role": "system", "content": SYSTEM_EXTRACT},
        {"role": "user", "content": user_content_with_memory_extract_prefix(user_query)},
    ]


def _decode_gen_ids(tokenizer, gen_ids: torch.Tensor) -> str:
    """R1 系：skip_special_tokens=True 时可能把可见内容一并剥成空串，故先 False 再兜底。"""
    if gen_ids.numel() == 0:
        return ""
    t = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
    for tok in (getattr(tokenizer, "eos_token", None), getattr(tokenizer, "bos_token", None), getattr(tokenizer, "pad_token", None)):
        if tok:
            t = t.replace(tok, "")
    t = t.strip()
    if not t:
        t = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return t


def generate_extract_completion(
    model,
    tokenizer,
    inputs: dict,
    max_new_tokens: int,
) -> str:
    """
    使用显式 generation 参数，避免 **inputs 展开 + generation_config 合并** 时带上模型里旧的 temperature。

    **与 `记忆模型/1.0/train_memory` 中记忆探针 `_generate_probe_answer` 一致：默认贪心（do_sample=False）**，
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
    max_new_tokens = min(max_new_tokens, 220)

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
