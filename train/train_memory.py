# -*- coding: utf-8 -*-
"""
使用 Hugging Face 基座（默认 Qwen/Qwen2.5-7B-Instruct）+ QLoRA 微调。

说明（与「GPT-4.1 自建模型」的关系）：
- OpenAI 云端「自定义 GPT / 微调」只能针对 OpenAI 提供的模型与 API，不能把本地 HF 基座权重
  拿去用 GPT-4.1 的结构训练；二者不是同一条管线。
- 本目录下 `GPT-4.1模型` 工程是自研 Transformer + FanTokenizer，若要对齐需单独做数据与训练对接；
  本脚本仅做「HF 开源基座 + LoRA」这一条可复现路径。

Session 约定（相对于本脚本所在目录下的 session/）：
- 多用户：`session/<用户ID>/YYYY-MM-DD/`（扁平日期目录）；训练时**对每个用户只取最新**一个 `YYYY-MM-DD` 目录汇总样本。
- 同 stem 的 raw 与 qa 须**同一日期目录**下。
- `{stem}.raw.txt`：会话原文，训练一条「内化会话」样本（SYSTEM_RAW + 全文）。推荐每行一条事实，格式 `[YYYY-MM-DD HH:MM:SS] …`（每事实独立时刻，便于「中午讨论过什么」等与时刻相关的检索对齐）；旧版仅首行「记忆时间：」仍可读，但不宜再新写。
- `{角色标识}.user.pic.md`（位于 ``session/<用户ID>/`` 根目录，非日期子目录）：由 ``session_to_memory`` 步骤 5 按需生成的**全局共享**角色画像（Markdown），与 raw 相同方式训一条「内化」样本；**同一用户**跨会话共用；日期目录内旧版 ``{stem}.{角色}.user.pic.md`` 不再作为训练来源。
- `{stem}.qa.jsonl`：从该会话抽取的问答，每行 `{"q":"...","a":"...","t":"YYYY-MM-DD HH:MM:SS"}`（`t` 可选但强烈建议：该条所对应事实的发生时刻，与 raw 中该行时间一致；也支持 question/answer/time）；训练时在 user 侧为带 `t` 的样本自动加 `【记忆时刻】` 行，并加 `[记忆提取]` 前缀。
- 同 stem 若**同时**有 raw 与 qa：先按 qa 行生成若干条，再**追加一条**该 raw 的内化样本；若仅有 raw 无 qa：只训 raw。

早停（内置默认开启）：训练 loss≤`--early-stop-loss` 可提前结束；记忆探针另需 loss≤`--memory-probe-enable-loss`（与早停 loss 无关）才允许按间隔抽检。详见 `--no-memory-probe` 等。

LoRA 输出目录：默认 `outputs/<memory_utils 中的目录名>`；若指定 ``--user_id`` 则默认 ``outputs/<用户ID>/该目录名``（只训练该用户数据）。可设环境变量 ``MEMORY_LORA_DIRNAME`` 或 ``--output_dir`` 覆盖。

右键运行：请改文首「HF + 训练/LoRA」内置区；其中 HF 端点/镜像项必须在最前、且在 import 任何 HF 相关库之前，勿只在 main 里设环境变量。
"""

from __future__ import annotations

import sys
from pathlib import Path
for _sub in ("core", "train", "serve"):
    _p = str(Path(__file__).resolve().parent.parent / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import os
import random
import sys
import traceback
import warnings
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# HF 端点（必须放在本文件最前，且在 import datasets/transformers/huggingface_hub 之前）
# huggingface_hub 在首次 import 时会读取 HF_ENDPOINT；若在 main() 里才设置，仍会直连
# huggingface.co，国内表现为 WinError 10060，日志里 URL 也一直是官方域名。
# 与紧接其后的「训练/LoRA 内置」同区，请勿重复定义。
# ---------------------------------------------------------------------------
BUILTIN_HF_MIRROR = True
BUILTIN_HF_ENDPOINT = ""
# True：仅用本地 hf_model + hf_tokenizer，不预置镜像、保存时临时去掉 HF_ENDPOINT；False：走 Hub/镜像（见 BUILTIN_HF_MIRROR）。
BUILTIN_TRAIN_USE_LOCAL_HF = False
# 单次 HTTP 读超时（秒）。官方默认约 10s，拉几 GB 的 safetensors 极易 read timed out，宜调大。
BUILTIN_HF_DOWNLOAD_TIMEOUT_SEC = "1800"
# 大文件默认走 XET（日志里 cas-bridge.xethub.hf.co），国内常超时；设为 True 则禁用 XET，改走传统下载（若仍失败可改回 False 再试）
BUILTIN_HF_DISABLE_XET = True
# Windows 缓存 symlink 提示，设为 1 可少刷屏
BUILTIN_HF_DISABLE_SYMLINKS_WARNING = True

# ---------------------------------------------------------------------------
# 训练/数据/LoRA 内置默认（与上方 HF 同区，右键只改本段；不 import memory_utils，避免与 HF 预置顺序冲突）
# 与 memory_utils 中 ``DEFAULT_BASE_MODEL``、``get_memory_lora_dirname`` 的默认取值一致；若设环境变量 ``MEMORY_LORA_DIRNAME`` 则与本段 ``BUILTIN_LORA_DIRNAME`` 同步读取。
# 若 CUDA OOM：优先把 BUILTIN_MAX_SEQ_LENGTH 降到 1536/1024，其次再把 BUILTIN_LORA_R 降到 96/64。
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
# LoRA 在 outputs/ 下子目录名；未设 ``MEMORY_LORA_DIRNAME`` 时与 memory_utils 默认「memory_lora_v2」一致
BUILTIN_LORA_DIRNAME = (os.environ.get("MEMORY_LORA_DIRNAME", "").strip() or "memory_lora_v2")
# Hub 基座 id；与 memory_utils.DEFAULT_BASE_MODEL 一致
BUILTIN_HUB_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# 只训练**单个用户**的 LoRA 时填写 session 下目录名；留空 = 多用户全量
BUILTIN_USER_ID = "user_001"
# 本地基座为 ``<本脚本目录>/hf_model``，与 memory_utils.memory_local_base_model_dir() 一致
BUILTIN_MODEL = (
    str(_script_dir / "hf_model")
    if BUILTIN_TRAIN_USE_LOCAL_HF
    else BUILTIN_HUB_BASE_MODEL
)
BUILTIN_SESSION_DIR = str(_script_dir / "session")
BUILTIN_OUTPUT_DIR = str(_script_dir / "outputs" / BUILTIN_LORA_DIRNAME)
BUILTIN_EPOCHS = 5
BUILTIN_LR = 3.5e-4
BUILTIN_BATCH_SIZE = 1
BUILTIN_GRAD_ACCUM = 2
BUILTIN_MAX_SEQ_LENGTH = 2048
BUILTIN_NO_4BIT = False
BUILTIN_FALLBACK_MODEL = False
BUILTIN_LORA_R = 128
BUILTIN_LORA_ALPHA = 256
BUILTIN_LORA_DROPOUT = 0.02
# 早停：loss 达阈值即停；或按间隔做随机 QA 探针，全部通过即停（见下方 argparse）
BUILTIN_EARLY_STOP_LOSS = 0.07
BUILTIN_EARLY_STOP_MIN_STEPS = 35
BUILTIN_MEMORY_PROBE_ENABLE = True
BUILTIN_MEMORY_PROBE_SAMPLES = 3
BUILTIN_MEMORY_PROBE_EVERY_STEPS = 20
# 记忆探针的 loss 上限（与 BUILTIN_EARLY_STOP_LOSS 无关）：仅当训练 loss≤此值时才允许跑探针
BUILTIN_MEMORY_PROBE_ENABLE_LOSS = 0.1
BUILTIN_MEMORY_PROBE_MAX_LOSS = BUILTIN_MEMORY_PROBE_ENABLE_LOSS
# ---------------------------------------------------------------------------


def _apply_hf_runtime_early() -> None:
    ep = BUILTIN_HF_ENDPOINT.strip().rstrip("/")
    if ep:
        os.environ["HF_ENDPOINT"] = ep
    elif BUILTIN_HF_MIRROR and not BUILTIN_TRAIN_USE_LOCAL_HF:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", BUILTIN_HF_DOWNLOAD_TIMEOUT_SEC)
    if BUILTIN_HF_DISABLE_XET:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if BUILTIN_HF_DISABLE_SYMLINKS_WARNING:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    out = os.environ.get("HF_ENDPOINT", "").strip()
    if out:
        print(
            f"[train_memory] 预置 HF_ENDPOINT={out}（在任何 huggingface 相关 import 之前）",
            flush=True,
        )
    print(
        f"[train_memory] 预置 HF_HUB_DOWNLOAD_TIMEOUT={os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}，"
        f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET', '')}",
        flush=True,
    )


_apply_hf_runtime_early()
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """行缓冲输出，避免 IDE/管道下长时间看不到日志。"""
    print(msg, flush=True)


def _configure_hf_download_logging() -> None:
    """打开 huggingface_hub 等日志，便于看到下载/解析进度。"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def apply_hf_runtime_env(args: argparse.Namespace) -> None:
    """在首次访问 HF 前设置端点（镜像），避免国内直连 huggingface.co 超时。"""
    import memory_utils as _mem

    if getattr(_mem, "TRAIN_USE_LOCAL_HF", False):
        if getattr(args, "hf_endpoint", None) and str(args.hf_endpoint).strip():
            v = str(args.hf_endpoint).strip().rstrip("/")
            os.environ["HF_ENDPOINT"] = v
            _log(f"[train_memory] 本地训练模式：仍使用 --hf_endpoint={v!r}")
        else:
            os.environ.pop("HF_ENDPOINT", None)
            _log(
                "[train_memory] 本地训练模式：已清除 HF_ENDPOINT（加载与保存尽量不走镜像/Hub；"
                "若仍需指定端点请传 --hf_endpoint）。"
            )
        return

    if getattr(args, "hf_endpoint", None):
        v = str(args.hf_endpoint).strip().rstrip("/")
        if v:
            os.environ["HF_ENDPOINT"] = v
            _log(f"[train_memory] 已设置 HF_ENDPOINT={v}（--hf_endpoint）")
    elif getattr(args, "hf_mirror", False):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        _log("[train_memory] 已设置 HF_ENDPOINT=https://hf-mirror.com（hf_mirror=True）")

    ep = os.environ.get("HF_ENDPOINT", "").strip()
    if ep:
        _log(f"[train_memory] 当前 Hugging Face 请求将走: HF_ENDPOINT={ep}")
    else:
        _log(
            "[train_memory] HF_ENDPOINT 未设置 → 将直连官方 huggingface.co。"
            " 若出现 WinError 10060 / 连接超时 / Retrying，请在文件顶部设 BUILTIN_HF_MIRROR=True 或命令行加 --hf-mirror"
        )


def _log_hf_connection_troubleshooting(exc: BaseException) -> None:
    _log("[train_memory] ---------- Hugging Face 网络访问失败 ----------")
    _log("[train_memory] 可任选其一重试：")
    _log("[train_memory]   1) 打开 train_memory.py 顶部将 BUILTIN_HF_MIRROR 设为 True（默认已为 True）")
    _log("[train_memory]   2) 或命令行: python train_memory.py --hf-mirror")
    _log('[train_memory]   3) 或 PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"')
    _log("[train_memory]   4) 使用系统/Clash 等 HTTP 代理后再运行")
    _log(f"[train_memory] 本次异常: {exc!r}")
    _log("[train_memory] --------------------------------------------")


def _looks_like_hf_connection_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc!s}".lower()
    needles = (
        "10060",
        "timed out",
        "timeout",
        "connection",
        "resolve",
        "network",
        "unreachable",
        "refused",
        "huggingface.co",
        "ssl",
    )
    return any(x in s for x in needles)


_log("[train_memory] 脚本已启动，正在加载 torch …")
import torch

# bitsandbytes + 新版 PyTorch：QLoRA 探针调用 model.generate 时可能刷 FutureWarning（_check_is_size），与训练无关
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*_check_is_size.*",
)

_log(f"[train_memory] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        _log(f"[train_memory] cuda_device={torch.cuda.get_device_name(0)}")
    except Exception:
        pass

_log("[train_memory] 正在加载 datasets / peft / transformers …")
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainerCallback

_log("[train_memory] 正在加载 memory_utils …")
from memory_user_paths import is_iso_date_dir_name, sanitize_path_user_id

from memory_utils import (
    FALLBACK_BASE_MODEL,
    _decode_gen_ids,
    apply_peft_save_pretrained_embedding_layers_explicit,
    per_user_lora_output_dir,
    qa_training_messages,
    raw_training_messages,
    save_trainer_model_and_tokenizer,
    SYSTEM_EXTRACT,
    clean_memory_generation_output,
    load_base_model_causal_lm,
    load_tokenizer,
    sanitize_memory_extract_output,
    user_content_with_memory_extract_prefix,
)

import memory_utils as memory_utils

memory_utils.TRAIN_USE_LOCAL_HF = BUILTIN_TRAIN_USE_LOCAL_HF


def _fail(msg: str, code: int = 1) -> None:
    """同时写到 stderr 与 stdout，避免 IDE 只显示其一导致「像没日志」。"""
    print(msg, file=sys.stderr, flush=True)
    print(msg, flush=True)
    sys.exit(code)


def _resolve_memory_probe_enable_loss(args: argparse.Namespace) -> float:
    """`--memory-probe-max-loss` 若指定则覆盖 `--memory-probe-enable-loss`。"""
    if getattr(args, "memory_probe_max_loss", None) is not None:
        return float(args.memory_probe_max_loss)
    return float(args.memory_probe_enable_loss)


def _stem_from_raw_path(p: Path) -> str:
    name = p.name
    if name.endswith(".raw.txt"):
        return name[: -len(".raw.txt")]
    return p.stem


def _read_qa_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stem_from_qa_path(p: Path) -> str:
    name = p.name
    if name.endswith(".qa.jsonl"):
        return name[: -len(".qa.jsonl")]
    return p.stem


def _qa_item_memory_time(item: dict) -> str | None:
    v = item.get("t") or item.get("time") or item.get("memory_time")
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _is_global_user_pic_filename(name: str) -> bool:
    """全局画像：``角色.user.pic.md``（角色名不含句点）；旧版 ``stem.角色.user.pic.md`` 不匹配。"""
    suf = ".user.pic.md"
    if not name.endswith(suf):
        return False
    rest = name[: -len(suf)]
    return "." not in (rest or "")


def collect_global_user_pics_all_users(session_root: Path) -> list[list[dict]]:
    """``session/<用户ID>/<角色>.user.pic.md``，按用户目录各加载一遍。"""
    out: list[list[dict]] = []
    if not session_root.is_dir():
        return out
    for ud in sorted(session_root.iterdir(), key=lambda p: p.name):
        if not ud.is_dir():
            continue
        for pic in sorted(ud.glob("*.user.pic.md")):
            if not pic.is_file():
                continue
            if not _is_global_user_pic_filename(pic.name):
                continue
            out.append(raw_training_messages(pic.read_text(encoding="utf-8")))
    return out


def collect_global_user_pics_one_user(session_root: Path, user_id: str) -> list[list[dict]]:
    """仅 ``session/<该用户ID>/`` 根目录下全局 ``*.user.pic.md``。"""
    out: list[list[dict]] = []
    ud = session_root / sanitize_path_user_id(user_id)
    if not ud.is_dir():
        return out
    for pic in sorted(ud.glob("*.user.pic.md")):
        if not pic.is_file():
            continue
        if not _is_global_user_pic_filename(pic.name):
            continue
        out.append(raw_training_messages(pic.read_text(encoding="utf-8")))
    return out


def _latest_date_dir_under_user_home(user_home: Path) -> list[Path]:
    """``user_home`` = ``session/<用户ID>/``，其下为 ``YYYY-MM-DD/``；返回最多一个「最新日期」目录。"""
    if not user_home.is_dir():
        return []
    subs = [d for d in user_home.iterdir() if d.is_dir() and is_iso_date_dir_name(d.name)]
    if not subs:
        return []
    subs.sort(key=lambda p: p.name)
    return [subs[-1]]


def _latest_date_dir_per_user(session_root: Path) -> list[Path]:
    """每个 ``session/<用户ID>/`` 下取字典序最新的 ``YYYY-MM-DD`` 目录。"""
    out: list[Path] = []
    if not session_root.is_dir():
        return out
    for ud in sorted(session_root.iterdir(), key=lambda p: p.name):
        if not ud.is_dir():
            continue
        subs = [d for d in ud.iterdir() if d.is_dir() and is_iso_date_dir_name(d.name)]
        if not subs:
            continue
        subs.sort(key=lambda p: p.name)
        out.append(subs[-1])
    return out


def _collect_training_messages_single_dir(session_dir: Path) -> list[list[dict]]:
    """单个日期目录内的样本（该目录下无用户子层级）。"""
    rows: list[list[dict]] = []
    qa_files = sorted(session_dir.glob("*.qa.jsonl"))
    for qa_path in qa_files:
        stem = _stem_from_qa_path(qa_path)
        base = qa_path.parent
        for item in _read_qa_jsonl(qa_path):
            q = item.get("q") or item.get("question")
            a = item.get("a") or item.get("answer")
            if not q or not a:
                continue
            rows.append(
                qa_training_messages(
                    str(q).strip(),
                    str(a).strip(),
                    memory_time_iso=_qa_item_memory_time(item),
                )
            )
        raw_path = base / f"{stem}.raw.txt"
        if raw_path.is_file():
            rows.append(raw_training_messages(raw_path.read_text(encoding="utf-8")))

    raw_files = sorted(session_dir.glob("*.raw.txt"))
    for rf in raw_files:
        stem = _stem_from_raw_path(rf)
        qa_path = rf.parent / f"{stem}.qa.jsonl"
        if not qa_path.is_file():
            rows.append(raw_training_messages(rf.read_text(encoding="utf-8")))
    return rows


def collect_training_messages(
    session_root: Path, *, train_user_id: str | None = None
) -> list[list[dict]]:
    """``session_root`` 为 ``session/``：默认汇总各用户**最新日期**目录；若 ``train_user_id`` 非空则**只**取该用户目录下最新日期与全局画像。"""
    rows: list[list[dict]] = []
    if train_user_id:
        uid = sanitize_path_user_id(train_user_id)
        uh = session_root / uid
        if not uh.is_dir():
            raise ValueError(f"未找到该用户 session 目录: {uh}")
        for latest in _latest_date_dir_under_user_home(uh):
            rows.extend(_collect_training_messages_single_dir(latest))
        rows.extend(collect_global_user_pics_one_user(session_root, uid))
        if not rows:
            raise ValueError(
                f"用户 {uid!r} 下未生成任何训练样本；请在该用户下的 YYYY-MM-DD/ 放置样本，"
                "和/或在该用户根目录放置 <角色>.user.pic.md。"
            )
        return rows
    for latest in _latest_date_dir_per_user(session_root):
        rows.extend(_collect_training_messages_single_dir(latest))
    rows.extend(collect_global_user_pics_all_users(session_root))
    if not rows:
        raise ValueError(
            "没有生成任何训练样本；请在 session/<用户ID>/YYYY-MM-DD/ 下放置 *.qa.jsonl、*.raw.txt，"
            "和/或在 session/<用户ID>/ 下放置 <角色>.user.pic.md，并确认各用户目录下存在日期子目录。"
        )
    return rows


def _collect_qa_pairs_probe_one_dir(session_dir: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for qa_path in sorted(session_dir.glob("*.qa.jsonl")):
        for item in _read_qa_jsonl(qa_path):
            q = item.get("q") or item.get("question")
            a = item.get("a") or item.get("answer")
            if not q or not a:
                continue
            mt = _qa_item_memory_time(item)
            q_str = str(q).strip()
            if mt:
                q_str = f"【记忆时刻】{mt}\n{q_str}"
            pairs.append((q_str, str(a).strip()))
    return pairs


def collect_qa_pairs_for_probe(
    session_root: Path, *, train_user_id: str | None = None
) -> list[tuple[str, str]]:
    """从各用户（或**仅** ``train_user_id``）最新日期目录下的 *.qa.jsonl 收集 (q,a)，供探针使用。"""
    pairs: list[tuple[str, str]] = []
    if train_user_id:
        uid = sanitize_path_user_id(train_user_id)
        for latest in _latest_date_dir_under_user_home(session_root / uid):
            pairs.extend(_collect_qa_pairs_probe_one_dir(latest))
        return pairs
    for latest in _latest_date_dir_per_user(session_root):
        pairs.extend(_collect_qa_pairs_probe_one_dir(latest))
    return pairs


def _qa_probe_accept(reference: str, generated: str) -> bool:
    ref, gen = reference.strip(), generated.strip()
    if not ref or not gen:
        return False
    if ref in gen or gen in ref:
        return True
    return SequenceMatcher(None, ref, gen).ratio() >= 0.72


def _generate_probe_answer(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_EXTRACT},
        {"role": "user", "content": user_content_with_memory_extract_prefix(question)},
    ]
    if getattr(tokenizer, "chat_template", None) is None:
        u = user_content_with_memory_extract_prefix(question)
        prompt = f"[SYSTEM]\n{SYSTEM_EXTRACT}\n[USER]\n{u}\n[ASSISTANT]\n"
    else:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    raw = _decode_gen_ids(tokenizer, gen_ids)
    return sanitize_memory_extract_output(clean_memory_generation_output(raw)).strip()


class MemoryEarlyStoppingCallback(TrainerCallback):
    """loss 阈值早停 + 随机 QA 探针全部通过则早停。"""

    def __init__(
        self,
        tokenizer,
        qa_pairs: list[tuple[str, str]],
        *,
        loss_threshold: float | None,
        min_steps: int,
        probe_enable: bool,
        probe_samples: int,
        probe_every_steps: int,
        probe_enable_loss: float,
        rng_seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_pairs = qa_pairs
        self.loss_threshold = loss_threshold
        self.min_steps = max(0, min_steps)
        self.probe_enable = probe_enable
        self.probe_samples = max(1, probe_samples)
        self.probe_every_steps = max(1, probe_every_steps)
        self.probe_enable_loss = probe_enable_loss
        self._rng = random.Random(rng_seed)
        self._model_ref = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._model_ref = kwargs.get("model")
        return control

    def _probe_all_pass(self, model) -> bool:
        if not self.qa_pairs:
            return False
        n = min(self.probe_samples, len(self.qa_pairs))
        sample = self._rng.sample(self.qa_pairs, n)
        was_training = model.training
        model.eval()
        try:
            for q, ref_a in sample:
                gen = _generate_probe_answer(model, self.tokenizer, q)
                if not _qa_probe_accept(ref_a, gen):
                    _log(
                        f"[train_memory] 记忆探针未通过：q={q[:40]!r}… ref[:40]={ref_a[:40]!r}… gen[:40]={gen[:40]!r}…"
                    )
                    return False
            return True
        finally:
            if was_training:
                model.train()

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs")
        model = kwargs.get("model") or self._model_ref
        if logs is None or "loss" not in logs or model is None:
            return control
        loss = float(logs["loss"])
        step = int(state.global_step)
        if step < self.min_steps:
            return control
        if self.loss_threshold is not None and loss <= self.loss_threshold:
            _log(
                f"[train_memory] 早停：loss={loss:.4f} ≤ 阈值 {self.loss_threshold}（step={step}）"
            )
            control.should_training_stop = True
            return control
        if not self.probe_enable or not self.qa_pairs or loss > self.probe_enable_loss:
            return control
        if step % self.probe_every_steps != 0:
            return control
        _log(
            f"[train_memory] 记忆探针 step={step} loss={loss:.4f} "
            f"（抽样 {min(self.probe_samples, len(self.qa_pairs))} 条 QA）…"
        )
        if self._probe_all_pass(model):
            _log("[train_memory] 早停：随机记忆探针全部通过。")
            control.should_training_stop = True
        return control


def main() -> None:
    parser = argparse.ArgumentParser(description="记忆 LoRA 微调（读取 session/）")
    parser.add_argument(
        "--model",
        type=str,
        default=BUILTIN_MODEL,
        help=f"本地基座目录（须含 config.json）；默认 BUILTIN_MODEL/MEMORY_BASE_MODEL_PATH（当前: {BUILTIN_MODEL}）",
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        default=BUILTIN_SESSION_DIR,
        help="session 根目录（其下为 <用户ID>/YYYY-MM-DD/；训练汇总各用户最新日期目录）",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default=BUILTIN_USER_ID,
        help="只训练该用户：数据仅来自 session/<user_id>/；留空为全量多用户。未指定 --output_dir 时输出到 outputs/<user_id>/<LoRA 子目录>。右键运行请改文件顶部 BUILTIN_USER_ID。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="LoRA 适配器输出目录；留空则全量为 outputs/<LoRA 子目录>，有 --user_id 时为 per-user 目录（见 per_user_lora_output_dir）",
    )
    parser.add_argument("--epochs", type=int, default=BUILTIN_EPOCHS, help="训练轮数")
    parser.add_argument("--lr", type=float, default=BUILTIN_LR, help="学习率")
    parser.add_argument("--batch_size", type=int, default=BUILTIN_BATCH_SIZE, help="per_device_train_batch_size")
    parser.add_argument("--grad_accum", type=int, default=BUILTIN_GRAD_ACCUM, help="gradient_accumulation_steps")
    parser.add_argument("--max_seq_length", type=int, default=BUILTIN_MAX_SEQ_LENGTH, help="最大序列长度")
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        default=BUILTIN_NO_4BIT,
        help="禁用 4bit 量化（需更大显存，约 16GB+ 用于 8B 级）；内置默认见 BUILTIN_NO_4BIT",
    )
    parser.add_argument(
        "--fallback_model",
        action="store_true",
        default=BUILTIN_FALLBACK_MODEL,
        help=f"加载失败时改用 {FALLBACK_BASE_MODEL}",
    )
    parser.add_argument(
        "--hf-mirror",
        dest="hf_mirror",
        default=BUILTIN_HF_MIRROR,
        action=argparse.BooleanOptionalAction,
        help="是否使用 HF 镜像（默认 True，对应 BUILTIN_HF_MIRROR；命令行可用 --no-hf-mirror 关闭）",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=BUILTIN_HF_ENDPOINT,
        help="自定义 HF 端点；非空则覆盖镜像。内置默认 BUILTIN_HF_ENDPOINT",
    )
    parser.add_argument(
        "--early-stop-loss",
        type=float,
        default=BUILTIN_EARLY_STOP_LOSS,
        help=f"训练 loss≤此值则提前停止（默认 {BUILTIN_EARLY_STOP_LOSS}）；配合 --no-early-stop-loss 可关闭",
    )
    parser.add_argument(
        "--no-early-stop-loss",
        action="store_true",
        default=False,
        help="关闭 loss 阈值早停（仍可按记忆探针早停，除非同时 --no-memory-probe）",
    )
    parser.add_argument(
        "--early-stop-min-steps",
        type=int,
        default=BUILTIN_EARLY_STOP_MIN_STEPS,
        help="早停生效前最少 global_step（避免前几步 loss 不稳定）",
    )
    parser.add_argument(
        "--no-memory-probe",
        action="store_true",
        default=False,
        help="关闭随机 QA 记忆探针早停",
    )
    parser.add_argument(
        "--memory-probe-samples",
        type=int,
        default=BUILTIN_MEMORY_PROBE_SAMPLES,
        help="每次探针随机抽几条 QA 做生成比对",
    )
    parser.add_argument(
        "--memory-probe-every",
        type=int,
        default=BUILTIN_MEMORY_PROBE_EVERY_STEPS,
        help="每隔多少 global_step 且在启用探针的 loss 条件满足时尝试一次探针",
    )
    parser.add_argument(
        "--memory-probe-enable-loss",
        type=float,
        default=BUILTIN_MEMORY_PROBE_ENABLE_LOSS,
        help="启用记忆探针的 loss 上限：仅当当前训练 loss≤此值时才允许运行探针（与 --early-stop-loss 无关）",
    )
    parser.add_argument(
        "--memory-probe-max-loss",
        type=float,
        default=None,
        help="兼容旧参数，语义同 --memory-probe-enable-loss；若指定则覆盖 --memory-probe-enable-loss",
    )
    args = parser.parse_args()

    apply_hf_runtime_env(args)

    _log("[train_memory] 参数解析完成，检查运行环境 …")
    if not torch.cuda.is_available():
        _fail(
            "错误：未检测到 CUDA。8B 级模型 QLoRA 微调通常需要 NVIDIA GPU。"
            " 若必须在 CPU 上调试，请先解决环境问题或使用带 CUDA 的解释器运行。",
            code=2,
        )

    session_dir = Path(args.session_dir)
    train_uid = (args.user_id or "").strip()
    if (args.output_dir or "").strip():
        output_dir = Path((args.output_dir or "").strip())
    else:
        if train_uid:
            output_dir = per_user_lora_output_dir(_script_dir, train_uid)
        else:
            output_dir = Path(BUILTIN_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"[train_memory] session_dir={session_dir}")
    if train_uid:
        _log(f"[train_memory] 单用户训练 user_id={train_uid!r} output_dir={output_dir}")
    else:
        _log(f"[train_memory] 多用户训练 output_dir={output_dir}")

    model_id = args.model
    if memory_utils.TRAIN_USE_LOCAL_HF:
        mp = Path(model_id)
        if not mp.is_dir() or not (mp / "config.json").is_file():
            _fail(
                "错误：本地训练模式未找到基座目录或缺少 config.json。\n"
                f"  当前 --model={model_id!r}\n"
                "  请将完整权重放入「记忆模型/1.0/hf_model」或设置 MEMORY_BASE_MODEL_PATH；"
                "或把 train_memory.py 顶部 BUILTIN_TRAIN_USE_LOCAL_HF 改为 False 使用 Hub。",
                code=2,
            )
    try:
        if memory_utils.TRAIN_USE_LOCAL_HF:
            _log(f"[train_memory] 准备从本地加载 tokenizer（基座目录: {model_id}）")
        else:
            _log(f"[train_memory] 准备加载 tokenizer: {model_id}")
            _log(
                "[train_memory] 提示：若长时间停在下一条 memory_utils 日志之后，"
                "多为正在从 Hugging Face 下载；国内可设 HF_ENDPOINT=https://hf-mirror.com 后重开。"
            )
        _configure_hf_download_logging()
        tokenizer = load_tokenizer(model_id)
        _log("[train_memory] tokenizer 已就绪，开始加载基座权重 …")
        if memory_utils.TRAIN_USE_LOCAL_HF:
            _log(
                "[train_memory] 从本地目录加载基座（local_files_only）。"
                + (" 当前：4bit 量化。" if not args.no_4bit else " 当前：未使用 4bit（显存占用更大）。")
            )
        else:
            _log(
                "[train_memory] 基座为首次下载时体积可达数 GB，耗时常达数十分钟，属正常现象。"
                + (" 当前：4bit 量化。" if not args.no_4bit else " 当前：未使用 4bit（显存占用更大）。")
            )
        model = load_base_model_causal_lm(model_id, use_4bit=not args.no_4bit)
        if memory_utils.TRAIN_USE_LOCAL_HF and Path(model_id).is_dir():
            try:
                model.config._name_or_path = str(Path(model_id).resolve())
            except Exception:
                pass
        _log("[train_memory] 基座模型已载入显存。")
    except Exception as e:
        if _looks_like_hf_connection_error(e):
            _log_hf_connection_troubleshooting(e)
        if args.fallback_model and model_id != FALLBACK_BASE_MODEL:
            _log(f"[train_memory] 加载 {model_id} 失败：{e}")
            _log(f"[train_memory] 改用 {FALLBACK_BASE_MODEL}")
            model_id = FALLBACK_BASE_MODEL
            tokenizer = load_tokenizer(model_id)
            model = load_base_model_causal_lm(model_id, use_4bit=not args.no_4bit)
        else:
            raise

    _log("[train_memory] 读取 session 并构造训练样本（TRL 全序列 loss，无按样本 assistant 掩码）…")
    messages_rows = collect_training_messages(
        session_dir, train_user_id=train_uid or None
    )
    qa_pairs = collect_qa_pairs_for_probe(
        session_dir, train_user_id=train_uid or None
    )
    dataset = Dataset.from_dict({"messages": messages_rows})
    _log(f"[train_memory] 训练样本数: {len(messages_rows)}（QA 条数用于探针: {len(qa_pairs)}）")

    if not args.no_4bit:
        _log("[train_memory] prepare_model_for_kbit_training …")
        model = prepare_model_for_kbit_training(model)

    _log("[train_memory] 注入 LoRA …")
    lora_config = LoraConfig(
        r=BUILTIN_LORA_R,
        lora_alpha=BUILTIN_LORA_ALPHA,
        lora_dropout=BUILTIN_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head",
        ],
    )
    model = get_peft_model(model, lora_config)
    apply_peft_save_pretrained_embedding_layers_explicit(model)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        _fail("需要安装 trl：pip install trl", code=1)

    loss_thr: float | None = None if args.no_early_stop_loss else float(args.early_stop_loss)
    probe_on = not args.no_memory_probe and bool(qa_pairs)
    probe_enable_loss = _resolve_memory_probe_enable_loss(args)
    _log(
        f"[train_memory] 早停：loss 阈值={'关闭' if loss_thr is None else loss_thr}，"
        f"min_steps={args.early_stop_min_steps}；"
        f"记忆探针={'开' if probe_on else '关'}（每 {args.memory_probe_every} step、"
        f"训练 loss≤{probe_enable_loss} 时才允许探针，抽 {args.memory_probe_samples} 条）"
    )

    _log("[train_memory] 构建 SFTTrainer，开始训练（会打印 loss 等）…")
    sft_args = SFTConfig(
        # 作用：检查点/Trainer 运行产物输出根目录。建议：与当前 LoRA 的 output_dir 一致；勿指向含无关大文件的盘。
        output_dir=str(output_dir),
        # 作用：全数据集重复训练轮数。建议：3～10，结合早停/探针，避免过拟合可减小。
        num_train_epochs=args.epochs,
        # 作用：每 GPU 每步的样本数。建议：1～4（7B+QLoRA 常 1，省显存）。
        per_device_train_batch_size=args.batch_size,
        # 作用：每若干 micro-step 再反传一次，等效调大「逻辑 batch」。建议：2～8，与上项相乘为有效 batch。
        gradient_accumulation_steps=args.grad_accum,
        # 作用：学习率。建议：1e-4～5e-4 量级 LoRA/QLoRA 常用；不收敛可略调。
        learning_rate=args.lr,
        # 作用：每多少步打印/记录一次 loss。建议：1～20；越小日志越密。
        logging_steps=5,
        # 作用：按 epoch 边界保存。建议：epoch（与下项搭配只保留末轮即可）；也可改 steps。
        save_strategy="epoch",
        # 作用：最多保留几个 checkpoint。建议：1 省盘；要对比中间结果可 2～3。
        save_total_limit=1,
        # 作用：GPU 支持时用 bfloat16 训练。建议：A100/30 系可 True；与 fp16 二选一，勿双 True。
        bf16=torch.cuda.is_bf16_supported(),
        # 作用：无 bf16 时用 fp16。建议：与 bf16 二选一，由 is_bf16_supported 自动二选一较稳。
        fp16=not torch.cuda.is_bf16_supported(),
        # 作用：单条序列截断长度。建议：与 BUILTIN_MAX_SEQ_LENGTH/显存平衡，OOM 时降低。
        max_length=args.max_seq_length,
        # 作用：是否仅对 assistant 段算 loss。建议：全序列 False 与本脚本 data collator 一致时保持 False。
        assistant_only_loss=False,
        # 作用：不连 wandb 等。建议：无外部跟踪保持 "none"。
        report_to="none",
        # 作用：以时间换显存、节省激活。建议：7B+ 常开；若异常可暂关试训。
        gradient_checkpointing=True,
        # 作用：优化器实现。建议：adamw_torch 通用；显存/速度如有要求可再换 paged_adamw_8bit 等。
        optim="adamw_torch",
    )

    early_cb = MemoryEarlyStoppingCallback(
        tokenizer,
        qa_pairs,
        loss_threshold=loss_thr,
        min_steps=args.early_stop_min_steps,
        probe_enable=probe_on,
        probe_samples=args.memory_probe_samples,
        probe_every_steps=args.memory_probe_every,
        probe_enable_loss=probe_enable_loss,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[early_cb],
    )
    trainer.train()
    save_trainer_model_and_tokenizer(trainer, str(output_dir), tokenizer)
    _log(f"[train_memory] 完成。适配器与 tokenizer 已保存到: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        err_path = _script_dir / "outputs" / "train_memory_last_error.txt"
        try:
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with err_path.open("w", encoding="utf-8") as ef:
                traceback.print_exc(file=ef)
        except OSError:
            err_path = None
        print("[train_memory] 运行出错，完整堆栈：", flush=True)
        traceback.print_exc()
        if err_path is not None:
            print(f"[train_memory] 堆栈已写入: {err_path}", flush=True)
        sys.exit(1)
