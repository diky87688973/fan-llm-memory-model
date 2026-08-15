# -*- coding: utf-8 -*-
"""
回放式记忆训练（按日目录全量 + 历史池抽样追加）。
该目录内**全部** raw 与**全部** QA 均纳入训练（同一日期的记忆文件一律最高优先级，彼此不做 token 比例裁切）。
再从**其余历史日期**目录汇总成历史池，按历史池总 token 的 5% **额外**抽样追加（不与当日样本抢预算）。

全量/首次训练仍用 train_memory.py。本地/镜像 由 train_memory.py 顶部 BUILTIN_TRAIN_USE_LOCAL_HF、BUILTIN_HF_MIRROR 控制；本文件在 import train_memory 前的 HF 预置须与之一致。路径与回放专用超参见文首 **HF 与下段同区**。从已有 LoRA 目录继续训练（若不存在适配器则新注入 LoRA）。
"""

from __future__ import annotations

import sys
from pathlib import Path
for _sub in ("core", "train", "serve"):
    _p = str(Path(__file__).resolve().parent.parent / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import os
import random
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# HF：须在 import train_memory（进而 import transformers）之前
# ---------------------------------------------------------------------------
BUILTIN_HF_MIRROR = True
BUILTIN_HF_ENDPOINT = ""
# 须与 train_memory.py 中同名常量一致（在 import memory_utils 之前即参与 HF 预置）。
BUILTIN_TRAIN_USE_LOCAL_HF = False
BUILTIN_HF_DOWNLOAD_TIMEOUT_SEC = "1800"
BUILTIN_HF_DISABLE_XET = True
BUILTIN_HF_DISABLE_SYMLINKS_WARNING = True

# ---------------------------------------------------------------------------
# 路径与回放专用内置（与上方 HF 同区；不 import memory_utils，与 train_memory 文首同构）
# 未设环境变量 ``MEMORY_LORA_DIRNAME`` 时与 memory_utils 默认「memory_lora_v2」一致
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
BUILTIN_LORA_DIRNAME = (os.environ.get("MEMORY_LORA_DIRNAME", "").strip() or "memory_lora_v2")
BUILTIN_SESSION_ROOT = str(_script_dir / "session")
BUILTIN_OUTPUT_DIR = str(_script_dir / "outputs" / BUILTIN_LORA_DIRNAME)
# 与 train_memory 相同：只训单用户时填写 session 下该用户目录名；留空 = 多用户、输出 BUILTIN_OUTPUT_DIR
BUILTIN_USER_ID = "user_001"
BUILTIN_HISTORY_RATIO = 0.05
BUILTIN_RANDOM_SEED = 42
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


_apply_hf_runtime_early()

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from memory_user_paths import is_iso_date_dir_name, sanitize_path_user_id

from memory_utils import (
    apply_peft_save_pretrained_embedding_layers_explicit,
    per_user_lora_output_dir,
    FALLBACK_BASE_MODEL,
    load_base_model_causal_lm,
    load_tokenizer,
    memory_training_messages_token_len,
    qa_training_messages,
    raw_training_messages,
    save_trainer_model_and_tokenizer,
)

import train_memory as tm

import memory_utils as _mem

_mem.TRAIN_USE_LOCAL_HF = tm.BUILTIN_TRAIN_USE_LOCAL_HF

from train_memory import (
    MemoryEarlyStoppingCallback,
    _resolve_memory_probe_enable_loss,
    apply_hf_runtime_env,
    collect_qa_pairs_for_probe,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _activate_loaded_peft_for_training(model: torch.nn.Module) -> None:
    """从磁盘 ``PeftModel.from_pretrained`` 加载时，PEFT 默认常为推理态（LoRA 的 requires_grad=False），须显式解冻后再训练。"""
    for _n, p in model.named_parameters():
        if "lora_" in _n:
            p.requires_grad = True
    model.train()


def _fail(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr, flush=True)
    print(msg, flush=True)
    sys.exit(code)


def _collect_split_samples(session_dir: Path) -> tuple[list[list[dict]], list[list[dict]]]:
    """同一日期目录内：全部 raw + 全部 QA（每条为 messages 列表）。"""
    raws: list[list[dict]] = []
    qas: list[list[dict]] = []
    qa_files = sorted(session_dir.glob("*.qa.jsonl"))
    for qa_path in qa_files:
        stem = tm._stem_from_qa_path(qa_path)
        for item in tm._read_qa_jsonl(qa_path):
            q = item.get("q") or item.get("question")
            a = item.get("a") or item.get("answer")
            if not q or not a:
                continue
            qas.append(
                qa_training_messages(
                    str(q).strip(),
                    str(a).strip(),
                    memory_time_iso=tm._qa_item_memory_time(item),
                )
            )
        raw_path = session_dir / f"{stem}.raw.txt"
        if raw_path.is_file():
            raws.append(raw_training_messages(raw_path.read_text(encoding="utf-8")))
    raw_files = sorted(session_dir.glob("*.raw.txt"))
    for rf in raw_files:
        stem = tm._stem_from_raw_path(rf)
        qa_path = session_dir / f"{stem}.qa.jsonl"
        if not qa_path.is_file():
            raws.append(raw_training_messages(rf.read_text(encoding="utf-8")))
    return raws, qas


def _flatten_historical_items(historical_dirs: list[Path]) -> list[list[dict]]:
    items: list[list[dict]] = []
    for d in historical_dirs:
        raws, qas = _collect_split_samples(d)
        items.extend(raws)
        items.extend(qas)
    return items


def _sample_history_by_token_ratio(
    tokenizer,
    items: list[list[dict]],
    ratio: float,
    rng: random.Random,
) -> list[list[dict]]:
    """历史池总 token 的 ratio（默认 5%）：打乱后依次加入直至累计 ≥ 目标。"""
    if not items or ratio <= 0:
        return []
    total = sum(memory_training_messages_token_len(tokenizer, x) for x in items)
    target = int(total * ratio)
    if target <= 0:
        return []
    shuffled = items[:]
    rng.shuffle(shuffled)
    out: list[list[dict]] = []
    acc = 0
    for it in shuffled:
        tl = memory_training_messages_token_len(tokenizer, it)
        if acc >= target:
            break
        if acc + tl >= target:
            out.append(it)
            break
        out.append(it)
        acc += tl
    return out


def build_incremental_messages(
    session_root: Path,
    tokenizer,
    *,
    rng: random.Random,
    history_ratio: float,
    user_id: str | None = None,
) -> tuple[list[list[dict]], Path, int, int]:
    """
    返回 (messages 样本列表, session_root, n_history_tokens, n_today_raw_count)。
    「当日」= 各 ``session/<用户ID>/`` 下字典序最新的 ``YYYY-MM-DD`` 目录之并集；历史池为其余日期目录。
    若 ``user_id`` 非空则**仅**该用户目录。
    """
    if user_id and str(user_id).strip():
        uid = sanitize_path_user_id(user_id)
        uh = session_root / uid
        if not uh.is_dir():
            raise ValueError(f"无该用户 session 目录: {uh}")
        latest_dirs = tm._latest_date_dir_under_user_home(uh)
        if not latest_dirs:
            raise ValueError(
                f"该用户 {uid!r} 下无 YYYY-MM-DD 日期子目录: {uh}"
            )
        today_all: list[list[dict]] = []
        n_raw_today = 0
        n_qa_today = 0
        for d in latest_dirs:
            raws_t, qas_t = _collect_split_samples(d)
            today_all.extend(raws_t)
            today_all.extend(qas_t)
            n_raw_today += len(raws_t)
            n_qa_today += len(qas_t)
        if not today_all:
            raise ValueError(
                f"该用户 {uid!r} 下最新日期目录中无训练样本: {latest_dirs!r}"
            )
        latest_set = {p.resolve() for p in latest_dirs}
        historical_dirs: list[Path] = []
        for d in sorted(uh.iterdir(), key=lambda p: p.name):
            if not d.is_dir() or not is_iso_date_dir_name(d.name):
                continue
            if d.resolve() not in latest_set:
                historical_dirs.append(d)
        hist_items = _flatten_historical_items(historical_dirs)
        hist_sel = _sample_history_by_token_ratio(tokenizer, hist_items, history_ratio, rng)
        rows = today_all + hist_sel
        pics = tm.collect_global_user_pics_one_user(session_root, uid)
        rows.extend(pics)
        rng.shuffle(rows)
        h_total = sum(
            memory_training_messages_token_len(tokenizer, x) for x in hist_items
        )
        _log(
            f"[train_memory_replay] 单用户 {uid!r}：raw={n_raw_today} QA行={n_qa_today}；"
            f"历史追加={len(hist_sel)} 条，历史池总 token≈{h_total}"
        )
        return rows, session_root, h_total, n_raw_today

    latest_dirs = tm._latest_date_dir_per_user(session_root)
    if not latest_dirs:
        raise ValueError(
            f"未找到任何 session/<用户ID>/YYYY-MM-DD 目录: {session_root}"
        )
    today_all: list[list[dict]] = []
    n_raw_today = 0
    n_qa_today = 0
    for d in latest_dirs:
        raws_t, qas_t = _collect_split_samples(d)
        today_all.extend(raws_t)
        today_all.extend(qas_t)
        n_raw_today += len(raws_t)
        n_qa_today += len(qas_t)
    if not today_all:
        raise ValueError(f"各用户最新日期目录下无训练样本: {session_root}")

    latest_set = {p.resolve() for p in latest_dirs}
    historical_dirs: list[Path] = []
    if session_root.is_dir():
        for ud in session_root.iterdir():
            if not ud.is_dir():
                continue
            for d in ud.iterdir():
                if not d.is_dir() or not is_iso_date_dir_name(d.name):
                    continue
                if d.resolve() not in latest_set:
                    historical_dirs.append(d)

    hist_items = _flatten_historical_items(historical_dirs)
    hist_sel = _sample_history_by_token_ratio(tokenizer, hist_items, history_ratio, rng)

    rows = today_all + hist_sel
    pics = tm.collect_global_user_pics_all_users(session_root)
    rows.extend(pics)
    rng.shuffle(rows)
    h_total = sum(memory_training_messages_token_len(tokenizer, x) for x in hist_items)
    _log(
        f"[train_memory_replay] 各用户最新日录合并：raw={n_raw_today} 条，QA={n_qa_today} 条；"
        f"历史追加样本={len(hist_sel)} 条（历史池总 token≈{h_total}）"
    )
    return rows, session_root, h_total, n_raw_today


def main() -> None:
    p = argparse.ArgumentParser(description="train_memory_replay：回放式记忆 LoRA（当日目录全量 + 历史池 5% 额外追加）")
    p.add_argument(
        "--session_root",
        type=str,
        default=BUILTIN_SESSION_ROOT,
        help="session 根目录（其下为 <用户ID>/YYYY-MM-DD/）",
    )
    p.add_argument(
        "--user_id",
        type=str,
        default=BUILTIN_USER_ID,
        help="仅该用户的回放与画像；未指定 --output_dir 时输出到 outputs/<user_id>/<LoRA 子目录>。右键运行请改本文件 BUILTIN_USER_ID。",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="LoRA 输出；留空则多用户为 outputs/<LoRA 名>，有 --user_id 为 per_user_lora_output_dir。",
    )
    p.add_argument("--model", type=str, default=tm.BUILTIN_MODEL, help="HF 基座 ID")
    p.add_argument("--epochs", type=int, default=tm.BUILTIN_EPOCHS)
    p.add_argument("--lr", type=float, default=tm.BUILTIN_LR)
    p.add_argument("--batch_size", type=int, default=tm.BUILTIN_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=tm.BUILTIN_GRAD_ACCUM)
    p.add_argument("--max_seq_length", type=int, default=tm.BUILTIN_MAX_SEQ_LENGTH)
    p.add_argument("--history-ratio", type=float, default=BUILTIN_HISTORY_RATIO, help="相对历史池总 token 的抽样比例（默认 0.05）")
    p.add_argument("--seed", type=int, default=BUILTIN_RANDOM_SEED)
    p.add_argument("--no_4bit", action="store_true", default=tm.BUILTIN_NO_4BIT)
    p.add_argument("--fallback_model", action="store_true", default=tm.BUILTIN_FALLBACK_MODEL)
    p.add_argument("--hf-mirror", dest="hf_mirror", default=tm.BUILTIN_HF_MIRROR, action=argparse.BooleanOptionalAction)
    p.add_argument("--hf_endpoint", type=str, default=tm.BUILTIN_HF_ENDPOINT)
    p.add_argument("--early-stop-loss", type=float, default=tm.BUILTIN_EARLY_STOP_LOSS)
    p.add_argument("--no-early-stop-loss", action="store_true", default=False)
    p.add_argument("--early-stop-min-steps", type=int, default=tm.BUILTIN_EARLY_STOP_MIN_STEPS)
    p.add_argument("--no-memory-probe", action="store_true", default=False)
    p.add_argument("--memory-probe-samples", type=int, default=tm.BUILTIN_MEMORY_PROBE_SAMPLES)
    p.add_argument("--memory-probe-every", type=int, default=tm.BUILTIN_MEMORY_PROBE_EVERY_STEPS)
    p.add_argument(
        "--memory-probe-enable-loss",
        type=float,
        default=tm.BUILTIN_MEMORY_PROBE_ENABLE_LOSS,
        help="启用记忆探针的 loss 上限（与 --early-stop-loss 无关）",
    )
    p.add_argument(
        "--memory-probe-max-loss",
        type=float,
        default=None,
        help="兼容旧参数，语义同 --memory-probe-enable-loss；若指定则覆盖 --memory-probe-enable-loss",
    )
    args = p.parse_args()

    apply_hf_runtime_env(args)

    if not torch.cuda.is_available():
        _fail(
            "错误：未检测到 CUDA。QLoRA 微调通常需要 NVIDIA GPU。",
            code=2,
        )

    session_root = Path(args.session_root)
    train_uid = (args.user_id or "").strip()
    if (args.output_dir or "").strip():
        output_dir = Path((args.output_dir or "").strip())
    else:
        if train_uid:
            output_dir = per_user_lora_output_dir(_script_dir, train_uid)
        else:
            output_dir = Path(BUILTIN_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    _log("[train_memory_replay] 参数解析完成 …")
    _log(f"[train_memory_replay] session_root={session_root}")
    if train_uid:
        _log(f"[train_memory_replay] 单用户 user_id={train_uid!r} output_dir={output_dir}")
    else:
        _log(f"[train_memory_replay] output_dir={output_dir}")

    model_id = args.model
    if tm.BUILTIN_TRAIN_USE_LOCAL_HF:
        mp = Path(model_id)
        if not mp.is_dir() or not (mp / "config.json").is_file():
            _fail(
                "错误：本地训练模式未找到基座目录或缺少 config.json。\n"
                f"  当前 --model={model_id!r}\n"
                "  请将完整权重放入 hf_model 或设置 MEMORY_BASE_MODEL_PATH；"
                "或将 train_memory.py 顶部 BUILTIN_TRAIN_USE_LOCAL_HF 改为 False。",
                code=2,
            )
    tm._configure_hf_download_logging()
    tokenizer = load_tokenizer(model_id)
    try:
        model = load_base_model_causal_lm(model_id, use_4bit=not args.no_4bit)
    except Exception as e:
        if tm._looks_like_hf_connection_error(e):
            tm._log_hf_connection_troubleshooting(e)
        if args.fallback_model and model_id != FALLBACK_BASE_MODEL:
            _log(f"[train_memory_replay] 加载 {model_id} 失败，改用 {FALLBACK_BASE_MODEL}")
            model_id = FALLBACK_BASE_MODEL
            tokenizer = load_tokenizer(model_id)
            model = load_base_model_causal_lm(model_id, use_4bit=not args.no_4bit)
        else:
            raise

    if tm.BUILTIN_TRAIN_USE_LOCAL_HF and Path(model_id).is_dir():
        try:
            model.config._name_or_path = str(Path(model_id).resolve())
        except Exception:
            pass

    try:
        messages_rows, _ref_root, h_pool_tokens, n_raw_today = build_incremental_messages(
            session_root,
            tokenizer,
            rng=rng,
            history_ratio=args.history_ratio,
            user_id=train_uid or None,
        )
    except ValueError as e:
        _fail(str(e), code=1)

    qa_pairs = collect_qa_pairs_for_probe(
        session_root, train_user_id=train_uid or None
    )
    dataset = Dataset.from_dict({"messages": messages_rows})

    n_tok = sum(memory_training_messages_token_len(tokenizer, m) for m in messages_rows)
    _log(
        f"[train_memory_replay] session 根: {session_root}（各用户最新日 raw 条数合计={n_raw_today}）"
    )
    _log(
        f"[train_memory_replay] 历史池总 token≈{h_pool_tokens}，"
        f"按 {args.history_ratio:.4f} 抽样并入；合并后样本数={len(messages_rows)}，总 token≈{n_tok}"
    )

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    adapter_config = output_dir / "adapter_config.json"
    if adapter_config.is_file():
        _log("[train_memory_replay] 从已有适配器继续训练 …")
        try:
            model = PeftModel.from_pretrained(
                model, str(output_dir), is_trainable=True
            )
        except TypeError:
            model = PeftModel.from_pretrained(model, str(output_dir))
        _activate_loaded_peft_for_training(model)
        apply_peft_save_pretrained_embedding_layers_explicit(model)
    else:
        _log("[train_memory_replay] 未找到适配器，新注入 LoRA（可与 train_memory.py 首次训练衔接）…")
        lora_config = LoraConfig(
            r=tm.BUILTIN_LORA_R,
            lora_alpha=tm.BUILTIN_LORA_ALPHA,
            lora_dropout=tm.BUILTIN_LORA_DROPOUT,
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
    except ImportError:
        _fail("需要安装 trl：pip install trl", code=1)

    loss_thr = None if args.no_early_stop_loss else float(args.early_stop_loss)
    probe_on = not args.no_memory_probe and bool(qa_pairs)
    probe_enable_loss = _resolve_memory_probe_enable_loss(args)
    _log(
        f"[train_memory_replay] 早停：loss 阈值={'关闭' if loss_thr is None else loss_thr}；"
        f"记忆探针={'开' if probe_on else '关'}（仅当日目录 QA；"
        f"训练 loss≤{probe_enable_loss} 时才允许探针）"
    )

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
        # 作用：是否仅对 assistant 段算 loss。建议：全序列 False 与 train_memory 的 data 约定一致时保持 False。
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
    _log(f"[train_memory_replay] 完成。适配器已保存到: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        err_path = _script_dir / "outputs" / "train_memory_replay_last_error.txt"
        try:
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with err_path.open("w", encoding="utf-8") as ef:
                traceback.print_exc(file=ef)
        except OSError:
            err_path = None
        print("[train_memory_replay] 运行出错：", flush=True)
        traceback.print_exc()
        if err_path is not None:
            print(f"[train_memory_replay] 堆栈已写入: {err_path}", flush=True)
        sys.exit(1)
