# -*- coding: utf-8 -*-
"""
记忆提取 HTTP API 服务：单基座 + 多用户 LoRA 常驻显存；请求时若 ``user_id`` 与当前激活适配器不同才 ``set_adapter``，相同则跳过。

默认监听与基座、路径等见文首 **HF 与同段内置**；显式以 ``--base_model`` 等覆盖。训练产物：``<outputs_root>/<user_id>/<MEMORY_LORA_DIRNAME>/``（与 ``train_memory --user_id`` 一致），
或旧版平铺 ``<outputs_root>/<LoRA 名>/``（名 ``__legacy__``）；服务默认**不**预载该平铺 LoRA 以省显存（见 BUILTIN_LOAD_LEGACY_SHARED_LORA / ``--load-legacy-shared-lora``）。训练写平铺产物可照旧。

流程：用户输入 -> POST /memory/extract，body: ``{ "query": "…", "user_id": "…" }``（多适配器时 user_id 必填）；
-> 返回 ``{"has_memory": bool, "memory": str}``；合批为 ``{"results": [ 同左 n 条 ]}`` 与 ``queries`` 同序（不重复印 query，与请求按序对齐即可）。

日志：``BUILTIN_LOG_LLM_INPUT_FULL`` / ``--log-llm-input-full`` 为真时，每次抽取会打印送本地模型的 ``messages`` JSON、展平 ``prompt``、``tokenizer`` 张量 shape 及按行 decode 对拍（与 ``memory_utils`` 内实现一致）。
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
import socket
import threading
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# HF：必须在 import memory_utils（进而 import transformers）之前设置。
# ---------------------------------------------------------------------------
BUILTIN_HF_MIRROR = True
BUILTIN_HF_ENDPOINT = ""
BUILTIN_HF_DOWNLOAD_TIMEOUT_SEC = "1800"
BUILTIN_HF_DISABLE_XET = True
BUILTIN_HF_DISABLE_SYMLINKS_WARNING = True

# ---------------------------------------------------------------------------
# 服务与路径内置（与上方 HF 同区；不依赖 import memory_utils；与 memory_utils.DEFAULT_BASE_MODEL 同串）
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
BUILTIN_HUB_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BUILTIN_BASE_MODEL = BUILTIN_HUB_BASE_MODEL
BUILTIN_OUTPUTS_ROOT = str(_script_dir / "outputs")
BUILTIN_LOCAL_TOKENIZER_DIR = str(_script_dir / "hf_tokenizer")
BUILTIN_HOST = "127.0.0.1"
BUILTIN_PORT = 8765
BUILTIN_NO_4BIT = False
# 服务启动时是否将平铺于 ``outputs/<LoRA 子目录名>/`` 的全局共享 LoRA（PEFT 名 ``__legacy__``）与多用户目录一并预载进显存。False=只预载 ``outputs/<user_id>/<LoRA 子目录>/``，省显存。train_memory 写全局/平铺产物可照旧，与本开关无关。
BUILTIN_LOAD_LEGACY_SHARED_LORA = False
# 是否在每次记忆抽取时打印「送入本地 CausalLM 前」的完整 messages 结构、展平 prompt、及 tokenizer 后张量形状与 decode 对拍（与 memory_utils 内单条/合批一致）。日志量大时可 --no-log-llm-input-full 关闭。
BUILTIN_LOG_LLM_INPUT_FULL = True
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

# bitsandbytes + 新版 PyTorch 在推理时可能刷 FutureWarning（_check_is_size），与业务无关
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*_check_is_size.*",
)

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from peft import PeftModel
from pydantic import BaseModel, Field

from memory_user_paths import sanitize_path_user_id

from memory_utils import (
    MEMORY_EXTRACT_MAX_NEW_TOKENS,
    build_memory_extract_messages_and_prompts_for_queries,
    get_memory_lora_dirname,
    clean_memory_generation_output,
    generate_extract_completion,
    generate_extract_completions_batched,
    load_base_model_causal_lm,
    load_tokenizer_for_inference,
    memory_extract_gate_by_query,
    sanitize_memory_extract_output,
)

_model = None
_tokenizer = None
_device = None
_outputs_root: Path | None = None
_lora_basename: str = ""
_load_lock = threading.Lock()
_adapter_paths: dict[str, Path] = {}
_active_adapter_id: str | None = None


def _discover_adapter_dirs(
    outputs_root: Path, lora_name: str
) -> list[tuple[str, Path]]:
    """
    返回 (adapter_name, 含 adapter_config.json 的目录)。
    支持平铺：``outputs/<lora_name>/`` → 名 ``__legacy__``；
    与每用户：``outputs/<user_id>/<lora_name>/``。
    """
    out: list[tuple[str, Path]] = []
    leg = outputs_root / lora_name
    if (leg / "adapter_config.json").is_file():
        out.append(("__legacy__", leg))
    if outputs_root.is_dir():
        for child in sorted(outputs_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name == lora_name:
                continue
            ap = child / lora_name
            if (ap / "adapter_config.json").is_file():
                out.append((child.name, ap))
    return out


def _sort_adapter_entries(entries: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    return sorted(
        entries,
        key=lambda x: (0 if x[0] == "__legacy__" else 1, x[0]),
    )


class ExtractRequest(BaseModel):
    query: str = Field(..., description="当前用户问题或检索意图")
    user_id: str = Field(
        default="",
        description="与 session/<user_id> 及 train_memory --user_id 一致；多适配器时须传；与 LoRA 目录名经 sanitize 后相同",
    )


class ExtractBatchRequest(BaseModel):
    queries: list[str] = Field(
        ..., min_length=1, description="检索问句列表（如路由条 memory_queries），一次前向合批，降低多轮串行 RT"
    )
    user_id: str = Field(
        default="",
        description="与单条 /memory/extract 的 user_id 含义相同",
    )


class ExtractResponse(BaseModel):
    has_memory: bool = Field(..., description="为 true 表示检索到与 query 相关的有效记忆")
    memory: str = Field("", description="记忆文本；无记忆时为空串")


class ExtractBatchResponse(BaseModel):
    results: list[ExtractResponse] = Field(
        default_factory=list, description="与请求 queries 同序、等长"
    )


def _log_api(msg: str) -> None:
    ts = datetime.now().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _build_local_model_input_log_payload(
    tokenizer,
    rows: list[tuple[list[dict], str]],
    enc: dict,
) -> dict:
    # 与 memory_utils 中分词+padding 一致；仅构造 dict，由调用方与 HTTP 体合并为一条 入参 日志
    n = len(rows)
    systems: list[str] = []
    users: list[str] = []
    for messages, _p in rows:
        s, u = "", ""
        if messages and messages[0].get("role") == "system":
            s = str(messages[0].get("content") or "")
        if len(messages) > 1 and messages[1].get("role") == "user":
            u = str(messages[1].get("content") or "")
        systems.append(s)
        users.append(u)
    shapes: dict = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            shapes[k] = [int(x) for x in v.shape]
    gen = {
        "do_sample": False,
        "max_new_tokens_max": MEMORY_EXTRACT_MAX_NEW_TOKENS,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if n > 1 and len(set(systems)) == 1 and (systems[0] or "") != "":
        return {"system": systems[0], "users": users, "enc": shapes, "gen": gen}
    return {
        "prompts": [p for _m, p in rows],
        "enc": shapes,
        "gen": gen,
    }


def _guess_lan_ipv4() -> str | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def _print_startup_listen_info(host: str, port: int) -> None:
    _log_api("======== memory_api_server 监听与访问路径 ========")
    _log_api(f"uvicorn 绑定：host={host!r}  port={port}")
    local_base = f"http://127.0.0.1:{port}"
    _log_api(
        f"本地访问（本机）：POST {local_base}/memory/extract  "
        f"POST {local_base}/memory/extract_batch    GET {local_base}/health"
    )

    if host in ("0.0.0.0", "::", "[::]"):
        lan = _guess_lan_ipv4()
        if lan:
            lan_base = f"http://{lan}:{port}"
            _log_api(f"局域网访问（示例，探测到本机 IPv4={lan}）：POST {lan_base}/memory/extract    GET {lan_base}/health")
        else:
            _log_api(
                "局域网访问：将 <本机局域网IP> 换成本机实际地址，"
                f"例如 POST http://<本机局域网IP>:{port}/memory/extract"
            )
        _log_api(
            "外网访问：依赖路由器端口映射 / 云安全组，将公网端口转到本机 "
            f"{port}；示例 POST http://<公网IP或域名>:<映射端口>/memory/extract"
        )
    else:
        bound = f"http://{host}:{port}"
        _log_api(f"当前仅绑定 {host}（非 0.0.0.0），其它机器请用：POST {bound}/memory/extract    GET {bound}/health")
    _log_api("==========================================")


def _resolve_local_tokenizer_dir() -> str | None:
    env = os.environ.get("MEMORY_TOKENIZER_PATH", "").strip()
    if env:
        return env
    return BUILTIN_LOCAL_TOKENIZER_DIR


def _load_for_inference(
    base_model_id: str,
    outputs_root: Path,
    lora_basename: str,
    use_4bit: bool,
    *,
    load_legacy_shared: bool = False,
) -> None:
    global _model, _tokenizer, _device, _outputs_root, _lora_basename, _adapter_paths, _active_adapter_id
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _outputs_root = outputs_root
    _lora_basename = lora_basename
    _tokenizer = load_tokenizer_for_inference(
        base_model_id,
        local_tokenizer_dir=_resolve_local_tokenizer_dir(),
    )
    base = load_base_model_causal_lm(base_model_id, use_4bit=use_4bit)
    entries = _sort_adapter_entries(_discover_adapter_dirs(outputs_root, lora_basename))
    if not load_legacy_shared:
        entries = [e for e in entries if e[0] != "__legacy__"]
    if not entries:
        raise FileNotFoundError(
            f"在 {outputs_root!r} 下未找到可预载的 LoRA 目录"
            f"（{outputs_root}/<user_id>/{lora_basename}）。"
            f"若仅有平铺目录 {outputs_root / lora_basename!s} 且需加载，"
            f"请设文首 BUILTIN_LOAD_LEGACY_SHARED_LORA=True 或命令行 --load-legacy-shared-lora；"
            f"并确认已 train_memory 或 MEMORY_LORA_DIRNAME 与 get_memory_lora_dirname 一致。"
        )
    _adapter_paths = {k: v for k, v in entries}
    first_name, first_path = entries[0]
    _model = PeftModel.from_pretrained(
        base, str(first_path), adapter_name=first_name
    )
    for aname, apath in entries[1:]:
        _model.load_adapter(str(apath), adapter_name=aname)
    _model.set_adapter(first_name)
    _active_adapter_id = first_name
    _model.eval()
    if _device == "cpu" and not use_4bit:
        _model = _model.to(_device)
    for aname, apath in entries:
        _log_api(
            f"已常驻 LoRA：adapter_name={aname!r}  path={apath.resolve()!s}  "
            f"base={base_model_id!r}  use_4bit={use_4bit}  切换=set_adapter（同基座、多路 LoRA 在显存）"
        )
    if not load_legacy_shared:
        leg = outputs_root / lora_basename
        if (leg / "adapter_config.json").is_file():
            _log_api(
                "已跳过平铺全局 LoRA（__legacy__）预载；见 BUILTIN_LOAD_LEGACY_SHARED_LORA / --load-legacy-shared-lora"
            )
    _log_api(
        f"共预载 {len(entries)} 套适配器；请求 POST 体可带 user_id 与之一致（多适配器时必填）"
    )


def _resolve_extract_adapter_id(req: "ExtractRequest") -> str | None:
    n = (req.user_id or "").strip()
    if n:
        return sanitize_path_user_id(n)
    if len(_adapter_paths) == 1:
        return next(iter(_adapter_paths.keys()))
    return None


def _ensure_adapter_in_memory(adapter_id: str) -> str | None:
    """磁盘上存在则返回 adapter_id 供 load_adapter；已常驻则同 id。否则 None。"""
    if adapter_id in _adapter_paths:
        return adapter_id
    if not _outputs_root or not _lora_basename:
        return None
    ap = _outputs_root / adapter_id / _lora_basename
    if (ap / "adapter_config.json").is_file():
        return adapter_id
    return None


def create_app(
    base_model_id: str,
    outputs_root: Path,
    lora_basename: str,
    use_4bit: bool,
    *,
    load_legacy_shared: bool = False,
    log_llm_input_full: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _load_for_inference(
            base_model_id,
            outputs_root,
            lora_basename,
            use_4bit,
            load_legacy_shared=load_legacy_shared,
        )
        yield

    application = FastAPI(title="Memory Extract API", lifespan=lifespan)

    @application.post("/memory/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest):
        global _active_adapter_id
        if not req.query or not req.query.strip():
            _log_api("memory/extract 出参 HTTP 400 query 空")
            return JSONResponse(status_code=400, content={"detail": "query 不能为空"})
        q = req.query.strip()
        aid0 = _resolve_extract_adapter_id(req)
        if aid0 is None:
            _log_api("memory/extract 出参 HTTP 400 缺 user_id")
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "已加载多用户 LoRA，请求 JSON 须包含 user_id 与其中一套目录名一致"
                },
            )
        with _load_lock:
            aid = aid0
            prev = _active_adapter_id
            if aid not in _adapter_paths:
                new_id = _ensure_adapter_in_memory(aid0)
                if not new_id:
                    return JSONResponse(
                        status_code=404,
                        content={"detail": f"未找到该用户的 LoRA：outputs/{aid0}/{_lora_basename}/"},
                    )
                ap = _outputs_root / new_id / _lora_basename
                assert _model is not None
                t_load0 = time.perf_counter()
                _model.load_adapter(str(ap), adapter_name=new_id)
                t_load_ms = (time.perf_counter() - t_load0) * 1000.0
                _adapter_paths[new_id] = ap
                aid = new_id
                _log_api(
                    f"LoRA 懒加载 load_adapter: name={new_id!r} path={ap.resolve()!s}  耗时 {t_load_ms:.2f} ms"
                )
            if prev != aid:
                assert _model is not None
                t_set0 = time.perf_counter()
                _model.set_adapter(aid)
                t_set_ms = (time.perf_counter() - t_set0) * 1000.0
                if prev is None:
                    _log_api(
                        f"LoRA 激活: adapter={aid!r}  set_adapter 耗时 {t_set_ms:.2f} ms"
                    )
                else:
                    _log_api(
                        f"LoRA 切换: {prev!r} -> {aid!r}  set_adapter 耗时 {t_set_ms:.2f} ms"
                    )
            else:
                _log_api(
                    f"LoRA 未切换: 仍为 adapter={aid!r}（已跳过 set_adapter）"
                )
            _active_adapter_id = aid
            assert _tokenizer is not None
            rows_one = build_memory_extract_messages_and_prompts_for_queries(
                [q], _tokenizer
            )
            enc_cpu = _tokenizer(
                rows_one[0][1],
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            )
            inp1: dict = {"query": q, "adapter": aid}
            if log_llm_input_full:
                inp1["llm"] = _build_local_model_input_log_payload(
                    _tokenizer, rows_one, enc_cpu
                )
            _log_api(
                f"memory/extract 入参 {json.dumps(inp1, ensure_ascii=False)}"
            )
            if _device == "cuda":
                inputs = {k: v.cuda() for k, v in enc_cpu.items()}
            else:
                inputs = enc_cpu
            raw = generate_extract_completion(_model, _tokenizer, inputs, 512)
            text = sanitize_memory_extract_output(clean_memory_generation_output(raw))
            text = memory_extract_gate_by_query(text, q)
        has_mem = bool(text.strip())
        out_mem = text if has_mem else ""
        _log_api(
            f"memory/extract 出参 {json.dumps({'has_memory': has_mem, 'memory': out_mem}, ensure_ascii=False)}"
        )
        return ExtractResponse(has_memory=has_mem, memory=out_mem)

    @application.post(
        "/memory/extract_batch",
        response_model=ExtractBatchResponse,
    )
    def extract_batch(req: ExtractBatchRequest):
        global _active_adapter_id
        if not req.queries:
            _log_api("memory/extract_batch 出参 HTTP 400 queries 空")
            return JSONResponse(
                status_code=400, content={"detail": "queries 不能为空列表"}
            )
        q_list = [str(x) for x in req.queries]
        uid_s = (req.user_id or "").strip()
        aid0 = _resolve_extract_adapter_id(
            ExtractRequest(query=req.queries[0] or " ", user_id=req.user_id)
        )
        if aid0 is None:
            _log_api(
                f"memory/extract_batch 出参 HTTP 400 缺 user_id 入参 {json.dumps({'user_id': uid_s, 'queries': q_list}, ensure_ascii=False)}"
            )
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "已加载多用户 LoRA，请求 JSON 须包含 user_id 与其中一套目录名一致"
                },
            )
        with _load_lock:
            aid = aid0
            prev = _active_adapter_id
            if aid not in _adapter_paths:
                new_id = _ensure_adapter_in_memory(aid0)
                if not new_id:
                    _log_api(
                        f"memory/extract_batch 出参 HTTP 404 {json.dumps({'adapter': aid0, 'queries': q_list}, ensure_ascii=False)}"
                    )
                    return JSONResponse(
                        status_code=404,
                        content={
                            "detail": f"未找到该用户的 LoRA：outputs/{aid0}/{_lora_basename}/"
                        },
                    )
                ap = _outputs_root / new_id / _lora_basename
                assert _model is not None
                t_load0 = time.perf_counter()
                _model.load_adapter(str(ap), adapter_name=new_id)
                t_load_ms = (time.perf_counter() - t_load0) * 1000.0
                _adapter_paths[new_id] = ap
                aid = new_id
                _log_api(
                    f"LoRA 懒加载 load_adapter: name={new_id!r} path={ap.resolve()!s}  耗时 {t_load_ms:.2f} ms"
                )
            if prev != aid:
                assert _model is not None
                t_set0 = time.perf_counter()
                _model.set_adapter(aid)
                t_set_ms = (time.perf_counter() - t_set0) * 1000.0
                if prev is None:
                    _log_api(
                        f"LoRA 激活: adapter={aid!r}  set_adapter 耗时 {t_set_ms:.2f} ms"
                    )
                else:
                    _log_api(
                        f"LoRA 切换: {prev!r} -> {aid!r}  set_adapter 耗时 {t_set_ms:.2f} ms"
                    )
            else:
                _log_api(
                    f"LoRA 未切换: 仍为 adapter={aid!r}（已跳过 set_adapter）"
                )
            _active_adapter_id = aid
            assert _model is not None and _tokenizer is not None
            rows: list | None = None
            enc_cpu = None
            if log_llm_input_full:
                rows = build_memory_extract_messages_and_prompts_for_queries(
                    q_list, _tokenizer
                )
                psl = [p for _, p in rows]
                pad_side = getattr(_tokenizer, "padding_side", "right")
                if pad_side != "left":
                    _tokenizer.padding_side = "left"
                try:
                    enc_cpu = _tokenizer(
                        psl,
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                        add_special_tokens=False,
                    )
                finally:
                    _tokenizer.padding_side = pad_side
            batch_in: dict = {
                "user_id": uid_s,
                "adapter": aid,
                "queries": q_list,
            }
            if log_llm_input_full and rows is not None and enc_cpu is not None:
                batch_in["llm"] = _build_local_model_input_log_payload(
                    _tokenizer, rows, enc_cpu
                )
            _log_api(
                f"memory/extract_batch 入参 {json.dumps(batch_in, ensure_ascii=False)}"
            )
            if log_llm_input_full and enc_cpu is not None:
                raws = generate_extract_completions_batched(
                    _model, _tokenizer, q_list, 512, prebuilt_enc=enc_cpu
                )
            else:
                raws = generate_extract_completions_batched(
                    _model, _tokenizer, q_list, 512
                )
            results: list[ExtractResponse] = []
            for q, raw in zip(q_list, raws):
                text = sanitize_memory_extract_output(
                    clean_memory_generation_output(raw)
                )
                text = memory_extract_gate_by_query(text, q)
                has_m = bool(text.strip())
                mem_out = text if has_m else ""
                r_item = ExtractResponse(has_memory=has_m, memory=mem_out)
                results.append(r_item)
        _log_api(
            f"memory/extract_batch 出参 {json.dumps(ExtractBatchResponse(results=results).model_dump(), ensure_ascii=False)}"
        )
        return ExtractBatchResponse(results=results)

    @application.get("/health")
    def health():
        return {
            "status": "ok",
            "adapters": sorted(_adapter_paths.keys()),
            "outputs_root": str(_outputs_root) if _outputs_root else None,
        }

    return application


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=BUILTIN_BASE_MODEL)
    parser.add_argument(
        "--outputs_root",
        type=str,
        default=BUILTIN_OUTPUTS_ROOT,
        help="训练产物根：其下为 <user_id>/<LoRA 子目录> 或平铺 <LoRA 子目录>（与 MEMORY_LORA_DIRNAME / 默认名一致）",
    )
    parser.add_argument("--host", type=str, default=BUILTIN_HOST)
    parser.add_argument("--port", type=int, default=BUILTIN_PORT)
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        default=BUILTIN_NO_4BIT,
        help="禁用 4bit；内置默认见 BUILTIN_NO_4BIT",
    )
    parser.add_argument(
        "--load-legacy-shared-lora",
        action=argparse.BooleanOptionalAction,
        default=BUILTIN_LOAD_LEGACY_SHARED_LORA,
        help="是否预载平铺于 outputs/<LoRA 子目录> 的全局 __legacy__ 适配器；默认 False 省显存。见 BUILTIN_LOAD_LEGACY_SHARED_LORA",
    )
    parser.add_argument(
        "--log-llm-input-full",
        action=argparse.BooleanOptionalAction,
        default=BUILTIN_LOG_LLM_INPUT_FULL,
        help="是否每次抽取打印送本地 CausalLM 的 messages、prompt 与 tokenized 对拍。见 BUILTIN_LOG_LLM_INPUT_FULL；可用 --no-log-llm-input-full 关",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    lora_basename = get_memory_lora_dirname()

    _print_startup_listen_info(args.host, args.port)

    import uvicorn

    uvicorn.run(
        create_app(
            args.base_model,
            outputs_root,
            lora_basename,
            use_4bit=not args.no_4bit,
            load_legacy_shared=args.load_legacy_shared_lora,
            log_llm_input_full=args.log_llm_input_full,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
