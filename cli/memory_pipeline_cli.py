# -*- coding: utf-8 -*-
"""
流水线：默认本机 **REPL**（``memory_pipeline_core``，多轮 history 在进程内）。

连接 ``memory_pipeline_server`` 时：日常请 **右键运行** ``memory_pipeline_client.py``（改文件内 ``BUILTIN_*``）；
本文件保留 ``--client`` 等参数仅作可选高级用法。
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
import sys
import uuid
from datetime import datetime
from pathlib import Path

import memory_pipeline_core as mpc
import memory_user_paths as mem_paths

BUILTIN_CLIENT_PIPELINE_BASE = "http://127.0.0.1:8890"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="记忆流水线 CLI：本机 REPL；连 HTTP 服务请优先用 memory_pipeline_client.py 右键运行",
        epilog="服务端：右键运行 memory_pipeline_server.py\n"
        "连服务端：右键运行 memory_pipeline_client.py（改 BUILTIN_PIPELINE_BASE 等）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--client", action="store_true", default=False, help="作为客户端连接流水线 HTTP 服务")
    ap.add_argument(
        "--pipeline-base",
        type=str,
        default=BUILTIN_CLIENT_PIPELINE_BASE,
        help="--client 时：memory_pipeline_server 根地址",
    )
    ap.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="终答流式：--client 时写入请求 stream（需服务端 deepseek）；本机 REPL 时仅 deepseek 生效",
    )
    ap.add_argument(
        "--llm-backend",
        choices=("ollama", "deepseek"),
        default=mpc.BUILTIN_LLM_BACKEND,
    )
    ap.add_argument("--deepseek-api-base", type=str, default=mpc.BUILTIN_DEEPSEEK_API_BASE)
    ap.add_argument("--deepseek-api-key", type=str, default="")
    ap.add_argument("--ollama-host", type=str, default=mpc.BUILTIN_OLLAMA_HOST)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--memory-api-base", type=str, default=mpc.BUILTIN_MEMORY_API_BASE)
    ap.add_argument("--server-host", type=str, default="")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--ollama-think", action="store_true", default=False)
    args = ap.parse_args()

    if args.client:
        mpc.pipeline_client_chat_loop(
            args.pipeline_base,
            stream=args.stream,
            timeout=args.timeout,
        )
        return

    model = (args.model or "").strip()
    if not model:
        model = (
            mpc.BUILTIN_DEEPSEEK_MODEL
            if args.llm_backend == "deepseek"
            else mpc.BUILTIN_OLLAMA_MODEL
        )
    deepseek_key = (
        (args.deepseek_api_key or "").strip()
        or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        or (mpc.BUILTIN_DEEPSEEK_API_KEY or "").strip()
    )
    if args.llm_backend == "deepseek" and not deepseek_key:
        print(
            "错误：--llm-backend deepseek 需要 API Key，请设置环境变量 DEEPSEEK_API_KEY 或传入 --deepseek-api-key；"
            "若仅用本机 Ollama，请加参数 --llm-backend ollama",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    ollama_host, memory_api_base = mpc._resolve_endpoints(
        args.server_host,
        args.ollama_host,
        args.memory_api_base,
    )
    deepseek_base = args.deepseek_api_base.strip().rstrip("/")
    host = ollama_host.rstrip("/")
    api_base = memory_api_base.rstrip("/")

    if args.llm_backend == "deepseek":
        llm_line = f"DeepSeek API={deepseek_base!r} model={model!r}"
    else:
        llm_line = f"Ollama={host!r} model={model!r}"

    print(
        f"流水线就绪（本机 REPL，多轮 history 在内存）。后端={args.llm_backend!r} {llm_line} "
        f"记忆 API={api_base}/memory/extract\n"
        "每步日志：[1-router] [2-extract] [3-final]。输入 quit/exit 退出。\n"
        "连流水线 HTTP：右键运行 memory_pipeline_client.py；本机关闭流式可在运行配置中加 --no-stream。\n",
        flush=True,
    )

    uid_raw = (os.environ.get("MEMORY_USER_ID") or "").strip() or mem_paths.generate_new_user_id()
    uid_s = mem_paths.sanitize_path_user_id(uid_raw)
    staging_log: Path | None = None
    staging_speaker = (os.environ.get("MEMORY_DISPLAY_NAME") or "User").strip() or "User"
    if mpc.BUILTIN_SAVE_SESSION_STAGING:
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        staging_root = Path(mpc.BUILTIN_SESSION_STAGING_DIR)
        staging_log = mem_paths.session_staging_user_dir(staging_root, uid_s) / f"session_{session_id}.txt"
        staging_log.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"会话记录 user_id={uid_s} 展示名={staging_speaker!r}: {staging_log}",
            flush=True,
        )

    history: list[tuple[str, str]] = []
    session_memory_retrieval_blocks: list[str] = []
    session_profile_fks: set[str] = set()
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
            if args.llm_backend == "deepseek" and args.stream:
                print("\n【答复】（流式）", flush=True)
                parts: list[str] = []
                for piece in mpc.iter_run_one_round_final_stream(
                    user,
                    history=history,
                    llm_backend=args.llm_backend,
                    ollama_host=host,
                    ollama_model=model,
                    deepseek_api_base=deepseek_base,
                    deepseek_api_key=deepseek_key,
                    deepseek_model=model,
                    memory_api_base=api_base,
                    timeout_sec=args.timeout,
                    ollama_think=bool(args.ollama_think),
                    dump_llm_requests=True,
                    memory_user_id=uid_s,
                    session_profile_fks=session_profile_fks,
                    session_memory_retrieval_blocks=session_memory_retrieval_blocks,
                ):
                    parts.append(piece)
                    sys.stdout.write(piece)
                    sys.stdout.flush()
                print(flush=True)
                out = "".join(parts).strip()
            else:
                out = mpc.run_one_round(
                    user,
                    history=history,
                    llm_backend=args.llm_backend,
                    ollama_host=host,
                    ollama_model=model,
                    deepseek_api_base=deepseek_base,
                    deepseek_api_key=deepseek_key,
                    deepseek_model=model,
                    memory_api_base=api_base,
                    timeout_sec=args.timeout,
                    ollama_think=bool(args.ollama_think),
                    dump_llm_requests=True,
                    memory_user_id=uid_s,
                    session_profile_fks=session_profile_fks,
                    session_memory_retrieval_blocks=session_memory_retrieval_blocks,
                )
                print("\n【答复】\n" + out, flush=True)
                print(flush=True)
            history.append((user, out))
            if staging_log is not None:
                mpc.append_session_staging_turn(
                    staging_log,
                    user,
                    out,
                    turn_time=turn_time,
                    user_speaker_label=staging_speaker,
                )
        except Exception as e:
            print(f"运行出错: {e!r}", flush=True)


if __name__ == "__main__":
    main()
