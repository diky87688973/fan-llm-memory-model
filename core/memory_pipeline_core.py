# -*- coding: utf-8 -*-
"""记忆流水线编排核心：路由 → memory_api_server → 终答；多轮 ``history`` 以 OpenAI messages 传入模型。"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

from session_to_memory import (
    _clean_model_text,
    deepseek_chat_messages,
    iter_deepseek_chat_messages_stream,
    ollama_chat_messages,
)

# ---------------------------------------------------------------------------
# 与 memory_pipeline_cli / memory_pipeline_server / memory_pipeline_client 共用内置默认
# ---------------------------------------------------------------------------
BUILTIN_LLM_BACKEND = "deepseek"  # "ollama" | "deepseek"
BUILTIN_OLLAMA_HOST = "http://127.0.0.1:11434"
BUILTIN_OLLAMA_MODEL = "deepseek-r1:8b"
BUILTIN_DEEPSEEK_API_BASE = "https://api.deepseek.com"
BUILTIN_DEEPSEEK_API_KEY = "sk-REPLACE_WITH_YOUR_KEY"  # 生产请改用环境变量，勿提交真实 Key
BUILTIN_DEEPSEEK_MODEL = "deepseek-v4-flash"  # 非推理默认；推理用 deepseek-reasoner 等并传 --deepseek-model
BUILTIN_DEEPSEEK_ROUTER_MAX_TOKENS = 4096
BUILTIN_DEEPSEEK_FINAL_MAX_TOKENS = 2048
BUILTIN_DEEPSEEK_ROUTER_TEMP = 0.2
BUILTIN_DEEPSEEK_FINAL_TEMP = 0.4
BUILTIN_MEMORY_API_BASE = "http://127.0.0.1:8765"
BUILTIN_OLLAMA_OPTIONS_ROUTER = {"num_predict": 4096, "temperature": 0.2}
# need_memory=true 时路由须生成的 memory_queries 条数 = 下游记忆 API 调用次数（与 SYSTEM_ROUTER 一致，不截断丢弃）
BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N = 5
BUILTIN_OLLAMA_OPTIONS_FINAL = {"num_predict": 2048, "temperature": 0.4}
BUILTIN_SAVE_SESSION_STAGING = True
BUILTIN_SESSION_STAGING_DIR = str(Path(__file__).resolve().parent / "session_staging")


def _resolve_endpoints(
    server_host: str,
    ollama_host: str,
    memory_api_base: str,
) -> tuple[str, str]:
    oo = ollama_host.strip().rstrip("/")
    mm = memory_api_base.strip().rstrip("/")
    sh = (server_host or "").strip()
    if not sh:
        return oo, mm
    if sh.startswith("http://") or sh.startswith("https://"):
        from urllib.parse import urlparse

        u = urlparse(sh)
        host_only = u.hostname or "127.0.0.1"
    else:
        host_only = sh.split(":")[0].split("/")[0]
    if oo == BUILTIN_OLLAMA_HOST:
        oo = f"http://{host_only}:11434"
    if mm == BUILTIN_MEMORY_API_BASE:
        mm = f"http://{host_only}:8765"
    return oo, mm


SYSTEM_ROUTER = f"""你是记忆检索路由模块：须**只**通过 API Tool 调用 `submit_memory_retrieval_route` 提交 need_memory 与 memory_queries（与 DeepSeek 官方 Tool Calls 一致）；**不要**在 assistant 的 content 里写面向用户的长段答句（可留空）。arguments 为 JSON 对象，键 need_memory、memory_queries。

【硬性约束】
- 禁止回答用户当前句子里的具体问题；禁止「根据你的记忆…」「你可以…」等对话式正文。你是在通过 function 交配置，不是在与用户聊天。
- 传参须可被 json 解析，前后不得有 markdown 围栏或多余说明性长文顶替代。

若 need_memory 为 false：memory_queries 必须为 []。

何时 need_memory 为 true：只要回答用户这句话时，有可能因已存储的个人信息而答得更准或更可个性化，就必须为 true；由你自行判断，禁止因本说明未描述某类场景就判 false；禁止在后续答复里向用户追问其个人信息却不在此先检索记忆。

若 need_memory 为 true：memory_queries 必须恰好 {BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N} 条互不重复、语义可区分的完整中文问句（下游逐条检索）；按与当前问题及可能命中记忆的相关性从高到低排序。须由你自行分析用户问题并设计问句角度；禁止照抄本段模板句。
每条须与当前用户问题直接相关；禁止为凑满条数而编造与回答无关的假设性追问。每条须为完整问句；禁止仅用「用户的xxx」这类名词短语。语义重复或近义只保留一条。

**submit_memory_retrieval_route 的 arguments 示例**（true/false 为占位，勿照抄问句）：
{{"need_memory": true, "memory_queries": ["问句1", "问句2"]}}
"""

ROUTER_JSON_RETRY_USER = (
    "上一条输出不符合约定：请仅输出一个 JSON 对象，键为 need_memory、memory_queries；"
    "不要 markdown、不要解释、不要回答用户原话里的具体问题。"
)

ROUTER_DS_TOOL_RETRY_USER = (
    "上一条未正确调用 submit_memory_retrieval_route：请仅通过该 function 再提交合法 arguments，键为 need_memory、memory_queries。"
)

SYSTEM_FINAL = """你是智能助手。用户消息里有两块：【已检索记忆】与【用户原话】。

【已检索记忆】中，每条有效记忆为两行：`[记忆检索]：` 与 `[记忆事实]：`（多段之间空行分隔）。
- **`[记忆事实]：` 之后为 AI 记忆库中的可引用内容（记录者视角的「我」，**用户侧信息**须与「用户告诉我…」「我了解到用户…」等一致；**禁止**把用户偏好误读为模型自己的「我喜欢…」。**不是**用户本人脑内独白）。
- **`[记忆检索]：` 之后仅为系统为检索而生成的问句（路由下发给记忆 API 的用语），不是用户原话，不是已证实经历；可能含未经验证假设。禁止把某条 [记忆检索] 里的具体说法写进答案，除非同一段对应的 [记忆事实] 里也有同样信息。
- 综合答复时：以 [记忆事实] 为准；[记忆检索] 仅作语义索引，避免孤立短词产生歧义。

硬性规则：
1）若【已检索记忆】不是「（无）」且含有实质文字，你必须以各条 [记忆事实] 后的内容为依据作答，不得再说「没有相关信息」「无法回答」「缺少上下文」等推脱语。
2）用户问及过往讨论、聊过内容时：请根据【已检索记忆】里 [记忆事实] 与主题概括；记忆很短也可作简要关联说明。
3）仅当【已检索记忆】为「（无）」或确实与问题无关时，才可简要说明记忆里没有这点。
4）用自然、简洁的中文直接回答，不要复述本说明。
5）禁止替用户做主：记忆只是辅助依据。若 [记忆事实] 仅表明对某类选项的接受、容忍或中立，且未排除其它选项，禁止据此把建议收窄为只推该类、或表现得像用户已选定该类；若信息不足以唯一确定偏好，应并列合理选项或说明由用户自行决定。
6）**自称与句首格式**：以第一人称「我」回答即可。介绍自己、身份或版本时**直接**用「我是…」等，**不要**在回答**最开头**写「[助手名/昵称] + 逗号 + 我…」（如「小忆，我是…」），以免用户误解为你在**称呼他**。若需带出助手名，可写在「我是」之后或主句中（如「我是小忆，是记忆模型 v1.0…」）。"""


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


def _emit_think(step: str, msg: str) -> dict:
    """同一条编排日志：``log`` 写控制台，``think.text`` 为 ``[{step}] {msg}`` 给 SSE 页面，内容与控制台一致（不压成单行）。"""
    log(step, msg)
    phase = "final" if step.startswith("3-") else ("memory" if step.startswith("2-") else "router")
    return {"kind": "think", "phase": phase, "text": f"[{step}] {msg}"}


def _step_break() -> None:
    print("", flush=True)


def _router_messages(history: list[tuple[str, str]], user_text: str) -> list[dict]:
    m: list[dict] = [{"role": "system", "content": SYSTEM_ROUTER}]
    for u, a in history:
        m.append({"role": "user", "content": u})
        m.append({"role": "assistant", "content": a})
    m.append({"role": "user", "content": user_text})
    return m


def _final_stage_think_message(memory_text: str) -> str:
    """终答前一条 think：仅当合并后的检索结果含实质记忆时，才用「结合记忆」类表述。"""
    if not (memory_text or "").strip():
        return "未命中可用记忆，正在直接生成回答…"
    return "已检索到记忆事实，正在据此生成回答…"


def _final_messages(history: list[tuple[str, str]], memory_text: str, user_text: str) -> list[dict]:
    block = (
        f"【已检索记忆】\n{memory_text if memory_text.strip() else '（无）'}\n\n"
        f"【用户原话】\n{user_text}"
    )
    m: list[dict] = [{"role": "system", "content": SYSTEM_FINAL}]
    for u, a in history:
        m.append({"role": "user", "content": u})
        m.append({"role": "assistant", "content": a})
    m.append({"role": "user", "content": block})
    return m


def _parse_router_json(text: str) -> tuple[bool, list[str]]:
    t = _clean_model_text(text)

    def _pull(obj: dict) -> tuple[bool, list[str]] | None:
        if not isinstance(obj, dict):
            return None
        need = bool(obj.get("need_memory"))
        raw_mq = obj.get("memory_queries")
        out: list[str] = []
        if isinstance(raw_mq, list):
            for x in raw_mq:
                s = str(x).strip()
                if s:
                    out.append(s)
        if not out:
            eq = str(obj.get("extract_query") or "").strip()
            if eq:
                out.append(eq)
        seen: set[str] = set()
        dedup: list[str] = []
        for q in out:
            if q not in seen:
                seen.add(q)
                dedup.append(q)
        return need, dedup

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            r = _pull(obj)
            if r is not None:
                return r
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(t[start : end + 1])
            if isinstance(obj, dict):
                r = _pull(obj)
                if r is not None:
                    return r
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法从路由模型输出解析 JSON：{text!r}")


def _deepseek_tool_memory_router() -> dict:
    n = BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N
    return {
        "type": "function",
        "function": {
            "name": "submit_memory_retrieval_route",
            "description": "提交本回合是否检索及 memory_queries（仅通过 arguments 传参）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "need_memory": {"type": "boolean"},
                    "memory_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"need_memory 为 false 时 []；为 true 时须 {n} 条中文问句。",
                    },
                },
                "required": ["need_memory", "memory_queries"],
            },
        },
    }


def _router_llm_once(
    messages: list[dict],
    *,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    timeout_sec: int,
    ollama_think: bool,
    dump_llm_requests: bool,
) -> str:
    """DeepSeek：官方 Tool Calls + thinking 关闭；Ollama：``format=json``。"""
    if llm_backend == "deepseek":
        return deepseek_chat_messages(
            deepseek_api_base,
            deepseek_api_key,
            deepseek_model,
            messages,
            timeout_sec=timeout_sec,
            max_tokens=BUILTIN_DEEPSEEK_ROUTER_MAX_TOKENS,
            temperature=BUILTIN_DEEPSEEK_ROUTER_TEMP,
            dump_raw_request_body=dump_llm_requests,
            tools=[_deepseek_tool_memory_router()],
            tool_choice="required",
        )
    return ollama_chat_messages(
        ollama_host,
        ollama_model,
        messages,
        timeout_sec=timeout_sec,
        think=ollama_think,
        options=BUILTIN_OLLAMA_OPTIONS_ROUTER,
        dump_raw_request_body=dump_llm_requests,
        response_format_json=True,
    )


def _execute_router_with_json_enforcement(
    rmsgs: list[dict],
    *,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    timeout_sec: int,
    ollama_think: bool,
    dump_llm_requests: bool,
    emit: Callable[[str, str], None] | None = None,
) -> tuple[str, bool, list[str]]:
    """在 API 层强制 JSON（见 ``_router_llm_once``）；解析失败则追加一条 user 重试一次；仍失败则抛 ``RuntimeError``。"""
    em = emit or log
    raw = _router_llm_once(
        rmsgs,
        llm_backend=llm_backend,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        deepseek_api_base=deepseek_api_base,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=deepseek_model,
        timeout_sec=timeout_sec,
        ollama_think=ollama_think,
        dump_llm_requests=dump_llm_requests,
    )
    em("1-router", f"路由返回：{raw}")
    try:
        need, qs = _parse_router_json(raw)
        return raw, need, qs
    except ValueError:
        em("1-router", "路由输出无法解析为合法 JSON，正在重试一次（仍须仅输出 JSON）…")
        _retry = (
            ROUTER_DS_TOOL_RETRY_USER
            if llm_backend == "deepseek"
            else ROUTER_JSON_RETRY_USER
        )
        raw2 = _router_llm_once(
            rmsgs + [{"role": "user", "content": _retry}],
            llm_backend=llm_backend,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            deepseek_api_base=deepseek_api_base,
            deepseek_api_key=deepseek_api_key,
            deepseek_model=deepseek_model,
            timeout_sec=timeout_sec,
            ollama_think=ollama_think,
            dump_llm_requests=dump_llm_requests,
        )
        em("1-router", f"路由重试返回：{raw2}")
        try:
            need2, qs2 = _parse_router_json(raw2)
            return raw2, need2, qs2
        except ValueError as e:
            snippet = (raw2 or "").strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "…(截断)"
            raise RuntimeError(
                "路由模型在 API JSON 模式与一次重试后仍返回无法解析的内容（不接受自然语言顶替）。"
                f"末次输出：{snippet!r}"
            ) from e


def http_memory_extract(
    api_base: str, query: str, timeout_sec: int, memory_user_id: str = ""
) -> tuple[bool, str, str]:
    url = api_base.rstrip("/") + "/memory/extract"
    body: dict = {"query": query}
    uid = (memory_user_id or "").strip()
    if uid:
        body["user_id"] = uid
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"记忆 API HTTP {e.code} {url}\n{body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接记忆 API ({url})。请先在本机运行 memory_api_server.py。\n{e}"
        ) from e
    data = json.loads(raw)
    mem = str(data.get("memory") or "").strip()
    if "has_memory" in data:
        has_mem = bool(data.get("has_memory")) and bool(mem)
    else:
        has_mem = bool(mem)
    if not has_mem:
        mem = ""
    return has_mem, mem, raw


def _iter_memory_extract_gen(
    need_memory: bool,
    memory_queries: list[str],
    memory_api_base: str,
    timeout_sec: int,
) -> Iterator[dict[str, str]]:
    """与 ``_run_memory_extracts`` 同一逻辑；yield 的 ``think.text`` 与控制台 ``log`` 一致，结束时 ``return memory_text``。"""
    ext_url = memory_api_base.rstrip("/") + "/memory/extract"
    memory_text = ""
    if need_memory and memory_queries:
        parts: list[str] = []
        seen_mem_norm: set[str] = set()
        for i, q in enumerate(memory_queries, start=1):
            has_mem, mem, raw_http = http_memory_extract(memory_api_base, q, timeout_sec)
            msg_http = f"[{i}/{len(memory_queries)}] POST {ext_url}  query={q!r}  响应={raw_http}"
            yield _emit_think("2-extract", msg_http)
            msg_mem = f"[{i}/{len(memory_queries)}] has_memory={has_mem}  [记忆事实]：{(mem if has_mem and mem else '（无）')}"
            yield _emit_think("2-extract", msg_mem)
            if has_mem and mem:
                m = mem.strip()
                if m in seen_mem_norm:
                    continue
                seen_mem_norm.add(m)
                parts.append(f"[记忆检索]：{q}\n[记忆事实]：{m}")
        memory_text = "\n\n".join(parts).strip()
        if memory_text:
            merge_msg = f"合并已检索记忆（{len(parts)} 条有效，每条 [记忆检索]+[记忆事实]）：\n{memory_text}"
            yield _emit_think("2-extract", merge_msg)
        else:
            yield _emit_think("2-extract", "合并已检索记忆：（全部为空）")
    elif need_memory and not memory_queries:
        yield _emit_think("2-extract", "未调用记忆 API（未给出检索问句）。")
    else:
        yield _emit_think("2-extract", "未调用记忆 API（无需提取）。")
    return memory_text


def _run_memory_extracts(
    need_memory: bool,
    memory_queries: list[str],
    memory_api_base: str,
    timeout_sec: int,
) -> str:
    it = _iter_memory_extract_gen(need_memory, memory_queries, memory_api_base, timeout_sec)
    try:
        while True:
            next(it)
    except StopIteration as e:
        return (e.value or "") if e.value is not None else ""


def run_one_round(
    user_text: str,
    *,
    history: list[tuple[str, str]] | None = None,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    memory_api_base: str,
    timeout_sec: int,
    ollama_think: bool,
    dump_llm_requests: bool = True,
) -> str:
    h = list(history or [])
    log(
        "1-router",
        f"用户输入：{user_text}",
    )
    rmsgs = _router_messages(h, user_text)
    router_raw, need_memory, memory_queries = _execute_router_with_json_enforcement(
        rmsgs,
        llm_backend=llm_backend,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        deepseek_api_base=deepseek_api_base,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=deepseek_model,
        timeout_sec=timeout_sec,
        ollama_think=ollama_think,
        dump_llm_requests=dump_llm_requests,
    )

    if need_memory and memory_queries:
        n_expect = BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N
        log("1-router", f"解析结果：需检索 {len(memory_queries)} 条 → {memory_queries!r}")
        if len(memory_queries) != n_expect:
            log(
                "1-router",
                f"警告：约定须恰好 {n_expect} 条 memory_queries，当前 {len(memory_queries)} 条。",
            )
    elif need_memory and not memory_queries:
        log("1-router", "解析结果：需提取记忆但未给出 memory_queries，跳过请求记忆 API")
    else:
        log("1-router", "解析结果：无需提取记忆")

    _step_break()
    memory_text = _run_memory_extracts(
        need_memory, memory_queries, memory_api_base, timeout_sec
    )
    _step_break()

    fmsgs = _final_messages(h, memory_text, user_text)
    if llm_backend == "deepseek":
        final_raw = deepseek_chat_messages(
            deepseek_api_base,
            deepseek_api_key,
            deepseek_model,
            fmsgs,
            timeout_sec=timeout_sec,
            max_tokens=BUILTIN_DEEPSEEK_FINAL_MAX_TOKENS,
            temperature=BUILTIN_DEEPSEEK_FINAL_TEMP,
            dump_raw_request_body=dump_llm_requests,
        )
    else:
        final_raw = ollama_chat_messages(
            ollama_host,
            ollama_model,
            fmsgs,
            timeout_sec=timeout_sec,
            think=ollama_think,
            options=BUILTIN_OLLAMA_OPTIONS_FINAL,
            dump_raw_request_body=dump_llm_requests,
        )
    log("3-final", f"终答：{final_raw}")
    _step_break()
    return final_raw.strip()


def iter_run_one_round_event_stream(
    user_text: str,
    *,
    history: list[tuple[str, str]] | None = None,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    memory_api_base: str,
    timeout_sec: int,
    ollama_think: bool,
    dump_llm_requests: bool = False,
) -> Iterator[dict]:
    """编排事件流：``think`` 与控制台 ``log`` 同文；``delta`` 为终答正文增量。供 SSE 等消费。"""
    h = list(history or [])
    yield _emit_think(
        "1-router",
        f"用户输入：{user_text}",
    )
    rmsgs = _router_messages(h, user_text)
    router_emit_buf: list[dict] = []

    def _emit_router(step: str, msg: str) -> None:
        router_emit_buf.append(_emit_think(step, msg))

    router_raw, need_memory, memory_queries = _execute_router_with_json_enforcement(
        rmsgs,
        llm_backend=llm_backend,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        deepseek_api_base=deepseek_api_base,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=deepseek_model,
        timeout_sec=timeout_sec,
        ollama_think=ollama_think,
        dump_llm_requests=dump_llm_requests,
        emit=_emit_router,
    )
    for _ev in router_emit_buf:
        yield _ev
    n_expect = BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N
    if need_memory and memory_queries:
        yield _emit_think("1-router", f"解析结果：需检索 {len(memory_queries)} 条 → {memory_queries!r}")
        if len(memory_queries) != n_expect:
            yield _emit_think(
                "1-router",
                f"警告：约定须恰好 {n_expect} 条 memory_queries，当前 {len(memory_queries)} 条。",
            )
    elif need_memory and not memory_queries:
        yield _emit_think("1-router", "解析结果：需提取记忆但未给出 memory_queries，跳过请求记忆 API")
    else:
        yield _emit_think("1-router", "解析结果：无需提取记忆")
    _step_break()
    memory_text = yield from _iter_memory_extract_gen(
        need_memory, memory_queries, memory_api_base, timeout_sec
    )
    _step_break()

    fmsgs = _final_messages(h, memory_text, user_text)
    if llm_backend == "deepseek":
        yield _emit_think("3-final", _final_stage_think_message(memory_text))
        stream_parts: list[str] = []
        for piece in iter_deepseek_chat_messages_stream(
            deepseek_api_base,
            deepseek_api_key,
            deepseek_model,
            fmsgs,
            timeout_sec=timeout_sec,
            max_tokens=BUILTIN_DEEPSEEK_FINAL_MAX_TOKENS,
            temperature=BUILTIN_DEEPSEEK_FINAL_TEMP,
            dump_raw_request_body=dump_llm_requests,
        ):
            stream_parts.append(piece)
            yield {"kind": "delta", "text": piece}
        final_streamed = "".join(stream_parts).strip()
        log("3-final", f"终答：{final_streamed}")
        _step_break()
    else:
        yield _emit_think("3-final", _final_stage_think_message(memory_text))
        final_raw = ollama_chat_messages(
            ollama_host,
            ollama_model,
            fmsgs,
            timeout_sec=timeout_sec,
            think=ollama_think,
            options=BUILTIN_OLLAMA_OPTIONS_FINAL,
            dump_raw_request_body=dump_llm_requests,
        )
        log("3-final", f"终答：{final_raw}")
        _step_break()
        yield {"kind": "delta", "text": final_raw.strip()}


def iter_run_one_round_final_stream(
    user_text: str,
    *,
    history: list[tuple[str, str]] | None = None,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    memory_api_base: str,
    timeout_sec: int,
    ollama_think: bool,
    dump_llm_requests: bool = False,
) -> Iterator[str]:
    """路由与记忆抽取同 ``run_one_round``；终答仅 DeepSeek 流式增量 yield，Ollama 为单次 yield 全文。"""
    for ev in iter_run_one_round_event_stream(
        user_text,
        history=history,
        llm_backend=llm_backend,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        deepseek_api_base=deepseek_api_base,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=deepseek_model,
        memory_api_base=memory_api_base,
        timeout_sec=timeout_sec,
        ollama_think=ollama_think,
        dump_llm_requests=dump_llm_requests,
    ):
        if ev.get("kind") == "delta":
            yield str(ev.get("text") or "")


def append_session_staging_turn(
    log_path: Path,
    user_query: str,
    final_answer: str,
    *,
    turn_time: datetime,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = turn_time.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    block = f"[轮次时间: {ts}]\nUser: {user_query}\nAssistant: {final_answer}\n\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(block)


def http_pipeline_json(
    method: str, url: str, payload: dict | None, timeout: int = 600
) -> tuple[int, str]:
    """流水线 HTTP 客户端：JSON 请求/整段响应。"""
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def iter_sse_lines_from_http_response(resp) -> Iterator[str]:
    """按块读 HTTP 体并按行拆分，减轻流式响应被整包缓冲。"""
    buf = b""
    while True:
        chunk = resp.read(512)
        if not chunk:
            break
        buf += chunk
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break
            line, buf = buf[:i], buf[i + 1 :]
            yield line.decode("utf-8", errors="replace").rstrip("\r")
    if buf:
        yield buf.decode("utf-8", errors="replace").rstrip("\r")


def pipeline_client_chat_loop(
    pipeline_base: str,
    *,
    stream: bool = True,
    timeout: int = 600,
) -> None:
    """连接 ``memory_pipeline_server`` 的交互循环（无 argparse，供右键运行入口脚本调用）。"""
    base = pipeline_base.rstrip("/")
    code, raw = http_pipeline_json("POST", f"{base}/sessions", {}, timeout=timeout)
    if code != 200:
        print(f"创建会话失败 HTTP {code}: {raw}", flush=True)
        sys.exit(1)
    sid = json.loads(raw).get("session_id", "")
    if not sid:
        print(f"创建会话失败: {raw}", flush=True)
        sys.exit(1)
    print(
        f"已连接流水线服务 session_id={sid}  流式终答={'开' if stream else '关'}（关则用整段 JSON）",
        flush=True,
    )
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
        url = f"{base}/sessions/{sid}/chat"
        body = {"message": user, "stream": stream}
        if not stream:
            code, raw = http_pipeline_json("POST", url, body, timeout=timeout)
            if code != 200:
                print(f"请求失败 HTTP {code}: {raw}", flush=True)
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                print(raw, flush=True)
                continue
            if "detail" in obj:
                print(obj["detail"], flush=True)
                continue
            out = str(obj.get("reply", "")).strip()
            print("\n【答复】\n" + out, flush=True)
            continue
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                print("\n【答复】（流式）", flush=True)
                for s in iter_sse_lines_from_http_response(resp):
                    s = s.strip()
                    if not s or s.startswith(":"):
                        continue
                    if s == "data: [DONE]":
                        break
                    if not s.startswith("data: "):
                        continue
                    chunk = s[6:].strip()
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        print(f"\n[错误] {obj['error']}", flush=True)
                        break
                    if obj.get("kind") == "think":
                        tx = str(obj.get("text") or "")
                        print(f"\n{tx}", flush=True)
                        continue
                    if obj.get("kind") == "delta":
                        t = str(obj.get("text", ""))
                        if t:
                            sys.stdout.write(t)
                            sys.stdout.flush()
                        continue
                    t = str(obj.get("text", ""))
                    if t:
                        sys.stdout.write(t)
                        sys.stdout.flush()
                print(flush=True)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}", flush=True)
        except urllib.error.URLError as e:
            print(f"连接失败: {e}", flush=True)
