# -*- coding: utf-8 -*-
"""
将 session 会话提纯为记忆训练数据：默认调用 **DeepSeek 官网** OpenAI 兼容 ``/chat/completions``；可用 ``--llm-backend ollama`` 改回本机 Ollama。

流程——
0）由模型判断是否有必要写入记忆；若判定「无需记住」则不生成 memory.json / raw / qa 等；
1）**二阶段事实核对**：先从对话摘录「事实 + 原文支持」，再经同一模型一致性过滤，仅保留可严格由原文支持的条目，供后续提取引用；
2）由模型输出结构化 JSON（memory_update_list + deleted_memory），**去重**后可选 **三阶段质审**（筛互斥/劣质条目，``BUILTIN_MEMORY_QC3`` / ``--no-memory-qc3``），再写入 {stem}.memory.json，并据此生成 {stem}.raw.txt；
3）据 raw 生成 QA；负向「未记录」样本由 ``BUILTIN_NEGATIVE_QA`` / ``--negative-qa`` 控制，默认不生成。
4）由模型判断是否需要**更新用户/角色画像**；若需要，在 ``session/<用户ID>/`` 下写入全局共享的 ``{角色标识}.user.pic.md``（Markdown，与 raw 相同方式参与 train_memory 内化训练；**同一用户**跨会话共用；阶段 1 输出完整 ``markdown`` 与本轮 ``session_delta`` 子串，阶段 2 **仅审查 session_delta**；合并替换后落盘；若本轮无 delta 则跳过阶段 2）。
全流程在提示词中携带**对话时间锚点**（reference_time 元数据）；raw 正文为**每事实一行**、`[YYYY-MM-DD HH:MM:SS]` 前缀表示该条事实的发生时刻（可与提纯运行时刻无关）。问答 JSON 须带 `t` 与 raw 对齐。

输入来自「临时会话目录」session_staging/<用户ID>/ 的会话记录文件时，无论最终是否写入记忆（含判定「无需记住」），均将该文件移入 session_bak/<用户ID>/YYYY-MM-DD/ 归档（不删除 session_staging 目录本身）。

输出文件名前缀 `{stem}`：默认当输入为 `session_staging/session_*.txt`（或旧版 `cli_*.txt`）时，`stem` 与该文件名（不含扩展名）一致，避免多轮临时会话都写入同一 `daily.*`；其它输入仍默认 `daily`。可显式传 `--stem` 覆盖。

右键运行：不传任何命令行参数时，行为由文件顶部 ``BUILTIN_STAGING_ON_RUN`` 决定（``"all"`` 为全部 ``session_*.txt`` / 旧版 ``cli_*.txt``，``"latest"`` 为仅最新一个）；命令行仍可传 ``--all-staging`` / ``--latest-staging`` 覆盖。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import memory_user_paths as mem_paths

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

_script_dir = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 右键运行：只改这里（人只点运行；命令行留给脚本/机器覆盖这些默认）
# ---------------------------------------------------------------------------
# 在 IDE 里无参右键运行时：如何处理 session_staging 下的 session_*.txt（含旧版 cli_*.txt）
#   "all"    → 全部文件，按修改时间从早到晚逐个提纯（有 tqdm 则显示进度条）
#   "latest" → 只处理最新的一个
BUILTIN_STAGING_ON_RUN = "all"
# QA 生成：是否包含负向「未记录」样本；False=仅正向（默认）
BUILTIN_NEGATIVE_QA = False
BUILTIN_OLLAMA_HOST = "http://127.0.0.1:11434"
# 与本机 `ollama list` 中名称一致，例如 deepseek-r1:8b
BUILTIN_OLLAMA_MODEL = "deepseek-r1:8b"
BUILTIN_OUTPUT_STEM = "daily"
BUILTIN_SESSION_BAK = str(_script_dir / "session_bak")
# 提纯默认关闭思考链（/api/chat 顶层 think:false，见 Ollama Thinking 文档）；R1/Qwen3 等否则会极慢。
BUILTIN_OLLAMA_THINK = False
# 各阶段 num_predict 上限，避免长思维占满预算；可按模型调大/调小。
BUILTIN_OLLAMA_OPTIONS_DECIDE = {"num_predict": 64, "temperature": 0.2}
BUILTIN_OLLAMA_OPTIONS_EXTRACT_JSON = {"num_predict": 4096, "temperature": 0.2}
BUILTIN_OLLAMA_OPTIONS_GEN_QA = {"num_predict": 8192, "temperature": 0.3}
BUILTIN_OLLAMA_OPTIONS_USER_PIC = {"num_predict": 8192, "temperature": 0.3}
BUILTIN_OLLAMA_OPTIONS_TWO_PHASE = {"num_predict": 4096, "temperature": 0.2}
BUILTIN_OLLAMA_OPTIONS_MEMORY_QC3 = {"num_predict": 4096, "temperature": 0.1}
# 在「提取 JSON 并去重」之后，再经大模型筛掉互斥/劣质条目（可 --no-memory-qc3 关闭）
BUILTIN_MEMORY_QC3 = True
BUILTIN_LLM_BACKEND = "deepseek"  # "ollama" | "deepseek"
BUILTIN_DEEPSEEK_API_BASE = "https://api.deepseek.com"
BUILTIN_DEEPSEEK_API_KEY = "sk-REPLACE_WITH_YOUR_KEY"  # 生产请改用环境变量，勿提交真实 Key
BUILTIN_DEEPSEEK_MODEL = "deepseek-v4-flash"
# 用户画像 JSON 阶段：DeepSeek 的 max_tokens（与 ollama num_predict 对应量级）
BUILTIN_DEEPSEEK_MAX_TOKENS_USER_PIC = 8192
# ---------------------------------------------------------------------------

USER_PIC_GLOB_SUFFIX = ".user.pic.md"


def _safe_user_pic_file_key(key: str) -> str:
    """用于全局文件名 `{key}.user.pic.md` 中角色段：去掉 Windows 非法字符等。"""
    k = (key or "").strip()
    k = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", k)
    k = re.sub(r"\s+", "_", k).strip("._") or "unknown"
    return k[:80]


def _is_global_user_pic_filename(name: str) -> bool:
    """``{角色}.user.pic.md``（角色段不含点），与旧版 ``stem.角色.user.pic.md`` 区分。"""
    if not name.endswith(USER_PIC_GLOB_SUFFIX):
        return False
    rest = name[: -len(USER_PIC_GLOB_SUFFIX)]
    return "." not in (rest or "")


def _existing_global_user_pic_context_block(user_profile_dir: Path) -> str:
    """``session/<用户ID>/`` 下全局 ``{角色}.user.pic.md`` 全文（单文件过长则截断），供模型增量修订。"""
    if not user_profile_dir.is_dir():
        return "（当前用户目录下尚无全局画像文件。）\n"
    paths = sorted(
        p
        for p in user_profile_dir.iterdir()
        if p.is_file() and _is_global_user_pic_filename(p.name)
    )
    if not paths:
        return f"（目录 {user_profile_dir.name!r} 下尚无 {USER_PIC_GLOB_SUFFIX} 全局画像文件。）\n"
    parts: list[str] = []
    for p in paths:
        body = p.read_text(encoding="utf-8")
        if len(body) > 20000:
            body = body[:20000] + "\n…(此处截断，后文略)…\n"
        parts.append(f"===== 已有文件: {p.name} =====\n{body}\n")
    return "\n".join(parts)


SYSTEM_USER_PROFILE_JSON = (
    "你是「用户与对话角色画像」维护助手。根据本轮对话、**已定稿 raw 事实行**、**二阶段核对摘录**、以及下面给出的**已有全局画像文件**（若有），"
    "自行判断：是否有必要新增或更新画像。\n"
    "画像可覆盖：用户本人、助手（Assistant）、对话中出现的第三方（可多个，如具体人名）。\n"
    "仅当存在值得长期检索的**相对稳定**、且**可由 raw 或二阶段摘录直接支持**的信息时才输出 need_update=true；"
    "纯寒暄、一次性无关细节、推断、或 raw/摘录中未出现的表述不要写入。\n"
    "字段与章节**不必固定**：可用 Markdown 表格、列表、小标题自由组织；也可在示例结构之外增加你认为有用的区块。\n"
    "若 need_update 为 false，则 profiles 必须为空数组 []。\n"
    "若 need_update 为 true，profiles 中每一项：\n"
    "  - file_key：仅表示角色侧短名（如「用户」「助手」「张三」）；实际落盘为当前用户目录下的 "
    f"file_key + 「{USER_PIC_GLOB_SUFFIX}」（例：用户 → 用户{USER_PIC_GLOB_SUFFIX}；全局共享，勿含会话 stem）。\n"
    "  file_key 禁止含路径分隔符、英文句点与英文引号。\n"
    "  - markdown：该角色的**完整**合并后 Markdown 正文（不要包在代码块里），含历史与本轮新增；建议以「## 用户画像」或「## 角色画像」类标题开头。\n"
    "  若某昵称/别名与已有 `file_key` 实为同一人、希望避免维护两份全文，可将本项 `markdown` 写为**整份恰一行**的关联格式：`[请调取\"某已有file_key\"的画像]`（引号用英文直引号成对；读盘时系统会以引号内键的正文为准）。\n"
    "  - session_delta：**必须**是 markdown 中一段与全文**逐字相同**的连续子串，且仅含本轮依据 raw/二阶段摘录**新写或修订**的片段；"
    "若本轮仅整理排版、未从 raw/摘录新增事实则必须为空串 \"\"；不得把仅来自历史的段落放进 session_delta。\n"
    "**只输出一个 JSON 对象**，不要其它说明文字。JSON 须可被标准库 json.loads 解析，键名固定为："
    'need_update（布尔）、profiles（数组，元素为 {"file_key": "...", "markdown": "...", "session_delta": "..."}）。\n'
    "示例键名示意（勿照抄空内容）：file_key=用户；markdown=完整 md；session_delta=本轮新增的那一段原文子串或空串。"
)


SYSTEM_USER_PROFILE_VERIFY = (
    "你是事实审查员。用户消息给出：对话轮次、已定稿 raw、二阶段核对摘录、以及「候选画像」JSON。\n"
    "**仅审查**每个 profile 的 **session_delta** 字段（本轮拟写入的片段）：判断其是否均可由 raw 或二阶段摘录中的客观内容**直接支持**。"
    "不要审查 markdown 全文里未出现在 session_delta 中的历史内容。\n"
    "对无法支持的句子从 session_delta 中删除或改写；若该片段全无依据则 session_delta 输出空串 \"\"。\n"
    "**只输出一个 JSON 对象**，键名固定为："
    'profiles（数组，元素为 {"file_key": "...", "session_delta": "审查后的片段（可与输入逐字不同，但须为去伪后的可支持表述）"}）。\n'
    "须覆盖候选 JSON 中每个 profile 的 file_key；不要输出 markdown 字段，不要输出 need_update 字段。"
)


def _refine_llm_chat_json_object(
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    system: str,
    user: str,
    *,
    timeout_sec: int,
    think: bool | None,
    ollama_options: dict | None,
    deepseek_max_tokens: int,
) -> str:
    """约束为 JSON 对象：DeepSeek 用 response_format；Ollama 用 format=json。"""
    if llm_backend == "deepseek":
        return deepseek_chat_messages(
            deepseek_api_base,
            deepseek_api_key,
            deepseek_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout_sec=timeout_sec,
            max_tokens=deepseek_max_tokens,
            temperature=0.2,
            dump_raw_request_body=False,
            response_format={"type": "json_object"},
        )
    return ollama_chat_messages(
        ollama_host,
        ollama_model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        timeout_sec=timeout_sec,
        think=think,
        options=ollama_options,
        response_format_json=True,
    )


def _merge_verified_user_profile_markdown(md_full: str, sd_before: str, sd_after: str) -> str:
    """用阶段 2 审查后的 ``sd_after`` 替换 ``md_full`` 中首次出现的 ``sd_before``。"""
    if not sd_before:
        return md_full
    if sd_before not in md_full:
        return md_full
    return md_full.replace(sd_before, sd_after, 1)


def run_user_profile_update(
    conversation: str,
    raw_text: str,
    grounded_facts_text: str,
    out_dir: Path,
    stem: str,
    *,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    reference_time_iso: str,
    timeout_sec: int,
    think: bool | None,
) -> None:
    """在 raw/qa 已定稿后调用：按需写入 ``session/<用户ID>/{file_key}.user.pic.md``（二阶段审查后落盘）。"""
    user_profile_dir = out_dir.parent
    try:
        user_profile_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log(f"[session_to_memory] 用户画像：无法创建用户目录 {user_profile_dir!r}：{e!r}")
        return
    turns = _parse_conversation_turns(conversation)
    dialog_fmt = _format_turns_for_refiner(turns) if turns else (conversation or "").strip()
    gf = (grounded_facts_text or "").strip()
    user_payload = (
        _time_block_for_prompt(reference_time_iso)
        + f"【本轮提纯会话 stem（仅作日志，勿写入文件名）】{stem}\n\n"
        + "【对话轮次（若有标准 User/Assistant 则已展开）】\n"
        + (dialog_fmt if dialog_fmt else "(无标准轮次结构，以下为原始全文。)\n" + (conversation or "").strip())
        + "\n\n【已定稿 raw（每行一条事实）】\n"
        + (raw_text or "").strip()
        + "\n\n【二阶段核对摘录】\n"
        + (gf if gf else "(空)")
        + "\n\n【本用户已有全局画像（可据此合并、修订）】\n"
        + _existing_global_user_pic_context_block(user_profile_dir)
    )
    try:
        raw = _refine_llm_chat_json_object(
            llm_backend,
            ollama_host,
            ollama_model,
            deepseek_api_base,
            deepseek_api_key,
            deepseek_model,
            SYSTEM_USER_PROFILE_JSON,
            user_payload,
            timeout_sec=timeout_sec,
            think=think,
            ollama_options=BUILTIN_OLLAMA_OPTIONS_USER_PIC,
            deepseek_max_tokens=BUILTIN_DEEPSEEK_MAX_TOKENS_USER_PIC,
        )
    except Exception as e:
        _log(f"[session_to_memory] 用户画像阶段 1 请求失败（已跳过）：{e!r}")
        return
    try:
        obj = _parse_json_object_from_model(raw)
    except ValueError:
        dump = out_dir / f"{stem}.user_pic.stage1.raw_model_output.txt"
        try:
            dump.write_text(raw or "", encoding="utf-8")
        except OSError:
            pass
        _log(
            f"[session_to_memory] 用户画像：阶段 1 无法解析 JSON，已跳过；"
            f"若已落盘则见 {dump.name}"
        )
        return
    need = bool(obj.get("need_update"))
    profiles = obj.get("profiles")
    if not need or not isinstance(profiles, list) or not profiles:
        _log("[session_to_memory] 用户画像：模型判定无需更新或未给出 profiles，跳过写入。")
        return

    def _any_session_delta() -> bool:
        for p in profiles:
            if isinstance(p, dict) and str(p.get("session_delta", "")).strip():
                return True
        return False

    vmap: dict[str, str] = {}
    if _any_session_delta():
        verify_user = (
            _time_block_for_prompt(reference_time_iso)
            + "【对话轮次】\n"
            + (dialog_fmt if dialog_fmt else (conversation or "").strip())
            + "\n\n【已定稿 raw】\n"
            + (raw_text or "").strip()
            + "\n\n【二阶段核对摘录】\n"
            + (gf if gf else "(空)")
            + "\n\n【候选画像 JSON】\n"
            + json.dumps(obj, ensure_ascii=False)
        )
        try:
            raw_v = _refine_llm_chat_json_object(
                llm_backend,
                ollama_host,
                ollama_model,
                deepseek_api_base,
                deepseek_api_key,
                deepseek_model,
                SYSTEM_USER_PROFILE_VERIFY,
                verify_user,
                timeout_sec=timeout_sec,
                think=think,
                ollama_options=BUILTIN_OLLAMA_OPTIONS_USER_PIC,
                deepseek_max_tokens=BUILTIN_DEEPSEEK_MAX_TOKENS_USER_PIC,
            )
        except Exception as e:
            _log(f"[session_to_memory] 用户画像阶段 2（事实审查）请求失败（已跳过落盘）：{e!r}")
            return
        try:
            vobj = _parse_json_object_from_model(raw_v)
        except ValueError:
            dump = out_dir / f"{stem}.user_pic.stage2.raw_model_output.txt"
            try:
                dump.write_text(raw_v or "", encoding="utf-8")
            except OSError:
                pass
            _log(
                f"[session_to_memory] 用户画像：阶段 2 无法解析 JSON，已跳过落盘；"
                f"若已落盘则见 {dump.name}"
            )
            return
        profiles_v = vobj.get("profiles")
        if not isinstance(profiles_v, list):
            profiles_v = []
        for vit in profiles_v:
            if not isinstance(vit, dict):
                continue
            vfk = _safe_user_pic_file_key(str(vit.get("file_key", "")))
            if vfk == "unknown" and not str(vit.get("file_key", "")).strip():
                continue
            vmap[vfk] = str(vit.get("session_delta", ""))
        if not profiles_v:
            _log(
                "[session_to_memory] 用户画像：阶段 2 返回空 profiles，本轮 session_delta 按未通过处理（将尝试从 markdown 中移除对应片段）。"
            )
    else:
        _log("[session_to_memory] 用户画像：本轮 session_delta 均为空，跳过阶段 2，仅落盘阶段 1 的 markdown。")

    n_written = 0
    for it in profiles:
        if not isinstance(it, dict):
            continue
        fk = _safe_user_pic_file_key(str(it.get("file_key", "")))
        md_full = str(it.get("markdown", "")).strip()
        sd1 = str(it.get("session_delta", ""))
        if fk == "unknown" and not str(it.get("file_key", "")).strip():
            continue
        if not md_full:
            continue
        if "." in fk:
            _log(f"[session_to_memory] 用户画像：跳过非法 file_key（含句点）：{it.get('file_key', '')!r}")
            continue
        if sd1.strip():
            sd2 = vmap.get(fk)
            if sd2 is None:
                sd2 = ""
                _log(
                    f"[session_to_memory] 用户画像：阶段 2 未返回 file_key={fk!r}，按未通过处理，移除本轮 session_delta。"
                )
            chunk = sd1 if sd1 in md_full else (sd1.strip() if sd1.strip() in md_full else "")
            if not chunk:
                _log(
                    f"[session_to_memory] 用户画像：警告 session_delta 不是 markdown 的子串，按阶段 1 markdown 原样落盘（{fk!r}）。"
                )
                final_md = md_full
            else:
                final_md = _merge_verified_user_profile_markdown(md_full, chunk, sd2)
        else:
            final_md = md_full
        final_md = (final_md or "").strip()
        if not final_md:
            _log(f"[session_to_memory] 用户画像：合并后正文为空，跳过写入 {fk!r}。")
            continue
        dest = user_profile_dir / f"{fk}{USER_PIC_GLOB_SUFFIX}"
        try:
            dest.write_text(final_md + ("\n" if not final_md.endswith("\n") else ""), encoding="utf-8")
        except OSError as e:
            _log(f"[session_to_memory] 用户画像：写入 {dest.name} 失败：{e!r}")
            continue
        _log(f"[session_to_memory] 用户画像：已写入 {dest}（{len(final_md)} 字）")
        n_written += 1
    if n_written == 0:
        _log("[session_to_memory] 用户画像：未写出任何有效文件（检查 file_key/markdown）。")


def _log(msg: str) -> None:
    print(msg, flush=True)


def reference_time_iso_local() -> str:
    """无法从输入确定事件发生时间时的回退：当前本地时刻（ISO，日期与时间之间为空格）。"""
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def _parse_absolute_event_time(s: str) -> str:
    """解析并规范为绝对时间字符串，须含日期与时分秒：YYYY-MM-DD HH:MM:SS。"""
    s = (s or "").strip().replace("T", " ", 1)
    if not s:
        raise ValueError("事件时间为空")
    if len(s) >= 19:
        s = s[:19]
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(
            f"事件时间须为绝对时间 YYYY-MM-DD HH:MM:SS（含时分秒），无法解析: {(s or '').strip()!r}"
        ) from e
    return dt.isoformat(sep=" ")


def resolve_session_event_time_iso(
    conversation: str,
    *,
    override: str | None = None,
) -> str:
    """确定事件时间锚点：优先 --event-time，否则首条可解析的「轮次时间」，否则当前本地时刻（均为绝对时间）。"""
    if override:
        return _parse_absolute_event_time(override)
    for _u, _a, t, _lbl in _parse_conversation_turns(conversation):
        tt = (t or "").strip()
        if tt:
            return _parse_absolute_event_time(tt)
    return reference_time_iso_local()


def _time_block_for_prompt(reference_time_iso: str) -> str:
    return (
        f"【对话时间锚点】{reference_time_iso}（绝对时间 YYYY-MM-DD HH:MM:SS，本条会话/记忆发生时刻；与提纯脚本运行时刻无关）\n\n"
    )


def _clean_model_text(text: str) -> str:
    """去掉常见思维链/代码块外壳，便于解析。"""
    t = text.strip()
    t = re.sub(r"<[^>]*redacted_think[^>]*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s*```\s*$", "", t, flags=re.MULTILINE)
    return t.strip()


def ollama_chat(
    host: str,
    model: str,
    system: str,
    user: str,
    *,
    timeout_sec: int = 600,
    think: bool | None = None,
    options: dict | None = None,
    dump_raw_request_body: bool = False,
) -> str:
    """POST /api/chat，非流式。think 为 False 时关闭思考模型推理链；options 可含 num_predict 等。
    dump_raw_request_body=True 时向 stdout 打印「正在请求 Ollama 接口 …」与「参数:」+ 请求体 JSON。"""
    url = host.rstrip("/") + "/api/chat"
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if think is not None:
        payload["think"] = think
    else:
        payload["think"] = BUILTIN_OLLAMA_THINK
    if options:
        payload["options"] = options
    raw_body_str = json.dumps(payload, ensure_ascii=False)
    if dump_raw_request_body:
        print(f"正在请求 Ollama 接口 POST {url}", flush=True)
        print(f"参数: {raw_body_str}", flush=True)
    body = raw_body_str.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接 Ollama ({url})。请确认已运行 ollama serve，且模型已 ollama pull。\n{e}"
        ) from e
    data = json.loads(raw)
    msg = data.get("message") or {}
    return str(msg.get("content") or "")


def _print_deepseek_usage_line(usage: dict) -> None:
    """从响应 ``usage`` 打一行（含官方硬盘缓存：prompt_cache_hit / miss_tokens）。"""
    if not isinstance(usage, dict):
        return
    parts: list[str] = []
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ):
        v = usage.get(key)
        if v is not None:
            parts.append(f"{key}={v}")
    rtd = (usage.get("completion_tokens_details") or {}) if isinstance(usage.get("completion_tokens_details"), dict) else {}
    if isinstance(rtd, dict) and rtd.get("reasoning_tokens") is not None:
        parts.append(f"reasoning_tokens={rtd.get('reasoning_tokens')}")
    if parts:
        print(f"[DeepSeek usage] {' ; '.join(parts)}", flush=True)


DEEPSEEK_ERROR_CODES_DOC = (
    "https://api-docs.deepseek.com/zh-cn/quick_start/error_codes"
)


def _deepseek_api_error_message(
    status_code: int, url: str, err_body: str
) -> str:
    """将 DeepSeek API 的 HTTP 错误格式化为可读的 RuntimeError 文本（含官方错误码说明，见上链）。"""
    # 与 https://api-docs.deepseek.com/zh-cn/quick_start/error_codes 表一致
    official = {
        400: "400 - 格式错误：请求体格式错误，请按返回信息修改请求体",
        401: "401 - 认证失败：API key 错误，请检查或到平台创建 API key",
        402: "402 - 余额不足：请确认账户余额并充值，或更换有效 Key",
        422: "422 - 参数错误：请求体参数错误，请按返回信息修改相关参数",
        429: "429 - 请求速率达到上限（TPM 或 RPM），请合理降速或稍后重试",
        500: "500 - 服务器故障：请稍后重试；若问题持续存在请联系官方",
        503: "503 - 服务器繁忙：请稍后重试",
    }
    line = official.get(
        status_code, f"HTTP {status_code}：请对照官方错误码说明排查"
    )
    b = (err_body or "").strip()
    if len(b) > 8000:
        b = b[:8000] + "\n…(响应体已截断)"
    out = [f"DeepSeek API：{line}", f"请求：{url}"]
    if b:
        out.append("响应体：\n" + b)
    if status_code == 402:
        out.append("提示：本机可在 memory_pipeline_cli 等使用 --llm-backend ollama 改用本地模型。")
    out.append("官方错误码表：" + DEEPSEEK_ERROR_CODES_DOC)
    return "\n".join(out)


def deepseek_chat(
    api_base: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    *,
    timeout_sec: int = 600,
    max_tokens: int = 4096,
    temperature: float | None = 0.2,
    dump_raw_request_body: bool = False,
) -> str:
    """POST DeepSeek 官网 OpenAI 兼容接口 ``/chat/completions``（与网页版同一 API；``api_base`` 如 ``https://api.deepseek.com`` 或 ``https://api.deepseek.com/v1``）。

    ``deepseek-reasoner`` 等推理模型按文档不支持 temperature，调用时不传该字段。
    返回 ``choices[0].message.content``（最终答复正文，不含 ``reasoning_content``）。
    ``dump_raw_request_body=True`` 时向 stdout 打印「正在请求 …」、请求体 JSON，以及响应中的 ``[DeepSeek usage]``（含 ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``，与官方「上下文硬盘缓存」说明一致）。"""
    url = api_base.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    if temperature is not None and "reasoner" not in (model or "").lower():
        payload["temperature"] = temperature
    raw_body_str = json.dumps(payload, ensure_ascii=False)
    if dump_raw_request_body:
        print(f"正在请求 DeepSeek 接口 POST {url}", flush=True)
        print(f"参数: {raw_body_str}", flush=True)
    body = raw_body_str.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            _deepseek_api_error_message(e.code, url, err_body)
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接 DeepSeek API ({url})。\n{e}") from e
    data = json.loads(raw)
    if dump_raw_request_body:
        u = data.get("usage")
        if isinstance(u, dict):
            _print_deepseek_usage_line(u)
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def deepseek_chat_messages(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    timeout_sec: int = 600,
    max_tokens: int = 4096,
    temperature: float | None = 0.2,
    dump_raw_request_body: bool = False,
    response_format: dict | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> str:
    """POST ``/chat/completions``，非流式；``messages`` 为 OpenAI 格式（须含 system 与多轮 user/assistant）。

    固定附带 ``thinking: {{"type": "disabled"}}``，不启用 API 思考模式（与官方「思考模式」开关一致）。
    ``response_format`` 例如 ``{{"type": "json_object"}}`` 时须**未**同时传 ``tools``（二者择一与官方用法一致）。
    若传 ``tools`` 且模型在 ``choices[0].message.tool_calls`` 中返回 function，本函数**返回**该 function 的 ``arguments`` 字符串（即 JSON 文本）；否则回退为 ``message.content``。
    ``response_format`` 与仅 content 的 JSON 模式：须在 system/user 中含「json」与示例。``dump_raw_request_body=True`` 时额外打印 ``[DeepSeek usage]``。"""
    url = api_base.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if response_format is not None and not tools:
        payload["response_format"] = response_format
    if temperature is not None and "reasoner" not in (model or "").lower():
        payload["temperature"] = temperature
    raw_body_str = json.dumps(payload, ensure_ascii=False)
    if dump_raw_request_body:
        print(f"正在请求 DeepSeek 接口 POST {url}", flush=True)
        print(f"参数: {raw_body_str}", flush=True)
    body = raw_body_str.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            _deepseek_api_error_message(e.code, url, err_body)
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接 DeepSeek API ({url})。\n{e}") from e
    data = json.loads(raw)
    if dump_raw_request_body:
        u = data.get("usage")
        if isinstance(u, dict):
            _print_deepseek_usage_line(u)
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    if tools is not None:
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if isinstance(fn, dict):
                    args = (fn.get("arguments") or "").strip()
                    if args:
                        return args
    return str(msg.get("content") or "")


def iter_deepseek_chat_messages_stream(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    timeout_sec: int = 600,
    max_tokens: int = 4096,
    temperature: float | None = 0.4,
    dump_raw_request_body: bool = False,
) -> Iterator[str]:
    """POST ``/chat/completions``，``stream: true``；逐段 yield ``choices[].delta.content``（正文增量）。
    请求附带 ``stream_options.include_usage``（与官方流式补全说明一致）；在 ``dump_raw_request_body`` 为真时流结束后再打一行 ``[DeepSeek usage]``。"""
    url = api_base.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
        "thinking": {"type": "disabled"},
    }
    if temperature is not None and "reasoner" not in (model or "").lower():
        payload["temperature"] = temperature
    raw_body_str = json.dumps(payload, ensure_ascii=False)
    if dump_raw_request_body:
        print(f"正在请求 DeepSeek 接口（流式）POST {url}", flush=True)
        print(f"参数: {raw_body_str}", flush=True)
    body = raw_body_str.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_stream_usage: dict | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                s = line.decode("utf-8", errors="replace").strip()
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
                u = obj.get("usage")
                if isinstance(u, dict) and u and any(
                    u.get(k) is not None
                    for k in ("total_tokens", "prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
                ):
                    last_stream_usage = u
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = str(delta.get("content") or "")
                    if piece:
                        yield piece
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            _deepseek_api_error_message(e.code, url, err_body)
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接 DeepSeek API ({url})。\n{e}") from e
    finally:
        if dump_raw_request_body and last_stream_usage is not None:
            _print_deepseek_usage_line(last_stream_usage)


def ollama_chat_messages(
    host: str,
    model: str,
    messages: list[dict],
    *,
    timeout_sec: int = 600,
    think: bool | None = None,
    options: dict | None = None,
    dump_raw_request_body: bool = False,
    response_format_json: bool = False,
) -> str:
    """POST ``/api/chat``，非流式；``messages`` 为多轮对话。

    ``response_format_json=True`` 时设置 ``format: \"json\"``，由 Ollama 约束输出为 JSON。"""
    url = host.rstrip("/") + "/api/chat"
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if response_format_json:
        payload["format"] = "json"
    if think is not None:
        payload["think"] = think
    else:
        payload["think"] = BUILTIN_OLLAMA_THINK
    if options:
        payload["options"] = options
    raw_body_str = json.dumps(payload, ensure_ascii=False)
    if dump_raw_request_body:
        print(f"正在请求 Ollama 接口 POST {url}", flush=True)
        print(f"参数: {raw_body_str}", flush=True)
    body = raw_body_str.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接 Ollama ({url})。请确认已运行 ollama serve，且模型已 ollama pull。\n{e}"
        ) from e
    data = json.loads(raw)
    msg = data.get("message") or {}
    return str(msg.get("content") or "")


def _refine_llm_chat(
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    system: str,
    user: str,
    *,
    timeout_sec: int,
    think: bool | None,
    options: dict | None,
) -> str:
    if llm_backend == "deepseek":
        opt = options or {}
        max_tok = int(opt.get("num_predict", 2048))
        temp = float(opt.get("temperature", 0.2))
        return deepseek_chat(
            deepseek_api_base,
            deepseek_api_key,
            deepseek_model,
            system,
            user,
            timeout_sec=timeout_sec,
            max_tokens=max_tok,
            temperature=temp,
            dump_raw_request_body=False,
        )
    return ollama_chat(
        ollama_host,
        ollama_model,
        system,
        user,
        timeout_sec=timeout_sec,
        think=think,
        options=options,
    )


def _parse_conversation_turns(text: str) -> list[tuple[str, str, str, str]]:
    """
    解析 staging 文本。
    新格式每轮以「[轮次时间: …]」开头；用户侧行为 ``User:`` 或 ``昵称:`` 等单行前缀 + ``\\nAssistant:``。
    返回 [(用户正文, 助手正文, 轮次时间, 用户侧显示名), …]。
    """
    text = (text or "").strip()
    if not text:
        return []
    if "[轮次时间:" in text:
        parts = re.split(r"\n\n(?=\[轮次时间:)", text)
    else:
        parts = re.split(r"\n\n(?=User:\s)", text)
    out: list[tuple[str, str, str, str]] = []
    time_head = re.compile(r"^\[轮次时间:\s*([^\]]+)\]\s*\n")
    for part in parts:
        part = part.strip()
        turn_time = ""
        m = time_head.match(part)
        if m:
            turn_time = m.group(1).strip()
            part = part[m.end() :].lstrip()
        if "\nAssistant:" not in part:
            continue
        left, a = part.split("\nAssistant:", 1)
        left = left.strip()
        a = a.strip()
        if left.startswith("User:"):
            u = left[len("User:") :].lstrip()
            label = "User"
        else:
            m2 = re.match(r"^([^:\n]+):\s*(.*)$", left, re.DOTALL)
            if m2:
                label, u = m2.group(1).strip(), m2.group(2).strip()
            else:
                u, label = left, "User"
        out.append((u, a, turn_time, label))
    return out


def _format_turns_for_refiner(turns: list[tuple[str, str, str, str]]) -> str:
    """按轮展开，显式标注用户与助手及轮次时间，供模型逐轮分析。"""
    lines: list[str] = []
    for i, (u, a, t, label) in enumerate(turns, 1):
        lines.append(f"【第{i}轮】")
        if t:
            lines.append(f"轮次时间：{t}")
        lines.append(f"用户（{label}）：{u}")
        lines.append(f"助手（Assistant）：{a}")
        lines.append("")
    return "\n".join(lines).strip()


# 模型首行输出须与之一致，便于解析（勿改字面）
MEMORY_DECISION_SKIP_LINE = "无需记住"
MEMORY_DECISION_KEEP_LINE = "需要记住"

SYSTEM_DECIDE_MEMORY = (
    "你是记忆必要性判断助手。根据对话内容，判断是否有必要写入长期记忆库（供后续训练记忆模型）。\n"
    "用户消息开头会给出【对话时间锚点】（绝对时间 YYYY-MM-DD HH:MM:SS）。若对话中出现相对时间表述，须结合该锚点理解；锚点本身为绝对时刻，非脚本运行时刻。\n"
    "无需记住：无信息增量、纯寒暄套话、无意义重复、或没有任何可复用的用户侧信息。\n"
    "需要记住：包含值得后续检索的用户信息、偏好、经历、约定、习惯、用户自述对某人物或话题的了解，或双方明确约定等。\n"
    "你的回复第一行必须是以下二者之一（整行一字不差，不要加引号、标点或空格）：\n"
    f"{MEMORY_DECISION_SKIP_LINE}\n"
    f"{MEMORY_DECISION_KEEP_LINE}\n"
    "除第一行外不要输出任何内容。"
)


def _parse_memory_decision_first_line(cleaned: str) -> bool | None:
    """True=需要记住，False=无需记住，None=无法解析。"""
    if not (cleaned or "").strip():
        return None
    first = cleaned.strip().splitlines()[0].strip()
    if first == MEMORY_DECISION_SKIP_LINE:
        return False
    if first == MEMORY_DECISION_KEEP_LINE:
        return True
    return None


def decide_memory_relevance(
    conversation: str,
    *,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    timeout_sec: int = 600,
    think: bool | None = None,
    reference_time_iso: str | None = None,
) -> str:
    """调用模型做必要性判断；返回清洗后的全文（至少应含首行决策）。"""
    if think is None:
        think = BUILTIN_OLLAMA_THINK
    if reference_time_iso is None:
        reference_time_iso = reference_time_iso_local()
    turns = _parse_conversation_turns(conversation)
    if turns:
        body = _format_turns_for_refiner(turns)
        user = (
            _time_block_for_prompt(reference_time_iso)
            + f"共 {len(turns)} 轮对话如下。请严格按 system 要求只输出第一行决策：\n\n"
            + body
        )
    else:
        user = (
            _time_block_for_prompt(reference_time_iso)
            + "以下文本未能解析为标准 User/Assistant 轮次，请仍根据全文判断是否有必要写入记忆。\n\n"
            + conversation.strip()
        )
    raw = _refine_llm_chat(
        llm_backend,
        ollama_host,
        ollama_model,
        deepseek_api_base,
        deepseek_api_key,
        deepseek_model,
        SYSTEM_DECIDE_MEMORY,
        user,
        timeout_sec=timeout_sec,
        think=think,
        options=BUILTIN_OLLAMA_OPTIONS_DECIDE,
    )
    return _clean_model_text(raw)


SYSTEM_MEMORY_EXTRACT_JSON = """# Role
你是「长期记忆提取器」：从多轮对话中提取用户**持久、稳定**的个人信息，以及用户**自述**的**自身知识**（对某人物或话题已知哪些要点）。不要冗长复述；subject / predicate / object 尽量短。

# Rules
1. **定稿**：只提取对话最后确认的状态；有冲突以用户明确陈述与最后一次有效纠正为准；反问、设错前提中的假设不得当作已承认事实。
2. **过滤**：无个人信息的纯工具型查询可忽略；但若用户表达的是**与个人相关的习惯、条件或自我认知**（即使表面像工具或百科场景），须按个人事实提取。用户**明确声明**自己掌握的理解或知识点须写入；**仅**索要客观信息且**未**陈述自身认知时可不提取。
3. **溯源**：可选 related_to、evidence；evidence 须短、可回到对话核对，禁止条条同一套空话。
4. **负向**：用户否认或纠正的内容，在可转为正向事实时按正向写入 object。
5. **evidence**：每条独立、可核验；说明来源或语境，勿与 object 无关；约 8～40 字为宜。
6. **去重**：同一 predicate + 相同 object 只保留一条。
7. **时刻**：每项须含 absolute_time（字符串 YYYY-MM-DD HH:MM:SS），取该条事实最后定稿所在轮次时间；勿多条共用一个时刻。
8. **memory_sentence**：一句完整中文事实句，不含方括号时间戳；与 subject、predicate、object 一致，用作训练 raw；勿与 evidence 照抄混用；避免主谓修饰叠用导致的病句。
9. **互斥自检**：若你列出的多条 memory_sentence 在语义上**互斥**（对同一事实不能同时为真），在 JSON 中**只输出一条**：以用户**最后一次**有效陈述为准，或与下方「经二阶段核对的事实」更一致的那条；**不要**把矛盾结论都输出给下游。

# Output
仅输出一个 JSON 对象，不要解释性文字。顶层：memory_update_list（数组）、deleted_memory（数组）。脚本会附加 reference_time，你不必输出。每项须含 subject、predicate、object、confidence、evidence、memory_sentence、absolute_time；可选 related_to。deleted_memory 无则 []。
"""

SYSTEM_GEN_QA_NO_NEGATIVE = (
    "你是记忆数据标注助手。根据 raw 生成问答对，用于训练「按问题检索并复述个人事实」的记忆模型。"
    "raw 每行形如「[YYYY-MM-DD HH:MM:SS] 事实句」，方括号内为该条事实的发生时刻（各条时刻可不同）。"
    "用户消息开头另有【对话时间锚点】供对照。"
    "每条一行 JSON，字段须含 q、a、t；每条问答的 t 须为该条所依据的 raw 事实行行首方括号内**绝对时刻**（与 raw 逐字一致）。\n"
    "【条数：不规定固定条数，以覆盖为纲】\n"
    "1）先把 raw 中**可独立被问到**的要点拆开（可理解为：不同事、不同人物、不同时间、不同偏好/关系/具体约定等；一句极短的 raw 往往只需 1～2 条 QA）。\n"
    "2）在这些要点中，应使**约 90%** 的要点在**至少一条**正向 QA 中被问到且答案 a 能据 raw 说清；余下难以化问可略。\n"
    "3）在达到上述覆盖的前提下，**能多则多、能少则少**，**禁止**为凑条数而写同义重复问法或空洞 QA。\n"
    "4）问法、答案完整度、多时刻分布等要求与此前一致：多角度、可换表述问法、答案非空壳、长 raw 时优先让不同时刻/不同行都有落点。\n"
    "只输出正向，不要负向或「未记录」类。只输出每行一个 JSON，不要 markdown 代码块、不要行外说明、不要重复 system。"
)
SYSTEM_GEN_QA = (
    "你是记忆数据标注助手。先按 system（仅正向版）的「条数不固定、约 90% 要点覆盖」原则生成**全部**正向问答；行数不预设。\n"
    "raw 与 t 的格式要求同前：每条 JSON 的 t 须为所据 raw 行首方括号内时刻（与 raw 一致）。\n"
    "用户消息会要求**另起**若干行再写 1～3 条**负向**样本：q 问 raw 中**完全未出现**的**具体**点，a 整句仅「未记录」二字，t 为指定统一时刻。\n"
    "不要输出 markdown 代码块、不要行外说明、不要重复 system。"
)

SYSTEM_TWO_PHASE_1 = """你是事实摘录助手。请从用户给出的「对话」中提取客观事实，不得添加对话中未出现的推断。
输出格式严格遵守（每个事实块共两行，块与块之间单独一行只写 ---）：
事实：[一句客观概括]
原文支持：[从对话中逐字拷贝的一小段原话，不要改写、不要翻译式概括]

不要输出 markdown 代码块、不要输出与上述格式无关的开场白或总结。"""

SYSTEM_TWO_PHASE_2 = """你是文本一致性审查员。用户消息的第一段为任务说明，其后为「第一步」已列出的事实与原文支持对。
请逐条判断：该「事实」是否严格仅由对应「原文支持」即可推出（无推断、无原文未写的内容）？
只输出仍成立的条目：格式与第一步完全相同（事实：/原文支持：成对，块之间一行 ---）。不要输出判定为「否」且无法修正的条目；若可小幅修正事实使其严格可由原文支持，则输出修正后的两行。
不要输出「判定：是/否」行或审查过程说明。"""

SYSTEM_MEMORY_QC_3 = """你是「长期记忆质审（第三阶段）」：在候选记忆**已被提取并去重**之后，你只做**删劣、去矛盾**，不编造新事实。

【输入】
用户消息含：① 候选 `memory_update_list` 的 JSON 数组；② 若有「经二阶段核对的事实摘录」，须作为**强依据**判断真伪与优先级；③ 附对话原文（节选）供核对**时间先后**与**用户最后定稿**。

【输出】
仅输出**一个** JSON 对象，顶层**只含** `memory_update_list`（数组）。数组内为**保留**的条目，字段名与输入每条一致，且不得缺关键字段（subject、predicate、object、confidence、evidence、memory_sentence、absolute_time 等，输入有则须保留）。**不要**输出其它顶层键，不要 markdown 代码块、不要说明文字。

【必须整组丢弃的条目】
- 与**另一条**在逻辑上**互斥**（对同一类事实给出不能同时为真的结论）时：**只留一条**——优先与**二阶段摘录**一致者；无摘录则优先**更晚时刻**的定稿、或**更具体**的表述，删除其余互斥项。
- **evidence 空洞**或与 **memory_sentence** 明显不对应、无法与摘录/对话对上的条目。
- **同义重复**：多条说同一事实时只留信息更全或 evidence 更具体的一条。

若全部应丢弃，输出 `{"memory_update_list":[]}` 。
【禁止】把两条互斥项合并成一条新结论、或改写为输入中不存在的内容。"""


def _build_dialog_for_extract(conversation: str) -> str:
    turns = _parse_conversation_turns(conversation)
    if turns:
        body = _format_turns_for_refiner(turns)
        return f"<对话历史>\n{body}\n</对话历史>"
    return f"<对话历史>\n{conversation.strip()}\n</对话历史>"


class MemoryJsonParseError(Exception):
    """模型返回无法解析为 JSON 对象。"""

    def __init__(self, raw_model_output: str) -> None:
        self.raw_model_output = raw_model_output
        super().__init__("无法从模型输出中解析出 JSON 对象")


def run_two_phase_fact_grounding(
    conversation: str,
    *,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    timeout_sec: int = 600,
    think: bool | None = None,
    reference_time_iso: str | None = None,
) -> str:
    """第一步摘录事实+原文支持；第二步同一链路模型做一致性过滤。返回供 ``extract_memory_json`` 附带的文本。"""
    if think is None:
        think = BUILTIN_OLLAMA_THINK
    if reference_time_iso is None:
        reference_time_iso = reference_time_iso_local()
    turns = _parse_conversation_turns(conversation)
    if turns:
        dialog_body = _format_turns_for_refiner(turns)
    else:
        dialog_body = (conversation or "").strip()
    user1 = (
        _time_block_for_prompt(reference_time_iso)
        + "请从以下对话中提取客观事实（不添加推断）。\n"
        + "每行格式：\n事实：[概括]\n原文支持：[直接引用对话中的原句，不要修改]\n"
        + "---\n"
        + "对话：\n"
        + dialog_body
    )
    phase1 = _refine_llm_chat(
        llm_backend,
        ollama_host,
        ollama_model,
        deepseek_api_base,
        deepseek_api_key,
        deepseek_model,
        SYSTEM_TWO_PHASE_1,
        user1,
        timeout_sec=timeout_sec,
        think=think,
        options=BUILTIN_OLLAMA_OPTIONS_TWO_PHASE,
    )
    phase1_clean = _clean_model_text(phase1).strip()
    if not phase1_clean:
        return ""
    user2 = (
        "以下「事实」是否严格基于「原文支持」？回答是/否，如果否，请修正事实。\n\n"
        + phase1_clean
        + "\n\n只保留回答「是」的事实（格式与第一步相同：事实：/原文支持：/---）。不要输出判定过程。"
    )
    phase2 = _refine_llm_chat(
        llm_backend,
        ollama_host,
        ollama_model,
        deepseek_api_base,
        deepseek_api_key,
        deepseek_model,
        SYSTEM_TWO_PHASE_2,
        user2,
        timeout_sec=timeout_sec,
        think=think,
        options=BUILTIN_OLLAMA_OPTIONS_TWO_PHASE,
    )
    phase2_clean = _clean_model_text(phase2).strip()
    return phase2_clean if phase2_clean else phase1_clean


def run_third_stage_memory_qc(
    memory_update_list: list,
    conversation: str,
    *,
    grounded_facts: str = "",
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    timeout_sec: int = 600,
    think: bool | None = None,
    reference_time_iso: str | None = None,
) -> list:
    """经 LLM 筛除互斥/劣质/重复的记忆条目，返回新 ``memory_update_list``。"""
    if not isinstance(memory_update_list, list) or not memory_update_list:
        return []
    if think is None:
        think = BUILTIN_OLLAMA_THINK
    if reference_time_iso is None:
        reference_time_iso = reference_time_iso_local()
    body = json.dumps(memory_update_list, ensure_ascii=False, indent=2)
    if len(body) > 24000:
        body = body[:24000] + "\n…(截断)…\n"
    conv = (conversation or "").strip()
    if len(conv) > 14000:
        conv = conv[:14000] + "\n…(截断)…\n"
    user = _time_block_for_prompt(reference_time_iso) + "【候选 memory_update_list】\n" + body + "\n\n"
    if (grounded_facts or "").strip():
        user += "【经二阶段核对的事实摘录】\n" + (grounded_facts or "").strip() + "\n\n"
    user += "【对话原文（节选，供核对矛盾与时间先后）】\n" + conv
    raw = _refine_llm_chat(
        llm_backend,
        ollama_host,
        ollama_model,
        deepseek_api_base,
        deepseek_api_key,
        deepseek_model,
        SYSTEM_MEMORY_QC_3,
        user,
        timeout_sec=timeout_sec,
        think=think,
        options=BUILTIN_OLLAMA_OPTIONS_MEMORY_QC3,
    )
    obj = _parse_json_object_from_model(_clean_model_text(raw))
    out = obj.get("memory_update_list")
    if not isinstance(out, list):
        raise ValueError("三阶段质审：JSON 中 memory_update_list 非数组")
    return out


def _parse_json_object_from_model(text: str) -> dict:
    t = _clean_model_text(text)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(t[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("无法从模型输出中解析出 JSON 对象")


def extract_memory_json(
    conversation: str,
    *,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    timeout_sec: int = 600,
    think: bool | None = None,
    reference_time_iso: str | None = None,
    grounded_facts_text: str | None = None,
) -> dict:
    if think is None:
        think = BUILTIN_OLLAMA_THINK
    if reference_time_iso is None:
        reference_time_iso = reference_time_iso_local()
    dialog = _build_dialog_for_extract(conversation)
    gf = (grounded_facts_text or "").strip()
    if gf:
        dialog += "\n\n<经二阶段核对的事实>\n" + gf + "\n</经二阶段核对的事实>"
    user = (
        _time_block_for_prompt(reference_time_iso)
        + "下面是一段用户与 AI 助手的对话历史"
        + ("（含经二阶段核对的事实摘录）" if gf else "")
        + "，请仔细分析并按 system 要求**只输出**一个 JSON 对象。\n\n"
        + dialog
    )
    raw = _refine_llm_chat(
        llm_backend,
        ollama_host,
        ollama_model,
        deepseek_api_base,
        deepseek_api_key,
        deepseek_model,
        SYSTEM_MEMORY_EXTRACT_JSON,
        user,
        timeout_sec=timeout_sec,
        think=think,
        options=BUILTIN_OLLAMA_OPTIONS_EXTRACT_JSON,
    )
    try:
        return _parse_json_object_from_model(raw)
    except ValueError:
        raise MemoryJsonParseError(raw) from None


def _format_raw_fact_sentence(subject: str, predicate: str, obj: str) -> str:
    """
    名词性谓词：用户的{谓词}：{宾语}。
    谓词已以「喜欢/偏爱…」等开头时：{主语}{谓词}：{宾语}。（避免「用户的喜欢的…」叠字）
    """
    sub = (subject or "").strip() or "用户"
    pred = (predicate or "").strip()
    o = (obj or "").strip()
    if pred.startswith(("喜欢", "偏爱", "不喜欢", "厌恶", "爱吃", "不爱吃")):
        return f"{sub}{pred}：{o}。"
    return f"{sub}的{pred}：{o}。"


def memory_update_list_to_raw_text(
    memory_update_list: object,
    *,
    reference_time_iso: str | None = None,
) -> str:
    """将 memory_update_list 压成 train_memory 用的 raw：每行「[绝对时刻] 事实句」，一刻一条。"""
    if not isinstance(memory_update_list, list):
        return ""
    lines: list[str] = []
    for it in memory_update_list:
        if not isinstance(it, dict):
            continue
        sub = str(it.get("subject", "用户")).strip() or "用户"
        pred = str(it.get("predicate", "")).strip()
        obj = str(it.get("object", "")).strip()
        if not pred or not obj:
            continue
        at = ""
        raw_at = str(it.get("absolute_time") or "").strip()
        if raw_at:
            try:
                at = _parse_absolute_event_time(raw_at)
            except ValueError:
                at = ""
        if not at and reference_time_iso:
            at = reference_time_iso
        if not at:
            continue
        ms = str(it.get("memory_sentence") or "").strip()
        if ms:
            body = ms
        else:
            body = _format_raw_fact_sentence(sub, pred, obj)
        lines.append(f"[{at}] {body}")
    return "\n".join(lines)


def dedupe_memory_update_list(memory_update_list: object) -> list:
    """按 (subject, predicate, object) 去重，保留最后一次出现的整条（与定稿一致）。"""
    if not isinstance(memory_update_list, list):
        return []
    by_key: dict[tuple[str, str, str], dict] = {}
    key_order: list[tuple[str, str, str]] = []
    for it in memory_update_list:
        if not isinstance(it, dict):
            continue
        sub = str(it.get("subject", "用户")).strip() or "用户"
        pred = str(it.get("predicate", "")).strip()
        obj = str(it.get("object", "")).strip()
        if not pred or not obj:
            continue
        key = (sub, pred, obj)
        if key not in by_key:
            key_order.append(key)
        by_key[key] = it
    return [by_key[k] for k in key_order]


def generate_qa_jsonl_lines(
    raw_text: str,
    *,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    timeout_sec: int = 600,
    think: bool | None = None,
    reference_time_iso: str | None = None,
    include_negative_qa: bool = False,
) -> str:
    if think is None:
        think = BUILTIN_OLLAMA_THINK
    if reference_time_iso is None:
        reference_time_iso = reference_time_iso_local()
    if include_negative_qa:
        system = SYSTEM_GEN_QA
        user = (
            _time_block_for_prompt(reference_time_iso)
            + f"raw 正文如下（约 {len(raw_text)} 字）。\n"
            "第一步：只输出**正向**问答，行数不固定。须满足 system 要求：不凑条数，以**约 90% 覆盖** raw 中可独立被问到的要点为主，再输出尽量多、互不重复、每行一个 JSON（q、a、t），不要其它文字。\n"
            "第二步：在正向行**全部写完后**，**另起新行**再写 **1～3 条**负向问答：q 问 raw 中**完全未提及**的**具体**信息；a 整句仅「未记录」二字；负向的 t 一律为：\n"
            f"{reference_time_iso}\n"
            "（正、负两段的每一行仍须是单独一行 JSON。）\n\n"
            + raw_text.strip()
        )
    else:
        system = SYSTEM_GEN_QA_NO_NEGATIVE
        user = (
            _time_block_for_prompt(reference_time_iso)
            + f"raw 正文如下（约 {len(raw_text)} 字）。\n"
            "请输出**只含正向**的问答，**行数不固定**：先满足对 raw 可考要点的**约 90% 覆盖**，并在此前提下尽量多写、不凑数、不同义刷条数；每行一个 JSON 对象，字段为 q、a、t，不要其它文字。\n"
            "每条 q、a 须可经 raw 中某行事实支持；t 须为该事实行行首方括号内时刻（与 raw 完全一致）。\n\n"
            + raw_text.strip()
        )
    out = _refine_llm_chat(
        llm_backend,
        ollama_host,
        ollama_model,
        deepseek_api_base,
        deepseek_api_key,
        deepseek_model,
        system,
        user,
        timeout_sec=timeout_sec,
        think=think,
        options=BUILTIN_OLLAMA_OPTIONS_GEN_QA,
    )
    return _clean_model_text(out)


def _parse_qa_lines(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = obj.get("q")
        a = obj.get("a")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            row: dict[str, str] = {"q": q.strip(), "a": a.strip()}
            tv = obj.get("t") or obj.get("time")
            if isinstance(tv, str) and tv.strip():
                row["t"] = tv.strip()
            rows.append(row)
    return rows


def _is_under_dir(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _stem_from_staging_cli_input(
    resolved_input: Path | None,
    staging_dir: Path,
    fallback: str,
) -> str:
    """输入为 session_staging 下 session_*.txt（或旧版 cli_*.txt）时，用该文件名（不含扩展名）作为输出前缀，否则 fallback。"""
    if resolved_input is None:
        return fallback
    try:
        p = resolved_input.resolve()
        sd = staging_dir.resolve()
    except OSError:
        return fallback
    if not _is_under_dir(p, sd):
        return fallback
    if mem_paths.is_staging_session_txt(p.name):
        return p.stem
    return fallback


def _archive_staging_file(src: Path, bak_root: Path, user_id_for_bak: str) -> Path:
    """将 session_staging 下的文件移入 session_bak/<user_id>/YYYY-MM-DD/，重名则加时间后缀。"""
    dest_dir = mem_paths.session_bak_day_dir(bak_root, user_id_for_bak)
    dest = dest_dir / src.name
    if dest.exists():
        stem = src.stem
        suf = src.suffix
        dest = dest_dir / f"{stem}_{datetime.now().strftime('%H%M%S')}{suf}"
    shutil.move(str(src), str(dest))
    return dest


def _pipeline_for_one_file(
    *,
    text: str,
    stem: str,
    staging_to_archive: Path | None,
    out_dir: Path,
    bak_root: Path,
    llm_backend: str,
    host: str,
    model: str,
    deepseek_base: str,
    deepseek_key: str,
    ollama_think: bool,
    timeout_sec: int,
    event_time_override: str,
    generate_negative_qa: bool,
    user_id_for_bak: str,
    memory_qc3: bool,
) -> int:
    memory_json_path = out_dir / f"{stem}.memory.json"
    raw_path = out_dir / f"{stem}.raw.txt"
    qa_path = out_dir / f"{stem}.qa.jsonl"

    if llm_backend == "ollama":
        _log(f"[session_to_memory] 后端=ollama think={ollama_think}（False=关闭推理链）")
    else:
        _log(f"[session_to_memory] 后端=deepseek api_base={deepseek_base!r} model={model!r}")

    et = (event_time_override or "").strip()
    try:
        session_reference_time = resolve_session_event_time_iso(
            text, override=et or None
        )
    except ValueError as e:
        _log(f"错误：{e}")
        return 1
    _log(f"[session_to_memory] 事件时间锚点 reference_time={session_reference_time}")

    turns = _parse_conversation_turns(text)
    if turns:
        _log(f"[session_to_memory] 已解析 {len(turns)} 轮对话（User/Assistant）。")
    else:
        _log("[session_to_memory] 未解析到标准 User/Assistant 轮次。")

    _log("[session_to_memory] 步骤 1/6：判断是否有必要写入记忆 …")
    decision_text = decide_memory_relevance(
        text,
        llm_backend=llm_backend,
        ollama_host=host,
        ollama_model=model,
        deepseek_api_base=deepseek_base,
        deepseek_api_key=deepseek_key,
        deepseek_model=model,
        timeout_sec=timeout_sec,
        think=ollama_think,
        reference_time_iso=session_reference_time,
    )
    decision = _parse_memory_decision_first_line(decision_text)
    if decision is None:
        _log(
            "错误：模型未给出可解析的首行决策（须为「无需记住」或「需要记住」一字不差）。"
            f"模型输出如下：\n{decision_text!r}"
        )
        return 4
    if not decision:
        _log(
            f"[session_to_memory] 判定：{MEMORY_DECISION_SKIP_LINE}，不生成 {stem}.memory.json / {stem}.raw.txt / {stem}.qa.jsonl。"
        )
        _log("[session_to_memory] 结束（跳过提纯与 QA）。")
        if staging_to_archive is not None and staging_to_archive.is_file():
            try:
                dest = _archive_staging_file(staging_to_archive, bak_root, user_id_for_bak)
                _log(f"[session_to_memory] 已归档临时会话（无需记住）: {dest}")
            except OSError as e:
                _log(f"[session_to_memory] 警告：归档失败（临时文件仍保留）: {e}")
        return 0

    _log(f"[session_to_memory] 判定：{MEMORY_DECISION_KEEP_LINE}，继续提纯。")

    _log("[session_to_memory] 步骤 2/6：二阶段事实与原文核对 …")
    grounded_facts = run_two_phase_fact_grounding(
        text,
        llm_backend=llm_backend,
        ollama_host=host,
        ollama_model=model,
        deepseek_api_base=deepseek_base,
        deepseek_api_key=deepseek_key,
        deepseek_model=model,
        timeout_sec=timeout_sec,
        think=ollama_think,
        reference_time_iso=session_reference_time,
    )
    if grounded_facts.strip():
        _log(
            f"[session_to_memory] 二阶段摘录长度 {len(grounded_facts)} 字（将附于提取请求）。"
        )
    else:
        _log("[session_to_memory] 二阶段摘录为空，跳过附带（仍仅按对话提取）。")

    _log("[session_to_memory] 步骤 3/6：提取结构化记忆 JSON …")
    try:
        memory_obj = extract_memory_json(
            text,
            llm_backend=llm_backend,
            ollama_host=host,
            ollama_model=model,
            deepseek_api_base=deepseek_base,
            deepseek_api_key=deepseek_key,
            deepseek_model=model,
            timeout_sec=timeout_sec,
            think=ollama_think,
            reference_time_iso=session_reference_time,
            grounded_facts_text=grounded_facts if grounded_facts.strip() else None,
        )
    except MemoryJsonParseError as e:
        dump_path = out_dir / f"{stem}.memory.json.raw_model_output.txt"
        dump_path.write_text(e.raw_model_output, encoding="utf-8")
        _log(
            "错误：无法解析模型返回的 JSON；"
            f"原文已写入: {dump_path}"
        )
        return 5
    if not isinstance(memory_obj.get("memory_update_list"), list):
        memory_obj["memory_update_list"] = []
    if "deleted_memory" not in memory_obj:
        memory_obj["deleted_memory"] = []

    _n_mu_before = (
        len(memory_obj["memory_update_list"])
        if isinstance(memory_obj["memory_update_list"], list)
        else 0
    )
    memory_obj["memory_update_list"] = dedupe_memory_update_list(
        memory_obj["memory_update_list"]
    )
    _n_mu_after = len(memory_obj["memory_update_list"])
    if _n_mu_before > _n_mu_after:
        _log(
            f"[session_to_memory] memory_update_list 已去重：{_n_mu_before} → {_n_mu_after} 条（同 subject+predicate+object 只保留末条）"
        )

    if memory_qc3 and _n_mu_after > 0:
        _log("[session_to_memory] 步骤 4/6：三阶段记忆质审（去矛盾/劣质/重复）…")
        try:
            _n_qc0 = len(memory_obj["memory_update_list"])
            memory_obj["memory_update_list"] = run_third_stage_memory_qc(
                memory_obj["memory_update_list"],
                text,
                grounded_facts=grounded_facts if (grounded_facts or "").strip() else "",
                llm_backend=llm_backend,
                ollama_host=host,
                ollama_model=model,
                deepseek_api_base=deepseek_base,
                deepseek_api_key=deepseek_key,
                deepseek_model=model,
                timeout_sec=timeout_sec,
                think=ollama_think,
                reference_time_iso=session_reference_time,
            )
            _n_qc1 = len(memory_obj["memory_update_list"])
            _log(
                f"[session_to_memory] 三阶段质审完成：{_n_qc0} → {_n_qc1} 条"
            )
        except (ValueError, OSError) as e:
            _log(f"[session_to_memory] 警告：三阶段质审失败，保留去重后列表：{e!r}")
    else:
        if not memory_qc3:
            _log("[session_to_memory] 步骤 4/6：三阶段质审已跳过（--no-memory-qc3）。")
        else:
            _log("[session_to_memory] 步骤 4/6：无条目，跳过三阶段质审。")

    memory_obj["reference_time"] = session_reference_time

    memory_json_path.write_text(
        json.dumps(memory_obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"[session_to_memory] 已写入: {memory_json_path}")

    raw_text = memory_update_list_to_raw_text(
        memory_obj.get("memory_update_list"),
        reference_time_iso=session_reference_time,
    )
    if not raw_text.strip():
        _log(
            "错误：memory_update_list 展开后 raw 为空（无有效 subject/predicate/object 条目）。"
            "请检查模型输出或对话内容。"
        )
        return 6

    raw_path.write_text(raw_text, encoding="utf-8")
    _log(f"[session_to_memory] 已写入: {raw_path}（{len(raw_text)} 字）")

    if generate_negative_qa:
        _log(
            "[session_to_memory] 步骤 5/6：生成 QA（行数不固定；正向以覆盖 raw 约 90% 可考要点为主，"
            "并另附 1～3 条负向「未记录」）…"
        )
    else:
        _log(
            "[session_to_memory] 步骤 5/6：生成 QA（行数不固定，正向以覆盖 raw 约 90% 可考要点为主，无负向）…"
        )
    qa_blob = generate_qa_jsonl_lines(
        raw_text,
        llm_backend=llm_backend,
        ollama_host=host,
        ollama_model=model,
        deepseek_api_base=deepseek_base,
        deepseek_api_key=deepseek_key,
        deepseek_model=model,
        timeout_sec=timeout_sec,
        think=ollama_think,
        reference_time_iso=session_reference_time,
        include_negative_qa=generate_negative_qa,
    )
    rows = _parse_qa_lines(qa_blob)
    if not rows:
        _log(
            "警告：未能从模型输出中解析出任何 JSONL 行；"
            f"已将模型原文写入「{stem}.qa.jsonl.raw_model_output.txt」供排查。"
        )
        (out_dir / f"{stem}.qa.jsonl.raw_model_output.txt").write_text(
            qa_blob, encoding="utf-8"
        )
        return 2

    with qa_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _log(f"[session_to_memory] 已写入: {qa_path}（{len(rows)} 条）")

    _log(
        f"[session_to_memory] 步骤 6/6：用户画像（按需写入 {out_dir.parent}/<角色>{USER_PIC_GLOB_SUFFIX}，二阶段核对）…"
    )
    run_user_profile_update(
        text,
        raw_text,
        grounded_facts,
        out_dir,
        stem,
        llm_backend=llm_backend,
        ollama_host=host,
        ollama_model=model,
        deepseek_api_base=deepseek_base,
        deepseek_api_key=deepseek_key,
        deepseek_model=model,
        reference_time_iso=session_reference_time,
        timeout_sec=timeout_sec,
        think=ollama_think,
    )

    if staging_to_archive is not None and staging_to_archive.is_file():
        try:
            dest = _archive_staging_file(staging_to_archive, bak_root, user_id_for_bak)
            _log(f"[session_to_memory] 已归档临时会话: {dest}")
        except OSError as e:
            _log(f"[session_to_memory] 警告：归档失败（临时文件仍保留）: {e}")

    _log("[session_to_memory] 完成。")
    return 0


def main() -> None:
    # 无参数：按顶部 BUILTIN_STAGING_ON_RUN 注入 staging 模式（人只右键运行；有 argv 时走 argparse 正常解析）
    if len(sys.argv) == 1:
        mode = (BUILTIN_STAGING_ON_RUN or "all").strip().lower()
        if mode == "latest":
            sys.argv.append("--latest-staging")
        else:
            sys.argv.append("--all-staging")

    ap = argparse.ArgumentParser(
        description="会话提纯为 memory JSON + raw + QA（JSONL）；默认 DeepSeek 官网 API，可选 Ollama。"
    )
    ap.add_argument(
        "--llm-backend",
        choices=("deepseek", "ollama"),
        default=BUILTIN_LLM_BACKEND,
        help="大模型后端：deepseek 为官网 chat/completions；ollama 为本机 /api/chat",
    )
    ap.add_argument(
        "--deepseek-api-base",
        type=str,
        default=BUILTIN_DEEPSEEK_API_BASE,
        help="DeepSeek API 根 URL",
    )
    ap.add_argument(
        "--deepseek-api-key",
        type=str,
        default="",
        help="DeepSeek API Key；可省略，从环境变量 DEEPSEEK_API_KEY 或内置 BUILTIN_DEEPSEEK_API_KEY 读取",
    )
    ap.add_argument(
        "--ollama-host",
        type=str,
        default=BUILTIN_OLLAMA_HOST,
        help="--llm-backend ollama 时：Ollama 根地址",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="",
        help="ollama 时为本地模型名；deepseek 时为 API 模型 id（空则各后端内置默认）",
    )
    ap.add_argument(
        "--input",
        type=str,
        default="",
        help="原始会话文本文件路径；若使用 --latest-staging 则不必填",
    )
    ap.add_argument(
        "--latest-staging",
        action="store_true",
        help="使用 session_staging 目录内修改时间最新的 session_*.txt（或旧版 cli_*.txt）",
    )
    ap.add_argument(
        "--all-staging",
        action="store_true",
        help="依次处理 staging-dir 下全部 session_*.txt / 旧版 cli_*.txt（按修改时间从早到晚）；与 --latest-staging 同用时本选项优先",
    )
    ap.add_argument(
        "--staging-dir",
        type=str,
        default=str(_script_dir / "session_staging"),
        help="与 --latest-staging 配合：staging 目录路径",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="输出目录；留空则写入 session/<用户ID>/YYYY-MM-DD/（由输入文件在 session_staging 下的路径推断用户ID）",
    )
    ap.add_argument(
        "--stem",
        type=str,
        default="",
        help="输出文件名前缀（生成 {stem}.memory.json 等）。留空则：输入为 session_staging/session_*.txt（或旧版 cli_*.txt）时用该文件名（不含扩展名），否则为 daily",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="单次请求大模型的超时时间（秒）",
    )
    ap.add_argument(
        "--ollama-think",
        action="store_true",
        default=False,
        help="启用思考链（默认关闭以缩短耗时；DeepSeek-R1/Qwen3 等建议保持关闭）",
    )
    ap.add_argument(
        "--bak-dir",
        type=str,
        default=BUILTIN_SESSION_BAK,
        help="归档 session_staging 源文件的根目录（判定无需记住或提纯成功后均会尝试移动；默认 session_bak）",
    )
    ap.add_argument(
        "--event-time",
        type=str,
        default="",
        help="会话/记忆的事件发生时间，须为绝对时间 YYYY-MM-DD HH:MM:SS。"
        "若不填：优先用输入中首条可解析的 [轮次时间: …]（同格式）；若无则回退为当前本地时刻（同为该格式）。",
    )
    ap.add_argument(
        "--negative-qa",
        action=argparse.BooleanOptionalAction,
        default=BUILTIN_NEGATIVE_QA,
        help="是否在 QA 中含负向「未记录」样本；默认由文件顶部 BUILTIN_NEGATIVE_QA 决定（默认关闭）",
    )
    ap.add_argument(
        "--memory-qc3",
        action=argparse.BooleanOptionalAction,
        default=BUILTIN_MEMORY_QC3,
        help="是否在去重后做大模型三阶段质审（去矛盾/劣质/重复）；默认开，加 --no-memory-qc3 关闭",
    )
    args = ap.parse_args()

    llm_backend = (args.llm_backend or "").strip().lower() or BUILTIN_LLM_BACKEND
    model = (args.model or "").strip()
    if not model:
        model = BUILTIN_DEEPSEEK_MODEL if llm_backend == "deepseek" else BUILTIN_OLLAMA_MODEL
    deepseek_base = (args.deepseek_api_base or "").strip().rstrip("/") or BUILTIN_DEEPSEEK_API_BASE
    deepseek_key = (args.deepseek_api_key or "").strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_key:
        deepseek_key = (BUILTIN_DEEPSEEK_API_KEY or "").strip()
    if llm_backend == "deepseek" and not deepseek_key:
        _log("错误：--llm-backend deepseek 需要 API Key，请设 DEEPSEEK_API_KEY 或 --deepseek-api-key，或改用 --llm-backend ollama")
        sys.exit(1)

    staging_dir = Path(args.staging_dir)
    bak_root = Path(args.bak_dir)
    session_data_root = _script_dir / "session"
    host = args.ollama_host.rstrip("/")
    ollama_think = bool(args.ollama_think)
    et = (args.event_time or "").strip()
    generate_negative_qa = bool(args.negative_qa)
    memory_qc3 = bool(args.memory_qc3)

    if args.all_staging:
        if args.latest_staging:
            _log(
                "[session_to_memory] 提示：已指定 --all-staging，将处理目录内全部 session_*.txt（含旧版 cli_*.txt），忽略 --latest-staging。"
            )
        files = mem_paths.iter_staging_cli_files_oldest_first(staging_dir)
        if not files:
            _log(f"错误：--all-staging 但在「{staging_dir}」下未找到 session_*.txt 或 cli_*.txt。")
            sys.exit(1)
        if (args.stem or "").strip():
            _log("错误：--all-staging 时每条输出 stem 取自文件名，请去掉 --stem。")
            sys.exit(1)
        last_rc = 0
        _log(f"[session_to_memory] --all-staging 共 {len(files)} 个文件，逐个提纯。")
        pbar = (
            tqdm(files, desc="staging 提纯", unit="个", ncols=100)
            if tqdm is not None
            else None
        )
        iterable = pbar if pbar is not None else files
        for idx, fp in enumerate(iterable, 1):
            if pbar is not None:
                pbar.set_postfix_str(fp.name[:56], refresh=False)
            else:
                _log("")
                _log(
                    f"[session_to_memory] ========== 批量 {fp.name} （{idx}/{len(files)}）=========="
                )
            text_i = fp.read_text(encoding="utf-8")
            stem_i = _stem_from_staging_cli_input(
                fp.resolve(), staging_dir, BUILTIN_OUTPUT_STEM
            )
            uid_i = mem_paths.user_id_from_staging_cli_path(fp.resolve(), staging_dir)
            out_i = (
                Path(args.output_dir).resolve()
                if args.output_dir.strip()
                else mem_paths.session_day_dir(session_data_root, uid_i)
            )
            stg_arc = fp if _is_under_dir(fp, staging_dir) else None
            _log(
                f"[session_to_memory] 输出前缀 stem={stem_i!r} → {stem_i}.memory.json / {stem_i}.raw.txt / {stem_i}.qa.jsonl"
                f"（及按需 {out_i.parent}/<角色>{USER_PIC_GLOB_SUFFIX}）"
            )
            _log(f"[session_to_memory] 用户目录 user_id={uid_i!r} → {out_i}")
            rc = _pipeline_for_one_file(
                text=text_i,
                stem=stem_i,
                staging_to_archive=stg_arc,
                out_dir=out_i,
                bak_root=bak_root,
                llm_backend=llm_backend,
                host=host,
                model=model,
                deepseek_base=deepseek_base,
                deepseek_key=deepseek_key,
                ollama_think=ollama_think,
                timeout_sec=args.timeout,
                event_time_override=et,
                generate_negative_qa=generate_negative_qa,
                user_id_for_bak=uid_i,
                memory_qc3=memory_qc3,
            )
            if rc != 0:
                last_rc = rc
                _log(f"[session_to_memory] 本条退出码 {rc}，继续下一条。")
        sys.exit(last_rc)

    staging_to_archive: Path | None = None
    resolved_input: Path | None = None
    text = ""
    if args.latest_staging:
        latest = mem_paths.latest_staging_cli_file(staging_dir)
        if latest is None:
            _log(
                f"错误：在「{args.staging_dir}」下未找到 session_*.txt 或 cli_*.txt。"
                "请先运行 memory_extract_cli 产生对话日志，或使用 --input 指定文件。"
            )
            sys.exit(1)
        text = latest.read_text(encoding="utf-8")
        staging_to_archive = latest
        resolved_input = latest
        _log(f"[session_to_memory] 输入文件: {latest}")
    elif args.input.strip():
        inp = Path(args.input)
        text = inp.read_text(encoding="utf-8")
        resolved_input = inp.resolve()
        if _is_under_dir(inp, staging_dir):
            staging_to_archive = resolved_input
        _log(f"[session_to_memory] 输入文件: {args.input}")
    else:
        _log(
            "错误：请指定 --input 会话文件路径，或加参数 --latest-staging / --all-staging。"
        )
        sys.exit(1)

    stem = (args.stem or "").strip()
    if not stem:
        stem = _stem_from_staging_cli_input(resolved_input, staging_dir, BUILTIN_OUTPUT_STEM)
    uid = mem_paths.user_id_from_staging_cli_path(resolved_input, staging_dir)
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir.strip()
        else mem_paths.session_day_dir(session_data_root, uid)
    )
    _log(
        f"[session_to_memory] 输出前缀 stem={stem!r} → {stem}.memory.json / {stem}.raw.txt / {stem}.qa.jsonl"
        f"（及按需 {out_dir.parent}/<角色>{USER_PIC_GLOB_SUFFIX}）"
    )
    _log(f"[session_to_memory] 用户目录 user_id={uid!r} → {out_dir}")
    sys.exit(
        _pipeline_for_one_file(
            text=text,
            stem=stem,
            staging_to_archive=staging_to_archive,
            out_dir=out_dir,
            bak_root=bak_root,
            llm_backend=llm_backend,
            host=host,
            model=model,
            deepseek_base=deepseek_base,
            deepseek_key=deepseek_key,
            ollama_think=ollama_think,
            timeout_sec=args.timeout,
            event_time_override=et,
            generate_negative_qa=generate_negative_qa,
            user_id_for_bak=uid,
            memory_qc3=memory_qc3,
        )
    )


if __name__ == "__main__":
    main()
