# -*- coding: utf-8 -*-
"""记忆流水线编排核心：路由 → 拼装记忆（memory_api + 端侧工具）→ 二阶段补充确认（可选再跑一轮）→ 终答；多轮 ``history`` 以 OpenAI messages 传入模型。"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import memory_user_paths as mem_paths

from user_profile_tool import file_key_from_cli_or_fields, memory_block_for_user_profile

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
# 记忆事实中含「[记忆提取]」时追加 POST /memory/extract；含路由初始问句的总次数上限。
BUILTIN_MEMORY_EXTRACT_EXPAND_MAX_CALLS = 40
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


SYSTEM_ROUTER = f"""你是记忆检索路由模块：须**只**通过 API 的 Tool 调用 `submit_memory_retrieval_route` 提交本回合配置，与 DeepSeek 官方 Tool Calls 机制对齐；**不要**在 assistant 的 content 里写整段自然语言答句或面向用户的说明（可留空）。提交参数为 JSON 对象，字段：need_memory、memory_queries、tool_calls。

【本模块 tool_calls 条目的含义（名称须原样出现在每条 tool 的 name 字段）】
- 「用户画像提取工具」：从本机已落盘的 Markdown 用户/角色画像文件读取（路径为 `session/<数据归属>/<画像键>.user.pic.md`）。**与 memory_queries 调用的记忆 API 是两条不同通路**：仅发问句**不会**自动读该文件；凡需利用画像文件，**必须**出现在你提交的 tool_calls 里。

【何时必须带 tool_calls（与 memory_queries 同时存在，不得二选一省掉）】
- 用户句子里出现**具体人名/明确第三方称谓**，并意在了解「是谁、认不认识、与我的关系、个人信息、是否聊过」等时：除须给出**恰好** {BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N} 条 memory_queries（与下段硬性条数相同）外，**必须**在 tool_calls 里为**每一个**在作答中可能用到的**画像键**各写（或合理合并为）对「用户画像提取工具」的调用（`file_key` 或 `cli_command: user_profile <画像键>`，如 `user_profile 谢翠萍`；涉及多人时允许多条）。若用户只泛问「你记得我朋友吗」未点名，可不在 tool 里写画像键；一旦点名，必须写，**由你在此 JSON 中一次性列全，勿假设下游会代你补全**。

【人物/事件与画像（无服务端代填，全靠本步 JSON）】
- 凡问题涉及**人物、关系、事件中的角色、第三人称/昵称/指代**等，且可能对应已落盘 `*.user.pic.md` 的，你须在 tool_calls 中**显式**列出要拉取的全部画像键；需要几人就写几条。memory_queries 只走记忆语言模型，**不能**替代画像文件；与人物侧写相关的题，**宜**同时带齐「相关人物的 user_profile」与覆盖事实检索的 memory_queries。

【硬性约束】
- 禁止回答用户当前句子里的具体问题；禁止「根据你的记忆…」「你可以…」等对话式正文。你是在提交路由配置，不是在与用户聊天。
- 通过 `submit_memory_retrieval_route` 的**参数**传 need_memory、memory_queries、tool_calls，结构须可被标准库 json 解析为对象；不得依赖 markdown 围栏或冗长自然语言顶替代。

若 need_memory 为 false：memory_queries 必须为 []。

何时 need_memory 为 true：只要回答用户这句话时，有可能因已存储的个人信息而答得更准或更可个性化，就必须为 true；由你自行判断。禁止在后续答复里向用户追问其个人信息却不在此先检索记忆。

若 need_memory 为 true：memory_queries 必须恰好 {BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N} 条互不重复、语义可区分的完整中文问句（下游对记忆模型逐条调用检索）；按相关性从高到低排序（第 1 条最相关）。若确实不足 {BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N} 个独立角度，也须用不重复、仍相关的问句凑满，禁止用占位废话凑数。
每条须为完整问句；禁止仅用「用户的xxx」这类名词短语。语义重复或近义只保留一条。

【tool_calls】
- 若用户**未**点名任何具体人名/角色名、且确定不需要读 `.user.pic.md`：tool_calls 为 []。一旦点名某人并可能用到其画像，**禁止**将 tool_calls 留空，**必须**至少含一条 `用户画像提取工具`。
- 单条示例如下（名称字段须为「用户画像提取工具」）：
  {{"name": "用户画像提取工具", "use_tool": true, "cli_command": "user_profile 谢翠萍"}}
- **路径语义（与「小明的 user_id」无关）**：落盘为 `session/<数据目录归属ID>/<人物画像键>.user.pic.md`。**人物画像键**（如 `小明`）是文件名里的角色/第三方/本人别名，**不是**该人物在系统里的「登录 user_id」；**数据目录归属ID** 由运行环境从「当前产品会话/账号」注入，只表示**这份记忆数据存在哪个用户目录下**，**不要把被问的人名（小明）理解成这个目录ID**。问「小明的信息」时，`file_key` / `cli_command` 里应写 **「小明」** 作为画像键，而不是把小明当成当前账号。
- cli_command 约定：仅写子命令与**画像键**（人物/角色名），形式为 `user_profile <画像键>`；不要手写磁盘路径。
- 若同时用 file_key 与 cli_command 给出画像键，优先以 file_key 为准（可二选一，不必重复）。

【当前时间（必看）】
- 本回合用户消息**首行**会附带「当前时间（本地）」。当用户原话含「昨天、今天、今天上午、上周」等**相对时间**时，你**须结合该行时间**理解指代，并在 memory_queries 中写成**可逐条去记忆库检索的完整问句**（若有利于命中，允许在问句中写明对「昨天」所对应的**公历日期**或时间范围，勿让下游猜「昨天是哪天」）。

**submit_memory_retrieval_route 的 arguments 结构示例**（键名与类型须一致，示例勿照搬）：
{{"need_memory": true, "memory_queries": ["……", "……"], "tool_calls": [{{"name": "用户画像提取工具", "use_tool": true, "cli_command": "user_profile 某某"}}]}}
"""

ROUTER_JSON_RETRY_USER = (
    "上一条输出不符合约定：请仅输出一个 JSON 对象，键为 need_memory、memory_queries、tool_calls；"
    "不要 markdown、不要解释、不要回答用户原话里的具体问题。"
    "若用户原话里出现具体人名、且你写了与该人相关的 memory_queries，则 tool_calls 中必须包含对「用户画像提取工具」的调用（cli_command: user_profile 该人名）。"
)

ROUTER_DS_TOOL_RETRY_USER = (
    "上一条未正确调用 submit_memory_retrieval_route：请仅通过该 function 再提交合法 arguments（JSON 对象，键为 need_memory、memory_queries、tool_calls）；"
    "不要 markdown、不要向用户作自然语言长答。若点名人名且需要画像，tool_calls 须含用户画像工具。"
)

SYSTEM_REFINER = f"""你是「记忆与工具二阶段确认」模块：在首轮已下发检索/工具并得到拼装结果后，须**只**通过 Tool 调用 `submit_memory_refiner_supplement` 提交配置（与官方 Tool Calls 对齐）；不回答用户、不对用户作闲聊。assistant 的 content 可留空。参数字段：need_supplement、memory_queries、tool_calls。

【硬性约束】
- 通过 `submit_memory_refiner_supplement` 的 **arguments** 传 JSON 对象，可被 json.loads 解析；无 markdown 围栏。
- 键：need_supplement（布尔）、memory_queries（字符串数组）、tool_calls（对象数组，格式与首阶段路由的 tool_calls 相同）。

当且仅当结合【用户原话】与【首轮已拼装记忆】，你认为仍**明显缺失**、且可通过**额外**向记忆 API 提问或**额外**拉取某人物画像才能更好作答时，need_supplement 为 true，并只填写**比首轮多出来的**、**不重复**的 memory_queries 与/或 tool_calls。
若无需补充、或已足够：need_supplement 为 false，且 memory_queries 为 []、tool_calls 为 []。

若【首轮已拼装记忆】里有多段 [记忆事实] 对**同一实体或同一问题**互斥，**不要**为「在叙述上圆场」而补写会与任一段直接冲突的假设性问句；此时优先 need_supplement=false，除非确有必要用**单条、中性、可区分**的澄清问句，且**不得**在问句中把未核实的某一方当已成立事实写死。

补充的 memory_queries：至多 {BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N} 条，须为可区分的完整中文问句，**禁止**与首轮或彼此语义重复。不需要补充问句时可为 []。
用户画像：规则与首段路由一致，点名某人物时可用 `file_key` 或 `cli_command`：`user_profile <画像键>`，数据目录由环境注入。

【画像关联行（与落盘 ``*.user.pic.md`` 一致）】
若某段 [记忆事实] 的**整段正文**仅为**一行**形如 ``[请调取"某画像键"的画像]``（引号内为**已有**的 `file_key`，如 用户、助手 或具体人名），表示：**本段 [记忆检索] 所对应的昵称/画像键** 与 引号内键 **为同一条画像数据的别名关联**，**不得**用其它 [记忆事实] 里对同一昵称/问句的**矛盾结论**去覆盖或与之合并（例如同轮若另有「X 是谁？→ 某人」的片段与上述关联冲突，**以本关联行所指向的画像键为准**；需补充时 tool_calls 只须确保已拉取 `user_profile <引号内键>`，**不要**为「自洽」再拉与关联目标冲突的画像键）。

**submit_memory_refiner_supplement 的 arguments 示例**：{{"need_supplement": false, "memory_queries": [], "tool_calls": []}}

二阶段中，【用户原话】**之前**会附与首轮相同的「当前时间（本地）」；补写 memory_queries 时须**同样**结合该时间，避免与首轮对「昨天/今天」的指代相矛盾。
"""

REFINER_JSON_RETRY_USER = (
    "上一条输出不符合约定：请仅输出一个 JSON 对象，键为 need_supplement、memory_queries、tool_calls；"
    "不要 markdown、不要解释、不要直接回答用户问题。"
)

REFINER_DS_TOOL_RETRY_USER = (
    "上一条未正确调用 submit_memory_refiner_supplement：请仅通过该 function 再提交合法 arguments（键为 need_supplement、memory_queries、tool_calls）；"
    "不要 markdown、不要直接答用户题。"
)


def _deepseek_tool_memory_router() -> dict:
    n = BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N
    return {
        "type": "function",
        "function": {
            "name": "submit_memory_retrieval_route",
            "description": "提交首段记忆检索路由：need_memory、memory_queries、tool_calls。仅通过本 function 的 arguments 传参（与 DeepSeek Tool Calls 一致）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "need_memory": {
                        "type": "boolean",
                        "description": "是否需向记忆 API 用 memory_queries 逐条检索。",
                    },
                    "memory_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            f"need_memory 为 false 时须为 []；为 true 时须凑满 {n} 条互不重复、可区分的中文问句。"
                        ),
                    },
                    "tool_calls": {
                        "type": "array",
                        "description": "端侧用户画像等；项含 name、use_tool、cli_command、file_key。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "use_tool": {"type": "boolean"},
                                "cli_command": {"type": "string"},
                                "file_key": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["need_memory", "memory_queries", "tool_calls"],
            },
        },
    }


def _deepseek_tool_memory_refiner() -> dict:
    n = BUILTIN_ROUTER_MEMORY_QUERIES_TOP_N
    return {
        "type": "function",
        "function": {
            "name": "submit_memory_refiner_supplement",
            "description": "提交二阶段是否补充 memory_queries 与/或端侧用户画像拉取。仅通过 arguments 传参。",
            "parameters": {
                "type": "object",
                "properties": {
                    "need_supplement": {
                        "type": "boolean",
                        "description": "是否需在此轮已拼装记忆之外再补检索/工具。",
                    },
                    "memory_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"无需补充时 []；需补充时最多 {n} 条。",
                    },
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "use_tool": {"type": "boolean"},
                                "cli_command": {"type": "string"},
                                "file_key": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["need_supplement", "memory_queries", "tool_calls"],
            },
        },
    }


SYSTEM_FINAL = """你是智能助手。用户消息里含：首行**当前时间（本地）**、再是【已检索记忆】、再是【用户原话】。

- 用首行**当前时间**理解用户说的「昨天、今天上午、上周」等相对时间；与 [记忆事实] 中涉及时间的表述对照看，不臆造未出现在记忆里的具体日期。
- **【已检索记忆】**与**【用户原话】**仍按下述规则使用。

【已检索记忆】中，每条有效记忆为两行：`[记忆检索]：` 与 `[记忆事实]：`（多段之间空行分隔）。
- **`[记忆事实]：` 之后才是从用户长期记忆里检索到的可引用内容；**作答应严格以同一段中 [记忆事实] 后的文字为依据**，不要虚构、不要凭 [记忆检索] 或常识补全未出现在该段事实里的信息。
- **`[记忆检索]：` 之后仅为系统为检索而生成的问句（路由下发给记忆 API 的用语），不是用户原话，不是已证实经历；可能含未经验证假设。禁止把 [记忆检索] 里的具体说法当事实写进答案，除非同一段 [记忆事实] 里也有同样信息。
- **[记忆检索] 仅作语义索引**；综合答复时以 [记忆事实] 为准，避免孤立短词产生歧义。
- **画像关联行**：若某段 [记忆事实] 仅有**一行** ``[请调取"某键"的画像]``，表示 [记忆检索] 里对应的昵称/人物与 引号内 ``某键`` **同一套画像**（读取时已解析为 ``某键`` 的正文，与其它段中「同昵称是谁」的**矛盾**条目不一并采信；若同时存在「## 用户画像」等已展开内容且与问句相关，**优先**采信**展开后**的姓名/关系，**不要**为合并矛盾而断言错误身份）。

**选用哪些 [记忆事实]：**
1）在有多段记忆时，**优先**采信**表述清楚、与【用户原话】当前问题关联度高**的 [记忆事实] 片段。
2）**主动忽略**与当前问题**明显无关**的片段，以及**信息量过低、无法支持作答**的片段（例如整段仅为无上下文的单字或词、如单独的「没有」「基座」、或与问题主题毫无关系的碎片）。
3）**互斥片段不得兼用为真**：若多段 [记忆事实] 对**同一主题**（同一人、同一事件结论、同一状态/数值/时间等）**互相矛盾**，**禁止**在答案中同时采信、禁止把它们**拼成一段**看似连贯的叙述。你必须**只采用其中与当前问题最相关、且单条相对完整**的一条，**或**直接说明「记忆中存在不同说法」、避免断言唯一结论，**不得**为消除表面冲突而脑补未出现在任何 [记忆事实] 中的情节。
3′）若经筛选后，仍没有任何 [记忆事实] 能**无矛盾地**支持回答，再按下面「（无）或无关」一条处理，不得把被忽略的碎片硬塞进答案。

硬性规则：
1）若【已检索记忆】不是「（无）」、且经上款筛选后仍有**可用**的 [记忆事实]，你须以这些内容为依据组织回答，不得再说「没有相关信息」「无法回答」「缺少上下文」等推脱语；但若可用片段在关键事实上互斥，按上款 3）与 3′）处理，**不得**为「能答上」而合并矛盾点。
2）用户问及过往讨论、聊过内容时：在符合上款「选用记忆」原则的前提下，根据相关 [记忆事实] 与主题概括；材料很短也可作简要关联说明。
3）仅当【已检索记忆】为「（无）」、或**筛选后**仍无与问题相关的可用事实时，才可简要说明记忆里没有这点、或就问题作不依赖记忆的回答。
4）用自然、简洁的中文直接回答，不要复述本说明。
5）禁止替用户做主：被采信的 [记忆事实] 只是辅助依据。若 [记忆事实] 仅表明对某类选项的接受、容忍或中立，且未排除其它选项，禁止据此把建议收窄为只推该类、或表现得像用户已选定该类；若信息不足以唯一确定偏好，应并列合理选项或说明由用户自行决定。
6）**自称与句首格式**：以第一人称「我」回答即可。介绍自己、身份或版本时**直接**用「我是…」「我是记忆模型 v1.0…」等，**不要**在回答**最开头**写「[助手名/昵称] + 逗号 + 我…」（如「小忆，我是…」），以免用户误解为你在**称呼他**。若需带出助手名，可写在「我是」之后或主句中（如「我是小忆，是记忆模型 v1.0…」），而不要用对方昵称**起头**当呼语。"""


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


def _emit_think(step: str, msg: str) -> dict:
    """同一条编排日志：``log`` 写控制台，``think.text`` 为 ``[{step}] {msg}`` 给 SSE 页面，内容与控制台一致（不压成单行）。"""
    log(step, msg)
    phase = (
        "final"
        if step.startswith("3-")
        else (
            "memory"
            if step.startswith("2-") or step.startswith("2b-")
            else "router"
        )
    )
    return {"kind": "think", "phase": phase, "text": f"[{step}] {msg}"}


def _step_break() -> None:
    print("", flush=True)


def _now_context_line() -> str:
    """本回合各阶段（路由 / 二阶段 / 终答）共用：本地时间，供「昨天、今天上午」等相对表述对齐。"""
    now = datetime.now()
    w = "一二三四五六日"[now.weekday()]
    return now.strftime(
        f"当前时间（本地）：%Y年%m月%d日 星期{w} %H:%M:%S"
    )


def _router_messages(history: list[tuple[str, str]], user_text: str) -> list[dict]:
    t = _now_context_line()
    m: list[dict] = [{"role": "system", "content": SYSTEM_ROUTER}]
    for u, a in history:
        m.append({"role": "user", "content": u})
        m.append({"role": "assistant", "content": a})
    m.append({"role": "user", "content": f"{t}\n\n{user_text}"})
    return m


def _refiner_messages(
    history: list[tuple[str, str]],
    user_text: str,
    memory_text_first: str,
) -> list[dict]:
    t = _now_context_line()
    m: list[dict] = [{"role": "system", "content": SYSTEM_REFINER}]
    for u, a in history:
        m.append({"role": "user", "content": u})
        m.append({"role": "assistant", "content": a})
    block = (
        f"{t}\n\n"
        f"【用户原话】\n{user_text}\n\n"
        f"【首轮已拼装记忆】\n{memory_text_first if (memory_text_first or '').strip() else '（无）'}"
    )
    m.append({"role": "user", "content": block})
    return m


def _final_stage_think_message(
    memory_text_this_round: str, prior_session_memory: str = ""
) -> str:
    """终答前 think：往轮+本轮 任一 有实质内容则按「有记忆」表述。"""
    p = (prior_session_memory or "").strip()
    c = (memory_text_this_round or "").strip()
    if p or c:
        return "已检索到记忆事实（含本会话往轮与/或本轮），正在据此生成回答…"
    return "未命中可用记忆，正在直接生成回答…"


def _join_prior_session_memory_blocks(blocks: list[str] | None) -> str:
    if not blocks:
        return ""
    segs = [(s or "").strip() for s in blocks]
    segs = [s for s in segs if s]
    if not segs:
        return ""
    return "\n\n---\n\n".join(segs)


def _final_user_block_memory_section(
    prior_session_memory: str, memory_text_this_round: str
) -> str:
    """终答里「已检索记忆」正文：含往轮（列表拼接）+ 本轮。"""
    p = (prior_session_memory or "").strip()
    c = (memory_text_this_round or "").strip()
    if p and c:
        return (
            f"【往轮已检索记忆（本会话）】\n{p}\n\n"
            f"【本轮新检索记忆】\n{c}"
        )
    if p:
        return f"【往轮已检索记忆（本会话）】\n{p}\n\n【本轮新检索记忆】\n（无）"
    if c:
        return c
    return "（无）"


def _final_messages(
    history: list[tuple[str, str]],
    memory_text: str,
    user_text: str,
    *,
    prior_session_memory: str = "",
) -> list[dict]:
    t = _now_context_line()
    mem_section = _final_user_block_memory_section(prior_session_memory, memory_text)
    block = (
        f"{t}\n\n"
        f"【已检索记忆】\n{mem_section}\n\n"
        f"【用户原话】\n{user_text}"
    )
    m: list[dict] = [{"role": "system", "content": SYSTEM_FINAL}]
    for u, a in history:
        m.append({"role": "user", "content": u})
        m.append({"role": "assistant", "content": a})
    m.append({"role": "user", "content": block})
    return m


def _log_final_submission_to_console(fmsgs: list[dict]) -> None:
    """在服务端控制台打印本次提交给终答模型的大致内容；不经 ``_emit_think``，故不会进入 SSE/聊天页 think。"""
    if not fmsgs:
        log("3-final-in", "（无终答 fmsgs，跳过）")
        return
    n = len(fmsgs)
    log("3-final-in", f"── 终答提交给大模型的 messages 共 {n} 条（仅控制台，不经 SSE/思考区）──")
    last = n - 1
    for i, m in enumerate(fmsgs):
        role = str(m.get("role", ""))
        c = m.get("content", "")
        if not isinstance(c, str):
            c = str(c)
        if i == last and role == "user":
            log(
                "3-final-in",
                "── 最后一条 user（含【已检索记忆】与【用户原话】，为排查重点）──",
            )
            if len(c) > 200_000:
                c = c[:200_000] + "\n…(截断 20 万字符)"
            log("3-final-in", c)
            continue
        if role == "system" and i == 0:
            log(
                "3-final-in",
                f"── [{i}] system 共 {len(c)} 字（与代码中 SYSTEM_FINAL 一致，此处不整段展开）──",
            )
            continue
        t = c
        if len(t) > 4000:
            t = t[:4000] + f"\n…(截断，该条全文 {len(c)} 字)"
        log("3-final-in", f"── [{i}] {role} ──\n{t}")


def _parse_tool_calls_from_obj(obj: dict) -> list[dict]:
    raw = obj.get("tool_calls")
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        if x.get("use_tool") in (False, 0, "false", "no", "0"):
            continue
        name = str(x.get("tool", x.get("name", ""))).strip()
        cc = str(x.get("cli_command", "")).strip()
        fk = str(x.get("file_key", "")).strip()
        if not name and not cc and not fk:
            continue
        out.append(
            {
                "name": name,
                "cli_command": cc,
                "file_key": fk,
            }
        )
    return out


def _is_user_profile_tool_item(item: dict) -> bool:
    n = (item.get("name") or "").strip()
    if n in (
        "用户画像提取工具",
        "用户画像提取",
        "user_profile",
        "user_profile_extract",
    ):
        return True
    if "用户画像" in n and "工具" in n:
        return True
    cc = (item.get("cli_command") or "").lower()
    if cc.startswith("user_profile") or "用户画像" in (item.get("cli_command") or ""):
        return True
    return bool((item.get("file_key") or "").strip() and not n)


def _parse_router_json(text: str) -> tuple[bool, list[str], list[dict]]:
    t = _clean_model_text(text)

    def _pull(obj: dict) -> tuple[bool, list[str], list[dict]] | None:
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
        tcalls = _parse_tool_calls_from_obj(obj)
        return need, dedup, tcalls

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


def _parse_refiner_json(text: str) -> tuple[bool, list[str], list[dict]]:
    t = _clean_model_text(text)

    def _pull(obj: dict) -> tuple[bool, list[str], list[dict]] | None:
        if not isinstance(obj, dict):
            return None
        need = bool(obj.get("need_supplement"))
        raw_mq = obj.get("memory_queries")
        out: list[str] = []
        if isinstance(raw_mq, list):
            for x in raw_mq:
                s = str(x).strip()
                if s:
                    out.append(s)
        seen: set[str] = set()
        dedup: list[str] = []
        for q in out:
            if q not in seen:
                seen.add(q)
                dedup.append(q)
        tcalls = _parse_tool_calls_from_obj(obj)
        if not need:
            return False, [], []
        return need, dedup, tcalls

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
    raise ValueError(f"无法从二阶段确认输出解析 JSON：{text!r}")


def _build_seen_dedup_from_first_round(
    memory_queries: list[str],
    tool_calls: list[dict] | None,
) -> tuple[set[str], set[str]]:
    qseen: set[str] = set()
    for q in memory_queries or []:
        n = _normalize_memory_query_for_dedup((q or "").strip())
        if n:
            qseen.add(n)
    fseen: set[str] = set()
    for it in tool_calls or []:
        if not _is_user_profile_tool_item(it):
            continue
        cc = str(it.get("cli_command", "")).strip()
        fk = file_key_from_cli_or_fields(
            cc, str(it.get("file_key", "")).strip() or None
        )
        if fk:
            fseen.add(fk)
    return qseen, fseen


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
    deepseek_task: str = "router",
) -> str:
    """首段路由与二阶段共用的单轮 LLM 调用。DeepSeek：``tools``+``tool_choice=required`` 与官方 Tool Calls 对齐，且 ``thinking=disabled``；Ollama：``format=json`` 文本。"""
    if llm_backend == "deepseek":
        tdef = (
            _deepseek_tool_memory_refiner()
            if deepseek_task == "refiner"
            else _deepseek_tool_memory_router()
        )
        return deepseek_chat_messages(
            deepseek_api_base,
            deepseek_api_key,
            deepseek_model,
            messages,
            timeout_sec=timeout_sec,
            max_tokens=BUILTIN_DEEPSEEK_ROUTER_MAX_TOKENS,
            temperature=BUILTIN_DEEPSEEK_ROUTER_TEMP,
            dump_raw_request_body=dump_llm_requests,
            tools=[tdef],
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
) -> tuple[str, bool, list[str], list[dict]]:
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
        deepseek_task="router",
    )
    em("1-router", f"路由返回：{raw}")
    try:
        need, qs, tcalls = _parse_router_json(raw)
        return raw, need, qs, tcalls
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
            deepseek_task="router",
        )
        em("1-router", f"路由重试返回：{raw2}")
        try:
            need2, qs2, tcalls2 = _parse_router_json(raw2)
            return raw2, need2, qs2, tcalls2
        except ValueError as e:
            snippet = (raw2 or "").strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "…(截断)"
            raise RuntimeError(
                "路由模型在 API JSON 模式与一次重试后仍返回无法解析的内容（不接受自然语言顶替）。"
                f"末次输出：{snippet!r}"
            ) from e


def _execute_refiner_with_json_enforcement(
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
) -> tuple[str, bool, list[str], list[dict]]:
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
        deepseek_task="refiner",
    )
    em("2b-supplement", f"二阶段确认返回：{raw}")
    try:
        need, qs, tcalls = _parse_refiner_json(raw)
        return raw, need, qs, tcalls
    except ValueError:
        em("2b-supplement", "二阶段输出无法解析为合法 JSON，重试一次…")
        _retry2 = (
            REFINER_DS_TOOL_RETRY_USER
            if llm_backend == "deepseek"
            else REFINER_JSON_RETRY_USER
        )
        raw2 = _router_llm_once(
            rmsgs + [{"role": "user", "content": _retry2}],
            llm_backend=llm_backend,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            deepseek_api_base=deepseek_api_base,
            deepseek_api_key=deepseek_api_key,
            deepseek_model=deepseek_model,
            timeout_sec=timeout_sec,
            ollama_think=ollama_think,
            dump_llm_requests=dump_llm_requests,
            deepseek_task="refiner",
        )
        em("2b-supplement", f"二阶段重试返回：{raw2}")
        try:
            need2, qs2, tcalls2 = _parse_refiner_json(raw2)
            return raw2, need2, qs2, tcalls2
        except ValueError as e:
            snippet = (raw2 or "").strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + "…(截断)"
            raise RuntimeError(
                "二阶段确认在 JSON 模式与一次重试后仍返回无法解析的内容。"
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


def http_memory_extract_batch(
    api_base: str, queries: list[str], timeout_sec: int, memory_user_id: str = ""
) -> tuple[list[tuple[bool, str, str]], str]:
    """
    调用 ``POST /memory/extract_batch`` 一次完成多条问句的合批前向；若服务端无该路由 (HTTP 404) 则退化为逐条
    ``http_memory_extract``，保证旧服务仍可跑。

    第二项为「客户端收到的整段 HTTP 响应体原文」（合批为单次响应；404 串行时为多段按序拼接、未做长度截断）。
    """
    if not queries:
        return [], ""
    if len(queries) == 1:
        h, m, r = http_memory_extract(api_base, queries[0], timeout_sec, memory_user_id)
        return [(h, m, r)], r
    url = api_base.rstrip("/") + "/memory/extract_batch"
    body: dict = {"queries": queries}
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
        if e.code == 404:
            rows: list[tuple[bool, str, str]] = []
            chunks: list[str] = []
            nq = len(queries)
            for k, q in enumerate(queries, 1):
                h, m, r = http_memory_extract(
                    api_base, q, timeout_sec, memory_user_id
                )
                rows.append((h, m, r))
                chunks.append(f"--- 串行 {k}/{nq}（extract_batch 404 回退）---\n{r}")
            return rows, "\n".join(chunks)
        b = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"记忆 API HTTP {e.code} {url}\n{b}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接记忆 API ({url})。请先在本机运行 memory_api_server.py。\n{e}"
        ) from e
    data = json.loads(raw)
    results = data.get("results")
    if not isinstance(results, list) or len(results) != len(queries):
        raise RuntimeError(
            f"memory/extract_batch 返回格式异常：需 results 为与 queries 等长的列表。原始：{raw!r}"
        )
    out: list[tuple[bool, str, str]] = []
    for it in results:
        if not isinstance(it, dict):
            it = {"has_memory": False, "memory": ""}
        mem = str(it.get("memory") or "").strip()
        if "has_memory" in it:
            has_mem = bool(it.get("has_memory")) and bool(mem)
        else:
            has_mem = bool(mem)
        if not has_mem:
            mem = ""
        item_raw = json.dumps(it, ensure_ascii=False)
        out.append((has_mem, mem, item_raw))
    return out, raw


def _normalize_memory_query_for_dedup(q: str) -> str:
    return " ".join((q or "").split()).strip()


def _is_interrogative_memory_followup_line(line: str) -> bool:
    """以「？」或「?」结尾的问句：只应走追加检索解析，不得写入合并后的 [记忆事实] 正文。"""
    t = (line or "").strip()
    if len(t) < 3:
        return False
    return t.endswith("？") or t.endswith("?")


_TRAINING_ACK_LINE_RE = re.compile(r"^好的[,，]我记住了[。.!！…\s]*$")


def _is_training_ack_memory_line(line: str) -> bool:
    """训练内化样本的 assistant 套话，无检索价值，不写入合并 [记忆事实]。"""
    t = (line or "").strip()
    return bool(_TRAINING_ACK_LINE_RE.match(t))


_MEM_EXTRACT_IN_FACT_RE = re.compile(r"\s*\[记忆提取\]\s*")


def _sanitize_memory_fact_for_merge(mem: str) -> str:
    """
    写入「合并已检索记忆」前清洗：
    - 去掉训练泄漏的 [记忆提取] 标记；
    - 去掉所有问句行（训练样本里的 q 混入事实，如「老板的数学能力如何？」），
      问句仅由 _queries_after_memory_extract_markers 从原文解析后入追加检索队列；
    - 去掉「好的，我记住了。」类训练套话行；
    - 规整换行。
    """
    s = (mem or "").strip()
    if not s:
        return ""
    s = _MEM_EXTRACT_IN_FACT_RE.sub("\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    lines: list[str] = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            continue
        if _is_interrogative_memory_followup_line(t):
            continue
        if _is_training_ack_memory_line(t):
            continue
        lines.append(t)
    return "\n".join(lines)


def _queries_after_memory_extract_markers(mem: str) -> list[str]:
    """
    从记忆 API 返回正文中解析「[记忆提取]」之后的片段行；
    仅 **问句**（行末为 ？ / ?）作为追加检索问句；其余行视为事实正文，不触发 memory/extract。
    块内按行去重保留顺序。
    """
    if not mem or "[记忆提取]" not in mem:
        return []
    parts = mem.split("[记忆提取]")
    raw_lines: list[str] = []
    for i in range(1, len(parts)):
        seg = parts[i].lstrip("\r\n").strip()
        if not seg:
            continue
        for line in seg.splitlines():
            t = line.strip()
            if t and _is_interrogative_memory_followup_line(t):
                raw_lines.append(t)
    seen_line: set[str] = set()
    out: list[str] = []
    for s in raw_lines:
        k = _normalize_memory_query_for_dedup(s)
        if k and k not in seen_line:
            seen_line.add(k)
            out.append(s)
    return out


def _collect_followup_queries_from_memory_raw(m_raw: str) -> list[str]:
    """
    追加检索问句来源：① [记忆提取] 后解析出的问句；② 全文任意行中的问句（无标记时也可能混入事实块）。
    顺序：先 marker 路径，再补全文扫描；规范化去重。
    """
    out: list[str] = []
    seen: set[str] = set()
    for s in _queries_after_memory_extract_markers(m_raw):
        k = _normalize_memory_query_for_dedup(s)
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    for ln in m_raw.splitlines():
        t = ln.strip()
        if not t or not _is_interrogative_memory_followup_line(t):
            continue
        k = _normalize_memory_query_for_dedup(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _iter_memory_extract_gen(
    need_memory: bool,
    memory_queries: list[str],
    memory_api_base: str,
    timeout_sec: int,
    tool_calls: list[dict] | None = None,
    memory_user_id: str = "",
    *,
    think_step: str = "2-extract",
    pre_seen_queries: set[str] | None = None,
    pre_seen_profile_fks: set[str] | None = None,
    session_profile_fks: set[str] | None = None,
) -> Iterator[dict[str, str]]:
    """与 ``_run_memory_extracts`` 同一逻辑；先执行端侧用户画像工具，再对记忆模型做 extract；结束时 ``return memory_text``。若传入 ``session_profile_fks``，每成功拉取一画像键会写入该集合，供多轮间去重。"""
    ext_url = memory_api_base.rstrip("/") + "/memory/extract"
    ext_url_batch = memory_api_base.rstrip("/") + "/memory/extract_batch"
    memory_text = ""
    parts: list[str] = []
    tcalls = list(tool_calls or [])
    uid = (memory_user_id or "").strip()
    ts = think_step if think_step else "2-extract"
    prof_done: set[str] = set(pre_seen_profile_fks or ())

    for item in tcalls:
        if not _is_user_profile_tool_item(item):
            if (item.get("name") or "").strip() or (item.get("cli_command") or "").strip():
                yield _emit_think(ts, f"[端侧工具] 未实现，已跳过：{item!r}")
            continue
        if not uid:
            yield _emit_think(
                ts,
                "[用户画像工具] 未提供会话数据根目录 id（session/<id>/），已跳过。",
            )
            continue
        cc = str(item.get("cli_command", "")).strip()
        fk_field = str(item.get("file_key", "")).strip()
        fk = file_key_from_cli_or_fields(cc, fk_field or None)
        if not fk:
            yield _emit_think(
                ts,
                f"[用户画像工具] 无法解析角色（需 cli_command 如 user_profile 某角色 或 file_key）：{item!r}",
            )
            continue
        if fk in prof_done:
            yield _emit_think(ts, f"[用户画像工具] 与本轮已处理重复，跳过 file_key={fk!r}")
            continue
        prof_done.add(fk)
        if session_profile_fks is not None:
            session_profile_fks.add(fk)
        label, body = memory_block_for_user_profile(uid, fk)
        has_body = bool((body or "").strip())
        yield _emit_think(
            ts,
            f"[用户画像工具] file_key={fk!r}  命中={has_body}",
        )
        m_disp = body if has_body else "（无）"
        parts.append(f"[记忆检索]：{label}\n[记忆事实]：{m_disp}")

    if not need_memory and not parts:
        yield _emit_think(ts, "未调用记忆 API（无需提取且无端侧画像）。")
        return memory_text
    if not need_memory:
        memory_text = "\n\n".join(parts).strip()
        if memory_text:
            yield _emit_think(ts, f"合并已检索记忆（仅端侧画像 {len(parts)} 条）…")
        return memory_text
    if not memory_queries and parts:
        memory_text = "\n\n".join(parts).strip()
        yield _emit_think(ts, "未调用 memory/extract（无检索问句），已合并端侧画像。")
        return memory_text
    if not memory_queries and not parts:
        yield _emit_think(ts, "未调用记忆 API（无检索问句且无端侧画像）。")
        return memory_text

    q_queue: deque[str] = deque()
    seen_queries: set[str] = set(pre_seen_queries or set())
    for q in memory_queries:
        nq = _normalize_memory_query_for_dedup(q)
        if nq and nq not in seen_queries:
            seen_queries.add(nq)
            q_queue.append((q or "").strip())

    seen_mem_norm: set[str] = set()
    call_idx = 0
    max_calls = BUILTIN_MEMORY_EXTRACT_EXPAND_MAX_CALLS

    while q_queue and call_idx < max_calls:
        n = min(len(q_queue), max_calls - call_idx)
        batch: list[str] = [q_queue.popleft() for _ in range(n)]
        j0 = call_idx
        if n == 1:
            raws: list[tuple[bool, str, str]] = [
                http_memory_extract(
                    memory_api_base, batch[0], timeout_sec, memory_user_id=uid
                )
            ]
        else:
            raws, _batch_raw_full = http_memory_extract_batch(
                memory_api_base, batch, timeout_sec, memory_user_id=uid
            )
        if n > 1:
            paired: list[dict] = []
            for i, q in enumerate(batch):
                has_m, m, _rh = raws[i]
                paired.append(
                    {
                        "query": q,
                        "has_memory": has_m,
                        "memory": (m or "").strip() if has_m and m else "",
                    }
                )
            yield _emit_think(
                ts,
                f"出参(配对) {json.dumps(paired, ensure_ascii=False)}",
            )
        for i, q in enumerate(batch):
            j = j0 + i + 1
            has_mem, mem, raw_http = raws[i]
            if n == 1:
                msg_http = f"出参 {ext_url} query={q!r} {raw_http}"
                yield _emit_think(ts, f"[{j}] {msg_http}")
            m_raw = (mem or "").strip() if has_mem and mem else ""
            m_disp = _sanitize_memory_fact_for_merge(m_raw) if m_raw else ""
            if n == 1:
                msg_mem = f"出参 has_memory={has_mem} memory={(m_disp if m_disp else '')}"
                yield _emit_think(ts, f"[{j}] {msg_mem}")
            if has_mem and m_raw:
                dedup_key = _normalize_memory_query_for_dedup(m_disp)
                if m_disp and dedup_key not in seen_mem_norm:
                    seen_mem_norm.add(dedup_key)
                    parts.append(f"[记忆检索]：{q}\n[记忆事实]：{m_disp}")
                for fq in _collect_followup_queries_from_memory_raw(m_raw):
                    nn = _normalize_memory_query_for_dedup(fq)
                    if nn and nn not in seen_queries:
                        seen_queries.add(nn)
                        q_queue.append(fq)
                        yield _emit_think(
                            ts,
                            f"[追加检索] 解析出待检索项并入队（与已检索去重）：{fq!r}",
                        )
        call_idx += n

    if call_idx >= max_calls and q_queue:
        yield _emit_think(
            ts,
            f"[追加检索] 已达 memory/extract 调用上限 {max_calls}，"
            f"剩余 {len(q_queue)} 条未检索已丢弃。",
        )

    memory_text = "\n\n".join(parts).strip()
    if memory_text:
        merge_msg = (
            f"合并已检索记忆（{len(parts)} 段，含端侧用户画像与 memory/extract；"
            f"每段为 [记忆检索]+[记忆事实]）：\n{memory_text}"
        )
        yield _emit_think(ts, merge_msg)
    else:
        yield _emit_think(ts, "合并已检索记忆：（全部为空）")
    return memory_text


def _run_memory_extracts(
    need_memory: bool,
    memory_queries: list[str],
    memory_api_base: str,
    timeout_sec: int,
    tool_calls: list[dict] | None = None,
    memory_user_id: str = "",
    *,
    think_step: str = "2-extract",
    pre_seen_queries: set[str] | None = None,
    pre_seen_profile_fks: set[str] | None = None,
    session_profile_fks: set[str] | None = None,
) -> str:
    it = _iter_memory_extract_gen(
        need_memory,
        memory_queries,
        memory_api_base,
        timeout_sec,
        tool_calls=tool_calls,
        memory_user_id=memory_user_id,
        think_step=think_step,
        pre_seen_queries=pre_seen_queries,
        pre_seen_profile_fks=pre_seen_profile_fks,
        session_profile_fks=session_profile_fks,
    )
    try:
        while True:
            next(it)
    except StopIteration as e:
        return (e.value or "") if e.value is not None else ""


def _run_refiner_and_append_extract(
    history: list[tuple[str, str]],
    user_text: str,
    memory_text: str,
    memory_queries_r1: list[str],
    tool_calls_r1: list[dict],
    *,
    memory_api_base: str,
    timeout_sec: int,
    memory_user_id: str,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    ollama_think: bool,
    dump_llm_requests: bool,
    session_profile_fks: set[str] | None = None,
) -> str:
    """二阶段 LLM 确认 + 视情况再跑一轮 memory/tools，拼到首轮 `memory_text` 之后。"""
    rmsgs = _refiner_messages(history, user_text, memory_text)
    try:
        _, need_sup, sup_qs, sup_tc = _execute_refiner_with_json_enforcement(
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
            emit=None,
        )
    except RuntimeError as e:
        log("2b-supplement", f"二阶段确认失败，沿用首轮已拼装记忆：{e!r}")
        return memory_text
    if not need_sup:
        log("2b-supplement", "二阶段确认：无需补充检索或端侧工具。")
        return memory_text
    if not sup_qs and not sup_tc:
        log("2b-supplement", "二阶段 need_supplement 为真但无问句与工具，沿用首轮。")
        return memory_text
    log("2b-supplement", f"二阶段执行补充：memory_queries={sup_qs!r} tool_calls={sup_tc!r}")
    qseen, pseen = _build_seen_dedup_from_first_round(memory_queries_r1, tool_calls_r1)
    need_mem2 = bool(sup_qs)
    extra = _run_memory_extracts(
        need_mem2,
        sup_qs,
        memory_api_base,
        timeout_sec,
        tool_calls=sup_tc,
        memory_user_id=memory_user_id,
        think_step="2b-extract",
        pre_seen_queries=qseen,
        pre_seen_profile_fks=pseen,
        session_profile_fks=session_profile_fks,
    )
    if not (extra or "").strip():
        return memory_text
    return f"{(memory_text or '').rstrip()}\n\n{extra.strip()}".strip()


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
    memory_user_id: str = "",
    session_profile_fks: set[str] | None = None,
    session_memory_retrieval_blocks: list[str] | None = None,
) -> str:
    # memory_user_id = session/<id>/ 数据归属；被查询人物 = tool_calls 的 file_key，二者不同。
    h = list(history or [])
    log(
        "1-router",
        f"用户输入：{user_text}",
    )
    rmsgs = _router_messages(h, user_text)
    router_raw, need_memory, memory_queries, tool_calls = _execute_router_with_json_enforcement(
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
    if tool_calls:
        log("1-router", f"端侧 tool_calls：{tool_calls!r}")
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
        need_memory,
        memory_queries,
        memory_api_base,
        timeout_sec,
        tool_calls=tool_calls,
        memory_user_id=memory_user_id,
        session_profile_fks=session_profile_fks,
    )
    _step_break()
    memory_text = _run_refiner_and_append_extract(
        h,
        user_text,
        memory_text,
        memory_queries,
        list(tool_calls or []),
        memory_api_base=memory_api_base,
        timeout_sec=timeout_sec,
        memory_user_id=memory_user_id,
        llm_backend=llm_backend,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        deepseek_api_base=deepseek_api_base,
        deepseek_api_key=deepseek_api_key,
        deepseek_model=deepseek_model,
        ollama_think=ollama_think,
        dump_llm_requests=dump_llm_requests,
        session_profile_fks=session_profile_fks,
    )
    _step_break()

    prior_mem = _join_prior_session_memory_blocks(session_memory_retrieval_blocks)
    fmsgs = _final_messages(
        h, memory_text, user_text, prior_session_memory=prior_mem
    )
    _log_final_submission_to_console(fmsgs)
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
    if (
        session_memory_retrieval_blocks is not None
        and (memory_text or "").strip()
    ):
        session_memory_retrieval_blocks.append((memory_text or "").strip())
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
    memory_user_id: str = "",
    session_profile_fks: set[str] | None = None,
    session_memory_retrieval_blocks: list[str] | None = None,
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

    router_raw, need_memory, memory_queries, tool_calls = _execute_router_with_json_enforcement(
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
    if tool_calls:
        yield _emit_think("1-router", f"端侧 tool_calls：{tool_calls!r}")
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
        need_memory,
        memory_queries,
        memory_api_base,
        timeout_sec,
        tool_calls=tool_calls,
        memory_user_id=memory_user_id,
        session_profile_fks=session_profile_fks,
    )
    _step_break()
    r2 = _refiner_messages(h, user_text, memory_text)
    ref_buf: list[dict] = []
    need_sup = False
    sup_qs: list[str] = []
    sup_tc: list[dict] = []
    refiner_failed = False

    def _emit_2b(step: str, msg: str) -> None:
        ref_buf.append(_emit_think(step, msg))

    try:
        _, need_sup, sup_qs, sup_tc = _execute_refiner_with_json_enforcement(
            r2,
            llm_backend=llm_backend,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            deepseek_api_base=deepseek_api_base,
            deepseek_api_key=deepseek_api_key,
            deepseek_model=deepseek_model,
            timeout_sec=timeout_sec,
            ollama_think=ollama_think,
            dump_llm_requests=dump_llm_requests,
            emit=_emit_2b,
        )
    except RuntimeError as e:
        ref_buf.append(
            _emit_think("2b-supplement", f"二阶段确认失败，沿用首轮已拼装记忆：{e!r}")
        )
        need_sup = False
        sup_qs, sup_tc = [], []
        refiner_failed = True
    for _ev in ref_buf:
        yield _ev
    if need_sup and (sup_qs or sup_tc):
        yield _emit_think(
            "2b-supplement",
            f"二阶段执行补充：memory_queries={sup_qs!r} tool_calls={sup_tc!r}",
        )
        qseen, pseen = _build_seen_dedup_from_first_round(
            memory_queries, list(tool_calls or [])
        )
        need_m2 = bool(sup_qs)
        extra = yield from _iter_memory_extract_gen(
            need_m2,
            sup_qs,
            memory_api_base,
            timeout_sec,
            tool_calls=sup_tc,
            memory_user_id=memory_user_id,
            think_step="2b-extract",
            pre_seen_queries=qseen,
            pre_seen_profile_fks=pseen,
            session_profile_fks=session_profile_fks,
        )
        if (extra or "").strip():
            memory_text = f"{(memory_text or '').rstrip()}\n\n{extra.strip()}".strip()
    elif not refiner_failed:
        if not need_sup:
            yield _emit_think("2b-supplement", "二阶段确认：无需补充检索或端侧工具。")
        else:
            yield _emit_think("2b-supplement", "二阶段无具体补充问句/工具，沿用首轮。")
    _step_break()

    prior_mem = _join_prior_session_memory_blocks(session_memory_retrieval_blocks)
    fmsgs = _final_messages(
        h, memory_text, user_text, prior_session_memory=prior_mem
    )
    _log_final_submission_to_console(fmsgs)
    if llm_backend == "deepseek":
        yield _emit_think(
            "3-final", _final_stage_think_message(memory_text, prior_mem)
        )
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
        yield _emit_think(
            "3-final", _final_stage_think_message(memory_text, prior_mem)
        )
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
    if (
        session_memory_retrieval_blocks is not None
        and (memory_text or "").strip()
    ):
        session_memory_retrieval_blocks.append((memory_text or "").strip())


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
    memory_user_id: str = "",
    session_profile_fks: set[str] | None = None,
    session_memory_retrieval_blocks: list[str] | None = None,
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
        memory_user_id=memory_user_id,
        session_profile_fks=session_profile_fks,
        session_memory_retrieval_blocks=session_memory_retrieval_blocks,
    ):
        if ev.get("kind") == "delta":
            yield str(ev.get("text") or "")


def append_session_staging_turn(
    log_path: Path,
    user_query: str,
    final_answer: str,
    *,
    turn_time: datetime,
    user_speaker_label: str = "User",
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = turn_time.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    label = (user_speaker_label or "").strip() or "User"
    # 冒号仅用于首行角色前缀；正文中的冒号保留在用户句内
    block = f"[轮次时间: {ts}]\n{label}: {user_query}\nAssistant: {final_answer}\n\n"
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
    code, raw = http_pipeline_json(
        "POST",
        f"{base}/sessions",
        {"user_id": mem_paths.generate_new_user_id(), "display_name": "User"},
        timeout=timeout,
    )
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
