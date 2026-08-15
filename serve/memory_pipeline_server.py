# -*- coding: utf-8 -*-
"""
记忆模型-chat HTTP 服务：维护会话多轮 history，在服务端写入 ``session_staging/<用户ID>/``（用户 ID 与展示名见 Cookie）；
每轮将完整对话以 OpenAI messages 交给路由与终答模型；终答支持 DeepSeek 流式（SSE）。
浏览器访问根路径 ``GET /`` 可得同 API 的流式（或整段）对话页。

监听地址默认 ``::``（IPv6 全接口；Windows 下对套接字设置 ``IPV6_V6ONLY=0`` 以便同端口接受 IPv4，避免仅 IPv6 时外网/本机访问异常）。
本机可试 ``http://127.0.0.1:端口`` 与 ``http://[::1]:端口``；若仍异常可改 ``--host 0.0.0.0`` 仅 IPv4。
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
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import memory_pipeline_core as mpc
import memory_user_paths as mem_paths

_script_dir = Path(__file__).resolve().parent

BUILTIN_HOST = "::"
BUILTIN_PORT = 8890
BUILTIN_SESSION_STAGING_DIR = str(_script_dir / "session_staging")
_SESSION_DATA_DIR = _script_dir / "session"
_SESSION_CHAT_DIR = _script_dir / "session_chat"

_CHAT_UI_HTML_PATH = _script_dir / "chat_ui.html"
_CHAT_UI_SUB_PLACEHOLDER = "__CHAT_UI_SUB__"
_CHAT_UI_MODEL_PLACEHOLDER = "__CHAT_UI_MODEL__"


class SessionState:
    def __init__(self, session_id: str) -> None:
        self.session_id: str = session_id
        self.history: list[tuple[str, str]] = []
        self.session_memory_retrieval_blocks: list[str] = []
        self.session_profile_fks: set[str] = set()
        self.staging_path: Path | None = None
        self.chat_dump_path: Path | None = None
        self.created_at: datetime = datetime.now()
        self.user_id: str = ""
        self.display_name: str = "User"


SESSIONS: dict[str, SessionState] = {}


def _safe_session_raw_path(rel: str, request: Request) -> Path:
    """``rel`` 为相对 ``session/`` 的路径（POSIX），仅允许 ``*.raw.txt``，且仅当前 cookie 用户目录下。"""
    uid = _cookie_user_id(request)
    if not uid:
        raise HTTPException(status_code=400, detail="缺少用户标识 cookie")
    if not rel or not isinstance(rel, str):
        raise HTTPException(status_code=400, detail="缺少路径")
    rel = rel.strip().replace("\\", "/").lstrip("/")
    if not rel or ".." in Path(rel).parts:
        raise HTTPException(status_code=400, detail="非法路径")
    root = _SESSION_DATA_DIR.resolve()
    p = (root / rel).resolve()
    try:
        rel_to_root = p.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法路径")
    parts = rel_to_root.parts
    if not parts or parts[0] != uid:
        raise HTTPException(status_code=403, detail="无权访问该路径")
    if not p.name.endswith(".raw.txt"):
        raise HTTPException(status_code=400, detail="仅支持 .raw.txt")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return p


def _session_staging_root() -> Path:
    return Path(BUILTIN_SESSION_STAGING_DIR).resolve()


def _cookie_user_id(request: Request) -> str:
    ck = (request.cookies.get(mem_paths.COOKIE_USER_ID) or "").strip()
    return mem_paths.sanitize_path_user_id(ck) if ck else ""


def _cookie_display_name(request: Request) -> str:
    raw = request.cookies.get(mem_paths.COOKIE_DISPLAY_NAME) or ""
    try:
        s = unquote(raw).strip()
    except Exception:
        s = (raw or "").strip()
    return s


def _safe_staging_file_basename(request: Request, name: str) -> Path:
    if not name or not isinstance(name, str):
        raise HTTPException(status_code=400, detail="缺少文件名")
    base = Path(name.strip().replace("\\", "/")).name
    if not base or ".." in base:
        raise HTTPException(status_code=400, detail="非法文件名")
    uid = _cookie_user_id(request)
    if not uid:
        raise HTTPException(status_code=400, detail="缺少用户标识 cookie，请刷新页面后先完成用户名设置")
    root = mem_paths.session_staging_user_dir(_session_staging_root(), uid)
    p = (root / base).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="非法文件名")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return p


def _session_chat_root() -> Path:
    return _SESSION_CHAT_DIR.resolve()


def _new_chat_dump_path(user_id: str, session_id: str) -> Path:
    d = mem_paths.session_chat_user_dir(_SESSION_CHAT_DIR, user_id)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return d / f"chat_{ts}_{session_id}.json"


def _write_json_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _session_chat_init_file(path: Path, session_id: str, *, user_id: str, display_name: str) -> None:
    doc = {
        "format": 1,
        "session_id": session_id,
        "memory_user_id": user_id,
        "display_name": display_name,
        "created_at": datetime.now().replace(microsecond=0).isoformat(sep=" "),
        "turns": [],
    }
    _write_json_atomic(path, doc)


def _session_chat_append_turn(path: Path, turn: dict) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") != 1:
        raise ValueError("session_chat 格式错误")
    doc.setdefault("turns", []).append(turn)
    _write_json_atomic(path, doc)


def _session_chat_append_turn_for_session(st: SessionState, turn: dict) -> None:
    """首轮对话前才创建 ``chat_*.json``；无对话则不落盘。"""
    if st.chat_dump_path is None:
        return
    p = st.chat_dump_path
    if not p.is_file():
        _session_chat_init_file(p, st.session_id, user_id=st.user_id, display_name=st.display_name)
    _session_chat_append_turn(p, turn)


def _find_staging_for_session(session_id: str, user_id: str) -> Path | None:
    root = _session_staging_root()
    if not root.is_dir():
        return None
    patterns = (f"session_*_{session_id}.txt", f"cli_*_{session_id}.txt")

    def _pick_newest_under(search_dir: Path) -> Path | None:
        if not search_dir.is_dir():
            return None
        candidates: list[Path] = []
        for pat in patterns:
            for p in search_dir.glob(pat):
                if p.is_file():
                    candidates.append(p)
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    if user_id:
        ud = mem_paths.session_staging_user_dir(root, user_id)
        hit = _pick_newest_under(ud)
        if hit is not None:
            return hit
    candidates: list[Path] = []
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _safe_session_chat_basename(request: Request, name: str) -> Path:
    if not name or not isinstance(name, str):
        raise HTTPException(status_code=400, detail="缺少文件名")
    base = Path(name.strip().replace("\\", "/")).name
    if not base.endswith(".json") or ".." in base:
        raise HTTPException(status_code=400, detail="非法文件名")
    uid = _cookie_user_id(request)
    if not uid:
        raise HTTPException(status_code=400, detail="缺少用户标识 cookie")
    root = mem_paths.session_chat_user_dir(_SESSION_CHAT_DIR, uid)
    p = (root / base).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="非法文件名")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return p


def _turns_out_from_file_doc(turns_raw: object) -> tuple[list[dict], list[tuple[str, str]]]:
    turns_out: list[dict] = []
    history: list[tuple[str, str]] = []
    if not isinstance(turns_raw, list):
        return turns_out, history
    for item in turns_raw:
        if not isinstance(item, dict):
            continue
        u = str(item.get("user") or "")
        a = str(item.get("assistant") or "")
        if not u and not a:
            continue
        history.append((u, a))
        turns_out.append(
            {
                "user": u,
                "think": str(item.get("think") or ""),
                "think_events": item.get("think_events")
                if isinstance(item.get("think_events"), list)
                else [],
                "stream_deltas": item.get("stream_deltas")
                if isinstance(item.get("stream_deltas"), list)
                else [],
                "assistant": a,
            }
        )
    return turns_out, history


def _new_staging_path(user_id: str, session_id: str) -> Path:
    """``session_staging/<user_id>/session_<时间>_<session_id>.txt``。"""
    d = mem_paths.session_staging_user_dir(_session_staging_root(), user_id)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return d / f"session_{ts}_{session_id}.txt"


class CreateSessionRequest(BaseModel):
    user_id: str = ""
    display_name: str = ""


class ChatRequest(BaseModel):
    message: str = Field(..., description="本轮用户输入")
    stream: bool = Field(
        False,
        description="SSE 事件流：含路由/记忆/终答准备等 think 与终答 delta；DeepSeek 终答逐字，Ollama 终答单条 delta",
    )


class OpenSessionChatBody(BaseModel):
    filename: str = Field(..., description="session_chat 目录下 JSON 文件名")


def _browser_chat_page_html(
    *, stream_capable: bool, llm_model_label: str
) -> str:
    """单页聊天：同域 SSE；左侧助手（含可折叠思考过程 + 气泡），右侧用户。"""
    # sub = (
    #     "DeepSeek 终答逐字流式"
    #     if stream_capable
    #     else "Ollama 终答整段返回（思考步骤仍流式展示）"
    # )
    sub = "流式生成"
    tpl = _CHAT_UI_HTML_PATH.read_text(encoding="utf-8")
    if _CHAT_UI_SUB_PLACEHOLDER not in tpl:
        raise RuntimeError("chat_ui.html 缺少占位符 " + _CHAT_UI_SUB_PLACEHOLDER)
    if _CHAT_UI_MODEL_PLACEHOLDER not in tpl:
        raise RuntimeError("chat_ui.html 缺少占位符 " + _CHAT_UI_MODEL_PLACEHOLDER)
    tpl = tpl.replace(_CHAT_UI_SUB_PLACEHOLDER, sub)
    return tpl.replace(_CHAT_UI_MODEL_PLACEHOLDER, llm_model_label)


def create_app(
    *,
    llm_backend: str,
    ollama_host: str,
    ollama_model: str,
    deepseek_api_base: str,
    deepseek_api_key: str,
    deepseek_model: str,
    memory_api_base: str,
    timeout_sec: int,
    ollama_think: bool,
) -> FastAPI:
    app = FastAPI(title="Memory Pipeline Server")
    stream_capable = llm_backend == "deepseek"

    def _effective_llm_model_label() -> str:
        if (llm_backend or "").lower() == "deepseek":
            s = (deepseek_model or mpc.BUILTIN_DEEPSEEK_MODEL or "").strip()
        else:
            s = (ollama_model or mpc.BUILTIN_OLLAMA_MODEL or "").strip()
        return s or str(llm_backend or "—")

    def _browser_ui_response() -> HTMLResponse:
        return HTMLResponse(
            content=_browser_chat_page_html(
                stream_capable=stream_capable,
                llm_model_label=_effective_llm_model_label(),
            ),
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/")
    @app.get("/ui")
    @app.get("/index.html")
    def browser_ui():
        return _browser_ui_response()

    @app.get("/ping")
    def ping():
        return Response(
            content=b"memory_pipeline_server ok\n",
            media_type="text/plain; charset=utf-8",
        )

    @app.post("/sessions")
    def create_session(request: Request, body: CreateSessionRequest = Body()):
        uid_raw = (body.user_id or request.cookies.get(mem_paths.COOKIE_USER_ID) or "").strip()
        if not uid_raw:
            uid_raw = mem_paths.generate_new_user_id()
            set_uid_cookie = True
        else:
            set_uid_cookie = False
        uid_s = mem_paths.sanitize_path_user_id(uid_raw)
        name = (body.display_name or _cookie_display_name(request) or "").strip()
        if not name:
            name = "User"
        sid = uuid.uuid4().hex
        st = SessionState(sid)
        st.user_id = uid_s
        st.display_name = name
        if mpc.BUILTIN_SAVE_SESSION_STAGING:
            st.staging_path = _new_staging_path(uid_s, sid)
        st.chat_dump_path = _new_chat_dump_path(uid_s, sid)
        SESSIONS[sid] = st
        payload = {
            "session_id": sid,
            "user_id": uid_s,
            "llm_model": _effective_llm_model_label(),
        }
        resp = JSONResponse(content=payload)
        if set_uid_cookie:
            resp.set_cookie(
                mem_paths.COOKIE_USER_ID,
                uid_s,
                max_age=365 * 86400,
                path="/",
                samesite="lax",
            )
        resp.set_cookie(
            mem_paths.COOKIE_DISPLAY_NAME,
            quote(name, safe=""),
            max_age=365 * 86400,
            path="/",
            samesite="lax",
        )
        return resp

    @app.post("/sessions/{session_id}/chat")
    def chat(session_id: str, body: ChatRequest):
        if session_id not in SESSIONS:
            raise HTTPException(status_code=404, detail="session_id 不存在，请先 POST /sessions")
        msg = (body.message or "").strip()
        if not msg:
            raise HTTPException(status_code=400, detail="message 不能为空")
        st = SESSIONS[session_id]
        h = st.history

        if body.stream:

            def gen():
                buf: list[str] = []
                think_events: list[dict[str, object]] = []
                think_concat_parts: list[str] = []
                delta_chunks: list[str] = []
                try:
                    for ev in mpc.iter_run_one_round_event_stream(
                        msg,
                        history=h,
                        llm_backend=llm_backend,
                        ollama_host=ollama_host,
                        ollama_model=ollama_model,
                        deepseek_api_base=deepseek_api_base,
                        deepseek_api_key=deepseek_api_key,
                        deepseek_model=deepseek_model,
                        memory_api_base=memory_api_base,
                        timeout_sec=timeout_sec,
                        ollama_think=ollama_think,
                        dump_llm_requests=False,
                        memory_user_id=st.user_id,
                        session_profile_fks=st.session_profile_fks,
                        session_memory_retrieval_blocks=st.session_memory_retrieval_blocks,
                    ):
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                        k = ev.get("kind")
                        if k == "think":
                            tx = str(ev.get("text") or "")
                            think_concat_parts.append(tx + "\n")
                            think_events.append(
                                {"phase": ev.get("phase"), "text": tx}
                            )
                        elif k == "delta":
                            piece = str(ev.get("text") or "")
                            buf.append(piece)
                            delta_chunks.append(piece)
                        elif k is None and ev.get("text") is not None:
                            piece = str(ev.get("text") or "")
                            buf.append(piece)
                            delta_chunks.append(piece)
                    full = "".join(buf).strip()
                    if mpc.BUILTIN_SAVE_SESSION_STAGING and st.staging_path is not None:
                        mpc.append_session_staging_turn(
                            st.staging_path,
                            msg,
                            full,
                            turn_time=datetime.now().replace(microsecond=0),
                            user_speaker_label=st.display_name,
                        )
                    st.history.append((msg, full))
                    _session_chat_append_turn_for_session(
                        st,
                        {
                            "user": msg,
                            "think": "".join(think_concat_parts).strip(),
                            "think_events": think_events,
                            "stream_deltas": delta_chunks,
                            "assistant": full,
                        },
                    )
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

            return StreamingResponse(
                gen(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            out = mpc.run_one_round(
                msg,
                history=h,
                llm_backend=llm_backend,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
                deepseek_api_base=deepseek_api_base,
                deepseek_api_key=deepseek_api_key,
                deepseek_model=deepseek_model,
                memory_api_base=memory_api_base,
                timeout_sec=timeout_sec,
                ollama_think=ollama_think,
                dump_llm_requests=False,
                memory_user_id=st.user_id,
                session_profile_fks=st.session_profile_fks,
                session_memory_retrieval_blocks=st.session_memory_retrieval_blocks,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        if mpc.BUILTIN_SAVE_SESSION_STAGING and st.staging_path is not None:
            mpc.append_session_staging_turn(
                st.staging_path,
                msg,
                out,
                turn_time=datetime.now().replace(microsecond=0),
                user_speaker_label=st.display_name,
            )
        st.history.append((msg, out))
        _session_chat_append_turn_for_session(
            st,
            {
                "user": msg,
                "think": "",
                "think_events": [],
                "stream_deltas": [],
                "assistant": out,
            },
        )
        return {"reply": out}

    @app.get("/memory-bank/raw-files")
    def list_session_raw_files(request: Request):
        """仅当前 cookie 用户的 ``session/<user_id>/`` 下 ``*.raw.txt``。"""
        uid = _cookie_user_id(request)
        if not uid:
            return {"files": []}
        root = _SESSION_DATA_DIR.resolve()
        user_dir = (root / uid).resolve()
        try:
            user_dir.relative_to(root)
        except ValueError:
            return {"files": []}
        if not user_dir.is_dir():
            return {"files": []}
        files: list[dict[str, str | int]] = []
        for p in user_dir.rglob("*.raw.txt"):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            st = p.stat()
            files.append({"rel": rel, "name": p.name, "mtime": int(st.st_mtime)})
        files.sort(key=lambda x: int(x["mtime"]), reverse=True)
        return {"files": files}

    @app.get("/memory-bank/raw-content")
    def get_session_raw_content(request: Request, rel: str):
        p = _safe_session_raw_path(rel, request)
        return Response(
            content=p.read_text(encoding="utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/today-sessions")
    def list_session_staging_files(request: Request):
        """当前 cookie 对应用户的 ``session_staging/<user_id>/`` 下文件。"""
        uid = _cookie_user_id(request)
        if not uid:
            return {"files": []}
        root = mem_paths.session_staging_user_dir(_session_staging_root(), uid)
        if not root.is_dir():
            return {"files": []}
        rows: list[dict[str, str | int]] = []
        for p in root.iterdir():
            if not p.is_file():
                continue
            st_t = p.stat()
            rows.append({"name": p.name, "mtime": int(st_t.st_mtime)})
        rows.sort(key=lambda x: int(x["mtime"]), reverse=True)
        return {"files": rows}

    @app.get("/today-session-content")
    def get_session_staging_content(request: Request, name: str):
        p = _safe_staging_file_basename(request, name)
        return Response(
            content=p.read_text(encoding="utf-8", errors="replace"),
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/session-chat/files")
    def list_session_chat_files(request: Request):
        uid = _cookie_user_id(request)
        if not uid:
            return {"files": []}
        root = mem_paths.session_chat_user_dir(_SESSION_CHAT_DIR, uid)
        if not root.is_dir():
            return {"files": []}
        rows: list[dict[str, str | int]] = []
        for p in root.glob("*.json"):
            if not p.is_file():
                continue
            st_t = p.stat()
            rows.append({"name": p.name, "mtime": int(st_t.st_mtime)})
        rows.sort(key=lambda x: int(x["mtime"]), reverse=True)
        return {"files": rows}

    @app.post("/session-chat/open")
    def session_chat_open(request: Request, body: OpenSessionChatBody):
        p = _safe_session_chat_basename(request, body.filename)
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("format") != 1:
            raise HTTPException(status_code=400, detail="不支持的 session_chat 格式")
        sid = str(doc.get("session_id") or "").strip().lower()
        if len(sid) != 32 or any(c not in "0123456789abcdef" for c in sid):
            raise HTTPException(status_code=400, detail="文件缺少有效 session_id")
        turns_raw = doc.get("turns")
        if not isinstance(turns_raw, list):
            turns_raw = []
        turns_out, hist_pairs = _turns_out_from_file_doc(turns_raw)

        if sid in SESSIONS:
            ex = SESSIONS[sid]
            try:
                same_file = (
                    ex.chat_dump_path is not None
                    and ex.chat_dump_path.resolve() == p.resolve()
                )
            except OSError:
                same_file = False
            if same_file:
                return {
                    "session_id": sid,
                    "turns": turns_out,
                    "llm_model": _effective_llm_model_label(),
                }
            raise HTTPException(
                status_code=409,
                detail="该会话已在服务内存中（且不是当前 JSON 文件），请关闭其它占用后重试",
            )

        uid_doc = str(doc.get("memory_user_id") or "").strip()
        uid_s = mem_paths.sanitize_path_user_id(uid_doc) if uid_doc else _cookie_user_id(request)
        if not uid_s:
            uid_s = mem_paths.sanitize_path_user_id(mem_paths.generate_new_user_id())
        dname = str(doc.get("display_name") or "").strip()
        if not dname:
            dname = _cookie_display_name(request) or "User"
        rst = SessionState(sid)
        rst.user_id = uid_s
        rst.display_name = dname
        rst.chat_dump_path = p
        ca = doc.get("created_at")
        if isinstance(ca, str):
            ca_s = ca.strip()
            try:
                rst.created_at = datetime.strptime(ca_s[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    rst.created_at = datetime.fromisoformat(ca_s)
                except ValueError:
                    rst.created_at = datetime.now()
        for pair in hist_pairs:
            rst.history.append(pair)
        rst.staging_path = _find_staging_for_session(sid, uid_s)
        if rst.staging_path is None and mpc.BUILTIN_SAVE_SESSION_STAGING:
            rst.staging_path = _new_staging_path(uid_s, sid)
        SESSIONS[sid] = rst
        return {
            "session_id": sid,
            "turns": turns_out,
            "llm_model": _effective_llm_model_label(),
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def _bind_ipv6_dual_stack_listen(port: int, *, backlog: int = 2048) -> socket.socket:
    """绑定 ``[::]:port``，并尽量 ``IPV6_V6ONLY=0``（Windows/Linux 上常能同端口收 IPv4）。"""
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    except OSError:
        pass
    s.bind(("::", port))
    s.setblocking(False)
    s.listen(backlog)
    try:
        s.set_inheritable(True)
    except OSError:
        pass
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="记忆模型-chat HTTP 服务（会话 + 多轮上下文 + 可选流式终答）")
    ap.add_argument(
        "--host",
        type=str,
        default=BUILTIN_HOST,
        help="uvicorn 绑定地址：::=IPv6 全接口（推荐 CGNAT+IPv6）；0.0.0.0=仅 IPv4 全接口；127.0.0.1=仅本机",
    )
    ap.add_argument("--port", type=int, default=BUILTIN_PORT)
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
        raise SystemExit(
            "错误：deepseek 需要 DEEPSEEK_API_KEY 或 --deepseek-api-key；或改用 --llm-backend ollama"
        )
    oo, mm = mpc._resolve_endpoints(args.server_host, args.ollama_host, args.memory_api_base)
    deepseek_base = args.deepseek_api_base.strip().rstrip("/")

    import uvicorn
    from uvicorn import Config, Server

    _h = (args.host or "").strip()
    _listen_display = f"http://[{_h}]:{args.port}" if _h == "::" else f"http://{_h}:{args.port}"

    print(
        f"[memory_pipeline_server] 监听 {_listen_display}（host={_h!r}）\n"
        f"  浏览器页面（任一路径均可）: / 、/ui 、/index.html ；自检: GET /ping\n"
        f"  POST /sessions → session_id\n"
        f"  POST /sessions/{{id}}/chat  body={{\"message\":\"...\",\"stream\":true|false}}\n"
        f"  stream=true：SSE 含 think（路由/记忆/终答准备）与 delta（终答），末行 data: [DONE]\n"
        f"  后端={args.llm_backend!r}  memory_api={mm!r}\n"
        f"  终端客户端：右键运行 memory_pipeline_client.py（改 BUILTIN_PIPELINE_BASE）\n",
        flush=True,
    )
    _app = create_app(
        llm_backend=args.llm_backend,
        ollama_host=oo.rstrip("/"),
        ollama_model=model,
        deepseek_api_base=deepseek_base,
        deepseek_api_key=deepseek_key,
        deepseek_model=model,
        memory_api_base=mm.rstrip("/"),
        timeout_sec=args.timeout,
        ollama_think=bool(args.ollama_think),
    )
    if _h == "::":
        try:
            _lsock = _bind_ipv6_dual_stack_listen(args.port)
        except OSError as e:
            raise SystemExit(f"绑定 IPv6 [::]:{args.port} 失败: {e}") from e
        _cfg = Config(_app, host="::", port=args.port)
        Server(_cfg).run(sockets=[_lsock])
    else:
        uvicorn.run(_app, host=_h, port=args.port)


if __name__ == "__main__":
    main()
