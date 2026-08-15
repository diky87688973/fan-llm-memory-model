# -*- coding: utf-8 -*-
"""
用户画像：从 ``session/<数据目录归属id>/<画像键>.user.pic.md`` 读 Markdown。
``user_id`` 为产品侧数据根（当前账号/租户）；``file_key`` 为人物/角色名（如 小明），不是前者的同义词。
"""

from __future__ import annotations

import re
from pathlib import Path

import memory_user_paths as mem_paths
from session_to_memory import USER_PIC_GLOB_SUFFIX, _safe_user_pic_file_key

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SESSION_ROOT = _SCRIPT_DIR / "session"


def get_session_root() -> Path:
    return DEFAULT_SESSION_ROOT


def _parse_user_pic_body_redirect(body: str) -> str | None:
    """
    若整份画像正文**仅为**「关联调取」的一行，则返回引号/书名号内的目标画像键；否则返回 None。
    约定示例（整文件一行）：``[请调取"用户"的画像]`` —— 表示本 file_key 的画像以 ``用户`` 的 ``.user.pic.md`` 为准。
    """
    t = (body or "").strip()
    if not t:
        return None
    nonempty_lines = [x.strip() for x in t.splitlines() if x.strip()]
    if len(nonempty_lines) != 1:
        return None
    first = nonempty_lines[0]
    for pat in (
        r'^\[请调取"([^"]+)"的画像\]$',  # ASCII 双引号
        r"^\[请调取\u201c([^\u201d]+)\u201d的画像\]$",  # U+201C / U+201D
    ):
        m = re.match(pat, first)
        if m:
            k = (m.group(1) or "").strip()
            return k or None
    m = re.match(r"^\[请调取「([^」]+)」的画像\]$", first)
    if m:
        k = (m.group(1) or "").strip()
        return k or None
    return None


def _split_cli_rest(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    t = re.sub(r"^用户画像(提取)?\s+", "", t)
    t = re.sub(r"^user_profile(_extract)?\s+", "", t, flags=re.IGNORECASE)
    return t.strip()


def file_key_from_cli_or_fields(cli_command: str, file_key: str | None) -> str:
    """优先 ``file_key`` 字段；否则从 ``cli_command`` 解析（``user_profile 小明`` 等）。"""
    fk = (file_key or "").strip()
    if fk:
        return fk
    rest = _split_cli_rest(cli_command)
    return rest


def read_user_profile_markdown(
    user_id: str,
    file_key_raw: str,
    *,
    session_root: Path | None = None,
    _redirect_chain: frozenset[str] | None = None,
) -> str:
    """
    若文件存在则返回正文（strip）；否则返回空串。
    若正文**整份**为 ``[请调取"某键"的画像]`` 类一行关联格式，则自动改为读取该目标键的文件正文（可链式、防循环）。
    ``user_id`` 须已 sanitize，与 session 目录名一致。
    """
    uid = (user_id or "").strip()
    fk = (file_key_raw or "").strip()
    if not uid or not fk:
        return ""
    root = (session_root or get_session_root()).resolve()
    safe = _safe_user_pic_file_key(fk)
    seen = _redirect_chain or frozenset()
    if safe in seen:
        return ""
    udir = root / mem_paths.sanitize_path_user_id(uid)
    p = udir / f"{safe}{USER_PIC_GLOB_SUFFIX}"
    if not p.is_file():
        return ""
    try:
        body = p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    target = _parse_user_pic_body_redirect(body)
    if not target:
        return body
    return read_user_profile_markdown(
        user_id,
        target,
        session_root=session_root,
        _redirect_chain=seen | frozenset([safe]),
    )


def memory_block_for_user_profile(
    user_id: str, file_key_raw: str, *, session_root: Path | None = None
) -> tuple[str, str]:
    """
    返回 ``([记忆检索] 行内标签用短句, markdown 事实正文)``。
    标签格式：``提取<角色>用户画像``（与流水线拼装一致）。
    """
    fk = (file_key_raw or "").strip() or "未知"
    body = read_user_profile_markdown(user_id, file_key_raw, session_root=session_root)
    if fk == "用户":
        label = "拉取用户侧角色画像"
    elif fk == "助手":
        label = "拉取助手侧角色画像"
    else:
        label = f"提取{fk}用户画像"
    return label, body
