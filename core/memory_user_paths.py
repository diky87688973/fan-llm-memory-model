# -*- coding: utf-8 -*-
"""多用户目录约定：session_staging/<user_id>/、session/<user_id>/YYYY-MM-DD/、session_bak/<user_id>/YYYY-MM-DD/、session_chat/<user_id>/。"""

from __future__ import annotations

import re
import threading
from datetime import date, datetime
from pathlib import Path

_counter_lock = threading.Lock()
_USER_ID_STATE = Path(__file__).resolve().parent / ".user_id_counter.state"


def generate_new_user_id() -> str:
    """``user_<YYYYMMDD>_<HHMMSS>_<6位循环自增>``，与同目录下 staging 时间串风格一致。"""
    with _counter_lock:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_ts = ""
        seq = 0
        if _USER_ID_STATE.is_file():
            try:
                lines = _USER_ID_STATE.read_text(encoding="utf-8").strip().splitlines()
                if len(lines) >= 2:
                    last_ts = lines[0].strip()
                    seq = int(lines[1].strip())
            except (OSError, ValueError):
                pass
        if last_ts == ts:
            seq = (seq % 999_999) + 1
        else:
            seq = 1
        try:
            _USER_ID_STATE.write_text(f"{ts}\n{seq}\n", encoding="utf-8")
        except OSError:
            pass
        return f"user_{ts}_{seq:06d}"

# 与 chat_ui / memory_pipeline_server 约定一致（仅 [a-zA-Z0-9_-]，便于路径安全）
COOKIE_USER_ID = "memory_user_id"
COOKIE_DISPLAY_NAME = "memory_display_name"

_ISO_DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def sanitize_path_user_id(raw: str, *, max_len: int = 80) -> str:
    """目录名用的 user_id：仅字母数字下划线连字符；空则 fallback。"""
    s = (raw or "").strip()
    t = re.sub(r"[^0-9A-Za-z_-]", "", s)
    if not t:
        return "user"
    return t[:max_len]


def is_iso_date_dir_name(name: str) -> bool:
    return bool(_ISO_DATE_DIR.match((name or "").strip()))


def session_staging_user_dir(staging_root: Path, user_id: str) -> Path:
    return staging_root / sanitize_path_user_id(user_id)


def session_day_dir(session_root: Path, user_id: str, d: date | None = None) -> Path:
    dd = d or date.today()
    day = f"{dd.year:04d}-{dd.month:02d}-{dd.day:02d}"
    p = session_root / sanitize_path_user_id(user_id) / day
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_bak_day_dir(bak_root: Path, user_id: str, d: date | None = None) -> Path:
    dd = d or date.today()
    day = f"{dd.year:04d}-{dd.month:02d}-{dd.day:02d}"
    p = bak_root / sanitize_path_user_id(user_id) / day
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_chat_user_dir(chat_root: Path, user_id: str) -> Path:
    p = chat_root / sanitize_path_user_id(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_staging_session_txt(name: str) -> bool:
    """今日会话 staging 文件名：``session_*.txt``；旧版 ``cli_*.txt`` 仍识别。"""
    n = (name or "").strip()
    if not n.lower().endswith(".txt"):
        return False
    return n.startswith("session_") or n.startswith("cli_")


def user_id_from_staging_cli_path(cli_path: Path | None, staging_root: Path) -> str:
    """
    从 ``…/session_staging/<user_id>/session_*.txt``（或旧版 ``cli_*.txt``）解析 user_id；
    若旧版平铺在 staging 根下则返回 ``_legacy``。
    """
    if cli_path is None:
        return "default"
    try:
        rel = cli_path.resolve().relative_to(staging_root.resolve())
    except ValueError:
        return "default"
    parts = rel.parts
    if len(parts) >= 2 and is_staging_session_txt(parts[-1]):
        return sanitize_path_user_id(parts[0])
    if len(parts) == 1 and is_staging_session_txt(parts[0]):
        return "_legacy"
    return "default"


def iter_staging_cli_files_oldest_first(staging_root: Path) -> list[Path]:
    """``session_staging`` 下任意深度的 ``session_*.txt`` 与旧版 ``cli_*.txt``，按 mtime 升序。"""
    if not staging_root.is_dir():
        return []
    out: list[Path] = []
    for p in staging_root.rglob("session_*.txt"):
        if p.is_file():
            out.append(p)
    for p in staging_root.rglob("cli_*.txt"):
        if p.is_file():
            out.append(p)
    out.sort(key=lambda p: p.stat().st_mtime)
    return out


def latest_staging_cli_file(staging_root: Path) -> Path | None:
    files = iter_staging_cli_files_oldest_first(staging_root)
    return files[-1] if files else None
