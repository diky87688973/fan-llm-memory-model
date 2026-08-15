# -*- coding: utf-8 -*-
"""session_id：``session_`` + 年月日时分秒毫秒(17 位数字) + 6 位循环自增（0～999999，再进位回 0）。"""
from __future__ import annotations

import threading
from datetime import datetime

_lock = threading.Lock()
_counter = 0


def next_session_id() -> str:
    global _counter
    with _lock:
        now = datetime.now()
        ts = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
        c = _counter
        _counter = (_counter + 1) % 1_000_000
    return f"session_{ts}{c:06d}"


def is_valid_session_id(s: str) -> bool:
    """兼容旧版 32 位 hex；新版 ``session_`` + 23 位数字。"""
    t = (s or "").strip().lower()
    if len(t) == 32 and all(c in "0123456789abcdef" for c in t):
        return True
    if len(t) != 31 or not t.startswith("session_"):
        return False
    rest = t[8:]
    return len(rest) == 23 and rest.isdigit()
