# -*- coding: utf-8 -*-
"""连接 ``memory_pipeline_server`` 的 REPL：专供 IDE 右键运行；改下方 ``BUILTIN_*`` 即可。"""
from __future__ import annotations

import sys
from pathlib import Path
for _sub in ("core", "train", "serve"):
    _p = str(Path(__file__).resolve().parent.parent / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import memory_pipeline_core as mpc

BUILTIN_PIPELINE_BASE = "http://127.0.0.1:8890"
BUILTIN_STREAM = True
BUILTIN_TIMEOUT = 600


def main() -> None:
    mpc.pipeline_client_chat_loop(
        BUILTIN_PIPELINE_BASE,
        stream=BUILTIN_STREAM,
        timeout=BUILTIN_TIMEOUT,
    )


if __name__ == "__main__":
    main()
