# -*- coding: utf-8 -*-
"""把 corpus 整份复制 TIMES 倍，输出与源文件同目录：<原名>_copyNx.txt。"""

from __future__ import annotations

from pathlib import Path

INPUT_PATH = r"..\..\..\PycharmProjects\AI\记忆模型\2.0\session\all_20260415_221653_corpus.txt"
TIMES = 100


def main() -> int:
    p = INPUT_PATH.strip()
    if not p:
        print("[session_corpus_duplicate] 请设置 INPUT_PATH", flush=True)
        return 1
    inp = Path(p)
    if TIMES < 1:
        print(f"[session_corpus_duplicate] TIMES 须 >= 1，当前为 {TIMES}", flush=True)
        return 1
    if not inp.is_file():
        print(f"[session_corpus_duplicate] 文件不存在: {inp}", flush=True)
        return 1
    raw = inp.read_text(encoding="utf-8", errors="replace")
    out_text = raw * TIMES
    out_path = inp.parent / f"{inp.stem}_copy{TIMES}x{inp.suffix}"
    out_path.write_text(out_text, encoding="utf-8")
    n_in = len(raw.splitlines()) if raw else 0
    n_out = len(out_text.splitlines()) if out_text else 0
    print(
        f"[session_corpus_duplicate] 已写入: {out_path} | 行数约 {n_in} × {TIMES} = {n_out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
