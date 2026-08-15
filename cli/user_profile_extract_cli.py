# -*- coding: utf-8 -*-
"""
从 ``session/<user_id>/<角色>.user.pic.md`` 读取用户/角色画像 Markdown 到 stdout；
无文件或读失败时输出空（仍 exit 0）。供命令行或 ``cli_command`` 子进程解析调试。

示例：
  python user_profile_extract_cli.py --user-id user_20260101_120000_000001 小明
"""
from __future__ import annotations

import sys
from pathlib import Path
for _sub in ("core", "train", "serve"):
    _p = str(Path(__file__).resolve().parent.parent / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import sys
from pathlib import Path

from user_profile_tool import get_session_root, read_user_profile_markdown


def main() -> None:
    ap = argparse.ArgumentParser(description="读取 session 下全局用户画像 .user.pic.md 正文到标准输出")
    ap.add_argument("--user-id", type=str, required=True, help="与 session/<user_id>/ 目录名一致")
    ap.add_argument(
        "--session-root",
        type=str,
        default="",
        help="默认同脚本目录下 session/，可改绝对路径",
    )
    ap.add_argument("file_key", help="角色标识（文件名 {file_key}.user.pic.md 经安全化后的基名来源）")
    args = ap.parse_args()
    root: Path | None = Path(args.session_root) if (args.session_root or "").strip() else None
    out = read_user_profile_markdown(args.user_id, args.file_key, session_root=root)
    sys.stdout.write(out)
    if out and not out.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
