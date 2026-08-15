# -*- coding: utf-8 -*-
"""从 hf-mirror 拉取 DeepSeek-R1-Distill-Llama-8B 的 tokenizer 文件（无代理可用）。

默认训练/推理基座已改为 Qwen/Qwen2.5-7B-Instruct；若需本地 tokenizer，请从同名 Qwen 仓库下载，
勿与 Qwen 权重混用本脚本拉取的 DeepSeek 词表。

用法：
  python download_hf_tokenizer.py
  python download_hf_tokenizer.py --out-dir D:\\路径\\hf_tokenizer
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/resolve/main"
FILES = ("tokenizer.json", "tokenizer_config.json")
USER_AGENT = "Mozilla/5.0 (compatible; download_hf_tokenizer/1.0)"


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 tokenizer 到本地目录（供 memory_extract_cli 使用）")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="保存目录；默认为本脚本同目录下的 hf_tokenizer",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="单次下载超时（秒），大文件 tokenizer.json 可能较慢",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / "hf_tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}", flush=True)

    for name in FILES:
        url = f"{BASE}/{name}"
        dest = out_dir / name
        print(f"GET {url}", flush=True)
        req = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=args.timeout_sec) as resp:
                data = resp.read()
        except HTTPError as e:
            print(f"  失败 HTTP {e.code}: {e.reason}", flush=True)
            raise SystemExit(1) from e
        except URLError as e:
            print(f"  失败: {e.reason}", flush=True)
            raise SystemExit(1) from e
        dest.write_bytes(data)
        print(f"  已写入 {dest} ({len(data)} bytes)", flush=True)

    print("完成。请运行 memory_extract_cli.py 验证「本地 tokenizer 中文探针通过」。", flush=True)


if __name__ == "__main__":
    main()
