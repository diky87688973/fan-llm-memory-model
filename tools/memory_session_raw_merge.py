# -*- coding: utf-8 -*-
"""将 ``session_raw/`` 下训练样本合并为 ``session/all_<合并时间>_corpus.txt``。

按 **stem**（文件名前缀）分组：每个 stem **先**输出对应的 ``.raw.txt``（整文件占一行，行内换行写成 ``\\n``），**再**逐行输出该 stem 的 ``.qa.jsonl``（**一行一条 QA**）。

每条 QA 的标签顺序为 **q → a**；若有时间 **t**，置于 **``</AI_answer>`` 之后**：``<AI_question>…</AI_question><AI_answer>…</AI_answer><AI_time>…</AI_time>``，便于生成答案后再预测事件发生时刻；丢弃 ``cluster_id`` 等其它字段。

stem 排序：按字符串排序后依次处理；同一 stem 下多个 raw/qa 文件时按路径排序。

默认仅纳入：``.raw.txt``、``.qa.jsonl``；不包含 ``.memory.json``。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _stem_raw(p: Path) -> str:
    n = p.name
    if not n.endswith(".raw.txt"):
        raise ValueError(f"非 .raw.txt: {p}")
    return n[: -len(".raw.txt")]


def _stem_qa(p: Path) -> str:
    n = p.name
    if not n.endswith(".qa.jsonl"):
        raise ValueError(f"非 .qa.jsonl: {p}")
    return n[: -len(".qa.jsonl")]


def _collect_by_stem(session_raw: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    raw_by_stem: dict[str, list[Path]] = {}
    qa_by_stem: dict[str, list[Path]] = {}
    if not session_raw.is_dir():
        return raw_by_stem, qa_by_stem
    for p in session_raw.rglob("*"):
        if not p.is_file():
            continue
        n = p.name
        if n.endswith(".raw.txt"):
            s = _stem_raw(p)
            raw_by_stem.setdefault(s, []).append(p)
        elif n.endswith(".qa.jsonl"):
            s = _stem_qa(p)
            qa_by_stem.setdefault(s, []).append(p)
    for paths in raw_by_stem.values():
        paths.sort(key=lambda x: x.as_posix().lower())
    for paths in qa_by_stem.values():
        paths.sort(key=lambda x: x.as_posix().lower())
    return raw_by_stem, qa_by_stem


def _strip_qa_obj(obj: dict) -> dict | None:
    q = obj.get("q") or obj.get("question")
    a = obj.get("a") or obj.get("answer")
    if not q or not a:
        return None
    out: dict = {"q": str(q).strip(), "a": str(a).strip()}
    tv = obj.get("t") or obj.get("time") or obj.get("memory_time")
    if tv is not None and str(tv).strip():
        out["t"] = str(tv).strip()
    return out


def _slim_to_tag_line(slim: dict) -> str:
    """标签顺序：q → a；若有 t，放在 </AI_answer> 之后为独立 <AI_time>。"""
    q = slim["q"]
    a = slim["a"]
    tv = slim.get("t")
    core = f"<AI_question>{q}</AI_question><AI_answer>{a}</AI_answer>"
    if tv is not None and str(tv).strip():
        tv = str(tv).strip()
        return f"{core}<AI_time>{tv}</AI_time>"
    return core


def _qa_jsonl_to_lines(path: Path) -> list[str]:
    lines_out: list[str] = []
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        slim = _strip_qa_obj(obj)
        if slim is None:
            continue
        lines_out.append(_slim_to_tag_line(slim))
    return lines_out


def _raw_file_to_one_line(path: Path) -> str:
    inner = path.read_text(encoding="utf-8", errors="replace")
    inner = inner.replace("\r\n", "\n").replace("\r", "\n")
    return inner.replace("\n", "\\n")


def _build_merged_lines(
    raw_by_stem: dict[str, list[Path]],
    qa_by_stem: dict[str, list[Path]],
) -> tuple[list[str], int, int]:
    """返回 (输出行列表, raw 行数, qa 行数)。"""
    stems = sorted(set(raw_by_stem) | set(qa_by_stem))
    out: list[str] = []
    n_raw_lines = 0
    n_qa_lines = 0
    for stem in stems:
        for rp in raw_by_stem.get(stem, []):
            out.append(_raw_file_to_one_line(rp))
            n_raw_lines += 1
        for qp in qa_by_stem.get(stem, []):
            for qa_line in _qa_jsonl_to_lines(qp):
                out.append(qa_line)
                n_qa_lines += 1
    return out, n_raw_lines, n_qa_lines


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="合并 session_raw：按 stem 先 raw 再逐行 qa，写入 session/all_*_corpus.txt")
    ap.add_argument(
        "--session_raw",
        type=str,
        default=str(script_dir / "session_raw/2026-04-12"),
        help="训练样本根目录（默认：本脚本同目录 session_raw）",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(script_dir / "session"),
        help="合并文件输出目录（默认：本脚本同目录 session）",
    )
    args = ap.parse_args()
    session_raw = Path(args.session_raw)
    out_dir = Path(args.out_dir)
    if not session_raw.is_dir():
        print(f"[memory_session_raw_merge] 错误：目录不存在: {session_raw}", flush=True)
        return 1
    raw_by_stem, qa_by_stem = _collect_by_stem(session_raw)
    if not raw_by_stem and not qa_by_stem:
        print(f"[memory_session_raw_merge] 未找到 .raw.txt / .qa.jsonl: {session_raw}", flush=True)
        return 1
    lines, n_raw, n_qa = _build_merged_lines(raw_by_stem, qa_by_stem)
    if not lines:
        print(f"[memory_session_raw_merge] 无有效内容: {session_raw}", flush=True)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"all_{ts}_corpus.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[memory_session_raw_merge] 已写入 {len(lines)} 行（raw 行={n_raw}, qa 行={n_qa}）→ {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
