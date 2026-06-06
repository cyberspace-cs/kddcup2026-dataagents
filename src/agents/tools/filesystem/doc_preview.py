"""文档（markdown/txt）预览。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.tools.filesystem import constants
from agents.tools.filesystem.path import resolve_context_path


def _normalize_line_count(line_count: int) -> int:
    """把读取行数限制在稳定的 observation 范围内。"""
    return min(300, max(1, int(line_count)))


def _split_paragraphs(lines: list[str]) -> list[tuple[int, int, str]]:
    """把行列表切成段落（连续非空行为一段；空行为分隔），返回 (start, end, text) 列表。"""
    paragraphs: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = 0
    for one_based_idx, ln in enumerate(lines, start=1):
        if ln.strip() == "":
            if current:
                paragraphs.append((current_start, one_based_idx - 1, "\n".join(current)))
                current = []
                current_start = 0
        else:
            if not current:
                current_start = one_based_idx
            current.append(ln)
    if current:
        paragraphs.append((current_start, len(lines), "\n".join(current)))
    return paragraphs


def _collect_keyword_matches(
    paragraphs: list[tuple[int, int, str]], keyword: str
) -> tuple[list[dict[str, Any]], int, bool]:
    """从段落集合中收集所有包含 keyword 的段落（大小写不敏感子串匹配）。"""
    needle = keyword.lower()
    match_count = 0
    matches: list[dict[str, Any]] = []
    cumulative_bytes = 0
    capacity_exhausted = False
    for start, end, text in paragraphs:
        if needle not in text.lower():
            continue
        match_count += 1
        if capacity_exhausted:
            continue
        if len(matches) >= constants.READ_DOC_KEYWORD_MATCH_CAP:
            capacity_exhausted = True
            continue
        paragraph_truncated = False
        text_to_emit = text
        if len(text) > constants.READ_DOC_KEYWORD_PARAGRAPH_BYTES:
            text_to_emit = text[: constants.READ_DOC_KEYWORD_PARAGRAPH_BYTES] + "\n..."
            paragraph_truncated = True
        if (
            cumulative_bytes + len(text_to_emit) > constants.READ_DOC_KEYWORD_TOTAL_BYTES
            and matches
        ):
            capacity_exhausted = True
            continue
        matches.append(
            {
                "start_line": start,
                "end_line": end,
                "text": text_to_emit,
                "paragraph_truncated": paragraph_truncated,
            }
        )
        cumulative_bytes += len(text_to_emit)
    matches_truncated = len(matches) < match_count
    return matches, match_count, matches_truncated


def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    start_line: int = 1,
    line_count: int = 80,
    keyword: str | None = None,
) -> dict[str, Any]:
    """文本文档预览。paging 模式与 keyword 模式互斥。"""
    path = resolve_context_path(task, relative_path)
    lines = path.read_text(errors="replace").splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return {
            "path": relative_path,
            "total_lines": 0,
            "returned_range": None,
            "has_more": False,
            "keyword_match": None,
            "preview": "",
        }

    if keyword:
        paragraphs = _split_paragraphs(lines)
        matches, match_count, matches_truncated = _collect_keyword_matches(paragraphs, keyword)
        return {
            "path": relative_path,
            "total_lines": total_lines,
            "returned_range": None,
            "has_more": False,
            "keyword_match": {
                "keyword": keyword,
                "found": match_count > 0,
                "match_count": match_count,
                "matches": matches,
                "matches_truncated": matches_truncated,
            },
            "preview": None,
        }

    normalized_start_line = max(1, int(start_line))
    normalized_line_count = _normalize_line_count(line_count)

    start_index = normalized_start_line - 1
    end_index = min(total_lines, start_index + normalized_line_count)
    selected_lines = lines[start_index:end_index]
    returned_range = None
    if selected_lines:
        returned_range = {
            "start_line": start_index + 1,
            "end_line": end_index,
        }

    return {
        "path": relative_path,
        "total_lines": total_lines,
        "returned_range": returned_range,
        "has_more": end_index < total_lines,
        "keyword_match": None,
        "preview": "\n".join(selected_lines),
    }


def summarize_doc(path: Path) -> dict[str, Any]:
    """文本文档：行数 + 前 3 行预览。"""
    lines = path.read_text(errors="replace").splitlines()
    return {
        "line_count": len(lines),
        "head": lines[: constants.INSPECT_FILES_DOC_HEAD_LINES],
    }
