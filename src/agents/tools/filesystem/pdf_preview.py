"""PDF 预览：页数、元信息、每页字符数 + 前几页文本摘要。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from agents.tools.filesystem import constants


def summarize_pdf(path: Path) -> dict[str, Any]:
    """PDF：返回页数、元信息、每页字符数和前几页文本预览。"""
    import pymupdf  # pyright: ignore[reportMissingImports]

    doc = pymupdf.open(str(path))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    try:
        page_count: int = len(doc)  # pyright: ignore[reportUnknownArgumentType]
        raw_meta = cast(dict[str, str], doc.metadata or {})  # pyright: ignore[reportUnknownMemberType]
        meta_filtered: dict[str, str] = {
            k: v
            for k, v in raw_meta.items()
            if v and k in ("title", "author", "subject", "keywords")
        }

        page_char_counts: list[int] = []
        head_texts: list[str] = []
        for i in range(page_count):
            text: str = cast(str, doc[i].get_text())  # pyright: ignore[reportUnknownMemberType,reportIndexIssue]
            page_char_counts.append(len(text))
            if i < constants.INSPECT_FILES_PDF_HEAD_PAGES:
                preview: str = text[: constants.INSPECT_FILES_PDF_HEAD_CHARS]
                if len(text) > constants.INSPECT_FILES_PDF_HEAD_CHARS:
                    preview += "\n..."
                head_texts.append(preview)

        result: dict[str, Any] = {
            "page_count": page_count,
            "page_char_counts": page_char_counts,
            "head": head_texts,
        }
        if meta_filtered:
            result["metadata"] = meta_filtered
        return result
    finally:
        doc.close()  # pyright: ignore[reportUnknownMemberType]
