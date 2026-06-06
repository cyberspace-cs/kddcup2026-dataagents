"""PDF → markdown conversion for the ETL pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)


def pdf_to_markdown(path: Path) -> str:
    """Extract text from a PDF and return it as markdown with page headings."""
    import pymupdf  # pyright: ignore[reportMissingImports]

    doc = pymupdf.open(str(path))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    try:
        parts: list[str] = []
        page_count: int = len(doc)  # pyright: ignore[reportUnknownArgumentType]
        for i in range(page_count):
            text: str = cast(str, doc[i].get_text())  # pyright: ignore[reportUnknownMemberType,reportIndexIssue]
            text = text.strip()
            if not text:
                continue
            parts.append(f"## Page {i + 1}\n\n{text}")
        return "\n\n".join(parts)
    finally:
        doc.close()  # pyright: ignore[reportUnknownMemberType]
