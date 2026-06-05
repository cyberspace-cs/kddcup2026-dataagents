"""Prose file detection and section parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_agent_baseline.benchmark.schema import PublicTask

MIN_LINES = 200
PROSE_EXTS = frozenset({".md", ".txt"})
ENTITY_LINE = re.compile(r"^ID:\s*", re.MULTILINE)
_SECTION_HEADING = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class ETLResult:
    """Metadata about a single pre-extracted CSV, for prompt injection."""

    source_file: str
    csv_path: str
    columns: list[str]
    row_count: int


def detect_sections(text: str) -> list[dict[str, Any]]:
    """Split text by heading markers into sections with metadata."""
    headings = list(_SECTION_HEADING.finditer(text))
    if not headings:
        return []
    top_level = min(len(m.group(1)) for m in headings)
    headings = [m for m in headings if len(m.group(1)) == top_level]

    sections: list[dict[str, Any]] = []
    for i, m in enumerate(headings):
        title = m.group(2).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        content = text[start:end]
        paras = [p.strip() for p in content.split("\n\n") if p.strip() and len(p.strip()) > 30]
        sections.append({"title": title, "content": content, "para_count": len(paras)})
    return sections


def detect_prose_files(task: PublicTask, *, force: bool = False) -> list[Path]:
    """Find .md/.txt files in context/doc/ and context/ root that exceed MIN_LINES.

    When *force* is True the line-count threshold is skipped — any non-empty
    prose file qualifies.
    """
    seen: set[Path] = set()
    results: list[Path] = []

    def _scan_dir(directory: Path) -> None:
        if not directory.is_dir():
            return
        for f in sorted(directory.iterdir()):
            if not f.is_file() or f.suffix.lower() not in PROSE_EXTS:
                continue
            if f.name.lower() == "knowledge.md":
                continue
            resolved = f.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                with f.open(encoding="utf-8", errors="replace") as handle:
                    line_count = sum(1 for _ in handle)
            except OSError:
                continue
            if line_count == 0:
                continue
            if not force and line_count < MIN_LINES:
                continue
            results.append(f)

    _scan_dir(task.context_dir / "doc")
    _scan_dir(task.context_dir)
    return results
