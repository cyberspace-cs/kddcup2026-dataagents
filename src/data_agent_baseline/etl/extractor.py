"""Core ETL: detect large prose files, compress noise, extract to CSV.

Two-phase strategy:
  1. **Compress** — LLM strips narrative filler from prose chunks, keeping only
     data-bearing sentences verbatim.  Output is a compact, near-structured text.
  2. **Extract** — LLM reads the compressed text and directly outputs CSV.
     Falls back to code-generation when the compressed text exceeds the
     direct-extraction token budget or when direct extraction fails.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_agent_baseline._constants import ETL_SCRATCH_ROOT
from data_agent_baseline.agents.model import (
    ModelMessage,
    ModelResponse,
    OpenAIModelAdapter,
)

if TYPE_CHECKING:
    from data_agent_baseline.benchmark.schema import PublicTask

logger = logging.getLogger(__name__)

_MIN_LINES = 200
_CODE_EXEC_TIMEOUT = 60
_MAX_RETRIES = 2
_SECTION_HEADING = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)
_COMPRESS_CHUNK_SIZE = 30
_DIRECT_EXTRACT_CHAR_LIMIT = 60_000
_MAX_WORKERS = 8


def _detect_sections(text: str) -> list[dict[str, Any]]:
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


@dataclass(frozen=True, slots=True)
class ETLResult:
    """Metadata about a single pre-extracted CSV, for prompt injection."""

    source_file: str
    csv_path: str
    columns: list[str]
    row_count: int


def _detect_prose_files(task: PublicTask) -> list[Path]:
    """Find .md files in context/doc/ (excluding knowledge.md) that exceed _MIN_LINES."""
    doc_dir = task.context_dir / "doc"
    if not doc_dir.is_dir():
        return []
    results: list[Path] = []
    for md_file in sorted(doc_dir.glob("*.md")):
        if md_file.name.lower() == "knowledge.md":
            continue
        try:
            line_count = sum(1 for _ in md_file.open(encoding="utf-8"))
        except OSError:
            continue
        if line_count >= _MIN_LINES:
            results.append(md_file)
    return results


def _parse_knowledge_schema(task: PublicTask, prose_stem: str) -> list[str] | None:
    """Parse knowledge.md to extract column names for the entity matching prose_stem.

    Returns the full field list prefixed with 'id', or None if no match.
    """
    km_path = task.context_dir / "knowledge.md"
    if not km_path.is_file():
        return None
    try:
        text = km_path.read_text(encoding="utf-8")
    except OSError:
        return None

    stem_lower = prose_stem.lower().replace("_", "").replace("-", "")
    entity_pattern = re.compile(r"^###\s+(.+)$", re.MULTILINE)
    field_pattern = re.compile(r"^\s*-\s+\*\*(\w+)\s*\(", re.MULTILINE)

    all_fields: list[str] = []
    best_name: str | None = None

    matches = list(entity_pattern.finditer(text))
    for idx, m in enumerate(matches):
        entity_name = m.group(1).strip()
        name_lower = entity_name.lower().replace("_", "").replace("-", "").replace(" ", "")
        if stem_lower not in name_lower and name_lower not in stem_lower:
            continue
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[start:end]
        fields = field_pattern.findall(section)
        if fields and (not all_fields or len(fields) > len(all_fields)):
            all_fields = fields
            best_name = entity_name

    if not all_fields:
        return None
    columns = ["id", *all_fields]
    logger.info("Schema from knowledge.md entity '%s': %s", best_name, columns)
    return columns


def _make_text_adapter(model_adapter: Any) -> OpenAIModelAdapter:
    """Build a plain-text adapter (no tools) from the existing adapter's credentials."""
    model: str | None = getattr(model_adapter, "model", None)
    api_base: str | None = getattr(model_adapter, "api_base", None)
    api_key: str | None = getattr(model_adapter, "api_key", None)
    backend_kind: Any = getattr(model_adapter, "backend_kind", None)

    if hasattr(model_adapter, "inner"):
        inner = model_adapter.inner
        model = model or getattr(inner, "model", None)
        api_base = api_base or getattr(inner, "api_base", None)
        api_key = api_key or getattr(inner, "api_key", None)
        backend_kind = backend_kind or getattr(inner, "backend_kind", None)

    if not model or not api_base or not api_key:
        raise RuntimeError("Cannot extract model credentials from adapter for ETL.")

    return OpenAIModelAdapter(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.1,
        enable_thinking=False,
        backend_kind=backend_kind,
    )


def _extract_code_block(text: str) -> str:
    """Extract Python code from a markdown fenced block, or return raw text."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _sample_paragraphs(text: str, first_n: int = 3, stride_n: int = 10) -> str:
    """Sample paragraphs: first few + evenly spaced from the rest of the file.

    This ensures the LLM sees all paragraph types (identity, physical stats,
    demographics, classification, etc.) rather than only the first section.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= first_n + stride_n:
        return "\n\n".join(paras)

    samples = paras[:first_n]
    remaining = paras[first_n:]
    step = max(1, len(remaining) // stride_n)
    for i in range(0, len(remaining), step):
        samples.append(remaining[i])
        if len(samples) >= first_n + stride_n:
            break

    return "\n\n".join(samples)


def _sample_per_section(
    text: str, first_n: int = 2, extra_n: int = 2
) -> tuple[str, list[dict[str, Any]]]:
    """Sample paragraphs from each section for representative coverage.

    Returns (sampled_text, section_info_list).
    Falls back to flat sampling when the document has no multi-section structure.
    """
    sections = _detect_sections(text)

    if len(sections) <= 1:
        info = [{"title": s["title"], "para_count": s["para_count"]} for s in sections]
        return _sample_paragraphs(text), info

    samples: list[str] = []
    info: list[dict[str, Any]] = []

    for sec in sections:
        paras = [
            p.strip() for p in sec["content"].split("\n\n") if p.strip() and len(p.strip()) > 30
        ]
        info.append({"title": sec["title"], "para_count": len(paras)})
        if not paras:
            continue

        samples.append(f"=== Section: {sec['title']} ({len(paras)} paragraphs) ===")
        cap = first_n + extra_n if len(paras) >= 10 else min(2, len(paras))
        selected = paras[: min(first_n, cap)]
        remaining = paras[len(selected) :]
        if remaining and len(selected) < cap:
            want = cap - len(selected)
            step = max(1, len(remaining) // want)
            for j in range(0, len(remaining), step):
                selected.append(remaining[j])
                if len(selected) >= cap:
                    break
        samples.extend(selected)

    return "\n\n".join(samples), info


def _compress_chunk(
    adapter: OpenAIModelAdapter,
    chunk_text: str,
    schema_columns: list[str] | None = None,
) -> str:
    """Ask LLM to extract labeled key-value pairs from prose paragraphs."""
    schema_hint = ""
    if schema_columns:
        cols = ", ".join(schema_columns)
        schema_hint = (
            f"\nUse EXACTLY these field names when applicable: {cols}\n"
            "Map every extracted value to the closest matching field name from this list. "
            "Do NOT invent synonyms or alternative names.\n"
        )

    prompt = textwrap.dedent(f"""\
        Below are paragraphs from a data document. Each paragraph describes one entity \
        and mixes data with narrative filler.

        Task: for each paragraph, extract data as labeled key-value pairs.
        {schema_hint}
        Format — one line per entity-paragraph:
          ID: <number> | entity_name: <name> | <field>: <value> | ...

        Rules:
        - Start each line with the entity's numeric ID AND its name/codename/label \
        as stated in the paragraph. These two fields anchor every line.
        - Every value MUST have a descriptive field label before it.
        - Include units where present (cm, kg, U/L, mg/dL, etc.).
        - If a field is explicitly stated as 0.0, None, NaN, placeholder, or \
        redacted, output it as-is with a "(placeholder)" marker — e.g. \
        "height_cm: 0.0 (placeholder)". Do NOT substitute another number from \
        the same paragraph (such as a weight or mass value) for a placeholder field.
        - Preserve corrections: "field: initially X, corrected to Y".
        - Preserve URLs, names, dates, times, categories, classifications.
        - Strip ALL narrative filler (hobbies, catering, anecdotes, etc.).
        - Separate entries with ONE blank line. No commentary, no markdown.

        ---
        {chunk_text}
        ---""")
    messages = [
        ModelMessage(role="system", content="You are a data extraction preprocessor."),
        ModelMessage(role="user", content=prompt),
    ]
    response: ModelResponse = adapter.complete(messages)
    return response.content.strip()


def _compress_prose(
    adapter: OpenAIModelAdapter,
    text: str,
    schema_columns: list[str] | None = None,
) -> str:
    """Compress prose by stripping noise. All chunks across all sections run in one pool."""
    sections = _detect_sections(text)

    work: list[tuple[str | None, str]] = []
    if not sections:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i in range(0, len(paras), _COMPRESS_CHUNK_SIZE):
            work.append((None, "\n\n".join(paras[i : i + _COMPRESS_CHUNK_SIZE])))
    else:
        for sec in sections:
            paras = [
                p.strip() for p in sec["content"].split("\n\n") if p.strip() and len(p.strip()) > 30
            ]
            first = True
            for i in range(0, len(paras), _COMPRESS_CHUNK_SIZE):
                title = sec["title"] if first else None
                work.append((title, "\n\n".join(paras[i : i + _COMPRESS_CHUNK_SIZE])))
                first = False

    if not work:
        return text

    compressed: list[str | None] = [None] * len(work)
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futs = {
            pool.submit(_compress_chunk, adapter, item[1], schema_columns): idx
            for idx, item in enumerate(work)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                compressed[idx] = fut.result()
            except Exception as exc:
                logger.warning("Compress chunk %d failed: %s", idx, exc)

    parts: list[str] = []
    for idx, (title, _) in enumerate(work):
        if title:
            parts.append(f"### {title}")
        chunk_result = compressed[idx]
        if chunk_result:
            parts.append(chunk_result)
    return "\n\n".join(parts)


def _extract_csv_from_text(
    adapter: OpenAIModelAdapter,
    text: str,
    schema_columns: list[str] | None,
    question: str,
) -> str | None:
    """Single LLM call: text → CSV."""
    schema_line = ""
    if schema_columns:
        schema_line = f"\nColumns: {','.join(schema_columns)}\n"
    else:
        schema_line = (
            "\nInfer column names from the data. First column must be the entity's "
            "numeric ID named 'id'.\n"
        )

    prompt = textwrap.dedent(f"""\
        Below is a compressed data document. Entities are identified by numeric ID. \
        Different sections may provide different attributes for the same entity.

        Merge all attributes for the same entity ID into one row and output a CSV.
        {schema_line}
        Rules:
        - Output ONLY valid CSV. First line is the header.
        - One row per entity. Merge data from all sections by matching ID.
        - For corrections ("initially X, corrected to Y"): use Y.
        - For missing values: leave empty.
        - Strip backticks and trailing punctuation from URLs.
        - If the same ID maps to DIFFERENT entity names in different lines, \
        those are DIFFERENT entities with a data-entry error. Use the name \
        from the FIRST occurrence of that ID and discard conflicting lines.
        - Values marked "(placeholder)" should be treated as empty/missing.
        - No markdown fences, no explanation.

        Question context: {question}

        Document:
        {text}

        Output CSV now:""")
    messages = [
        ModelMessage(role="system", content="You are a CSV data extractor. Output only CSV."),
        ModelMessage(role="user", content=prompt),
    ]
    response: ModelResponse = adapter.complete(messages)
    csv_text = response.content.strip()
    if csv_text.startswith("```"):
        csv_text = csv_text.split("\n", 1)[1]
        csv_text = csv_text.rsplit("```", 1)[0].strip()
    try:
        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader, None)
        if not header or len(header) < 2:
            return None
        ncols = len(header)
        clean_rows = [header]
        for row in reader:
            if len(row) > ncols:
                row = row[:ncols]
            clean_rows.append(row)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerows(clean_rows)
        return buf.getvalue()
    except csv.Error:
        return None


def _direct_extract_csv(
    adapter: OpenAIModelAdapter,
    compressed_text: str,
    schema_columns: list[str] | None,
    question: str,
) -> str | None:
    """Extract CSV from compressed text. Parallelises across sections when possible."""
    sections = _detect_sections(compressed_text)
    data_sections = [s for s in sections if s["para_count"] > 0]

    # Single call when: schema provided, small text, or ≤2 sections
    if schema_columns or len(data_sections) <= 2:
        return _extract_csv_from_text(adapter, compressed_text, schema_columns, question)

    # Parallel per-section extraction
    import pandas as pd

    partial: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(data_sections), _MAX_WORKERS)) as pool:
        futs = {
            pool.submit(
                _extract_csv_from_text,
                adapter,
                f"### {s['title']}\n{s['content']}",
                None,
                question,
            ): idx
            for idx, s in enumerate(data_sections)
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                result = fut.result()
                if result:
                    partial[idx] = result
            except Exception as exc:
                logger.warning("Section %d extraction failed: %s", idx, exc)

    if not partial:
        return _extract_csv_from_text(adapter, compressed_text, None, question)

    dfs: list[pd.DataFrame] = []
    for idx in sorted(partial):
        try:
            df = pd.read_csv(io.StringIO(partial[idx]))
            if len(df) == 0:
                continue
            first_col = df.columns[0]
            if first_col != "id":
                df = df.rename(columns={first_col: "id"})
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        overlap = sorted(set(merged.columns) & set(df.columns) - {"id"})
        if overlap:
            df = df.drop(columns=overlap)
        if len(df.columns) > 1:
            merged = merged.merge(df, on="id", how="outer")
    buf = io.StringIO()
    merged.to_csv(buf, index=False)
    return buf.getvalue()


def _build_codegen_prompt(
    prose_path: Path,
    csv_path: Path,
    sample_text: str,
    total_paras: int,
    schema_columns: list[str] | None,
    question: str,
    section_info: list[dict[str, Any]] | None = None,
) -> str:
    """Build the prompt asking the LLM to generate extraction code."""
    schema_section = ""
    if schema_columns:
        schema_section = (
            f"\nTarget CSV columns (use exactly these, in this order):\n{schema_columns}\n"
        )

    structure_section = ""
    if section_info and len(section_info) > 1:
        lines = [
            f'  {i + 1}. "{s["title"]}" (~{s["para_count"]} paragraphs)'
            for i, s in enumerate(section_info)
        ]
        structure_section = (
            "Document structure (detected sections):\n"
            + "\n".join(lines)
            + "\n\nThe SAME entity (identified by its numeric ID) likely appears across "
            "multiple sections with different attributes. Merge all attributes for the "
            "same entity into a single CSV row.\n"
        )

    return textwrap.dedent(f"""\
        Write a Python script that reads the prose file and extracts ALL entities into a CSV.

        File: {prose_path}
        Total paragraphs: ~{total_paras} (separated by blank lines)
        Output CSV: {csv_path}
        {schema_section}
        Question context: {question}

        {structure_section}\
        Sampled paragraphs (labelled by section when the document has multiple sections):
        ---
        {sample_text}
        ---

        Requirements:
        - Read the ENTIRE file, split by blank lines into paragraphs.
        - For each paragraph, extract the entity's numeric ID using a broad regex — \
        e.g. r'(?:registration\\s+(?:number|code)|identifier|reference\\s+code|\
        (?:under|with)\\s+(?:the\\s+)?(?:unique\\s+)?(?:ID|identifier)|\
        (?:Case\\s+ID|[Pp]atient|event|race|docket(?:ed)?(?:\\s+as)?))\\s+(?:#\\s*)?(\\d+)'. \
        EVERY paragraph mentions an entity — if your regex misses some, broaden it.
        - Extract whatever attributes are present in that paragraph. Different \
        sections of the document contain different attributes for the same entity.
        - Merge all attributes for the same ID into a single row (dict keyed by ID).
        - For text fields: extract the VALUE only, not surrounding prose. \
        Handle "None"/"-"/"redacted"/"N/A"/"not available" → NaN.
        - For numeric fields with corrections: "initially 188.0, corrected to 193.0" → 193.0. \
        Use r'(?:corrected|amended|adjusted|revised|rectified|finalized)\\s+(?:to|at)\\s+([\\d.]+)' \
        to find the corrected value; if no correction, take the first numeric value.
        - Wrap EVERY regex match in try/except so one bad paragraph doesn't crash the script.
        - Use pandas to build the final DataFrame; write to CSV with index=False.
        - Print shape and first 5 rows for verification.
        - Output ONLY the Python code in a ```python``` block. No explanation.""")


def _build_retry_prompt(code: str, error: str) -> str:
    """Build a retry prompt with the error from the failed execution."""
    return textwrap.dedent(f"""\
        The extraction script failed. Fix the error and output the corrected script.

        Previous code:
        ```python
        {code}
        ```

        Error:
        {error}

        Output ONLY the corrected Python code in a ```python``` block. No explanation.""")


def _call_llm(adapter: OpenAIModelAdapter, prompt: str) -> str:
    """Single LLM call."""
    messages = [
        ModelMessage(
            role="system",
            content="You are a Python code generator for data extraction. Output only code.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    response: ModelResponse = adapter.complete(messages)
    return response.content


def _exec_code(code: str) -> tuple[bool, str]:
    """Execute Python code in a subprocess, return (success, stdout_or_stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=_CODE_EXEC_TIMEOUT,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {_CODE_EXEC_TIMEOUT}s"


_KV_LINE = re.compile(r"^ID:\s*(\d+)\s*\|\s*entity_name:\s*(.+?)\s*\|", re.MULTILINE)


def _resolve_id_conflicts(compressed: str) -> str:
    """Detect and fix systematic ID-shift errors in compressed prose.

    Pattern: the prose defines entity names in the first section (identity).
    Later sections sometimes shift IDs by a fixed offset — e.g. entity "X"
    is ID 365 in section 1, but sections 2-5 assign ID 369 to "X" instead
    (a chain shift where every ID is off by the same amount).

    Strategy:
      1. Build the canonical name→ID mapping from the FIRST occurrence of each name.
      2. Scan all lines — if a line's (ID, name) pair disagrees with the canonical
         mapping AND the canonical ID exists for that name, drop the line.
    """
    _placeholder_set = {"-", "none", "nan", "placeholder", "- (placeholder)", ""}

    name_to_first_id: dict[str, int] = {}
    id_to_first_name: dict[int, str] = {}
    for m in _KV_LINE.finditer(compressed):
        eid = int(m.group(1))
        name = m.group(2).strip()
        if name.lower() in _placeholder_set:
            continue
        name_to_first_id.setdefault(name, eid)
        id_to_first_name.setdefault(eid, name)

    fixed_lines: list[str] = []
    fixed_count = 0
    for line in compressed.splitlines():
        m = _KV_LINE.match(line)
        if m:
            eid = int(m.group(1))
            name = m.group(2).strip()
            if name.lower() not in _placeholder_set:
                canonical_id = name_to_first_id.get(name)
                if canonical_id is not None and canonical_id != eid:
                    line = line.replace(f"ID: {eid}", f"ID: {canonical_id}", 1)
                    if fixed_count < 10:
                        logger.info(
                            "ETL ID-fix: '%s' ID %d → %d",
                            name,
                            eid,
                            canonical_id,
                        )
                    fixed_count += 1
        fixed_lines.append(line)
    if fixed_count:
        logger.info("ETL ID-fix: corrected %d lines total", fixed_count)
    return "\n".join(fixed_lines)


def _validate_csv(csv_path: Path) -> tuple[list[str], int] | None:
    """Read a CSV and return (header, row_count), or None if invalid."""
    if not csv_path.is_file():
        return None
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            row_count = sum(1 for _ in reader)
        return (header, row_count) if row_count > 0 else None
    except csv.Error:
        return None


def extract_prose_file(
    adapter: OpenAIModelAdapter,
    path: Path,
    task: PublicTask,
    schema_columns: list[str] | None = None,
) -> ETLResult | None:
    """Extract structured data from a prose file.

    Pipeline: compress prose → try direct CSV extraction → fallback to code-gen.
    """
    full_text = path.read_text(encoding="utf-8")
    etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"
    etl_dir.mkdir(parents=True, exist_ok=True)
    csv_path = etl_dir / f"{path.stem}.csv"

    # ---- Phase 1: compress ----
    logger.info("ETL compress: %s (%d chars)", path.name, len(full_text))
    compressed = _compress_prose(adapter, full_text, schema_columns=schema_columns)
    compressed_path = etl_dir / f"{path.stem}_clean.md"
    compressed_path.write_text(compressed, encoding="utf-8")
    logger.info(
        "ETL compressed: %d → %d chars (%.0f%%)",
        len(full_text),
        len(compressed),
        len(compressed) / len(full_text) * 100,
    )

    # ---- Phase 1b: resolve ID↔name conflicts deterministically ----
    compressed = _resolve_id_conflicts(compressed)
    compressed_path.write_text(compressed, encoding="utf-8")

    # ---- Phase 2: LLM extraction ----

    # ---- Phase 2b: LLM extraction (if compressed text is small enough) ----
    if len(compressed) <= _DIRECT_EXTRACT_CHAR_LIMIT:
        logger.info("ETL: attempting LLM CSV extraction from compressed text")
        csv_text = _direct_extract_csv(
            adapter,
            compressed,
            schema_columns,
            task.question,
        )
        if csv_text:
            csv_path.write_text(csv_text, encoding="utf-8")
            result = _validate_csv(csv_path)
            if result:
                header, row_count = result
                logger.info(
                    "ETL direct: %s → %s (%d cols, %d rows)",
                    path.name,
                    csv_path,
                    len(header),
                    row_count,
                )
                return ETLResult(
                    source_file=path.name,
                    csv_path=str(csv_path),
                    columns=header,
                    row_count=row_count,
                )
        logger.warning("ETL direct extraction failed, falling back to code-gen")

    # ---- Phase 2b: code-generation fallback (on compressed file) ----
    paras = [p.strip() for p in compressed.split("\n\n") if p.strip()]
    sample_text, section_info = _sample_per_section(compressed)

    prompt = _build_codegen_prompt(
        compressed_path,
        csv_path,
        sample_text,
        len(paras),
        schema_columns,
        task.question,
        section_info=section_info,
    )

    code = ""
    for attempt in range(_MAX_RETRIES + 1):
        raw = _call_llm(adapter, prompt)
        code = _extract_code_block(raw)
        ok, output = _exec_code(code)
        if ok:
            logger.info("ETL code exec succeeded (attempt %d): %s", attempt + 1, output[:200])
            break
        logger.warning("ETL code exec failed (attempt %d): %s", attempt + 1, output[:300])
        if attempt < _MAX_RETRIES:
            prompt = _build_retry_prompt(code, output)
    else:
        logger.error("ETL code generation failed after %d attempts", _MAX_RETRIES + 1)
        return None

    result = _validate_csv(csv_path)
    if not result:
        logger.error("ETL code ran but CSV is empty or missing: %s", csv_path)
        return None

    header, row_count = result
    logger.info(
        "ETL codegen: %s → %s (%d cols, %d rows)",
        path.name,
        csv_path,
        len(header),
        row_count,
    )
    return ETLResult(
        source_file=path.name,
        csv_path=str(csv_path),
        columns=header,
        row_count=row_count,
    )


def run_etl_for_task(
    task: PublicTask,
    model_adapter: Any,
) -> list[ETLResult] | None:
    """Orchestrator: detect large prose files, extract each, return metadata list.

    Returns None if no large prose files found. Non-fatal: callers should
    catch exceptions and fall back to manual parsing.
    """
    prose_files = _detect_prose_files(task)
    if not prose_files:
        return None

    adapter = _make_text_adapter(model_adapter)

    def _process(pf: Path) -> ETLResult | None:
        schema = _parse_knowledge_schema(task, pf.stem)
        return extract_prose_file(adapter, pf, task, schema_columns=schema)

    if len(prose_files) == 1:
        result = _process(prose_files[0])
        return [result] if result else None

    results: list[ETLResult] = []
    with ThreadPoolExecutor(max_workers=min(len(prose_files), _MAX_WORKERS)) as pool:
        futs = {pool.submit(_process, pf): pf for pf in prose_files}
        for fut in as_completed(futs):
            try:
                result = fut.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.error("ETL failed for %s: %s", futs[fut].name, exc)

    return results if results else None
