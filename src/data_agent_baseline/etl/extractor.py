"""Core ETL: detect large prose files, compress noise, extract to CSV.

Pipeline (when knowledge.md schema is available):
  1. **Compress** — LLM strips narrative filler from prose chunks, producing
     pipe-separated key-value entity lines.
  2. **Pre-merge** — Python collapses multi-line entity records into one line
     per unique primary key.
  3. **Normalize** — reorder fields to match schema, drop unknowns.
  4. **Deterministic CSV** — Python parses normalized KV lines into CSV
     (no LLM needed).

Fallback (no schema): LLM direct extraction → code-generation.
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
    ModelAdapter,
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
_LLM_CALL_TIMEOUT = 180
_ETL_HTTP_TIMEOUT = 300.0
_EXTRACT_CHUNK_LINES = 30
_ENTITY_LINE = re.compile(r"^ID:\s*", re.MULTILINE)
_PROSE_EXTS = frozenset({".md", ".txt"})


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


def _detect_prose_files(task: PublicTask, *, force: bool = False) -> list[Path]:
    """Find .md/.txt files in context/doc/ and context/ root that exceed _MIN_LINES.

    When *force* is True the line-count threshold is skipped — any non-empty
    prose file qualifies.
    """
    seen: set[Path] = set()
    results: list[Path] = []

    def _scan_dir(directory: Path) -> None:
        if not directory.is_dir():
            return
        for f in sorted(directory.iterdir()):
            if not f.is_file() or f.suffix.lower() not in _PROSE_EXTS:
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
            if not force and line_count < _MIN_LINES:
                continue
            results.append(f)

    _scan_dir(task.context_dir / "doc")
    _scan_dir(task.context_dir)
    return results


def _parse_knowledge_schema(
    task: PublicTask, prose_path: Path, adapter: ModelAdapter
) -> tuple[list[str], str] | None:
    """Use LLM to extract column names and primary key from knowledge.md.

    Samples the prose file so the LLM can discover fields present in the data
    but missing from the governance document (e.g. record IDs).

    Returns (columns, primary_key) or None if no matching entity.
    """
    km_path = task.context_dir / "knowledge.md"
    if not km_path.is_file():
        return None
    try:
        km_text = km_path.read_text(encoding="utf-8")
    except OSError:
        return None

    prose_sample = ""
    try:
        prose_text = prose_path.read_text(encoding="utf-8")
        prose_sample = _sample_paragraphs_by_section(prose_text, per_section=1)
    except OSError:
        pass

    prose_stem = prose_path.stem
    prose_section = ""
    if prose_sample:
        prose_section = (
            f"\n\nSample paragraphs from EVERY SECTION of the prose file "
            f'"{prose_stem}.md" (use these to discover ALL numeric/categorical '
            f"data fields present in the prose, even if the governance document "
            f"does not list them):\n"
            f"---\n{prose_sample}\n---"
        )

    prompt = textwrap.dedent(f"""\
        Below is a data governance document AND a sample from a prose data file.
        Find the entity in the governance document that best matches \
        the prose file "{prose_stem}".

        If NO entity in the governance document matches, output exactly: NONE

        Otherwise output exactly TWO lines:
        Line 1: PK: <primary_key_field_name>
        Line 2: all field names as a comma-separated list (including the primary key)

        IMPORTANT:
        - PK and column names must be SHORT GENERIC FIELD NAMES (like "event_id", \
        "molecule_id", "patient_id"), NEVER a specific data value.
        - Start with fields from the governance document.
        - ALSO scan the prose sample for ADDITIONAL structured data fields \
        that are present in the data but MISSING from the governance document. \
        Add these to the list with descriptive short names matching the \
        abbreviations used in the prose.
        - Do NOT add narrative/filler fields — only structured data fields.

        Governance document:
        {km_text}{prose_section}""")
    messages = [
        ModelMessage(
            role="system",
            content="You extract database schema. Output PK line then comma-separated columns.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("LLM schema extraction failed: %s", exc)
        return None

    result = response.content.strip().strip("`").strip()
    if not result or result.upper() == "NONE":
        return None

    lines = [line.strip() for line in result.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    pk_line = lines[0]
    pk = pk_line.split(":", 1)[-1].strip() if pk_line.upper().startswith("PK") else None
    columns = [c.strip() for c in lines[1].split(",") if c.strip()]
    if len(columns) < 2 or not pk:
        return None
    logger.info("Schema from knowledge.md (LLM) for '%s': pk=%s, cols=%s", prose_stem, pk, columns)
    return columns, pk


def _make_text_adapter(model_adapter: Any) -> ModelAdapter:
    """Build a plain-text adapter (no tools) from the existing adapter's credentials.

    When *model_adapter* is a ``RateLimitedAdapter``, the returned adapter is
    also rate-limited — sharing the same token-bucket state files so ETL and
    agent calls compete for the same RPM/TPM budget.
    """
    from data_agent_baseline.agents.rate_limit import RateLimitedAdapter

    source = model_adapter
    rate_limit_cfg: Any = None
    rate_limit_key: str | None = None
    if isinstance(model_adapter, RateLimitedAdapter):
        source = model_adapter.inner
        rate_limit_cfg = model_adapter.rate_limit_config
        rate_limit_key = model_adapter.api_key

    model: str | None = getattr(source, "model", None)
    api_base: str | None = getattr(source, "api_base", None)
    api_key: str | None = getattr(source, "api_key", None)
    backend_kind: Any = getattr(source, "backend_kind", None)

    if not model or not api_base or not api_key:
        raise RuntimeError("Cannot extract model credentials from adapter for ETL.")

    has_rate_limiter = rate_limit_cfg is not None and rate_limit_key
    inner: ModelAdapter = OpenAIModelAdapter(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.1,
        enable_thinking=False,
        backend_kind=backend_kind,
        max_retries=0 if has_rate_limiter else 2,
        timeout=_ETL_HTTP_TIMEOUT,
    )

    if rate_limit_cfg is not None and rate_limit_key:
        return RateLimitedAdapter(
            inner=inner,
            api_key=rate_limit_key,
            rate_limit=rate_limit_cfg,
        )
    return inner


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


def _sample_paragraphs_by_section(text: str, per_section: int = 1) -> str:
    """Sample data-bearing paragraphs for field discovery.

    For multi-section files (>=3 sections), picks the longest paragraph from
    each section so every field subset is represented.  For single/few-section
    files, falls back to stride sampling for broader coverage.
    """
    sections = _detect_sections(text)
    if not sections or len(sections) < 3:
        return _sample_paragraphs(text, first_n=3, stride_n=5)

    samples: list[str] = []
    for sec in sections:
        samples.append(f"### {sec['title']}")
        paras = [
            p.strip() for p in sec["content"].split("\n\n") if p.strip() and len(p.strip()) > 50
        ]
        if not paras:
            continue
        paras.sort(key=len, reverse=True)
        for p in paras[:per_section]:
            samples.append(p[:1000])
    return "\n\n".join(samples)


def _compress_chunk(
    adapter: ModelAdapter,
    chunk_text: str,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
) -> str:
    """Ask LLM to extract labeled key-value pairs from prose paragraphs."""
    if schema_columns and primary_key:
        cols = ", ".join(schema_columns)
        schema_hint = (
            f"\nUse EXACTLY these field names: {cols}\n"
            "Map every extracted value to the closest matching field name from this list. "
            "Do NOT invent synonyms or alternative names.\n"
        )
        format_hint = (
            f"Format — one line per entity-paragraph, pipe-separated key-value pairs:\n"
            f"  {primary_key}: <value> | field_a: <value> | field_b: <value> | ...\n"
            "ONLY include fields that have actual values in the paragraph. "
            "OMIT fields not mentioned — do NOT output empty fields.\n"
        )
        anchor_rule = (
            f"- Start each line with {primary_key} as stated in the paragraph. "
            "This field anchors every line."
        )
    else:
        schema_hint = ""
        format_hint = (
            "Format — one line per entity-paragraph:\n"
            "  ID: <number> | entity_name: <name> | <field>: <value> | ...\n"
        )
        anchor_rule = (
            "- Start each line with the entity's numeric ID AND its name/codename/label "
            "as stated in the paragraph. These two fields anchor every line."
        )

    sparse = bool(schema_columns and len(schema_columns) > 10)
    if sparse:
        placeholder_rule = (
            "- If a field is not mentioned or its value is None/NaN/placeholder/redacted, "
            "SKIP IT entirely — do NOT include it in the output line."
        )
    else:
        placeholder_rule = (
            "- If a field's ENTIRE value is 0.0, None, NaN, placeholder, or "
            'redacted (with no real data), output it with a "(placeholder)" marker — '
            'e.g. "height_cm: 0.0 (placeholder)". Do NOT mark corrected values as '
            'placeholder — "initially X, corrected to Y" means Y is the real value.'
        )

    prompt = textwrap.dedent(f"""\
        Below are paragraphs from a data document. Each paragraph describes one entity \
        and mixes data with narrative filler.

        Task: for each paragraph, extract data as labeled key-value pairs.
        {schema_hint}
        {format_hint}
        Rules:
        {anchor_rule}
        - Every value MUST have a descriptive field label before it.
        - Output numeric values WITHOUT unit suffixes (e.g. 193.0, not 193.0 cm).
        {placeholder_rule}
        - For corrections: output the FINAL corrected value directly — e.g. \
        "label: positive", not "label: positive (placeholder)".
        - Preserve URLs, names, dates, times, categories, classifications.
        - Strip only narrative filler that contains no structured field, schema field, \
        ID, value, date, unit, label, status, or question-relevant fact.
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
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
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
            pool.submit(_compress_chunk, adapter, item[1], schema_columns, primary_key): idx
            for idx, item in enumerate(work)
        }
        try:
            for fut in as_completed(futs, timeout=_LLM_CALL_TIMEOUT * len(work)):
                idx = futs[fut]
                try:
                    compressed[idx] = fut.result(timeout=_LLM_CALL_TIMEOUT)
                except TimeoutError:
                    logger.warning("Compress chunk %d timed out", idx)
                except Exception as exc:
                    logger.warning("Compress chunk %d failed: %s", idx, exc)
        except TimeoutError:
            logger.warning("Compress pool timed out, proceeding with partial results")

    parts: list[str] = []
    for idx, (title, _) in enumerate(work):
        if title:
            parts.append(f"### {title}")
        chunk_result = compressed[idx]
        if chunk_result:
            parts.append(chunk_result)
    return "\n\n".join(parts)


def _extract_csv_from_text(
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str] | None,
    question: str,
) -> str | None:
    """Single LLM call: text → CSV.  Used as fallback when no schema is available."""
    schema_line = (
        f"\nColumns: {','.join(schema_columns)}\n"
        if schema_columns
        else "\nInfer column names from the data. First column must be the entity's "
        "numeric ID named 'id'.\n"
    )

    prompt = textwrap.dedent(f"""\
        Below is a compressed data document. Each line describes one entity \
        as pipe-separated key-value pairs (e.g. "key: value | key: value").

        Output a CSV with one row per entity.
        {schema_line}
        Rules:
        - Output ONLY valid CSV. First line is the header.
        - Placeholder/None/NaN/redacted values → leave the cell empty.
        - Strip unit suffixes (cm, kg, etc.) from numeric fields.
        - For genuinely missing values: leave empty.
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


def _split_entity_chunks(
    text: str,
    chunk_size: int = _EXTRACT_CHUNK_LINES,
    primary_key: str | None = None,
) -> list[str]:
    """Split compressed text into chunks of ~chunk_size entity lines each."""
    pk_token = f"{primary_key}:" if primary_key else None
    entity_lines: list[str] = []
    non_entity_header: list[str] = []
    for line in text.splitlines():
        is_entity = (pk_token and pk_token in line) if pk_token else _ENTITY_LINE.match(line)
        if is_entity:
            entity_lines.append(line)
        elif not entity_lines:
            non_entity_header.append(line)

    if len(entity_lines) <= chunk_size:
        return [text]

    header = "\n".join(non_entity_header).strip()
    chunks: list[str] = []
    for i in range(0, len(entity_lines), chunk_size):
        batch = "\n".join(entity_lines[i : i + chunk_size])
        chunks.append(f"{header}\n\n{batch}" if header else batch)
    return chunks


def _direct_extract_csv(
    adapter: ModelAdapter,
    compressed_text: str,
    schema_columns: list[str] | None,
    question: str,
    primary_key: str | None = None,
) -> str | None:
    """Extract CSV from compressed text. Splits into entity-line chunks for parallel extraction."""
    chunks = _split_entity_chunks(compressed_text, primary_key=primary_key)

    if len(chunks) == 1:
        return _extract_csv_from_text(adapter, compressed_text, schema_columns, question)

    import pandas as pd

    logger.info("ETL extract: splitting into %d chunks for parallel extraction", len(chunks))
    partial: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(chunks), _MAX_WORKERS)) as pool:
        futs = {
            pool.submit(
                _extract_csv_from_text,
                adapter,
                chunk,
                schema_columns,
                question,
            ): idx
            for idx, chunk in enumerate(chunks)
        }
        try:
            for fut in as_completed(futs, timeout=_LLM_CALL_TIMEOUT * len(chunks)):
                idx = futs[fut]
                try:
                    result = fut.result(timeout=_LLM_CALL_TIMEOUT)
                    if result:
                        partial[idx] = result
                except TimeoutError:
                    logger.warning("Extract chunk %d timed out", idx)
                except Exception as exc:
                    logger.warning("Extract chunk %d failed: %s", idx, exc)
        except TimeoutError:
            logger.warning("Extract pool timed out, proceeding with partial results")

    if not partial:
        return None

    dfs: list[pd.DataFrame] = []
    for idx in sorted(partial):
        try:
            df = pd.read_csv(io.StringIO(partial[idx]))
            if len(df) == 0:
                continue
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    dedup_col = (
        primary_key
        if primary_key and primary_key in merged.columns
        else next((c for c in ("entity_name", "id") if c in merged.columns), None)
    )
    if dedup_col:
        merged = merged.groupby(dedup_col, sort=False).first().reset_index()
    buf = io.StringIO()
    merged.to_csv(buf, index=False)
    return buf.getvalue()


def _build_codegen_prompt(
    compressed_path: Path,
    csv_path: Path,
    sample_text: str,
    total_lines: int,
    schema_columns: list[str] | None,
    question: str,
    **_kwargs: Any,
) -> str:
    """Build the prompt asking the LLM to generate extraction code.

    The input file is the compressed/normalized KV text, NOT the original prose.
    """
    schema_section = ""
    if schema_columns:
        schema_section = (
            f"\nTarget CSV columns (use exactly these, in this order):\n{schema_columns}\n"
        )

    return textwrap.dedent(f"""\
        Write a Python script that reads a pre-processed data file and converts it to CSV.

        Input file: {compressed_path}
        Format: each non-blank line is one entity as pipe-separated key-value pairs, e.g.
          field_a: value | field_b: value | field_c: value
        Total entity lines: ~{total_lines}
        Output CSV: {csv_path}
        {schema_section}
        Question context: {question}

        Sample lines:
        ---
        {sample_text}
        ---

        Requirements:
        - Read the ENTIRE file line by line.
        - For each non-blank, non-heading line, split by " | " then by ": " \
        to extract key-value pairs.
        - Values that are empty, "None", "-", "NaN", "redacted", or contain \
        "(placeholder)" should be treated as missing (NaN).
        - Strip unit suffixes (cm, kg, etc.) from numeric values.
        - Build a pandas DataFrame with one row per entity; write to CSV with index=False.
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


def _call_llm(adapter: ModelAdapter, prompt: str) -> str:
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

_PLACEHOLDER_VALS = frozenset(
    {
        "",
        "-",
        "none",
        "nan",
        "n/a",
        "placeholder",
        "none (placeholder)",
        "nan (placeholder)",
        "- (placeholder)",
        "0.0 (placeholder)",
        "redacted",
    }
)
_UNIT_SUFFIX = re.compile(r"^([\d.]+)\s*(?:cm|kg|U/L|mg/dL|mmHg|%)\s*$")


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


def _pre_merge_entity_lines(
    text: str,
    primary_key: str | None = None,
) -> str:
    """Merge multi-line entity records into one line per unique entity.

    Compression can produce several lines for the same entity (each carrying
    a different attribute subset).  This collapses them so downstream
    extraction sees one complete record per entity.
    """
    pk_token = f"{primary_key}:" if primary_key else None

    entity_lines: list[str] = []
    header_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (pk_token in stripped) if pk_token else _ENTITY_LINE.match(stripped)
        if is_entity:
            entity_lines.append(stripped)
        elif not entity_lines:
            header_lines.append(line)

    if len(entity_lines) <= 1:
        return text

    def _is_placeholder(val: str) -> bool:
        return val.lower().strip() in _PLACEHOLDER_VALS or "(placeholder)" in val.lower()

    def _parse_kv(line: str) -> tuple[str | None, list[tuple[str, str]]]:
        parts = line.split("|")
        pk_val: str | None = None
        pairs: list[tuple[str, str]] = []
        for part in parts:
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = key.strip()
            val = val.strip()
            if (pk_token and key == primary_key) or (not pk_token and key.upper() == "ID"):
                pk_val = val
            pairs.append((key, val))
        return pk_val, pairs

    groups: dict[str, dict[str, str]] = {}
    order: list[str] = []

    for line in entity_lines:
        pk_val, pairs = _parse_kv(line)
        if pk_val is None:
            continue
        if pk_val not in groups:
            groups[pk_val] = {}
            order.append(pk_val)
        rec = groups[pk_val]
        for k, v in pairs:
            if k not in rec or (_is_placeholder(rec[k]) and not _is_placeholder(v)):
                rec[k] = v

    merged: list[str] = []
    for pk_val in order:
        parts = [f"{k}: {v}" for k, v in groups[pk_val].items()]
        merged.append(" | ".join(parts))

    before = len(entity_lines)
    after = len(merged)
    if before != after:
        logger.info("ETL pre-merge: %d entity lines → %d unique entities", before, after)

    header = "\n".join(header_lines).strip()
    body = "\n".join(merged)
    return f"{header}\n\n{body}" if header else body


def _normalize_entity_lines(
    text: str,
    schema_columns: list[str],
    primary_key: str | None = None,
) -> str:
    """Reorder fields in every entity line to match schema_columns order.

    Drops fields not in the schema and fills missing fields with empty
    values, so downstream CSV extraction sees a uniform tabular layout.
    """
    pk_token = f"{primary_key}:" if primary_key else None
    col_set = set(schema_columns)

    normalized: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (pk_token in stripped) if pk_token else _ENTITY_LINE.match(stripped)
        if not is_entity:
            normalized.append(line)
            continue

        fields: dict[str, str] = {}
        for part in stripped.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = key.strip()
            if key in col_set:
                fields[key] = val.strip()

        parts = [f"{c}: {fields.get(c, '')}" for c in schema_columns]
        normalized.append(" | ".join(parts))

    return "\n".join(normalized)


def _clean_cell(val: str) -> str:
    """Normalize a single KV value into a CSV-ready cell."""
    if not val or val.lower().strip() in _PLACEHOLDER_VALS or "(placeholder)" in val.lower():
        return ""
    m = _UNIT_SUFFIX.match(val)
    if m:
        return m.group(1)
    return val


def _deterministic_kv_to_csv(
    text: str,
    schema_columns: list[str],
    primary_key: str | None = None,
) -> str | None:
    """Parse normalized KV entity lines directly into CSV without LLM.

    Each entity line is expected to be in the form:
      col_a: val | col_b: val | ...
    with columns in schema_columns order (produced by _normalize_entity_lines).
    """
    pk_token = f"{primary_key}:" if primary_key else None
    col_set = set(schema_columns)

    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (pk_token in stripped) if pk_token else _ENTITY_LINE.match(stripped)
        if not is_entity:
            continue

        fields: dict[str, str] = {}
        for part in stripped.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = key.strip()
            if key in col_set:
                fields[key] = _clean_cell(val.strip())

        row = [fields.get(c, "") for c in schema_columns]
        if any(row):
            rows.append(row)

    if not rows:
        return None

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(schema_columns)
    writer.writerows(rows)
    return buf.getvalue()


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
    adapter: ModelAdapter,
    path: Path,
    task: PublicTask,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
) -> ETLResult | None:
    """Extract structured data from a prose file.

    Pipeline: compress → pre-merge → normalize → deterministic CSV.
    Fallback: LLM direct extraction → code-gen.
    """
    full_text = path.read_text(encoding="utf-8")
    etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"
    etl_dir.mkdir(parents=True, exist_ok=True)
    csv_path = etl_dir / f"{path.stem}.csv"

    cached = _validate_csv(csv_path)
    if cached:
        header, row_count = cached
        logger.info("ETL cache hit: %s (%d cols, %d rows)", csv_path, len(header), row_count)
        return ETLResult(
            source_file=path.name,
            csv_path=str(csv_path),
            columns=header,
            row_count=row_count,
        )

    # ---- Phase 1: compress ----
    logger.info("ETL compress: %s (%d chars)", path.name, len(full_text))
    compressed = _compress_prose(
        adapter, full_text, schema_columns=schema_columns, primary_key=primary_key
    )
    compressed_path = etl_dir / f"{path.stem}_clean.md"
    compressed_path.write_text(compressed, encoding="utf-8")
    logger.info(
        "ETL compressed: %d → %d chars (%.0f%%)",
        len(full_text),
        len(compressed),
        len(compressed) / len(full_text) * 100,
    )

    if not primary_key:
        compressed = _resolve_id_conflicts(compressed)

    compressed = _pre_merge_entity_lines(compressed, primary_key=primary_key)
    if schema_columns:
        compressed = _normalize_entity_lines(compressed, schema_columns, primary_key)
    compressed_path.write_text(compressed, encoding="utf-8")

    # ---- Phase 2a: deterministic KV→CSV (when normalized) ----
    if schema_columns:
        csv_text = _deterministic_kv_to_csv(compressed, schema_columns, primary_key)
        if csv_text:
            csv_path.write_text(csv_text, encoding="utf-8")
            result = _validate_csv(csv_path)
            if result:
                header, row_count = result
                logger.info(
                    "ETL deterministic: %s → %s (%d cols, %d rows)",
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
            logger.warning("ETL deterministic extraction produced invalid CSV, falling back")

    # ---- Phase 2b: LLM extraction (if compressed text is small enough) ----
    if len(compressed) <= _DIRECT_EXTRACT_CHAR_LIMIT:
        logger.info("ETL: attempting LLM CSV extraction from compressed text")
        csv_text = _direct_extract_csv(
            adapter,
            compressed,
            schema_columns,
            task.question,
            primary_key=primary_key,
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

    # ---- Phase 2c: code-generation fallback (on compressed file) ----
    entity_lines = [
        ln.strip() for ln in compressed.splitlines() if ln.strip() and not ln.startswith("#")
    ]
    sample_text = "\n".join(entity_lines[:5] + (["..."] if len(entity_lines) > 5 else []))

    prompt = _build_codegen_prompt(
        compressed_path,
        csv_path,
        sample_text,
        len(entity_lines),
        schema_columns,
        task.question,
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
    force_etl = task.difficulty.lower() in {"hard", "extreme"}
    prose_files = _detect_prose_files(task, force=force_etl)
    if not prose_files:
        return None

    adapter = _make_text_adapter(model_adapter)

    def _process(pf: Path) -> ETLResult | None:
        parsed = _parse_knowledge_schema(task, pf, adapter)
        if not parsed:
            logger.info("ETL skip: no matching entity for %s", pf.name)
            return None
        schema, pk = parsed
        return extract_prose_file(adapter, pf, task, schema_columns=schema, primary_key=pk)

    if len(prose_files) == 1:
        result = _process(prose_files[0])
        return [result] if result else None

    results: list[ETLResult] = []
    orch_timeout = _LLM_CALL_TIMEOUT * max(4, len(prose_files))
    with ThreadPoolExecutor(max_workers=min(len(prose_files), _MAX_WORKERS)) as pool:
        futs = {pool.submit(_process, pf): pf for pf in prose_files}
        try:
            for fut in as_completed(futs, timeout=orch_timeout):
                try:
                    result = fut.result(timeout=orch_timeout)
                    if result is not None:
                        results.append(result)
                except TimeoutError:
                    logger.error("ETL timed out for %s", futs[fut].name)
                except Exception as exc:
                    logger.error("ETL failed for %s: %s", futs[fut].name, exc)
        except TimeoutError:
            logger.error("ETL orchestrator timed out, proceeding with partial results")

    return results if results else None
