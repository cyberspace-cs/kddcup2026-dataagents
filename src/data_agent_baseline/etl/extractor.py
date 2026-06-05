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

import logging
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_agent_baseline._constants import ETL_SCRATCH_ROOT
from data_agent_baseline.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    OpenAIModelAdapter,
)
from data_agent_baseline.etl._codegen import (
    MAX_RETRIES,
    build_codegen_prompt,
    build_retry_prompt,
    call_llm,
    exec_code,
    extract_code_block,
)
from data_agent_baseline.etl._compress import (
    compress_prose,
    direct_extract_csv,
    sample_paragraphs_by_section,
)
from data_agent_baseline.etl._detect import (
    ETLResult,
    detect_prose_files,
)
from data_agent_baseline.etl._merge import (
    deterministic_kv_to_csv,
    normalize_entity_lines,
    pre_merge_entity_lines,
    resolve_id_conflicts,
    validate_csv,
)

if TYPE_CHECKING:
    from data_agent_baseline.benchmark.schema import PublicTask

logger = logging.getLogger(__name__)

_MAX_WORKERS = 4
_LLM_CALL_TIMEOUT = 180
_ETL_HTTP_TIMEOUT = 300.0
_DIRECT_EXTRACT_CHAR_LIMIT = 60_000


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
        prose_sample = sample_paragraphs_by_section(prose_text, per_section=1)
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

    cached = validate_csv(csv_path)
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
    compressed = compress_prose(
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
        compressed = resolve_id_conflicts(compressed)

    compressed = pre_merge_entity_lines(compressed, primary_key=primary_key)
    if schema_columns:
        compressed = normalize_entity_lines(compressed, schema_columns, primary_key)
    compressed_path.write_text(compressed, encoding="utf-8")

    # ---- Phase 2a: deterministic KV→CSV (when normalized) ----
    if schema_columns:
        csv_text = deterministic_kv_to_csv(compressed, schema_columns, primary_key)
        if csv_text:
            csv_path.write_text(csv_text, encoding="utf-8")
            result = validate_csv(csv_path)
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
        csv_text = direct_extract_csv(
            adapter,
            compressed,
            schema_columns,
            task.question,
            primary_key=primary_key,
        )
        if csv_text:
            csv_path.write_text(csv_text, encoding="utf-8")
            result = validate_csv(csv_path)
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

    prompt = build_codegen_prompt(
        compressed_path,
        csv_path,
        sample_text,
        len(entity_lines),
        schema_columns,
        task.question,
    )

    code = ""
    for attempt in range(MAX_RETRIES + 1):
        raw = call_llm(adapter, prompt)
        code = extract_code_block(raw)
        ok, output = exec_code(code)
        if ok:
            logger.info("ETL code exec succeeded (attempt %d): %s", attempt + 1, output[:200])
            break
        logger.warning("ETL code exec failed (attempt %d): %s", attempt + 1, output[:300])
        if attempt < MAX_RETRIES:
            prompt = build_retry_prompt(code, output)
    else:
        logger.error("ETL code generation failed after %d attempts", MAX_RETRIES + 1)
        return None

    result = validate_csv(csv_path)
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
    prose_files = detect_prose_files(task, force=force_etl)
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
