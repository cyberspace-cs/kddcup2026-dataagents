"""Core ETL: detect large prose files, compress noise, extract to CSV.

Pipeline (when knowledge.md schema is available):
  1. **Compress** — LLM strips narrative filler from prose chunks, producing
     pipe-separated key-value entity lines.
  2. **Pre-merge** — Python collapses multi-line entity records into one line
     per unique primary key.
  3. **Normalize** — reorder fields to match schema, drop unknowns.
  4. **Deterministic CSV** — Python parses normalized KV lines into CSV
     (no LLM needed).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeout,
    as_completed,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.config import ETL_SCRATCH_ROOT
from agents.etl._compress import compress_prose, sample_paragraphs_by_section
from agents.etl._constants import (
    ETL_HTTP_TIMEOUT,
    ETL_MAX_OUTPUT_TOKENS,
    ETL_ORCHESTRATOR_LLM_CALL_TIMEOUT,
    ETL_ORCHESTRATOR_MAX_WORKERS,
    ETL_RESERVE_OUTPUT_TOKENS,
)
from agents.etl._detect import ETLResult, detect_prose_files
from agents.etl._identity import validate_csv
from agents.etl._merge import (
    deterministic_kv_to_csv,
    normalize_entity_lines,
    pre_merge_entity_lines,
    resolve_id_conflicts,
)
from agents.etl._pdf import pdf_to_markdown
from agents.etl._reconcile import (
    fix_malformed_lines,
    reconcile_field_names,
    unify_csv_synonyms,
)
from agents.etl._schema import (
    infer_field_units_from_prose,
    infer_schema_from_sample,
    merge_prose_schema_into_governance,
    parse_knowledge_schema,
    repair_csv_numeric_identities,
)
from agents.etl._threading import submit_in_context
from agents.etl._trace import (
    ETLFileTrace,
    etl_phase,
    reset_etl_trace,
    serialize_trace,
    set_etl_trace,
)
from agents.etl._units import (
    apply_proportion_convention,
    infer_proportion_conventions,
    vote_target_units,
    write_units_sidecar,
)
from agents.etl._verify import (
    dedup_cross_field_copies,
    retry_missing_values,
    verify_field_values,
)
from agents.etl.knowledge import km_table_fields
from agents.llm import ModelAdapter, OpenAIModelAdapter

if TYPE_CHECKING:
    from agents.benchmark.schema import PublicTask

logger = logging.getLogger(__name__)


def _make_text_adapter(model_adapter: Any) -> ModelAdapter:
    """Build a plain-text adapter (no tools) from the existing adapter's credentials.

    When *model_adapter* is a ``RateLimitedAdapter``, the returned adapter is
    also rate-limited — sharing the same token-bucket state files so ETL and
    agent calls compete for the same RPM/TPM budget.  Likewise, when the
    incoming adapter is wrapped with ``TracedModelAdapter``, the returned
    adapter is re-wrapped so ETL LLM calls produce generation spans.
    """
    from agents.llm.rate_limit import RateLimitedAdapter
    from agents.tracing.instrument import TracedModelAdapter

    source = model_adapter
    traced = False
    rate_limit_cfg: Any = None
    rate_limit_key: str | None = None

    if isinstance(source, TracedModelAdapter):
        traced = True
        source = source.inner

    if isinstance(source, RateLimitedAdapter):
        rate_limit_cfg = source.rate_limit_config
        rate_limit_key = source.api_key
        source = source.inner

    model: str | None = getattr(source, "model", None)
    api_base: str | None = getattr(source, "api_base", None)
    api_key: str | None = getattr(source, "api_key", None)
    backend_kind: Any = getattr(source, "backend_kind", None)

    if not model or not api_base or not api_key:
        raise RuntimeError("Cannot extract model credentials from adapter for ETL.")

    has_rate_limiter = rate_limit_cfg is not None and rate_limit_key
    seed: int | None = getattr(source, "seed", None)
    top_p: float | None = getattr(source, "top_p", None)
    top_k: int | None = getattr(source, "top_k", None)
    is_deepseek = backend_kind == "deepseek" or getattr(source, "thinking_mode", None) is not None

    inner: ModelAdapter = OpenAIModelAdapter(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.1,
        max_tokens=ETL_MAX_OUTPUT_TOKENS,
        enable_thinking=None if is_deepseek else False,
        thinking_mode="disabled" if is_deepseek else None,
        backend_kind=backend_kind,
        seed=seed,
        top_p=top_p,
        top_k=top_k,
        max_retries=0 if has_rate_limiter else 2,
        timeout=ETL_HTTP_TIMEOUT,
    )

    if rate_limit_cfg is not None and rate_limit_key:
        inner = RateLimitedAdapter(
            inner=inner,
            api_key=rate_limit_key,
            # ETL 用更小的 reserve 预扣；少数超出的输出靠 complete() 的 usage 对账补扣。
            rate_limit=dataclasses.replace(
                rate_limit_cfg, reserve_output_tokens=ETL_RESERVE_OUTPUT_TOKENS
            ),
        )

    if traced:
        inner = TracedModelAdapter(inner)

    return inner


def extract_prose_file(
    adapter: ModelAdapter,
    path: Path,
    task: PublicTask,
    prose_text: str,
    schema_columns: list[str] | None = None,
    anchor_keys: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
    field_units: dict[str, str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
    governance_columns: list[str] | None = None,
    proportion_conventions: dict[str, str] | None = None,
) -> ETLResult | None:
    """Extract structured data from a prose file.

    Pipeline: compress → pre-merge → normalize → deterministic CSV.
    """
    primary_key = anchor_keys[0] if anchor_keys else None

    etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"
    etl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = etl_dir / f"{path.stem}.csv"
    full_text = prose_text

    def _finalize(phase: str) -> ETLResult | None:
        """Validate the materialized CSV, unify value-split synonym columns,
        repair approximate numerals from mined identities, build result."""
        result = validate_csv(csv_path)
        if not result:
            return None
        changed = False
        with etl_phase("identity_repair") as p:
            repaired = repair_csv_numeric_identities(csv_path)
            p.metrics["repaired"] = bool(repaired)
            if repaired:
                changed = True
        if governance_columns:
            with etl_phase("synonym_unify") as p:
                applied = unify_csv_synonyms(
                    adapter,
                    csv_path,
                    path.stem,
                    governance_columns,
                    anchor_keys or [],
                    schema_field_defs or {},
                    task.question,
                    cache_dir / f"{path.stem}_units.json",
                )
                p.metrics["applied"] = bool(applied)
                if applied:
                    changed = True
        if changed:
            result = validate_csv(csv_path) or result
        header, row_count = result
        logger.info(
            "ETL %s: %s → %s (%d cols, %d rows)",
            phase,
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

    if path.suffix.lower() == ".pdf":
        md_path = cache_dir / f"{path.stem}.md"
        md_path.write_text(full_text, encoding="utf-8")

    # ---- Cache: reuse existing CSV if present and valid ----
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

    if not full_text.strip():
        logger.info("ETL skip: no extractable text in %s", path.name)
        return None

    # ---- Phase 0: infer schema from prose sample (when no schema from knowledge.md) ----
    if not schema_columns:
        with etl_phase("schema_infer", input_size=len(full_text)) as p:
            inferred = infer_schema_from_sample(
                adapter, full_text, task.question, table_name=path.stem
            )
            if inferred:
                schema_columns, anchor_keys, schema_field_types, field_units, schema_field_defs = (
                    inferred
                )
                primary_key = anchor_keys[0] if anchor_keys else None
                p.metrics["column_count"] = len(schema_columns)
                p.metrics["anchor_count"] = len(anchor_keys)
            else:
                p.skipped = True

    # ---- Phase 1: compress ----
    logger.info("ETL compress: %s (%d chars)", path.name, len(full_text))
    with etl_phase("compress", input_size=len(full_text)) as p:
        compressed, entity_groups, compress_guide = compress_prose(
            adapter,
            full_text,
            schema_columns=schema_columns,
            primary_key=primary_key,
            anchor_keys=anchor_keys,
            schema_field_types=schema_field_types,
            schema_field_defs=schema_field_defs,
        )
        p.metrics["output_chars"] = len(compressed)
        p.metrics["compression_ratio"] = (
            round(len(compressed) / len(full_text), 3) if full_text else 0
        )
        p.metrics["entity_count"] = len(entity_groups) if entity_groups else 0
    compressed_path = cache_dir / f"{path.stem}_clean.md"
    compressed_path.write_text(compressed, encoding="utf-8")
    logger.info(
        "ETL compressed: %d → %d chars (%.0f%%)",
        len(full_text),
        len(compressed),
        len(compressed) / len(full_text) * 100,
    )

    if not primary_key:
        compressed = resolve_id_conflicts(compressed)

    with etl_phase("pre_merge", input_size=len(compressed)) as p:
        compressed = pre_merge_entity_lines(
            compressed,
            primary_key=primary_key,
            anchor_keys=anchor_keys,
            source_text=full_text,
        )
        p.output_size = len(compressed)
    if schema_columns:
        with etl_phase("fix_format", input_size=len(compressed)) as p:
            compressed = fix_malformed_lines(
                adapter,
                compressed,
                schema_columns,
                schema_field_defs,
            )
            p.output_size = len(compressed)
        with etl_phase("reconcile", input_size=len(compressed)) as p:
            compressed = reconcile_field_names(
                adapter,
                compressed,
                schema_columns,
                schema_field_defs,
            )
            p.output_size = len(compressed)
        with etl_phase("normalize", input_size=len(compressed)) as p:
            compressed = normalize_entity_lines(compressed, schema_columns, primary_key)
            p.output_size = len(compressed)
        with etl_phase("dedup", input_size=len(compressed)) as p:
            compressed, dedup_cleared = dedup_cross_field_copies(
                adapter,
                compressed,
                schema_columns,
                primary_key,
                anchor_keys,
                full_text,
                schema_field_types,
            )
            p.output_size = len(compressed)
            p.metrics["cleared_count"] = len(dedup_cleared) if dedup_cleared else 0
        with etl_phase("retry", input_size=len(compressed)) as p:
            compressed = retry_missing_values(
                adapter,
                compressed,
                schema_columns,
                primary_key,
                anchor_keys,
                entity_groups,
                schema_field_types,
                schema_field_defs,
                forced_gaps=dedup_cleared,
            )
            p.output_size = len(compressed)
        with etl_phase("verify", input_size=len(compressed)) as p:
            compressed = verify_field_values(
                adapter,
                compressed,
                schema_columns,
                primary_key,
                anchor_keys,
                entity_groups,
                schema_field_types,
                schema_field_defs,
                compress_guide=compress_guide,
            )
            p.output_size = len(compressed)
    compressed_path.write_text(compressed, encoding="utf-8")

    if field_units:
        apply_proportion_convention(field_units, proportion_conventions)
        write_units_sidecar(task, path.stem, field_units)

    # ---- Phase 2a: deterministic KV→CSV (when normalized) ----
    if schema_columns:
        with etl_phase("csv", input_size=len(compressed)) as p:
            csv_text = deterministic_kv_to_csv(
                compressed,
                schema_columns,
                primary_key,
                schema_field_types,
            )
            p.metrics["produced"] = csv_text is not None
        if csv_text:
            csv_path.write_text(csv_text, encoding="utf-8")
            result = _finalize("deterministic")
            if result:
                return result
            logger.warning("ETL deterministic extraction produced invalid CSV")
    return None


def run_etl_for_task(
    task: PublicTask,
    model_adapter: Any,
) -> list[ETLResult] | None:
    """Orchestrator: detect prose files, extract each, return metadata list.

    Returns None if no prose files were converted. Individual file failures are
    logged and skipped so one bad document does not block usable ETL output from
    sibling documents.
    """
    prose_files = detect_prose_files(task)
    if not prose_files:
        return None

    adapter = _make_text_adapter(model_adapter)

    # Read knowledge.md once for target-unit voting (shared across all files).
    km_path = task.context_dir / "knowledge.md"
    km_text = ""
    if km_path.is_file():
        with contextlib.suppress(OSError):
            km_text = km_path.read_text(encoding="utf-8")

    # Route: select only relevant prose files for ETL.
    if len(prose_files) >= 2:
        from agents.etl._router import route_prose_files

        prose_files = route_prose_files(adapter, task.question, prose_files, km_text)
    if not prose_files:
        return None

    # Observe once how sibling structured sources store proportions; the
    # per-subfamily verdicts deterministically override the heuristic
    # %→ratio decision for fields of the same naming subfamily.
    proportion_conventions = infer_proportion_conventions(task.context_dir)
    if proportion_conventions:
        logger.info("ETL: sibling proportion conventions: %s", proportion_conventions)

    def _process(pf: Path) -> ETLResult | None:
        # Early cache check: skip file reading and all LLM calls if CSV exists.
        etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"
        csv_path = etl_dir / f"{pf.stem}.csv"
        cached = validate_csv(csv_path)
        if cached:
            header, row_count = cached
            logger.info(
                "ETL cache hit (early): %s (%d cols, %d rows)", csv_path, len(header), row_count
            )
            return ETLResult(
                source_file=pf.name,
                csv_path=str(csv_path),
                columns=header,
                row_count=row_count,
            )

        from agents.tracing.create import function_span

        file_trace = ETLFileTrace(task.task_id, pf.stem)
        tokens = set_etl_trace(file_trace)
        cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with function_span(f"etl/file/{pf.stem}"):
                return _process_inner(pf)
        finally:
            try:
                trace_json = serialize_trace(file_trace)
                trace_path = cache_dir / f"{pf.stem}_trace.json"
                trace_path.write_text(
                    json.dumps(trace_json, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                logger.debug("Failed to write ETL trace for %s", pf.stem, exc_info=True)
            reset_etl_trace(tokens)

    def _process_inner(pf: Path) -> ETLResult | None:
        if pf.suffix.lower() == ".pdf":
            logger.info("ETL PDF→markdown: %s", pf.name)
            text: str = pdf_to_markdown(pf)  # pyright: ignore[reportUnknownVariableType]
        else:
            text = pf.read_text(encoding="utf-8")

        with etl_phase("schema_parse", input_size=len(text)) as p:
            parsed = parse_knowledge_schema(task, pf, adapter, text)
            p.metrics["found"] = parsed is not None
        if parsed:
            schema, anchors, field_types, units, defs = parsed
            with etl_phase("schema_merge") as p:
                extra_cols, extra_types, extra_defs = merge_prose_schema_into_governance(
                    adapter,
                    text,
                    schema,
                    defs,
                    field_types,
                    task.question,
                    table_name=pf.stem,
                )
                p.metrics["extra_columns"] = len(extra_cols) if extra_cols else 0
            if extra_cols:
                schema.extend(extra_cols)
                field_types.update(extra_types)
                defs.update(extra_defs)
                logger.info(
                    "ETL schema union: added %d prose-discovered fields: %s",
                    len(extra_cols),
                    extra_cols,
                )
            units.pop("_default", None)
            prose_units = infer_field_units_from_prose(
                adapter,
                sample_paragraphs_by_section(text) if text else "",
                schema,
                defs,
                units,
            )
            units.update(prose_units)
            if km_text and units:
                targets = vote_target_units(adapter, km_text, units, question=task.question)
                if targets:
                    for field, (target_unit, factor) in targets.items():
                        units[f"_target_{field}"] = target_unit
                        units[f"_factor_{field}"] = str(factor)
                    logger.info("ETL: voted target units for %s: %s", pf.stem, targets)
            governance_cols = sorted(km_table_fields(km_text, pf.stem)) if km_text else []
            return extract_prose_file(
                adapter,
                pf,
                task,
                text,
                schema_columns=schema,
                anchor_keys=anchors,
                schema_field_types=field_types,
                field_units=units,
                schema_field_defs=defs,
                governance_columns=governance_cols or None,
                proportion_conventions=proportion_conventions,
            )
        logger.info("ETL: no knowledge.md schema for %s, will infer from sample", pf.name)
        return extract_prose_file(
            adapter, pf, task, text, proportion_conventions=proportion_conventions
        )

    def _run_one(pf: Path) -> ETLResult | None:
        try:
            result = _process(pf)
        except Exception as exc:
            logger.warning("ETL failed for %s; skipping file: %s", pf.name, exc)
            return None
        if result is None:
            logger.warning("ETL produced no CSV for %s; skipping file", pf.name)
            return None
        return result

    if len(prose_files) == 1:
        result = _run_one(prose_files[0])
        return [result] if result is not None else None

    results: list[ETLResult] = []
    orch_timeout = ETL_ORCHESTRATOR_LLM_CALL_TIMEOUT * max(4, len(prose_files))
    pool = ThreadPoolExecutor(max_workers=min(len(prose_files), ETL_ORCHESTRATOR_MAX_WORKERS))
    try:
        futs = {submit_in_context(pool, _run_one, pf): pf for pf in prose_files}
        try:
            for fut in as_completed(futs, timeout=orch_timeout):
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.warning(
                        "ETL worker failed for %s; skipping file: %s", futs[fut].name, exc
                    )
                    continue
                if result is not None:
                    results.append(result)
        except FuturesTimeout:
            unfinished = [pf.name for fut, pf in futs.items() if not fut.done()]
            logger.warning("ETL orchestrator timed out; skipping unfinished files: %s", unfinished)
            for fut in futs:
                if not fut.done():
                    fut.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return results or None
