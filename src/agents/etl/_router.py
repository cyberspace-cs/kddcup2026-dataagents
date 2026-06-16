"""Document routing: select which prose files are relevant to the task question.

Given ≥2 prose files, builds a lightweight profile per file (schema fields from
knowledge.md + sampled paragraphs) and asks the LLM which files contain data
needed to answer the question. Files not selected are skipped from ETL entirely
and excluded from the virtual context.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from agents.etl._compress import sample_paragraphs_by_section
from agents.etl._pdf import pdf_to_markdown
from agents.etl.knowledge import km_table_fields

if TYPE_CHECKING:
    from agents.llm import ModelAdapter

logger = logging.getLogger(__name__)

_ROUTER_SAMPLE_TOKEN_BUDGET = 2000
_MAX_FIELDS_PER_FILE = 15
_MAX_SAMPLE_CHARS = 500
_MAX_DESC_CHARS = 200


class FileProfile(TypedDict):
    filename: str
    stem: str
    sample: str
    schema_fields: list[str]
    schema_descriptions: dict[str, str]
    schema_source: str
    char_count: int


def read_prose_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return pdf_to_markdown(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _infer_schema_lite(adapter: ModelAdapter, sample: str, question: str) -> list[str] | None:
    from agents.llm.types import ModelMessage

    if not sample or len(sample) < 50:
        return None

    prompt = (
        f"Below is a sample from a prose data document. Each paragraph describes one entity.\n\n"
        f"Question for context: {question}\n\n"
        f"Sample:\n{sample}\n\n"
        f"List ALL structured data field names present (snake_case). "
        f"Ignore narrative filler. Output ONLY a comma-separated list on one line."
    )
    messages = [
        ModelMessage(
            role="system", content="Extract field names from prose. One line, comma-separated."
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response = adapter.complete(messages)
    except Exception as exc:
        logger.warning("Lite schema inference failed: %s", exc)
        return None

    line = response.content.strip().splitlines()[0] if response.content.strip() else ""
    fields = [f.strip().lower().replace(" ", "_") for f in line.split(",") if f.strip()]
    return fields if fields else None


def empty_file_profile(path: Path) -> FileProfile:
    return {
        "filename": path.name,
        "stem": path.stem,
        "sample": "",
        "schema_fields": [],
        "schema_descriptions": {},
        "schema_source": "error",
        "char_count": 0,
    }


def build_file_profile(
    path: Path, km_text: str, question: str, adapter: ModelAdapter | None = None
) -> FileProfile:
    text = read_prose_text(path)
    sample = sample_paragraphs_by_section(text, token_limit=_ROUTER_SAMPLE_TOKEN_BUDGET)
    km_fields = km_table_fields(km_text, path.stem) if km_text else {}

    schema_fields: list[str] = list(km_fields.keys())
    schema_descriptions: dict[str, str] = dict(km_fields)
    schema_source = "knowledge.md" if km_fields else "none"

    if not km_fields and adapter is not None:
        fields = _infer_schema_lite(adapter, sample, question)
        if fields:
            schema_fields = fields
            schema_source = "inferred"

    return {
        "filename": path.name,
        "stem": path.stem,
        "sample": sample[:3000],
        "schema_fields": schema_fields,
        "schema_descriptions": schema_descriptions,
        "schema_source": schema_source,
        "char_count": len(text),
    }


def format_profiles_for_prompt(profiles: list[FileProfile]) -> str:
    parts: list[str] = []
    for i, p in enumerate(profiles, 1):
        fields = p["schema_fields"][:_MAX_FIELDS_PER_FILE]
        fields_str = ", ".join(fields) if fields else "(unknown)"
        descs = p.get("schema_descriptions", {})
        desc_str = "; ".join(f"{k}={v}" for k, v in list(descs.items())[:8]) if descs else ""
        parts.append(
            f"[{i}] {p['filename']}\n"
            f"    Fields: {fields_str}\n"
            f"    Descriptions: {desc_str[:_MAX_DESC_CHARS]}\n"
            f"    Sample: {p['sample'][:_MAX_SAMPLE_CHARS]}\n"
        )
    return "\n".join(parts)


def route_file_profiles(
    adapter: ModelAdapter, question: str, profiles: list[FileProfile]
) -> list[str]:
    """Ask the LLM which file profiles are relevant. Returns selected stems."""
    from agents.llm.types import ModelMessage

    profiles_text = format_profiles_for_prompt(profiles)
    filenames = [p["filename"] for p in profiles]

    prompt = (
        f"Question: {question}\n\n"
        f"Documents:\n{profiles_text}\n\n"
        f"Select ONLY documents whose fields are DIRECTLY needed to answer the question.\n"
        f"Output a JSON array of filenames. Available: {json.dumps(filenames)}"
    )
    messages = [
        ModelMessage(role="system", content="Select relevant documents. Output ONLY a JSON array."),
        ModelMessage(role="user", content=prompt),
    ]

    response = adapter.complete(messages)
    content = response.content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if not match:
        logger.warning("Router returned unparseable response, selecting all: %s", content[:200])
        return [p["stem"] for p in profiles]

    try:
        selected = cast(list[object], json.loads(match.group()))
    except json.JSONDecodeError:
        logger.warning("Router JSON parse failed, selecting all: %s", content[:200])
        return [p["stem"] for p in profiles]

    logger.info("Router raw selection: %s", selected)

    name_to_stem = {p["filename"]: p["stem"] for p in profiles}
    stem_to_stem = {p["stem"]: p["stem"] for p in profiles}
    name_lower_to_stem = {p["filename"].lower(): p["stem"] for p in profiles}
    stem_lower_to_stem = {p["stem"].lower(): p["stem"] for p in profiles}
    result: list[str] = []
    seen: set[str] = set()
    for item in selected:
        name = str(item).strip()
        matched: str | None = None
        if name in name_to_stem:
            matched = name_to_stem[name]
        elif name in stem_to_stem:
            matched = stem_to_stem[name]
        else:
            stem = Path(name).stem
            nl = name.lower()
            sl = stem.lower()
            if nl in name_lower_to_stem:
                matched = name_lower_to_stem[nl]
            elif sl in stem_lower_to_stem:
                matched = stem_lower_to_stem[sl]
        if matched and matched not in seen:
            seen.add(matched)
            result.append(matched)

    if not result:
        logger.warning("Router selected no valid files (raw: %s), returning all", selected)
        return [p["stem"] for p in profiles]
    return result


def route_prose_files(
    adapter: ModelAdapter,
    question: str,
    prose_files: list[Path],
    km_text: str,
) -> list[Path]:
    """Select prose files relevant to the task question.

    Returns the subset of prose_files that should undergo ETL extraction.
    Conservative: returns all files on parse failure or when fewer than 2 files.
    """
    if len(prose_files) < 2:
        return list(prose_files)

    profiles: list[FileProfile] = []
    for pf in prose_files:
        try:
            profiles.append(build_file_profile(pf, km_text, question, adapter=adapter))
        except Exception as exc:
            logger.warning("Failed to build profile for %s: %s", pf.name, exc)
            profiles.append(empty_file_profile(pf))

    try:
        selected_stems = route_file_profiles(adapter, question, profiles)
    except Exception as exc:
        logger.warning("Router LLM call failed, selecting all: %s", exc)
        return list(prose_files)

    stem_to_path = {pf.stem: pf for pf in prose_files}
    result: list[Path] = []
    seen: set[str] = set()
    for stem in selected_stems:
        matched = stem_to_path.get(stem)
        if matched and matched.name not in seen:
            seen.add(matched.name)
            result.append(matched)

    if not result:
        logger.warning("Router selected no valid files (raw: %s), returning all", selected_stems)
        return list(prose_files)

    skipped = [pf.name for pf in prose_files if pf not in result]
    if skipped:
        logger.info("Router: skipping %d files: %s", len(skipped), skipped)
    return result
