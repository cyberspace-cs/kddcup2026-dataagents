"""Code-generation fallback: prompt building, LLM call, subprocess execution."""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_agent_baseline.agents.model import ModelAdapter

CODE_EXEC_TIMEOUT = 60
MAX_RETRIES = 2


def extract_code_block(text: str) -> str:
    """Extract Python code from a markdown fenced block, or return raw text."""
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def build_codegen_prompt(
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


def build_retry_prompt(code: str, error: str) -> str:
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


def call_llm(adapter: ModelAdapter, prompt: str) -> str:
    """Single LLM call."""
    from data_agent_baseline.agents.model import ModelMessage, ModelResponse

    messages = [
        ModelMessage(
            role="system",
            content="You are a Python code generator for data extraction. Output only code.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    response: ModelResponse = adapter.complete(messages)
    return response.content


def exec_code(code: str) -> tuple[bool, str]:
    """Execute Python code in a subprocess, return (success, stdout_or_stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=CODE_EXEC_TIMEOUT,
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {CODE_EXEC_TIMEOUT}s"
