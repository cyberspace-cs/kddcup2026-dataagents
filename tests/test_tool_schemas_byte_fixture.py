"""Lock `to_openai_tools()` byte output against pre-change fixtures.

This guards the prompt-cache contract: any drift in JSON Schema bytes (whether
from a refactor of `registry.py`, a Pydantic version bump, or an accidental
description edit) causes a measurable cache miss on rollout.

Fixtures captured on the pre-refactor `master` head via `scripts/_dump_tool_schemas.py`
(deleted after capture). Each fixture is the per-tool entry serialized with
`json.dumps(..., sort_keys=True, ensure_ascii=False, indent=2)` plus a trailing
newline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent_baseline.tools.registry import create_default_tool_registry

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tool_schemas"

EXPECTED_TOOLS = {
    "answer",
    "execute_context_sql",
    "execute_python",
    "inspect_files",
    "inspect_sqlite_schema",
    "list_context",
    "read_csv",
    "read_doc",
    "read_json",
}


def test_fixture_directory_contains_one_file_per_default_tool() -> None:
    fixtures = {path.stem for path in FIXTURE_DIR.glob("*.json")}
    assert fixtures == EXPECTED_TOOLS


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOLS))
def test_to_openai_tools_entry_matches_fixture_bytes(tool_name: str) -> None:
    registry = create_default_tool_registry()
    entries = {entry["function"]["name"]: entry for entry in registry.to_openai_tools()}
    assert tool_name in entries, f"{tool_name} not surfaced by to_openai_tools()"
    serialized = json.dumps(entries[tool_name], sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    expected = (FIXTURE_DIR / f"{tool_name}.json").read_text(encoding="utf-8")
    assert serialized == expected, (
        f"{tool_name} schema bytes drifted from fixture; refresh fixture only after "
        "an intentional wire-format change."
    )
