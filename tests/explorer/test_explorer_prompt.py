"""Explorer prompt regression tests."""

from __future__ import annotations

from agents.explorer.prompt import EXPLORER_SYSTEM_PROMPT


def test_explorer_prompt_stops_after_video_rules_and_schema() -> None:
    prompt = EXPLORER_SYSTEM_PROMPT

    assert "=== STOP CONDITIONS ===" in prompt
    assert "explore_video` provides filter rules" in prompt
    assert "one structured source exposes every field needed" in prompt
    assert "MUST call `report` next" in prompt
    assert "Do NOT inspect video example rows" in prompt
    assert "Do NOT grep, paginate, or `read_doc` through JSON/CSV" in prompt
    assert "Alternate-source checking is forbidden" in prompt
    assert "do NOT inspect SQLite, query `sqlite_master`" in prompt
    assert "If SQLite is needed, use `inspect_sqlite_schema`" in prompt
    assert "do NOT query SQLite metadata with `execute_context_sql`" in prompt
    assert "Turn 3+: Call `report` unless a required rule" in prompt
