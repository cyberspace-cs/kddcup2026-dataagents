from __future__ import annotations

from agents.explorer.prompt import EXPLORER_SYSTEM_PROMPT


def test_explorer_prompt_requires_knowledge_md_mapping_evidence() -> None:
    prompt = EXPLORER_SYSTEM_PROMPT
    lower = prompt.lower()

    assert "if `inspect_files` lists `knowledge.md`, you must read it before `report`" in lower
    assert "a knowledge.md mapping/disambiguation/example" in lower
    assert "column mapping" in lower
    assert "disambiguation rule" in lower
    assert "example/use-case sql" in lower
    assert "knowledge_md_evidence" in prompt
    assert "trading volume" in prompt
    assert "turnoverdeals" in prompt
