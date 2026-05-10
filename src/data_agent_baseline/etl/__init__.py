"""Prose ETL: pre-extract structured data from large Markdown files before the agent loop."""

from __future__ import annotations

from data_agent_baseline.etl.extractor import ETLResult, run_etl_for_task

__all__ = ["ETLResult", "run_etl_for_task"]
