"""Extraction prompt templates for prose-to-CSV conversion."""

from __future__ import annotations

EXTRACTION_PROMPT = """\
You are a data extraction engine. Your ONLY job is to read the document below \
and output a single CSV table.

Rules:
- Output ONLY valid CSV: no markdown fences, no explanation, no preamble.
- The first line is the header row, using EXACTLY the column names listed below.
- Extract EVERY entity mentioned in the document — do NOT skip any. Each entity gets one row.
- If the same entity appears in multiple paragraphs/sections, merge all attributes into one row.
- Use corrected values only: if the text says "initially X, corrected to Y", use Y.
- Preserve exact numeric values and string values as they appear in the data.
- For missing/redacted/None values, leave the cell empty (two consecutive commas).
- If a value contains a comma, wrap it in double quotes.
- If a value contains double quotes, escape them by doubling ("").

Document:
{document}

Question context:
{question}

Output the CSV now:"""

EXTRACTION_PROMPT_WITH_SCHEMA = """\
You are a data extraction engine. Your ONLY job is to read the document below \
and output a single CSV table with the EXACT columns specified.

Target columns (use these as the CSV header, in this exact order):
{schema}

Rules:
- Output ONLY valid CSV: no markdown fences, no explanation, no preamble.
- The first line MUST be the header row using exactly the column names above.
- Extract EVERY entity mentioned in the document — do NOT skip any. Each entity gets one row.
- If the same entity appears in multiple paragraphs/sections, merge all attributes into one row.
- Use corrected values only: if the text says "initially X, corrected to Y", use Y.
- Preserve exact numeric values and string values as they appear in the data.
- For missing/redacted/None values, leave the cell empty (two consecutive commas).
- If a value contains a comma, wrap it in double quotes.
- If a value contains double quotes, escape them by doubling ("").

Document:
{document}

Question context:
{question}

Output the CSV now:"""

CONTINUATION_PROMPT = """\
You are continuing a data extraction. The header row is:
{header}

Rules:
- Output ONLY data rows (NO header row) in valid CSV format.
- No markdown fences, no explanation, no preamble.
- Follow the exact column order from the header above.
- Extract EVERY entity in this document chunk — do NOT skip any. Each entity gets one row.
- If the same entity appears in multiple paragraphs, merge all attributes into one row.
- Use corrected values only.
- Preserve exact numeric values and string values as they appear.
- For missing/redacted/None values, leave the cell empty.
- If a value contains a comma, wrap it in double quotes.

Document (continuation):
{document}

Question context:
{question}

Output CSV data rows now:"""
