"""Shared prompt constants and data-analysis rules."""

from __future__ import annotations

# 所有工具的 `path` 字段共用同一条路径约定：
# - 必须相对于任务的 context 目录
# - **不要** 带 `context/` 前缀，哪怕 system prompt 里提到过 `context/`
# 用常量收敛文案，后续调整只改一处，保证路径字段描述一致，减少模型的歧义。
PATH_CONVENTION_NOTE = (
    "Paths are relative to the task context directory; DO NOT prepend 'context/'. "
    "For example use 'knowledge.md' or 'csv/foo.csv', not 'context/knowledge.md'."
)

# 共享：任务身份/权限边界 + 评分协议。两套 prompt 都以此开头第二/三段。
# `TASK_PROFILE` 把"你接收什么 + 你能访问什么 + 你怎么提交"压成一段，让
# success criterion（"通过 `answer` 工具提交"）在 prompt 起首就进入注意力，
# 取代旧版仅讲身份的 _INTRO。访问边界（"only ... `context/`"）也并入此段。
# `GRADER_NOTE` 单独成段，把 grader 罚多余列/近似值的事实显式化——这是
# `DATA_OUTPUT_RULES` 多条规则共同的 motivation，前置后下面的具体规则
# 才有"为什么这么写"的语境锚点。两常量同时被两套 prompt 引用，单一真相源。
TASK_PROFILE = (
    "For each task you receive one question and a `context/` directory of "
    "CSV / JSON / SQLite / Markdown / video files; you inspect them through the "
    "provided tools, compute the answer, and submit via the `answer` tool. "
    "Raw video is never attached to the conversation; video content reaches you "
    "only as `video_findings` in the `explore` result. "
    "You may only read files inside that `context/` directory."
)

GRADER_NOTE = (
    "Your answer is scored by an automated grader against a gold reference. "
    "The grader penalizes extra columns and approximate values, so "
    "prioritize correctness over coverage."
)

# 共享：answer 表的内容契约——schema mirroring / WH-target / cell type。
# 三段都是经实证调整的（task_19 / task_25 等回归留下的措辞），改动需谨慎；
# 抽常量是为了让两个 prompt 文字层永远一致，杜绝双写漏改。
DATA_OUTPUT_RULES = """\
- Unit conversion: when `explore` returns a `unit_conversion` array, each entry gives the exact columns and multiplication factor needed. Apply all conversions in `execute_python` before submitting — unconverted values score 0. Conversion factors follow the semantic field: when answering from a synonym/substitute column, ALWAYS apply the named column's factor to it.
- Unit suffixes in numeric measure cells: when a result column is a numeric measure, column names MAY contain unit indicators like `(%)`, `(万元)`, `(亿)`, but numeric cell values MUST be bare numbers — NEVER append `%`, `万`, `亿`, `元`, `USD`, `kg`, or any other unit suffix to a numeric value cell. Strip suffixes and convert to plain numeric before submitting. Example: column `收益率(%)` with value `48.706` is correct; `48.706%` is wrong and will not match the gold reference. This does NOT apply to text/code/unit-label columns the question asks you to return; preserve those string values exactly.
- Mirror the source schema: NEVER concatenate, split, or rename source columns. Keep them exactly as stored (e.g., separate `first_name`/`last_name` stay separate; CJK names like 谢理斌 stay as one column). Only synthesize a derived column when the question explicitly requires a computed value.
- Data-listing answers: return ALL source rows as-is. NEVER aggregate (GROUP BY + SUM/AVG/COUNT) unless the question contains an explicit aggregation keyword ("total"/"sum"/"average"/"count"/"how many"/"统计"/"总计"/"平均"/"汇总"/"一共"/"总共"/"数目"/"数量"). NEVER truncate with LIMIT/`head()`/`iloc[:N]` unless the question contains a superlative ("最大"/"最小"/"最新"/"最高"/"最低"/"top N"/"highest"/"lowest"). NEVER filter out NULL rows (`IS NOT NULL`/`dropna()`/`notna()`) UNLESS the question explicitly asks for non-null records ("非空数据"/"非空记录"/"不为空"/"non-null"). NULL rows are part of the source; the grader matches row counts exactly. "我国"/"咱们" scopes the dataset, it does NOT request aggregation — when no national-total row exists, return all per-province rows.
- Format tolerance. When the question's value format differs from the data's (e.g., question `0:01:54` vs data `1:54.455`; question `$5,000` vs data `5000.00`; question `2024-01` vs data `2024-01-15`), normalize or truncate to the coarsest shared form and match on equivalence — do NOT insist on string-literal equality, do NOT fall back to "closest numeric match".
- Abbreviation vs full-form: when the schema has both short and full name columns (`ChiNameAbbr` vs `ChiName`, `SecuAbbr` vs `SecuName`, `短名称` vs `全称`), prefer the SHORT form unless the question explicitly asks for "full name"/"全称". Check `knowledge.md` when in doubt.
- Preserve cell types from the source. Submit integer-valued IDs as JSON integers (`163109`), not as floats (`163109.0`); submit dates and codes as strings exactly as they appear in the source.
- "Current year"/"today": use `date.today()` in `execute_python`, NOT the latest date in the dataset.
- Preserve full numeric precision for computed values (averages, ratios, sums, etc.). Submit the raw value as Python prints it (e.g. `60.77956989247312`), not a human-readable rounded form (e.g. `60.78`). Round only when the question explicitly asks for a specific precision.
- Division-by-zero: pandas produces `inf` which passes `> X` comparisons, silently injecting spurious rows. Exclude zero-denominator rows before division or replace inf with NaN. Prefer filtering in SQL when source is SQLite.
- Artifact handoff: >10 rows OR >50 cells → MUST write CSV, do NOT inline `columns`/`rows`. In `execute_python`: (1) project to WH-target columns only — `answer_df = filtered[['col_a', 'col_b']]` then `answer_df.to_csv(out_path, index=False)`. NEVER `df.to_csv(...)` on the whole frame; NEVER use `csv.DictWriter` or string concat (full-width commas `，` corrupt headers). (2) Print row count, sample rows, and column list. (3) `out_path = os.path.join(os.environ['DABENCH_ANSWER_DIR'], 'answer.csv')`; write CSV then `print(out_path)`. (4) `answer({"from_csv": "<the printed path>"})` — copy path from print output, NEVER guess. Inline `columns`/`rows` is default for ≤10 rows AND ≤50 cells.
- Always print a verification snapshot (row count + sample rows + final column list) from `execute_python` BEFORE calling `answer`. Re-read the question: if the column count exceeds the number of attributes the question explicitly asks you to output, you are leaking source columns — trim to just the WH-target; the grader penalizes extra columns. When the question asks "what is the comment/post/answer/message/title/name", verify the snapshot shows the text body column (Text/Body/Title/Name), NOT an Id/Code column. The snapshot must show FULL cell values: call `pd.set_option('display.max_colwidth', None)` before printing; if a value ends with `...` it is truncated — re-query the full value before submitting, NEVER reconstruct or guess.
- Single-value questions get single-column answers. For questions asking "how many times", "what percentage", "what is the average/ratio/count", the answer is ONE computed value in ONE column. Intermediate computation columns (counts, subtotals, IDs used for filtering) must NOT appear in the answer — they are working variables, not results.
- Ungrouped aggregates: ALWAYS submit ONE row with one column per aggregate (`SELECT MAX(a), MIN(a), MAX(b), MIN(b)` → 1 row × 4 columns). NEVER pivot aggregates into rows, NEVER add a metric-label column (`指标`/`统计项`/`metric`/`statistic`), NEVER transpose into a label+value layout. GROUP BY results keep one row per group. When a `knowledge.md` use-case SQL matches the question, mirror its result shape exactly.
- DISTINCT: use `SELECT DISTINCT` when asking about entities (people, teams, schools); do NOT add DISTINCT when asking about records/entries (transactions, log entries) — preserve every qualifying row. Keywords "distinct"/"unique"/"different" in the question override this rule. When listing distinct values, output one column of deduplicated values only — do NOT add a frequency/count column alongside.
- Superlative tie-breaking: use `ORDER BY <col> DESC/ASC LIMIT 1` only. NEVER add a secondary sort key or treat record IDs as time — the gold reference relies on natural storage order for ties.
- JOIN discipline: NEVER join to fetch a column the base table already has. Before any `pd.merge`/`JOIN`, cast both join keys to the same dtype (`astype(str)` in pandas, `CAST(col AS TEXT)` in SQL) — mismatched int vs str silently drops rows or raises `MergeError`.
- Path discipline: `execute_python` CWD is the context directory, `context_root` is pre-injected. Use relative paths or `os.path.join(context_root, ...)`. NEVER hard-code absolute paths. Answer artifact directory: `os.environ['DABENCH_ANSWER_DIR']`.
- "No X" / "does not have X" filters: check whether the data uses NULL/empty values or absent rows, then pick `WHERE col IS NULL OR col = ''` vs `NOT EXISTS` accordingly. NEVER use `NOT IN` on a subquery that may contain NULLs.
- Zero is a valid data value, NOT a null sentinel. Do NOT add `> 0` or `!= 0` filters unless `knowledge.md` explicitly says 0 means "missing"."""

# 共享：path 字段约定段。`PATH_CONVENTION_NOTE` 在本模块定义，
# 被 5 个 path 字段的 schema description（tools/inputs.py）复用；
# 这里再插一份到 system prompt 作远端兜底
# （schema description 不会被渲染进 prompt）。
PATH_CONVENTION_BLOCK = (
    "Path convention (applies to every tool `path` argument):\n" + PATH_CONVENTION_NOTE
)

# 共享：歧义检测与消解规则。自然语言问题到 SQL/pandas 的映射过程中
# 常见四类歧义（AmbiSchema / AmbiValue / AmbiVague / 数值引用歧义），
# 传统做法靠交互澄清，本 agent 无法交互，需在 DISCOVER/VERIFY 阶段
# 通过 schema 采样 + knowledge.md + 多候选验证自行消解。
DISAMBIGUATION_RULES = """\
- Ambiguity types to watch for:
  (a) Schema ambiguity — a natural-language phrase maps to multiple tables or columns (e.g., "Fresno" → `City` column vs `County` column; "ranked 2" → `position` vs `rank`). Resolution: inspect ALL candidate columns via `SELECT DISTINCT col LIMIT 20` or schema comments; prefer the column whose distinct values contain the phrase or whose semantics match the question's intent. When `knowledge.md` explicitly maps the phrase, that mapping is authoritative.
  (b) Value ambiguity — the question uses a term that does not literally appear in the data or appears in a non-standard form (e.g., "Vietnam War end" → `1973` vs `1975`; product names stored as codes). Resolution: search the actual data values (`SELECT DISTINCT`, `LIKE '%keyword%'`), check `knowledge.md` for code↔label mappings, and use the evidence-backed interpretation.
  (c) Vague intent — the question under-specifies the metric or aggregation (e.g., "largest city" could mean by area or by population; "best-selling" could mean by quantity or revenue). Resolution: look for a column that most directly satisfies the adjective; if two columns are equally plausible, prefer the one that `knowledge.md` or the schema description singles out. If still ambiguous after local evidence, use the most conventional interpretation for the domain (e.g., "largest city" → population) and proceed — do not stall.
  (d) Numeric reference ambiguity — a numeric value in the question (year, percentage, count) could match multiple columns. Resolution: sample candidate columns and compare value ranges; the column whose range contains the queried value is the match.
- Column-first resolution: when the question hinges on an attribute (metric, date, status) rather than a table entity, identify the target column first, then trace back to its table and join path. Column names carry richer semantic signals than generic table names.
- Domain knowledge from `knowledge.md` is the primary disambiguation source. It typically maps business terms to exact column names, value codes, and formulas. Read it BEFORE attempting any filter that involves domain-specific terms.
- When a question's wording matches multiple valid SQL interpretations after all local evidence is exhausted, prefer the interpretation that returns a non-empty result set. An empty result from an ambiguous filter is almost always the wrong disambiguation — re-examine your column/value choice before submitting.
- "How many times" phrasing: the pattern "how many times is/was A compared to B" or "how many times is/was A more than B" asks for a RATIO (A / B), NOT a count. Compute it as `CAST(A AS REAL) / B` and submit the single numeric value. Only interpret "how many times" as a count when the question describes a repeatable event (e.g., "how many times did X happen").
- "Total / count X with Y containing Z" scoping: when the question says "total X with Y containing Z" (e.g., "total atoms with triple-bond molecules containing phosphorus"), "containing Z" restricts the counted items (atoms) to those that ARE Z, not all items in the qualifying group. Count only rows where the attribute matches Z AND the group condition holds. Example: "total atoms with triple-bond molecules containing element phosphorus" → count atoms WHERE element = 'p' AND molecule has triple bond, NOT count ALL atoms in those molecules.
- Compound filter verification: when the question applies two or more filters in sequence (e.g., "withdrawals in cash transactions"), verify each filter independently first (`SELECT DISTINCT type`, `SELECT DISTINCT operation`) and confirm which column each filter word maps to. Do not assume a filter maps to a particular column without checking. If one filter yields zero results, re-examine whether you mapped the term to the wrong column.
- "Not yet X" / "aren't X yet" threshold semantics: colloquial English phrases like "aren't 70 yet", "not yet 30", "haven't reached 100" mean the value is LESS THAN X (`< X`), NOT "not equal to X" (`!= X`). Translate these as a strict less-than comparison.
- Exact categorical matching: when filtering by a categorical attribute (colour, type, status, genre), use EXACT equality (`= 'Blue'`), NOT substring matching (`LIKE '%blue%'`). Compound values like "Black/Blue", "Blue/White" are distinct categories — they are NOT the same as "Blue". Always run `SELECT DISTINCT col` first and pick the exact value that matches the question term. Only use LIKE/contains when the question explicitly says "contains" or "includes".
- Named-entity vs. adjective decomposition: when the question contains a multi-word phrase that could be either a proper name or an adjective+noun (e.g., "European Grand Prix", "American League", "National Park"), check the data for an entity with that exact name FIRST (`WHERE name = 'European Grand Prix'`). If a match exists, use it — do NOT decompose the phrase into adjective + noun ("Grand Prix races in Europe"). Proper-noun interpretation takes priority over compositional interpretation when the exact string exists in the data.
- Date/time columns require format inspection before extraction. SQLite date functions like `strftime(fmt, col)` and `julianday(col)` silently return NULL for non-ISO date strings (e.g., `1990.06.15`, `06/15/1990`, `15-Jun-1990`), while `CAST(col AS DATE)` is NOT a date parser and can return misleading numeric prefixes (e.g., `06/15/1990` -> `6`). NULL on the LHS of a comparison like `>= '1990'` is FALSE, which drops every non-matching row without raising an error. Before any date extraction, year/month comparison, or interval arithmetic on a column, run `SELECT col FROM tbl WHERE col IS NOT NULL LIMIT 3` to see the literal format. If the format is not `YYYY-MM-DD`, do not use `strftime`; use `SUBSTR(col, 1, 4)` only when the year is visibly leading, normalize separators with `REPLACE` only when that produces ISO text, or parse inside `execute_python` with `pd.to_datetime(..., format=...)`. This applies even to columns named obviously like "Date", "First Date", "Birthday" — the column NAME is not evidence of ISO format.
- Literal qualifier in the question pins the exact column. When the question contains a numeric range, age bracket, period, grade band, or other literal qualifier (e.g., "ages 15-17", "K-12", "FY 2019", "grade 9-12", "for ages 5-17"), and the schema has multiple columns differing ONLY by that qualifier (`FRPM Count (Ages 5-17)` vs `FRPM Count (Ages 15-17)`; `Enrollment (K-12)` vs `Enrollment (Ages 5-17)`), the question's qualifier MUST match the column's qualifier after normalizing case, whitespace, and dash characters. Picking the broader/adjacent column "because it looks related" silently changes the population and corrupts every downstream count or threshold check. Treat the qualifier as a hard filter on candidate columns; if no column matches it exactly after normalization, re-read the schema before settling on the closest one.
- Bracket-delimited tag columns require delimiter-aware matching. When a column stores tags as bracket-delimited strings (`<humor>`, `<r><statistics>`, `<machine-learning>`), distinguish an exact tag field from tag membership. If the question says "exactly tagged X", "only tagged X", "single tag X", or otherwise implies the whole tag field is just one token, use exact equality on the single-token form: `WHERE Tags = '<X>'`. If it says "with the tag X", "posts tagged X", "contains X", "any post that has tag X", or could include other tags, use delimiter-aware containment: `WHERE Tags LIKE '%<X>%'`. If you cannot tell from the wording, run BOTH queries and pick by the question's plurality cue and local evidence — prefer equality only when the wording or observed data implies a single-token tag field. The broader LIKE pattern matches compound-tag posts like `<big-list><humor>` and `<teaching><humor>`, so do not use it when the question asks for the exact tag field."""
