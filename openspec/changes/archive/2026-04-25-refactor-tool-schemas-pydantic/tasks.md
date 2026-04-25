## 1. Pre-change byte fixture capture (lock current schema bytes)

- [x] 1.1 On the current `master` head (no implementation changes yet), write a one-shot script `scripts/_dump_tool_schemas.py` (delete after fixture capture) that builds `create_default_tool_registry()` and dumps each tool's OpenAI schema entry to `tests/fixtures/tool_schemas/<tool_name>.json` using `json.dumps(entry, sort_keys=True, ensure_ascii=False, indent=2)`
- [x] 1.2 Run the script under `uv run python scripts/_dump_tool_schemas.py`; verify exactly 8 fixture files are written (one per default tool, named `list_context.json` … `answer.json`)
- [x] 1.3 Add `tests/test_tool_schemas_byte_fixture.py` that loads each fixture and asserts byte-equality against the current `to_openai_tools()` output (this test must pass on the pre-change baseline before any refactor)
- [x] 1.4 Delete the one-shot dump script; commit the fixtures and the byte-equality test together as the first commit on the change branch

## 2. Pydantic input models (one per tool)

- [x] 2.1 In `src/data_agent_baseline/tools/registry.py`, add the `_path_field(*, file_kind, examples)` Field factory and the `_positive_int_field(*, default, description)` Field factory below the existing `_PATH_CONVENTION_NOTE`
- [x] 2.2 Define `ListContextInput(BaseModel)` with `model_config = ConfigDict(extra="forbid")`, field `max_depth: Annotated[int, _positive_int_field(default=4, description=...)]`, and classmethod `example_value() -> dict[str, Any]` returning `{"max_depth": 4}`
- [x] 2.3 Define `ReadCsvInput` with `path` (via `_path_field`), `max_rows` (via `_positive_int_field`, default=20), and `example_value()`
- [x] 2.4 Define `ReadJsonInput` with `path`, `max_chars` (default=4000), and `example_value()`
- [x] 2.5 Define `ReadDocInput` with `path`, `start_line` (default=1), `line_count` (default=80), `keyword: str | None = None` (optional, with description), and `example_value()`
- [x] 2.6 Define `InspectSqliteSchemaInput` with `path` only, and `example_value()`
- [x] 2.7 Define `ExecuteContextSqlInput` with `path`, `sql: str` (required, description), `limit` (default=200), and `example_value()`
- [x] 2.8 Define `ExecutePythonInput` with `code: str` (required, description includes `EXECUTE_PYTHON_TIMEOUT_SECONDS` interpolation), and `example_value()`
- [x] 2.9 Define `AnswerInput` with `columns: list[str]` (`min_length=1`), `rows: list[list[str | float | bool | None]]`, `example_value()`, and a `@model_validator(mode="after")` that asserts every row's length equals `len(columns)` and raises `ValueError(f"row {i} has {len(row)} cells but columns has {width}")`

## 3. Schema generation + byte-stability normalizer

- [x] 3.1 Implement `_normalize_for_vllm(schema: dict[str, Any]) -> dict[str, Any]` in `registry.py`: recursively (a) remove every `title` key from every dict at every nesting depth, (b) collapse `{"anyOf": [{"type": T}, {"type": "null"}]}` (and the swap-ordered variant) into `{"type": [T, "null"]}` while preserving sibling keys (`default`, `description`, etc.), (c) leave all other structures untouched
- [x] 3.2 Add unit tests in a new file `tests/test_schema_normalizer.py`: cover the title-stripping at top level + nested properties + array items, the anyOf-nullable rewrite (both orderings), preservation of sibling keys (`default`, `description`, `minimum`), and identity for schemas that need no rewrite
- [x] 3.3 Replace the eight `_*_SCHEMA` dict literals in `registry.py` with `_normalize_for_vllm(<Model>.model_json_schema())` lookups; remove the literal definitions
- [x] 3.4 Run `uv run pytest tests/test_tool_schemas_byte_fixture.py` from task 1.3 and confirm byte-equality against the captured fixtures (this proves the normalizer correctness)
- [x] 3.5 Run `uv run pytest tests/test_registry_json_schema.py` and confirm all existing assertions still pass without modification

## 4. Handler simplification + structured ValueError translation

- [x] 4.1 Add `_format_validation_error(tool_name: str, error: pydantic.ValidationError) -> ValueError` private helper in `registry.py` that translates the first error in `error.errors()` into a single-line `ValueError("<tool>: <field_path> <reason>")` per the message format pinned in Decision 4
- [x] 4.2 Add a private mapping `_INPUT_MODELS: dict[str, type[BaseModel]]` keyed by tool name to its input-model class
- [x] 4.3 Modify `ToolRegistry.execute(task, action, action_input)` to: (a) `raise KeyError(f"Unknown tool: {action}")` if `action` not in definitions (preserve current behavior), (b) call `_INPUT_MODELS[action].model_validate(action_input)`, (c) on `ValidationError` raise the translated `ValueError` from 4.1, (d) call the handler with the parsed model attributes (or pass-through `action_input` for handlers that haven't been migrated yet — see 4.4)
- [x] 4.4 Rewrite each `_xxx` handler to accept a typed model instance instead of `dict[str, Any]`: change signature to `(_task: PublicTask, args: <ModelClass>) -> ToolExecutionResult`; remove all manual `int(action_input.get(...))` / `str(action_input[...])` / `cast(...)` calls; access fields via `args.path`, `args.max_rows`, etc.
- [x] 4.5 Update `ToolHandler` type alias in `registry.py` from `Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]` to a generic-aware form (or keep the `dict` alias and have the registry pass `model.model_dump()` to handlers — pick whichever keeps the public `ToolHandler` type alias usable for downstream user-defined tools)
- [x] 4.6 Replace the hand-rolled column/row validation in `_answer` (current `registry.py:162-186`) with the `AnswerInput`-resolved fields; the `model_validator` runs at validation time so the handler can trust `args.columns` / `args.rows` are already shape-correct
- [x] 4.7 Verify in code review that `action_input` (caller's dict) is never mutated by the registry edge — `model_validate` copies values into the model; the original dict identity is untouched

## 5. Test additions (lock the new spec scenarios)

- [x] 5.1 In `tests/test_dispatcher.py`, add tests for the four ADDED requirements scenarios: missing required field (`read_csv` with no path), wrong type (`read_csv` with `path=42`), bound violation (`read_doc` with `start_line=0`), unknown field (`read_csv` with `ttl=5`)
- [x] 5.2 Each test asserts `pytest.raises(ValueError, match=r"^read_csv: ...")` (or `read_doc:`/`answer:`) — the `^` anchor pins that the tool name is the first token
- [x] 5.3 Add tests for `answer` row-width mismatch and empty-columns rejection (mirrors the spec scenarios)
- [x] 5.4 Add a test that calls `registry.execute(task, "read_csv", payload)` with a frozen-snapshot copy of `payload` and asserts `payload == frozen_snapshot` after both successful and failing calls (locks the no-mutation requirement)
- [x] 5.5 Run the full suite: `uv run pytest` — confirm 0 failures, 0 new warnings; existing assertions on schema structure (`additionalProperties: false`, required fields per tool, `to_openai_tools` byte stability) continue to pass

## 6. Quality gates + CHANGELOG

- [x] 6.1 Run `uv run ruff format src/data_agent_baseline/tools/registry.py tests/test_dispatcher.py tests/test_schema_normalizer.py tests/test_tool_schemas_byte_fixture.py` and apply formatting
- [x] 6.2 Run `uv run ruff check src/data_agent_baseline/tools/ tests/`; resolve all warnings without `# noqa` suppressions
- [x] 6.3 Run `uv run pyright src/data_agent_baseline/tools/`; resolve all errors without adding `# type: ignore`
- [x] 6.4 Run `uv run pre-commit run --all-files` if `.pre-commit-config.yaml` is configured; resolve all hook failures
- [x] 6.5 Add a `[Unreleased]` entry to `CHANGELOG.md` under `### Changed`: "Tool input validation now uses Pydantic v2 models; missing-required / wrong-type / unknown-field errors surface as structured `ValueError` messages naming the tool and field."
- [x] 6.6 Confirm `CHANGELOG.md` does not advertise any wire-format change (because there is none)

## 7. OpenSpec validation + final verification

- [x] 7.1 Run `openspec validate refactor-tool-schemas-pydantic --strict` and resolve all reported errors
- [x] 7.2 Run `openspec show refactor-tool-schemas-pydantic` and visually verify the proposal/design/specs/tasks summary
- [x] 7.3 Run end-to-end: `uv run dabench run-task <task_id> --config configs/react_baseline.example.yaml --limit 1` against a small task to confirm a real ReAct loop still functions (smoke-only; no benchmark sweep needed)
- [x] 7.4 Confirm `git diff --stat` shows changes only in `src/data_agent_baseline/tools/registry.py`, `tests/test_*.py`, `tests/fixtures/tool_schemas/*.json`, `CHANGELOG.md`, `openspec/changes/refactor-tool-schemas-pydantic/**` (no other files touched)
