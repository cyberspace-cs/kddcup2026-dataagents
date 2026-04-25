# tool-registry Specification

## Purpose
TBD - created by archiving change refactor-architecture-and-prune. Update Purpose after archive.
## Requirements
### Requirement: Single ToolDefinition replaces ToolSpec + handler split

The tool registry SHALL expose tools as `ToolDefinition` instances with the fields: `name: str`, `description: str`, `input_schema: dict[str, Any]`, `json_schema: dict[str, Any]`, `handler: Callable`, `is_terminal: bool`. The legacy `ToolSpec` (without handler) and the parallel `handlers` mapping MUST be unified into this single shape.

#### Scenario: Each registered tool has a complete ToolDefinition
- **WHEN** `create_default_tool_registry(...)` is called and the returned registry is iterated
- **THEN** every entry is a `ToolDefinition` with all six fields populated and `handler` is callable

### Requirement: Default tool inventory is exactly 8 tools

The default registry built by `create_default_tool_registry` SHALL contain exactly these tool names, in registration order: `list_context`, `read_csv`, `read_doc`, `read_json`, `inspect_sqlite_schema`, `execute_context_sql`, `execute_python`, `answer`.

#### Scenario: Default registry tool inventory
- **WHEN** `create_default_tool_registry(task=task, ...)` is called for any valid task
- **THEN** `list(registry.definitions.keys()) == ["list_context", "read_csv", "read_doc", "read_json", "inspect_sqlite_schema", "execute_context_sql", "execute_python", "answer"]`

### Requirement: Registry is immutable post-construction

After construction, the registry SHALL refuse mutation. Adding, removing, or replacing a `ToolDefinition` MUST raise `TypeError` (or equivalent) so that downstream caches (e.g. `NativeToolsOpenAIAdapter`'s schema cache) cannot become stale.

#### Scenario: Attempting to assign a new tool raises
- **WHEN** code tries `registry.definitions["new_tool"] = some_definition` after construction
- **THEN** the assignment raises `TypeError` and the registry contents are unchanged

#### Scenario: Attempting to overwrite a tool raises
- **WHEN** code tries to reassign an existing tool after construction
- **THEN** the operation raises and the existing definition is preserved

### Requirement: OpenAI tool schema output is byte-stable and sorted by name

`registry.to_openai_tools()` SHALL return a list of OpenAI-format tool dicts sorted by `name` ascending. Repeated calls MUST return byte-identical output.

#### Scenario: to_openai_tools() is sorted by name
- **WHEN** `to_openai_tools()` is invoked
- **THEN** the returned list's `[t["function"]["name"] for t in result]` equals `sorted([t["function"]["name"] for t in result])`

#### Scenario: to_openai_tools() is invariant across calls
- **WHEN** `to_openai_tools()` is invoked twice on the same registry
- **THEN** both invocations return objects whose JSON serialization (`json.dumps(..., sort_keys=True)`) is byte-identical

### Requirement: Exactly one terminal tool, named `answer`

Within the default registry, exactly one `ToolDefinition` SHALL carry `is_terminal=True`, and its `name` MUST be `answer`.

#### Scenario: Only answer is terminal
- **WHEN** the default registry is built
- **THEN** `[d.name for d in registry.definitions.values() if d.is_terminal] == ["answer"]`

### Requirement: Filesystem and SQLite tools enforce task-scoped sandboxing

Tools that read files (`read_csv`, `read_doc`, `read_json`) and access SQLite (`inspect_sqlite_schema`, `execute_context_sql`) SHALL reject any path that resolves outside the task's `context_dir`. Resolution MUST follow symlinks and reject `..` traversal.

#### Scenario: Path traversal is rejected
- **WHEN** a tool is invoked with `path="../../../etc/passwd"` (or any path resolving outside `context_dir`)
- **THEN** the handler raises with a message referencing the sandbox boundary, and no file outside the sandbox is read

#### Scenario: Absolute path outside context is rejected
- **WHEN** a tool is invoked with an absolute path not contained in `context_dir`
- **THEN** the handler raises and no file outside the context is read

### Requirement: `execute_python` and `execute_context_sql` produce non-terminal results regardless of inner success

Tools other than `answer` SHALL NOT report `is_terminal=True` even when they succeed. A failed inner script (e.g. Python with `ok=False`) MUST be a non-terminal `tool_error` from the loop's perspective.

#### Scenario: Failed Python execution is non-terminal
- **WHEN** `execute_python` runs code that raises an exception
- **THEN** the resulting `ToolExecutionResult` has `is_terminal=False`, `ok=False`, and the dispatcher records a `tool_error` event without halting the loop

### Requirement: Tool input validation produces a structured `ValueError` naming the tool and field

`ToolRegistry.execute(task, action, action_input)` SHALL validate `action_input` against the input model declared for `action` before invoking the handler. Validation failures (missing required field, wrong type, value outside declared bounds, unknown field) MUST be raised as a single `ValueError` whose first line is `"<tool_name>: <field_path> <reason>"`. Raw Pydantic `ValidationError` objects MUST NOT propagate from `ToolRegistry.execute`.

#### Scenario: Missing required field is reported by name
- **WHEN** `registry.execute(task, "read_csv", {})` is called (the required `path` field is absent)
- **THEN** a `ValueError` is raised whose first line begins with `"read_csv: "`, names the field `'path'`, and indicates the field is required

#### Scenario: Wrong type for a required field is reported with the offending field name
- **WHEN** `registry.execute(task, "read_csv", {"path": 42})` is called (path is an int, not a string)
- **THEN** a `ValueError` is raised whose first line names the tool `read_csv`, names the field `'path'`, and identifies the type mismatch

#### Scenario: Value violates declared minimum bound
- **WHEN** `registry.execute(task, "read_doc", {"path": "knowledge.md", "start_line": 0})` is called (start_line is below the declared minimum of 1)
- **THEN** a `ValueError` is raised whose first line names the tool `read_doc`, names the field `'start_line'`, and indicates a minimum-bound violation

### Requirement: Unknown fields in `action_input` are rejected

Each tool's input model SHALL be configured with `extra="forbid"` so that the generated JSON schema declares `additionalProperties: false` and the runtime validator rejects keys not declared on the model. Rejected unknown fields MUST surface via the same structured `ValueError` shape defined above, naming the unknown field.

#### Scenario: Unknown field on a tool input is rejected
- **WHEN** `registry.execute(task, "read_csv", {"path": "csv/foo.csv", "max_rows": 20, "ttl": 5})` is called (`ttl` is not declared)
- **THEN** a `ValueError` is raised whose first line names the tool `read_csv` and names the unknown field `'ttl'`

#### Scenario: `additionalProperties: false` is preserved in the generated JSON schema
- **WHEN** `registry.to_openai_tools()` is invoked
- **THEN** every returned `function.parameters` object has `additionalProperties: false`

### Requirement: `answer` tool input enforces row width matching column count

The `answer` tool's input model SHALL include a cross-field validator that requires every row in `rows` to be a list whose length equals `len(columns)`. Mismatched row width MUST raise a `ValueError` naming the offending row index and the observed vs. expected widths.

#### Scenario: Row width mismatch is reported with row index
- **WHEN** `registry.execute(task, "answer", {"columns": ["a", "b"], "rows": [["x"]]})` is called
- **THEN** a `ValueError` is raised whose first line names the tool `answer`, identifies row index `0`, and reports the observed width (`1`) vs. expected (`2`)

#### Scenario: Empty columns list is rejected
- **WHEN** `registry.execute(task, "answer", {"columns": [], "rows": []})` is called
- **THEN** a `ValueError` is raised whose first line names the tool `answer` and the field `'columns'` and indicates the list must be non-empty

### Requirement: Pre-validation `action_input` mutation is forbidden

The validation step in `ToolRegistry.execute` MUST NOT modify the caller-provided `action_input` dict. Validation SHALL produce a parsed model instance from a copy or via attribute access; the original `action_input` MUST be byte-identical (same keys, same values, same `id()`) before and after the call regardless of whether validation succeeded or failed.

#### Scenario: Successful validation does not mutate caller's dict
- **WHEN** a caller passes a dict `payload = {"path": "csv/foo.csv", "max_rows": 20}` to `registry.execute(task, "read_csv", payload)` and the call succeeds
- **THEN** `payload` after the call equals `{"path": "csv/foo.csv", "max_rows": 20}` and contains no additional keys (e.g. no Pydantic-injected aliases)

#### Scenario: Failed validation does not mutate caller's dict
- **WHEN** a caller passes a dict `payload = {}` to `registry.execute(task, "read_csv", payload)` and the call raises
- **THEN** `payload` after the call equals `{}`

