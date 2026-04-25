## ADDED Requirements

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
