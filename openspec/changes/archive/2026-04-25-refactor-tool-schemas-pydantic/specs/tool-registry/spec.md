## ADDED Requirements

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
