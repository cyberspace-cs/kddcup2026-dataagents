## Why

`src/data_agent_baseline/tools/registry.py` currently maintains the same input contract in three hand-written places per tool: a non-strict `input_schema` demo dict for the text-protocol prompt, an 8-section strict JSON Schema literal (`_LIST_CONTEXT_SCHEMA`, `_READ_DOC_SCHEMA`, …) for native function calling, and per-handler manual coercion (`int(action_input.get("max_rows", 20))`, `cast(list[str], columns)`, etc.). The redundancy is roughly 180 lines of schema literals plus per-handler coercion boilerplate; the three views can drift in subtle ways (a new field added to the prompt demo but missed in the JSON Schema, a default value updated in the schema but not in the handler), and missing required keys surface to the model as the unhelpful `KeyError: 'path'` rather than a typed validation message.

Pydantic v2 is already a project dependency (`config_models.py`), pyright strict is enforced, and vLLM 0.19.1's xgrammar backend tolerates both the current `type: ["string","null"]` form and Pydantic's default `anyOf` nullable form — so a single Pydantic input model per tool can become the one source of truth for validation, JSON schema generation, and the prompt-side demo example, removing the drift surface and producing actionable error messages without changing observable wire behavior.

## What Changes

**Tool input modeling (one source of truth)**
- Introduce one Pydantic v2 `BaseModel` per default tool (8 models): `ListContextInput`, `ReadCsvInput`, `ReadJsonInput`, `ReadDocInput`, `InspectSqliteSchemaInput`, `ExecuteContextSqlInput`, `ExecutePythonInput`, `AnswerInput`. Each declares `model_config = ConfigDict(extra="forbid")` so the generated schema retains `additionalProperties: false`.
- Each model carries field-level `description` text (currently duplicated in the JSON Schema property dicts) and an `example_value()` classmethod returning the demo dict currently held in `ToolDefinition.input_schema`.
- `_PATH_CONVENTION_NOTE` becomes a single `Annotated[str, Field(description=...)]` factory shared by all path fields; integer-bound parameters share a `PositiveIntField(default, description)` factory.

**Schema generation + byte-stability normalizer**
- Replace each `_*_SCHEMA` dict literal with `Model.model_json_schema()` filtered through a small in-module `_normalize_for_vllm(schema)` helper that: (a) strips Pydantic's `title` keys at every nesting level (cosmetic; xgrammar tolerates them, but removal preserves current request bytes), (b) rewrites the `anyOf: [{type: "string"}, {type: "null"}]` shape Pydantic emits for `Optional[str]` back to the `type: ["string", "null"]` shape currently in use, (c) preserves field declaration order so `to_openai_tools()` output stays byte-stable across calls.
- Net effect: `to_openai_tools()` returns dicts that are byte-equivalent to today's output (zero prompt-cache invalidation on rollout). Equivalence is locked by a new fixture-based test that captures the pre-change schema bytes per tool and compares post-change.

**Handler simplification**
- Each `_xxx` handler replaces its manual `int(...)` / `str(...)` / `action_input.get(default)` chain with `args = XxxInput.model_validate(action_input)` and accesses typed attributes directly. The `_answer` handler's hand-rolled column/row validation (`registry.py:162-186`) collapses into `AnswerInput`'s field validators (non-empty string columns; each row is a list whose length matches `len(columns)`; cells are `str | float | bool | None`).
- Pydantic `ValidationError` is caught at the registry edge and re-raised as a single `ValueError` whose message names the offending tool and field (e.g., `"read_csv: field 'path' is required"` instead of `KeyError: 'path'`). The driver's `tool_error` recording path is unchanged — only the message text the model sees improves.

**Capability impacted**
- `tool-registry` only. Spec-level deltas are limited to the input-validation contract: error messages name the tool and field; extra unknown fields are rejected with a structured message. Existing requirements (8-tool inventory, immutable post-construction, byte-stable `to_openai_tools()`, sandbox path enforcement, only-`answer`-is-terminal) remain in force unchanged.

**Out of scope**
- No changes to `ToolRegistry`, `ToolDefinition`, `ToolExecutionResult` shapes or public exports.
- No changes to the underlying file/sql/python implementation modules (`filesystem.py`, `sqlite.py`, `python_exec.py`).
- No changes to `agents/dispatcher.py` or the ReAct loop's terminal-detection contract (D3).
- No new tools added; the 8-tool inventory and registration order remain locked per the existing spec.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `tool-registry`: Tool inputs are validated against per-tool Pydantic v2 models. Missing-required-field and unknown-field errors raise structured `ValueError` whose message names the tool and field. JSON Schema output remains byte-equivalent to the pre-change form via an in-module normalizer.

## Impact

**Affected source code**
- `src/data_agent_baseline/tools/registry.py` — add 8 Pydantic input models, the `_normalize_for_vllm` helper, and the `path`/`positive-int` field factories; replace 8 `_*_SCHEMA` dict literals with `Model.model_json_schema()` calls; rewrite 7 `_xxx` handlers and `_answer` to validate via Pydantic; raise structured `ValueError` from `ToolRegistry.execute` on `ValidationError`.

**Affected tests**
- `tests/test_registry_json_schema.py` — keep all existing assertions; add a fixture-locked byte-equivalence assertion (`schema_bytes_pre.json` per tool) so future drift in the normalizer is caught.
- `tests/test_registry_immutable.py` — unchanged.
- `tests/test_dispatcher.py` / `tests/test_react_native.py` / `tests/test_react_text_fallback.py` — add cases that pass malformed `action_input` and assert the new error message shape (`"<tool>: field '<field>' …"`).
- New: `tests/fixtures/tool_schemas/<tool>.json` — one frozen JSON file per tool, captured from the pre-change `to_openai_tools()` output.

**Affected configuration & build**
- `pyproject.toml` — no dependency change (`pydantic>=2.10` already present).
- `configs/react_baseline.example.yaml` — no change.

**Wire-format / external contracts (UNCHANGED)**
- `to_openai_tools()` byte output (locked by fixture).
- `ToolDefinition` field shape (`name`, `description`, `input_schema`, `json_schema`, `handler`, `is_terminal`).
- `ToolRegistry.definitions` ordering and immutability.
- `ToolExecutionResult` shape and `is_terminal` semantics.
- `dabench` CLI, `trace.json`, `summary.json`, `prediction.csv`.

**Behavioral changes (intentional, spec-relevant)**
- Tool errors caused by missing required fields, unknown fields, or type-mismatched fields surface as `ValueError("<tool>: …")` instead of `KeyError`/`TypeError`/silent `int()`-cast `ValueError` from the handler. Message contents are stable and tested.

**Dependencies**
- None added. None removed.

**Breaking changes**
- None at the public-API level. Models in third-party code calling tools through the registry will see a different error class only if they introspect the previously-raised `KeyError`/`TypeError` (which the dispatcher swallows internally — no production caller does this).
