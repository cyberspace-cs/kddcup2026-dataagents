## Context

The tool layer in `src/data_agent_baseline/tools/registry.py` currently uses three parallel representations of every tool's input contract:

1. `ToolDefinition.input_schema` — a non-strict demo dict shown to the model in text-protocol prompts (e.g. `{"path": "relative/path/to/file.csv", "max_rows": 20}`).
2. `_LIST_CONTEXT_SCHEMA` … `_ANSWER_SCHEMA` — strict JSON Schema literals fed into native function-calling requests via `to_openai_tools()`. These rely on per-schema constants and a shared `_PATH_CONVENTION_NOTE` string for the `path` field text.
3. The `_xxx` handler bodies — manual `int(action_input.get("max_rows", 20))`, `cast(list[str], columns)`, `if "rows" not in action_input: raise …` style validation that re-implements (loosely) what the schema is supposed to declare.

The runtime is vLLM 0.19.1 (Stable, April 18 2026 release), which uses xgrammar as its default structured-output backend. xgrammar tolerates both `{"type": ["string","null"]}` and `{"anyOf": [{"type":"string"},{"type":"null"}]}` for nullable fields, and ignores Pydantic-emitted `title` keys (they have no constraint semantics). The model parser in use is `qwen3_coder` (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`), which works at the token-extraction layer and is independent of the JSON Schema shape.

Pydantic v2 is already a runtime dependency (in use by `config.py` / `config_models.py`). pyright strict and ruff are enforced project-wide (`chore(quality): wire pyright strict + pre-commit, eliminate 122 type errors` — commit `a74eaa1`). The project has explicit byte-stability guarantees on `to_openai_tools()` output (`test_to_openai_tools_is_byte_stable_across_calls`) so prompt cache hits remain stable.

## Goals / Non-Goals

**Goals**
- Reduce the schema-related surface in `registry.py` from three parallel views to one Pydantic model per tool, while preserving the public `ToolDefinition` dataclass shape.
- Replace handler manual coercion with `Model.model_validate(...)` so type-correctness is enforced at the registry edge once, not in each handler.
- Improve error legibility: missing required fields surface as `ValueError("<tool>: field '<name>' is required")` (or equivalent typed message), not raw `KeyError('path')`.
- Preserve byte-equivalence of `to_openai_tools()` output across the migration so downstream prompt-cache hit rates are unaffected on the day of rollout.
- Keep the migration confined to `tools/registry.py` plus its tests; no spec or behavior changes leak into agents/, run/, or model-adapter layers.

**Non-Goals**
- Adding new tools, removing existing tools, or changing tool registration order.
- Changing the `ToolDefinition` dataclass shape, `ToolRegistry` interface, `ToolExecutionResult` shape, or any package-level `__all__`.
- Switching the structured-output backend (still xgrammar) or the tool-call parser (still `qwen3_coder`).
- Validating `task` (the `PublicTask` argument) via Pydantic — handlers still receive it as-is from the dispatcher.
- Migrating `_answer`'s post-validation observation payload (status / column_count / row_count) to Pydantic; that's an output side concern.
- Touching `filesystem.py`, `python_exec.py`, or `sqlite.py` — those modules already have stable, well-typed interfaces and are out of scope.

## Decisions

### Decision 1: One Pydantic input model per tool, colocated in `registry.py`

Define eight `BaseModel` subclasses inside `registry.py` directly above the `create_default_tool_registry()` builder. They are tightly coupled to the registry (each is paired 1:1 with a `ToolDefinition`); a separate `tools/inputs.py` module would only obscure that coupling for no readability gain at this scale.

Each model declares:
```python
model_config = ConfigDict(extra="forbid", populate_by_name=False)
```
- `extra="forbid"` produces `additionalProperties: false` in the generated schema (matches current contract; xgrammar's strictness expects this).
- `populate_by_name=False` ensures action_input keys are matched literally — no aliasing surprises.

Rejected alternative — putting models in a new `tools/schemas.py`: would split a single concept (tool = name + description + input contract + handler) across two files; the registry already centralizes the other two, so input contracts belong here too.

### Decision 2: Reusable field factories for `path` and bounded positive integers

Two helpers near the top of the schema section:

```python
def _path_field(*, file_kind: str, examples: tuple[str, ...]) -> Any:
    example_text = ", ".join(repr(ex) for ex in examples)
    return Field(
        description=(
            f"Relative path to a {file_kind}. "
            f"Examples: {example_text}. " + _PATH_CONVENTION_NOTE
        ),
    )

def _positive_int_field(*, default: int, description: str) -> Any:
    return Field(default=default, ge=1, description=description)
```

Used as `path: Annotated[str, _path_field(...)]` and `max_rows: Annotated[int, _positive_int_field(...)]`. This keeps `_PATH_CONVENTION_NOTE` as the single source of truth for path semantics across all five path-bearing tools and centralizes the `minimum: 1` invariant (which the current handcrafted schemas duplicate six times).

Rejected alternative — `class PathField(str): ...` typed-string subclass: Pydantic v2 supports custom types but they require `__get_pydantic_core_schema__` boilerplate that's heavier than the `Field` factory approach for the same expressive gain.

### Decision 3: Byte-stability normalizer with two pinned rewrites

Pydantic 2.10's `model_json_schema()` differs from the current handcrafted schemas in two known ways:

1. It adds `"title": "FieldName"` (capitalized) to every property and `"title": "ModelName"` at the top level. xgrammar ignores these but they bloat the request body.
2. For `Optional[str]` it emits `{"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}`, whereas the current `_READ_DOC_SCHEMA` writes `{"type": ["string", "null"], "default": null}`.

A small in-module helper rewrites both:

```python
def _normalize_for_vllm(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite Pydantic-generated schema into the byte-stable form vLLM expects.

    Two pinned transformations:
    - Strip every `title` key recursively (Pydantic-only metadata; not load-bearing).
    - Collapse `{"anyOf": [{"type": T}, {"type": "null"}]}` (and the symmetric ordering)
      into `{"type": [T, "null"]}` — preserves the wire bytes the project committed to.
    """
    ...
```

Both transformations are syntactically equivalent under JSON Schema 2020-12 and under xgrammar's PDA construction; the rewrite exists purely to keep the `to_openai_tools()` byte output identical on the day of rollout (no one-time prompt-cache miss).

The normalizer is locked by a fixture: `tests/fixtures/tool_schemas/<tool>.json` captures the pre-change schema bytes per tool. The post-change `to_openai_tools()` must produce byte-identical JSON when serialized with `json.dumps(..., sort_keys=True)`.

Rejected alternative — accept the prompt-cache miss and let the bytes change: defensible, but the project explicitly tests for byte stability and the normalizer is ~25 lines. Not worth giving up the property.

### Decision 4: Centralized validation at the registry edge, message format pinned

`ToolRegistry.execute(task, action, action_input)` wraps the per-tool `Model.model_validate(action_input)` call and translates `pydantic.ValidationError` into a single `ValueError` with a stable message format:

```
<tool_name>: <field_path> <reason>
```

Examples:
- `read_csv: field 'path' is required`
- `read_doc: field 'start_line' must be ≥ 1, got 0`
- `answer: extra field 'metadata' is not permitted`

The format is one-line and human-readable so the model can correct itself in the next ReAct turn. The format is locked by tests in `test_dispatcher.py` so future schema or message changes can't silently drift.

Rejected alternative — let `ValidationError` propagate raw: its multi-line, JSON-pointer-style format (`1 validation error for ReadCsvInput\npath\n  Field required [type=missing, ...`) wastes observation tokens and embeds an internal model name (`ReadCsvInput`) the agent shouldn't have to reason about.

### Decision 5: `AnswerInput` keeps the column-count cross-field check

The current `_answer` handler validates that each row's length equals `len(columns)` (`registry.py:182`). This is preserved as a `model_validator(mode="after")` on `AnswerInput`:

```python
@model_validator(mode="after")
def _check_row_widths(self) -> "AnswerInput":
    width = len(self.columns)
    for index, row in enumerate(self.rows):
        if len(row) != width:
            raise ValueError(
                f"row {index} has {len(row)} cells but columns has {width}"
            )
    return self
```

The cell-type contract (`str | float | bool | None`) becomes `list[list[str | float | bool | None]]`, which Pydantic emits as `anyOf: [{string}, {number}, {boolean}, {null}]` — already the shape `test_answer_schema_accepts_heterogeneous_row_cells` checks. No normalizer rewrite needed for this case.

### Decision 6: `input_schema` (text-prompt demo dict) generated from each model

Replace each hardcoded `input_schema={...}` literal in `create_default_tool_registry()` with a per-model `example_value()` classmethod that returns the demo dict. This couples the example to the model so that adding/removing a field in the model also surfaces in the prompt without manual sync.

```python
class ReadCsvInput(BaseModel):
    ...
    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"path": "relative/path/to/file.csv", "max_rows": 20}
```

The example dicts intentionally do not match the strict schema (e.g. `"path": "relative/path/to/file.csv"` is a placeholder, not a real path). This is preserved verbatim: empirically the demo-shaped form is more reliable than a strict-shaped example for text-protocol generation. The decision is documented in code via existing comments at `registry.py:11-15` and `:474-476`.

## Risks / Trade-offs

- **[Pydantic schema-emission drift across versions]** A future Pydantic minor version could change how `Optional[str]` or `model_config` flags get serialized, breaking the byte-stability normalizer. → **Mitigation**: the fixture-locked test fails immediately on drift; the normalizer is ~25 LOC and easy to extend with another rewrite case. Pin `pydantic>=2.10,<3.0` (already the case in `pyproject.toml`).

- **[xgrammar feature-gap on schema differences we didn't anticipate]** Our search confirms xgrammar handles `anyOf`-nullable, `additionalProperties: false`, and `title` keys; it could regress on edge cases we don't exercise (e.g. recursive `$ref`). → **Mitigation**: scope is restricted to flat object schemas with primitive properties — no `$ref`, no nested `oneOf`/`allOf`, no recursion. The fixture-byte-equivalence test is the canary.

- **[Cross-process schema imports]** Pydantic model classes live in `registry.py`. Subprocess workers (e.g. `python_exec.py`'s `multiprocessing.Process`) don't import `registry.py`; they receive primitive args. So no pickling concern. → **Mitigation**: none required, but documented here so the constraint isn't lost.

- **[`ValidationError` formatting becomes part of an implicit contract]** Once tests assert `"read_csv: field 'path' is required"`, callers (and prompts) will rely on it. Future Pydantic upgrades that change `ValidationError.errors()[0]["loc"]` shape could ripple into our message. → **Mitigation**: the translation lives in one private function (`_format_validation_error`) inside `registry.py`. Tests pin the message format, not the internal `ValidationError` shape, so we control the wire surface.

- **[`example_value` drifting from the strict schema]** The whole point of the demo dict is that it doesn't match the schema (it's a placeholder). A reviewer could reasonably ask "shouldn't these match?" and try to align them. → **Mitigation**: the empirical reasoning (placeholder is more reliable for text-protocol prompts) is recorded in `registry.py` comments today and is reiterated in this design's Decision 6 and in the tasks.md acceptance criteria. The prompt-side test that compares descriptive output remains.

## Migration Plan

This is an in-place refactor of one module. There is no rollout phase, no feature flag, no service to redeploy.

1. **Pre-change fixture capture (Task 1.x)**: while on the current `master` head, dump `to_openai_tools()` JSON per tool into `tests/fixtures/tool_schemas/<tool>.json` with `json.dumps(..., sort_keys=True, ensure_ascii=False, indent=2)`. Commit the fixtures.
2. **Implementation (Task 2.x–4.x)**: introduce models, normalizer, handler rewrites in a single PR. Tests run green on the pre-existing suite plus the new fixture comparison.
3. **Verification (Task 5.x)**: run the full test suite (`uv run pytest`) and `uv run pyright src/data_agent_baseline/tools/`. No `# type: ignore` introductions allowed.
4. **Rollback strategy**: revert the single PR. Because no public API or wire format changed, the revert is a `git revert <sha>` with no migration of state.

There is no separate deploy step — the code ships as part of the branch.

## Open Questions

None at this time. The proposal nailed down: (a) byte-stability is preserved (Decision 3), (b) the error message format is owned and pinned (Decision 4), (c) `example_value()` decoupling from strict schema is intentional (Decision 6), (d) `pydantic>=2.10,<3.0` pin is already in place. Remaining work is mechanical.
