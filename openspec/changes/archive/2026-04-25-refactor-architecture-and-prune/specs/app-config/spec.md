## ADDED Requirements

### Requirement: Pydantic v2 powers config loading; legacy ValueError contract preserved

`load_app_config(path: Path) -> AppConfig` SHALL parse YAML and validate it via Pydantic v2 models, but SHALL return frozen dataclass instances (`AppConfig`, `AgentConfig`, `RunConfig`, `RateLimitConfig`, `DatasetConfig`) — not Pydantic model instances — and SHALL raise `ValueError` (not `pydantic.ValidationError`) with the same message strings the pre-change loader produced.

#### Scenario: Valid YAML returns the expected dataclass shape
- **WHEN** `configs/react_baseline.example.yaml` is loaded
- **THEN** the returned object is an `AppConfig` instance whose subfields are dataclass instances of `AgentConfig`, `RunConfig`, etc.

#### Scenario: Invalid `tool_protocol` raises legacy ValueError
- **WHEN** YAML sets `agent.tool_protocol: "wrong"`
- **THEN** `load_app_config` raises `ValueError` with a message containing `agent.tool_protocol must be either 'native' or 'text'`

### Requirement: Strict bool/int coercion (intentional behavior tightening)

Boolean fields SHALL accept only YAML booleans (`true`, `false`); quoted strings (`"false"`, `"yes"`) MUST raise. Integer fields SHALL accept only YAML integers; numeric strings (`"5"`) and booleans-as-int MUST raise. This is an intentional change from the pre-Pydantic behavior and SHALL be documented in `CHANGELOG.md` `[Unreleased]`.

#### Scenario: Quoted bool string raises
- **WHEN** YAML sets `agent.allow_parallel_tool_calls: "false"`
- **THEN** `load_app_config` raises `ValueError` referencing the field

#### Scenario: Numeric string for int field raises
- **WHEN** YAML sets `agent.max_tokens: "8192"`
- **THEN** `load_app_config` raises `ValueError` referencing the field

#### Scenario: Plain YAML scalars work as before
- **WHEN** YAML sets `agent.allow_parallel_tool_calls: false` (unquoted) and `agent.max_tokens: 8192`
- **THEN** the resulting dataclass has `allow_parallel_tool_calls=False` and `max_tokens=8192`

### Requirement: `tool_protocol` is `Literal["native","text"]` end-to-end

The `AgentConfig.tool_protocol` field SHALL be typed `Literal["native", "text"]`. Pyright in strict mode MUST accept the assignment from `config.agent.tool_protocol` to call sites that expect `Literal["native","text"]` without a `# type: ignore`.

#### Scenario: No type-ignore at runner.py call site
- **WHEN** static analysis runs over `src/data_agent_baseline/run/runner.py`
- **THEN** the previously needed `# type: ignore[arg-type]` at the `protocol=config.agent.tool_protocol` site is removed and the call type-checks cleanly under strict mode

### Requirement: Range validation at load time

`max_steps`, `max_workers`, `max_tokens`, `max_empty_tool_call_retries`, and rate-limit numeric fields SHALL be validated at load time. Out-of-range values MUST raise `ValueError` naming the field.

#### Scenario: max_steps must be positive
- **WHEN** YAML sets `agent.max_steps: 0` or a negative integer
- **THEN** `load_app_config` raises `ValueError` referencing `agent.max_steps`

#### Scenario: rate_limit.safety_factor must be in (0, 1]
- **WHEN** YAML sets `agent.rate_limit.safety_factor: 0` or `1.5`
- **THEN** `load_app_config` raises `ValueError` referencing `agent.rate_limit.safety_factor`

### Requirement: Key-pool ↔ rate-limit pairing

`agent.api_keys` and `agent.rate_limit` MUST both be configured or both be omitted. Configuring only one MUST raise.

#### Scenario: api_keys without rate_limit raises
- **WHEN** YAML provides `agent.api_keys` but no `agent.rate_limit`
- **THEN** `load_app_config` raises `ValueError` mentioning that the two must be configured together

#### Scenario: Neither configured is allowed
- **WHEN** YAML provides neither `agent.api_keys` nor `agent.rate_limit`
- **THEN** `load_app_config` returns successfully with `api_keys=()` and `rate_limit=None`

### Requirement: Path resolution is project-root-anchored

Path-typed fields (`run.output_dir`, `dataset.input_root`, `agent.rate_limit.state_dir` when set) SHALL be resolved to absolute paths anchored at `PROJECT_ROOT` when the YAML supplies a relative path.

#### Scenario: Relative path is anchored at project root
- **WHEN** YAML sets `run.output_dir: "artifacts/runs"` (relative)
- **THEN** the loaded `run.output_dir` is `PROJECT_ROOT / "artifacts/runs"` as an absolute path

### Requirement: Configs round-trip byte-stably for valid inputs

For every fixture under `tests/fixtures/configs/valid/`, loading the YAML SHALL produce a dataclass whose `dataclasses.asdict` JSON serialization MUST match a frozen `expected.normalized.json` byte-for-byte.

#### Scenario: Valid fixtures match their golden serializations
- **WHEN** the test suite iterates `tests/fixtures/configs/valid/`
- **THEN** every input file produces a normalized output that byte-equals its companion `expected.normalized.json`
