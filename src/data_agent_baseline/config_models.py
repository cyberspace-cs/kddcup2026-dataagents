"""Pydantic v2 模型：用于 `load_app_config` 的内部校验层。

这些模型是 `config.py` 的实现细节，不向外暴露：

- 字段命名/默认值与 `AppConfig`/`AgentConfig`/`RunConfig`/`RateLimitConfig`/
  `DatasetConfig` 一一对应。
- 所有路径字段在 `mode="before"` 校验器里完成 `PROJECT_ROOT` 锚定，
  保留 `_path_value` 的语义（空/None → 默认值；绝对路径原样保留；相对路径以
  PROJECT_ROOT 为基；与 `state_dir` 不同：dataset/run 的路径会再 `.resolve()`）。
- `bool` / `int` 字段使用 `StrictBool` / `StrictInt`，关闭 Pydantic 的字符串
  →bool / 字符串→int 隐式转换，避免 `allow_parallel_tool_calls: "false"` 这类陷阱
  （详见 design.md D6 与 CHANGELOG `[Unreleased]`）。
- `str` 字段（model / api_base / api_key）保留旧 `str(...)` 强制转换，
  即 `api_key: 123 → "123"`、`api_key: null → "None"`，避免无声破坏现有 YAML。
- 校验器抛出的 `ValueError` 文本与 `config.py:179-293` 的旧 raise 站点字节一致；
  上层 `load_app_config` 捕获 `ValidationError` 后只取第一条错误并剥去
  `"Value error, "` 前缀重新抛 `ValueError`。

校验器声明顺序 = 错误暴露顺序，必须与旧实现保持一致，保证 fixture 报文不漂移：

    tool_protocol → enable_thinking → max_empty_tool_call_retries
        → api_keys → rate_limit{rpm → tpm → safety → reserve > 0
        → reserve_vs_max_tokens → max_retries → backoff_init → backoff_max}
        → key/limit pairing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

# 从 config.py 复用 PROJECT_ROOT、默认路径工厂、tool_protocol 字面量，保证两套实现指向同一基准
from data_agent_baseline.config import (
    PROJECT_ROOT,
    ToolProtocolLiteral,
    _default_dataset_root,  # pyright: ignore[reportPrivateUsage]
    _default_run_output_dir,  # pyright: ignore[reportPrivateUsage]
)

# RateLimitConfigModel.reserve_output_tokens 校验器借助 context 拿到父级的 max_tokens，
# 保持 "reserve > 0" 与 "reserve <= max_tokens" 在旧实现里的紧邻顺序。
_CONTEXT_KEY_MAX_TOKENS = "_max_tokens"


def _resolve_anchor_path(raw: Any, default: Path) -> Path:
    """模仿旧 `_path_value`：falsy → default；绝对 → 原样；相对 → PROJECT_ROOT 解析。"""
    if not raw:
        return default
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_state_dir(raw: Any) -> Path | None:
    """`rate_limit.state_dir` 专用：空/None → None；不调 `.resolve()` 与旧实现一致。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


class _StrictModel(BaseModel):
    """基类：未知键忽略；字段级 strict 通过 `StrictBool`/`StrictInt` 显式声明。"""

    model_config = ConfigDict(extra="ignore")


class DatasetConfigModel(_StrictModel):
    root_path: Path = Field(default_factory=_default_dataset_root)

    @field_validator("root_path", mode="before")
    @classmethod
    def _resolve(cls, v: Any) -> Path:
        return _resolve_anchor_path(v, _default_dataset_root())


class RateLimitConfigModel(_StrictModel):
    rpm_per_key: StrictInt = 600
    tpm_per_key: StrictInt = 1_000_000
    safety_factor: float = 0.85
    reserve_output_tokens: StrictInt = 16384
    max_retries: StrictInt = 6
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    state_dir: Path | None = None

    @field_validator("rpm_per_key")
    @classmethod
    def _check_rpm(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("agent.rate_limit.rpm_per_key must be positive.")
        return v

    @field_validator("tpm_per_key")
    @classmethod
    def _check_tpm(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("agent.rate_limit.tpm_per_key must be positive.")
        return v

    @field_validator("safety_factor")
    @classmethod
    def _check_safety_factor(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError("agent.rate_limit.safety_factor must be in the interval (0, 1].")
        return v

    @field_validator("reserve_output_tokens")
    @classmethod
    def _check_reserve(cls, v: int, info: ValidationInfo) -> int:
        # 旧实现里 reserve > 0 与 reserve <= max_tokens 是相邻两步；
        # 把后者也放进同一个校验器，保证多错误场景下首个错误次序与旧实现一致。
        if v <= 0:
            raise ValueError("agent.rate_limit.reserve_output_tokens must be positive.")
        ctx: dict[str, Any] = info.context or {}
        max_tokens = ctx.get(_CONTEXT_KEY_MAX_TOKENS, 0)
        if isinstance(max_tokens, int) and max_tokens > 0 and v > max_tokens:
            raise ValueError(
                "agent.rate_limit.reserve_output_tokens must not exceed agent.max_tokens."
            )
        return v

    @field_validator("max_retries")
    @classmethod
    def _check_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("agent.rate_limit.max_retries must be >= 0.")
        return v

    @field_validator("backoff_initial_seconds")
    @classmethod
    def _check_backoff_initial(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("agent.rate_limit.backoff_initial_seconds must be positive.")
        return v

    @field_validator("state_dir", mode="before")
    @classmethod
    def _resolve_state_dir_field(cls, v: Any) -> Path | None:
        return _resolve_state_dir(v)

    @model_validator(mode="after")
    def _check_backoff_max(self) -> RateLimitConfigModel:
        if self.backoff_max_seconds < self.backoff_initial_seconds:
            raise ValueError(
                "agent.rate_limit.backoff_max_seconds must be >= backoff_initial_seconds."
            )
        return self


class AgentConfigModel(_StrictModel):
    # 字段顺序 = 错误暴露顺序：tool_protocol → enable_thinking → max_empty_tool_call_retries
    # → api_keys → rate_limit。详见模块顶部 docstring。
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: StrictInt = 16
    temperature: float = 0.0
    tool_protocol: ToolProtocolLiteral = "native"
    allow_parallel_tool_calls: StrictBool = True
    max_tokens: StrictInt = 16384
    enable_thinking: StrictBool | None = None
    max_empty_tool_call_retries: StrictInt = 8
    api_keys: tuple[str, ...] = Field(default_factory=tuple)
    rate_limit: RateLimitConfigModel | None = None

    @field_validator("model", "api_base", "api_key", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        # 旧实现 `str(...)`：保留 `123 → "123"` 与 `null → "None"`，避免对历史 YAML 静默破坏。
        return str(v)

    @field_validator("tool_protocol", mode="before")
    @classmethod
    def _normalize_protocol(cls, v: Any) -> Any:
        normalized = str(v).strip().lower()
        if normalized not in {"native", "text"}:
            raise ValueError(
                f"agent.tool_protocol must be either 'native' or 'text'; got {normalized!r}."
            )
        return normalized

    @field_validator("enable_thinking", mode="before")
    @classmethod
    def _check_enable_thinking(cls, v: Any) -> Any:
        if v is None or isinstance(v, bool):
            return v
        raise ValueError(f"agent.enable_thinking must be a bool or null; got {type(v).__name__}.")

    @field_validator("max_empty_tool_call_retries")
    @classmethod
    def _check_max_empty(cls, v: int) -> int:
        if v < 0:
            raise ValueError("agent.max_empty_tool_call_retries must be >= 0.")
        return v

    @field_validator("api_keys", mode="before")
    @classmethod
    def _check_api_keys(cls, v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if not isinstance(v, list):
            raise ValueError("agent.api_keys must be a list of strings.")
        v_list = cast(list[Any], v)
        parsed: list[str] = []
        for idx, item in enumerate(v_list):
            if not isinstance(item, str):
                raise ValueError(
                    f"agent.api_keys[{idx}] must be a string; got {type(item).__name__}."
                )
            stripped = item.strip()
            if not stripped:
                raise ValueError(f"agent.api_keys[{idx}] must be a non-empty string.")
            parsed.append(stripped)
        if len(parsed) != len(set(parsed)):
            raise ValueError("agent.api_keys contains duplicate entries.")
        return tuple(parsed)

    @field_validator("rate_limit", mode="before")
    @classmethod
    def _validate_rate_limit(cls, v: Any, info: ValidationInfo) -> Any:
        # 1. shape 校验在嵌套校验之前先做（与旧实现 line 230-232 顺序一致）
        # 2. 用 context 把已校验的 max_tokens 注入子模型，使 `reserve <= max_tokens`
        #    与 `reserve > 0` 在子模型内部相邻执行（旧实现 line 262-267）
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("agent.rate_limit must be a mapping.")
        max_tokens_value = info.data.get("max_tokens", 16384)
        return RateLimitConfigModel.model_validate(
            v, context={_CONTEXT_KEY_MAX_TOKENS: max_tokens_value}
        )

    @model_validator(mode="after")
    def _check_pairing(self) -> AgentConfigModel:
        has_keys = bool(self.api_keys)
        has_rate_limit = self.rate_limit is not None
        if has_keys != has_rate_limit:
            raise ValueError(
                "agent.api_keys and agent.rate_limit must be configured together; "
                "leave both unset for the single-key / no-limit default path."
            )
        return self


class RunConfigModel(_StrictModel):
    output_dir: Path = Field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: StrictInt = 4
    task_timeout_seconds: StrictInt = 600
    blocklist: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("output_dir", mode="before")
    @classmethod
    def _resolve_output_dir(cls, v: Any) -> Path:
        return _resolve_anchor_path(v, _default_run_output_dir())

    @field_validator("run_id", mode="before")
    @classmethod
    def _normalize_run_id(cls, v: Any) -> str | None:
        # 旧实现：显式空白串视为 None（触发自动生成）
        if v is None:
            return None
        normalized = str(v).strip()
        return normalized or None

    @field_validator("blocklist", mode="before")
    @classmethod
    def _check_blocklist(cls, v: Any) -> tuple[str, ...]:
        # 与 cli.py `_coerce_task_number` 对齐：接受 int / "5" / "task_5"，全部归一化到 "task_<n>"。
        # 拒绝非 list、负数、空串、未知前缀，避免静默丢弃运维误配。
        if v is None:
            return ()
        if not isinstance(v, list):
            raise ValueError("run.blocklist must be a list of task ids.")
        v_list = cast(list[Any], v)
        normalized: list[str] = []
        for idx, item in enumerate(v_list):
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                # bool 是 int 子类，单独排除避免 `- true` 被当成 task_1
                raise ValueError(
                    f"run.blocklist[{idx}] must be an int or string; got {type(item).__name__}."
                )
            if isinstance(item, int):
                number = item
            else:
                cleaned = item.strip()
                if not cleaned:
                    raise ValueError(f"run.blocklist[{idx}] must be a non-empty string.")
                token = cleaned.removeprefix("task_") if cleaned.startswith("task_") else cleaned
                try:
                    number = int(token)
                except ValueError as exc:
                    raise ValueError(
                        f"run.blocklist[{idx}] must be an int or 'task_<n>'; got {item!r}."
                    ) from exc
            if number < 1:
                raise ValueError(f"run.blocklist[{idx}] task number must be >= 1; got {number}.")
            normalized.append(f"task_{number}")
        if len(normalized) != len(set(normalized)):
            raise ValueError("run.blocklist contains duplicate entries.")
        return tuple(normalized)


class AppConfigModel(_StrictModel):
    dataset: DatasetConfigModel = Field(default_factory=DatasetConfigModel)
    agent: AgentConfigModel = Field(default_factory=AgentConfigModel)
    run: RunConfigModel = Field(default_factory=RunConfigModel)
