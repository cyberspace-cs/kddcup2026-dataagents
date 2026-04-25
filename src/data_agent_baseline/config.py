"""应用配置模块。

负责把 YAML 配置文件读成 `AppConfig`，并把可选字段退化为合理的默认值，
使得上层（CLI、runner、ReAct agent）不需要关心缺省处理。

校验由 `config_models` 里的 Pydantic v2 模型承担：
- `bool` / `int` 字段使用 `StrictBool` / `StrictInt`，
  关闭 YAML 字符串到 bool/int 的隐式转换（详见 design.md D6 与 CHANGELOG）。
- `tool_protocol` 端到端为 `Literal["native", "text"]`。
- 校验失败时取首条错误并剥去 `"Value error, "` 前缀，再用 `ValueError` 抛出，
  与旧实现的 17 个 raise 站点字节一致。

`load_app_config` 仍然返回 `AppConfig` 这套冻结 dataclass，
不向调用方泄漏 Pydantic 类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError

# 项目根目录：由本文件所在位置向上两级得到
# 注：src/data_agent_baseline/config.py → parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    """数据集根目录的默认值：`<project>/data/public/input`。"""
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    """运行产物的默认落盘根目录：`<project>/artifacts/runs`。"""
    return PROJECT_ROOT / "artifacts" / "runs"


# 工具协议字面量：所有调用方对此字段的取值假设都收敛到这里
ToolProtocolLiteral = Literal["native", "text"]


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """数据集相关配置。

    仅包含输入数据的根路径，其下应按 `task_<id>/` 组织每个任务。
    """

    # 公开数据集 input 目录；YAML 里缺省时回落到 PROJECT_ROOT/data/public/input
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """本地调限速 LLM 时的速率限制参数（RPM + TPM 双桶）。

    只在 `AgentConfig.rate_limit` 被显式配置时启用；线上 vLLM 路径请留空以保持 PR-1 行为。

    - rpm_per_key / tpm_per_key: 单 Key 的名义上限（与上游账号配额一致）
    - safety_factor: 实际容量 = cap * safety_factor，留出突发/时钟抖动的 buffer（0 < f <= 1）
    - reserve_output_tokens: 调用前按 "estimate_prompt + reserve_output" 预扣 TPM；
      实际 usage 回来后按差值补扣或退款，因此这里给的是输出上限的近似值（与 max_tokens 对齐）
    - max_retries: 429/5xx 重试次数上限（不含首次），>=0
    - backoff_initial_seconds / backoff_max_seconds: 指数退避起步和封顶；若响应带 Retry-After 则优先用 header
    - state_dir: 状态/锁文件根目录；None 时由 runner 自动推导为 `<artifacts_root>/ratelimit/<run_id>/`
    """

    rpm_per_key: int = 600
    tpm_per_key: int = 1_000_000
    safety_factor: float = 0.85
    reserve_output_tokens: int = 16384
    max_retries: int = 6
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    state_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """ReAct Agent 及底层模型调用的配置。

    api_key 作为**明文**从 YAML 读取；生产部署应通过环境变量或 secret store 注入,
    而不是写进版本化的配置文件。

    tool_protocol 决定使用哪种 LLM 协议：
    - "native":  OpenAI 原生 function calling（默认，vLLM 需 --enable-auto-tool-choice）
    - "text":    文本 JSON fence ReAct 协议（端点不支持 tools= 时的降级路径）

    max_tokens 用于 chat.completions 请求的同名字段（总输出上限，含 reasoning tokens）：
    - 0:   不发送该字段，沿用服务端默认（线上 vLLM 一般不限）
    - >0:  显式下发。本地调用百炼 Qwen3.5 限 TPM 的场景下建议填，
           预扣 token 误差会小很多；同时兜住单次 runaway 生成

    max_empty_tool_call_retries 是 native 协议下的空 tool_calls 独立重试预算：
    空 tool_calls 会写纠偏 observation，但不消耗业务 max_steps；达到该上限后失败，
    避免模型 reasoning-only 无限循环。

    enable_thinking 控制思考模式（Qwen3 特有；经 OpenAI SDK `extra_body` 透传）：
    - None:  不发送该字段，沿用服务端默认（vLLM `--reasoning-parser qwen3` 为 on）
    - True:  显式开；与线上行为对齐
    - False: 显式关；本地省 token 时可选，但线上不可用→慎用

    api_keys / rate_limit 是 PR-2 新增的 "本地 Key 池 + 限速" 入口：
    - 两者都为空/None（默认）: 走单 Key 直连路径，与 PR-1 行为一致
    - 两者均配置:             runner 按 task_index 轮询 api_keys，并用 RateLimitedAdapter 包壳
    - 只有其一:               load 时报错，避免 "只池不限" / "只限无池" 的误配
    """

    model: str = "gpt-4.1-mini"  # OpenAI 兼容 chat.completions 的模型标识符
    api_base: str = "https://api.openai.com/v1"  # 兼容 OpenAI 协议的推理服务根地址
    api_key: str = ""  # 认证 Key；空串会在实际调用时抛 RuntimeError
    max_steps: int = 16  # 单任务 ReAct 循环上限，防止无限思考
    temperature: float = 0.0  # 采样温度；0 保证可复现
    tool_protocol: ToolProtocolLiteral = "native"  # 端到端字面量；runner 直接转交 ReActAgent
    # 原生协议下，是否允许模型一轮返回多个 tool_calls；个别 OpenAI-兼容后端不支持该字段，可关闭
    allow_parallel_tool_calls: bool = True
    max_tokens: int = 16384  # 单次输出上限；0=不下发（用服务端默认）
    enable_thinking: bool | None = None  # None=不下发；True/False=走 extra_body 透传
    # PR-2：本地 Key 池；runner 按 task_index 轮询并下推单 Key 视图给子进程。空元组=禁用
    api_keys: tuple[str, ...] = ()
    # PR-2：限速配置；None=不启用限速器（线上 vLLM 路径默认如此）
    rate_limit: RateLimitConfig | None = None
    max_empty_tool_call_retries: int = 8


@dataclass(frozen=True, slots=True)
class RunConfig:
    """批量运行（run-benchmark / run-task）的调度与产物落盘配置。"""

    output_dir: Path = field(default_factory=_default_run_output_dir)  # 运行产物落盘根目录
    run_id: str | None = None  # 显式指定的 run 子目录名；None 时按 UTC 时间戳生成
    max_workers: int = 4  # 并行线程数（ThreadPoolExecutor 大小）
    task_timeout_seconds: int = 600  # 单任务硬超时，超时会 kill 子进程


@dataclass(frozen=True, slots=True)
class AppConfig:
    """顶层聚合配置，`load_app_config` 返回的完整对象。"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_app_config(config_path: Path) -> AppConfig:
    """读取 YAML 配置并构造 `AppConfig`。

    校验由 `AppConfigModel`（Pydantic v2）承担；通过后再实例化冻结 dataclass，
    保证调用方拿到的仍然是历史以来的 `AppConfig` 类型，不暴露 Pydantic 实例。
    """
    # 防御性导入：在函数内导入避免 config 模块 import 时的循环依赖隐患
    from data_agent_baseline.config_models import AppConfigModel

    # safe_load 防止任意 Python 对象实例化；空文件返回 None 时退化为空 dict
    payload = yaml.safe_load(config_path.read_text()) or {}

    try:
        validated = AppConfigModel.model_validate(payload)
    except ValidationError as exc:
        # Pydantic 在自定义校验器抛 ValueError 时会把 msg 包成 "Value error, <text>"。
        # 取第一条错误并剥去前缀，与旧实现的 ValueError 字节一致。
        first_error = exc.errors()[0]
        message = str(first_error["msg"]).removeprefix("Value error, ")
        raise ValueError(message) from exc

    rate_limit_model = validated.agent.rate_limit
    rate_limit: RateLimitConfig | None
    if rate_limit_model is None:
        rate_limit = None
    else:
        rate_limit = RateLimitConfig(
            rpm_per_key=rate_limit_model.rpm_per_key,
            tpm_per_key=rate_limit_model.tpm_per_key,
            safety_factor=rate_limit_model.safety_factor,
            reserve_output_tokens=rate_limit_model.reserve_output_tokens,
            max_retries=rate_limit_model.max_retries,
            backoff_initial_seconds=rate_limit_model.backoff_initial_seconds,
            backoff_max_seconds=rate_limit_model.backoff_max_seconds,
            state_dir=rate_limit_model.state_dir,
        )

    agent_config = AgentConfig(
        model=validated.agent.model,
        api_base=validated.agent.api_base,
        api_key=validated.agent.api_key,
        max_steps=validated.agent.max_steps,
        temperature=validated.agent.temperature,
        tool_protocol=validated.agent.tool_protocol,
        allow_parallel_tool_calls=validated.agent.allow_parallel_tool_calls,
        max_tokens=validated.agent.max_tokens,
        enable_thinking=validated.agent.enable_thinking,
        api_keys=validated.agent.api_keys,
        rate_limit=rate_limit,
        max_empty_tool_call_retries=validated.agent.max_empty_tool_call_retries,
    )

    return AppConfig(
        dataset=DatasetConfig(root_path=validated.dataset.root_path),
        agent=agent_config,
        run=RunConfig(
            output_dir=validated.run.output_dir,
            run_id=validated.run.run_id,
            max_workers=validated.run.max_workers,
            task_timeout_seconds=validated.run.task_timeout_seconds,
        ),
    )
