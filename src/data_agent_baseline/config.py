"""应用配置模块。

负责把 YAML 配置文件读成 `AppConfig`，并把可选字段退化为合理的默认值，
使得上层（CLI、runner、ReAct agent）不需要关心缺省处理。

一切路径都以项目根 (PROJECT_ROOT = src/.. 的上两级) 为基准，保证在任何
工作目录下 `dabench` 命令都能定位到 `data/` 与 `artifacts/`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 项目根目录：由本文件所在位置向上两级得到
# 注：src/data_agent_baseline/config.py → parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    """数据集根目录的默认值：`<project>/data/public/input`。"""
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    """运行产物的默认落盘根目录：`<project>/artifacts/runs`。"""
    return PROJECT_ROOT / "artifacts" / "runs"


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
    tool_protocol: str = "native"  # "native" | "text"；默认原生，text 仅作降级
    # 原生协议下，是否允许模型一轮返回多个 tool_calls；个别 OpenAI-兼容后端不支持该字段，可关闭
    allow_parallel_tool_calls: bool = True
    max_tokens: int = 16384  # 单次输出上限；0=不下发（用服务端默认）
    enable_thinking: bool | None = None  # None=不下发；True/False=走 extra_body 透传
    # PR-2：本地 Key 池；runner 按 task_index 轮询并下推单 Key 视图给子进程。空元组=禁用
    api_keys: tuple[str, ...] = ()
    # PR-2：限速配置；None=不启用限速器（线上 vLLM 路径默认如此）
    rate_limit: RateLimitConfig | None = None


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


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    """把 YAML 里的路径字符串规整成绝对 `Path`。

    规则：
    - 空串或 None → 采用默认值
    - 绝对路径 → 原样保留
    - 相对路径 → 以 `PROJECT_ROOT` 为锚点解析
    """
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    # 相对路径以项目根为基，避免受 CWD 影响
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    """读取 YAML 配置并构造 `AppConfig`。

    YAML 顶层允许的三个键：`dataset` / `agent` / `run`。任何字段缺省时
    回落到对应 dataclass 的默认值，从而允许使用者只写自己关心的字段。
    """
    # safe_load 防止任意 Python 对象实例化；空文件返回 None 时退化为空 dict
    payload = yaml.safe_load(config_path.read_text()) or {}

    # 预先实例化默认对象，作为 “字段缺省” 的来源
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    # 数据集仅包含 root_path，直接做路径规整
    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )

    # Agent 字段全部显式 str/int/float 转换，避免 YAML 把裸数值当成 int/float 时类型错位
    tool_protocol_raw = (
        str(agent_payload.get("tool_protocol", agent_defaults.tool_protocol)).strip().lower()
    )
    if tool_protocol_raw not in {"native", "text"}:
        raise ValueError(
            f"agent.tool_protocol must be either 'native' or 'text'; got {tool_protocol_raw!r}."
        )

    # enable_thinking 三值严格校验：None / bool；其他类型（int/str）直接拒收，
    # 避免 `enable_thinking: 1` 被静默当 True，读者/运维都会困惑
    raw_enable_thinking = agent_payload.get("enable_thinking", agent_defaults.enable_thinking)
    if raw_enable_thinking is None:
        enable_thinking: bool | None = None
    elif isinstance(raw_enable_thinking, bool):
        enable_thinking = raw_enable_thinking
    else:
        raise ValueError(
            "agent.enable_thinking must be a bool or null; "
            f"got {type(raw_enable_thinking).__name__}."
        )

    max_tokens_value = int(agent_payload.get("max_tokens", agent_defaults.max_tokens))

    # api_keys：严格要求 list-of-str，非字符串立即拒收；空串、重复也拒（YAML 误配非常好修）
    api_keys_raw = agent_payload.get("api_keys", None)
    api_keys: tuple[str, ...] = ()
    if api_keys_raw is not None:
        if not isinstance(api_keys_raw, list):
            raise ValueError("agent.api_keys must be a list of strings.")
        parsed_keys: list[str] = []
        for idx, item in enumerate(api_keys_raw):
            if not isinstance(item, str):
                raise ValueError(
                    f"agent.api_keys[{idx}] must be a string; got {type(item).__name__}."
                )
            stripped = item.strip()
            if not stripped:
                raise ValueError(f"agent.api_keys[{idx}] must be a non-empty string.")
            parsed_keys.append(stripped)
        if len(parsed_keys) != len(set(parsed_keys)):
            raise ValueError("agent.api_keys contains duplicate entries.")
        api_keys = tuple(parsed_keys)

    # rate_limit：缺省为 None；显式写了子节则逐字段解析+校验
    rate_limit_raw = agent_payload.get("rate_limit", None)
    rate_limit: RateLimitConfig | None = None
    if rate_limit_raw is not None:
        if not isinstance(rate_limit_raw, dict):
            raise ValueError("agent.rate_limit must be a mapping.")
        rl_defaults = RateLimitConfig()
        rpm_per_key = int(rate_limit_raw.get("rpm_per_key", rl_defaults.rpm_per_key))
        tpm_per_key = int(rate_limit_raw.get("tpm_per_key", rl_defaults.tpm_per_key))
        safety_factor = float(rate_limit_raw.get("safety_factor", rl_defaults.safety_factor))
        reserve_output_tokens = int(
            rate_limit_raw.get("reserve_output_tokens", rl_defaults.reserve_output_tokens)
        )
        max_retries = int(rate_limit_raw.get("max_retries", rl_defaults.max_retries))
        backoff_initial = float(
            rate_limit_raw.get("backoff_initial_seconds", rl_defaults.backoff_initial_seconds)
        )
        backoff_max = float(
            rate_limit_raw.get("backoff_max_seconds", rl_defaults.backoff_max_seconds)
        )
        state_dir_raw = rate_limit_raw.get("state_dir")
        state_dir: Path | None = None
        if state_dir_raw is not None:
            # 复用 _path_value 的语义：空串视为未填；相对路径以 PROJECT_ROOT 为基
            state_dir_str = str(state_dir_raw).strip()
            if state_dir_str:
                candidate = Path(state_dir_str)
                state_dir = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)

        if rpm_per_key <= 0:
            raise ValueError("agent.rate_limit.rpm_per_key must be positive.")
        if tpm_per_key <= 0:
            raise ValueError("agent.rate_limit.tpm_per_key must be positive.")
        if not (0 < safety_factor <= 1):
            raise ValueError("agent.rate_limit.safety_factor must be in the interval (0, 1].")
        if reserve_output_tokens <= 0:
            raise ValueError("agent.rate_limit.reserve_output_tokens must be positive.")
        if max_tokens_value > 0 and reserve_output_tokens > max_tokens_value:
            raise ValueError(
                "agent.rate_limit.reserve_output_tokens must not exceed agent.max_tokens."
            )
        if max_retries < 0:
            raise ValueError("agent.rate_limit.max_retries must be >= 0.")
        if backoff_initial <= 0:
            raise ValueError("agent.rate_limit.backoff_initial_seconds must be positive.")
        if backoff_max < backoff_initial:
            raise ValueError(
                "agent.rate_limit.backoff_max_seconds must be >= backoff_initial_seconds."
            )

        rate_limit = RateLimitConfig(
            rpm_per_key=rpm_per_key,
            tpm_per_key=tpm_per_key,
            safety_factor=safety_factor,
            reserve_output_tokens=reserve_output_tokens,
            max_retries=max_retries,
            backoff_initial_seconds=backoff_initial,
            backoff_max_seconds=backoff_max,
            state_dir=state_dir,
        )

    # 池与限速必须成对启用/关闭：避免 "只有池无限速" / "只限速不分池" 的误配
    if (api_keys and rate_limit is None) or (rate_limit is not None and not api_keys):
        raise ValueError(
            "agent.api_keys and agent.rate_limit must be configured together; "
            "leave both unset for the single-key / no-limit default path."
        )

    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        tool_protocol=tool_protocol_raw,
        allow_parallel_tool_calls=bool(
            agent_payload.get("allow_parallel_tool_calls", agent_defaults.allow_parallel_tool_calls)
        ),
        max_tokens=max_tokens_value,
        enable_thinking=enable_thinking,
        api_keys=api_keys,
        rate_limit=rate_limit,
    )

    # run_id 的特殊处理：显式写了空白串也视为 None（触发自动生成）
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(
            run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)
        ),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
