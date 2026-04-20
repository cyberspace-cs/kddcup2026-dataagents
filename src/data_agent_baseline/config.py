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
class AgentConfig:
    """ReAct Agent 及底层模型调用的配置。

    api_key 作为**明文**从 YAML 读取；生产部署应通过环境变量或 secret store 注入，
    而不是写进版本化的配置文件。
    """

    model: str = "gpt-4.1-mini"              # OpenAI 兼容 chat.completions 的模型标识符
    api_base: str = "https://api.openai.com/v1"  # 兼容 OpenAI 协议的推理服务根地址
    api_key: str = ""                        # 认证 Key；空串会在实际调用时抛 RuntimeError
    max_steps: int = 16                      # 单任务 ReAct 循环上限，防止无限思考
    temperature: float = 0.0                 # 采样温度；0 保证可复现


@dataclass(frozen=True, slots=True)
class RunConfig:
    """批量运行（run-benchmark / run-task）的调度与产物落盘配置。"""

    output_dir: Path = field(default_factory=_default_run_output_dir)  # 运行产物落盘根目录
    run_id: str | None = None                # 显式指定的 run 子目录名；None 时按 UTC 时间戳生成
    max_workers: int = 4                     # 并行线程数（ThreadPoolExecutor 大小）
    task_timeout_seconds: int = 600          # 单任务硬超时，超时会 kill 子进程


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
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
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
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
