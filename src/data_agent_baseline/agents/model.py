"""模型调用抽象层。

通过 `ModelAdapter` 协议解耦具体 LLM 供应商，方便在部署时切到不同的 OpenAI 兼容
端点（含本地 vLLM、Anthropic/Gemini 的兼容网关等）。测试用的预置响应适配器位于
`tests.helpers.scripted_adapters`。

本模块同时承载两种对模型"一次输出"的抽象：
- `ModelStep`：旧文本 ReAct 协议下，解析 ```json fence 的结果（见 react.parse_model_step）
- `ModelResponse` + `ModelToolCall`：统一返回格式，同时覆盖文本协议与原生 function calling 协议
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast

from openai import APIError, OpenAI
from openai.types.chat import ChatCompletion

if TYPE_CHECKING:
    from data_agent_baseline.tools.registry import ToolRegistry


class _ToolCallFunctionDump(TypedDict, total=False):
    name: str
    arguments: str | dict[str, Any]


class _ToolCallDump(TypedDict, total=False):
    id: str
    type: str
    function: _ToolCallFunctionDump


# 用于 defensively 剥离 `<think>...</think>`（vLLM `--reasoning-parser qwen3` 未启用时的兜底）
_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """把潜在的 `<think>...</think>` 块剔除，返回干净的外部可见文本。

    即便 vLLM 启用了 `--reasoning-parser qwen3`，少数旧版本或并非 Qwen 的后端仍可能
    把思考段塞进 `message.content`，让文本 ReAct 解析器困惑，这里做一层防御性处理。
    """
    return _THINK_TAG_PATTERN.sub("", text).strip()


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """chat.completions 风格的单条消息。

    为了让同一结构同时表达 text/native 两种协议，字段做了如下扩展：
    - tool_calls:    仅 role="assistant" + native 协议下使用，承载原生 OpenAI tool_calls payload
    - tool_call_id:  仅 role="tool" 使用，OpenAI 要求每个 tool 消息必须回指其 tool_call_id

    常规 text 协议下只用 role + content 两个字段。
    """

    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ModelStep:
    """对文本 ReAct 协议下模型一次原始输出的结构化抽象。

    由 `parse_model_step` 从 JSON fence 块里抽取而成；`raw_response` 保留
    原始文本，方便后续回放/调试。
    """

    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """统一的工具调用表示，覆盖 native 与 text 两种协议。

    - id:         native 模式下是上游 tool_call_id；text 模式下用 "text:{turn}" 占位
    - name:       = action（工具名）
    - arguments:  = action_input，已解析好的 dict
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """单次 chat.completions 调用的 token 用量快照。

    覆盖 OpenAI 兼容后端常见字段（含 Qwen3 经由百炼/vLLM 时的嵌套扩展）：
    - prompt_tokens / completion_tokens / total_tokens：`usage.*` 顶层字段
    - cached_tokens:    `usage.prompt_tokens_details.cached_tokens`（前缀缓存命中量）
    - reasoning_tokens: `usage.completion_tokens_details.reasoning_tokens`（思维链耗量）

    全部默认 0，方便构造空实例（未捕获 usage / Scripted 测试路径）。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """asdict 包一层，保证 JSON 序列化时字段顺序与命名稳定。"""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """模型一次调用的统一返回结构。

    - content:          assistant 文本（text 模式 = 原始响应；native 模式 = thought 或空串）
    - tool_calls:       本轮模型请求执行的工具列表；text 模式下为空，由 ReAct 层从 content 里解析
    - raw_response:     原始 content（text 模式下用于回放成 assistant 消息）
    - raw_tool_calls:   原始 OpenAI tool_calls payload（SDK 对象的 dict 形式），
                        native 模式下用来把 assistant 消息重放回 messages 数组
    - reasoning_content:vLLM `--reasoning-parser qwen3` / Qwen 百炼会把 <think> 拆到
                        `message.reasoning_content`；这里留作透传用于 trace 可观测
    - usage:            单次调用的 token 用量（供限流器校准与成本统计）
    - latency_ms:       API 往返耗时（含 SDK 解析），0 表示未测
    """

    content: str
    tool_calls: list[ModelToolCall] = field(default_factory=lambda: [])  # noqa: PIE807
    raw_response: str = ""
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=lambda: [])  # noqa: PIE807
    reasoning_content: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: int = 0


class ModelAdapter(Protocol):
    """鸭子类型协议：任何暴露 `complete(messages) -> ModelResponse` 的对象都可充当模型。"""

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        raise NotImplementedError


def _message_to_payload(message: ModelMessage) -> dict[str, Any]:
    """把 `ModelMessage` 序列化成 OpenAI SDK 期望的 dict。

    规则：
    - tool 角色必须带 tool_call_id
    - assistant 角色若带 tool_calls，则 content 可为空串
    - 其他角色只保留 role+content
    """
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id or "",
        }
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "assistant" and message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    return payload


def _extract_reasoning_content(message: Any) -> str:
    """从 `choices[0].message` 里防御性提取 `reasoning_content`。

    vLLM `--reasoning-parser qwen3` 与百炼 Qwen3 均会把思维链塞进这个字段。
    OpenAI 官方 SDK 的 pydantic 模型不一定声明该字段，兜底走 `model_extra`。
    """
    direct = getattr(message, "reasoning_content", None)
    if isinstance(direct, str) and direct:
        return direct
    extras = getattr(message, "model_extra", None)
    if isinstance(extras, dict):
        value = cast(dict[str, Any], extras).get("reasoning_content")
        if isinstance(value, str):
            return value
    return ""


def _as_int(value: Any) -> int:
    """把 usage 里的数值字段强制转 int，None/非法值归零。

    百炼偶尔返回字符串数字；SDK 基本保证 int，但做一层保护成本微乎其微。
    """
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_usage(response: Any) -> TokenUsage:
    """从 `chat.completions` 响应构造 `TokenUsage`。

    对嵌套字段（`prompt_tokens_details.cached_tokens` /
    `completion_tokens_details.reasoning_tokens`）全部走 `getattr(..., None)`，
    不同后端返回的字段集不一致，缺失一律视为 0。
    """
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return TokenUsage()

    prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
    completion_details = getattr(usage_obj, "completion_tokens_details", None)
    cached = getattr(prompt_details, "cached_tokens", None) if prompt_details is not None else None
    reasoning = (
        getattr(completion_details, "reasoning_tokens", None)
        if completion_details is not None
        else None
    )

    return TokenUsage(
        prompt_tokens=_as_int(getattr(usage_obj, "prompt_tokens", 0)),
        completion_tokens=_as_int(getattr(usage_obj, "completion_tokens", 0)),
        total_tokens=_as_int(getattr(usage_obj, "total_tokens", 0)),
        cached_tokens=_as_int(cached),
        reasoning_tokens=_as_int(reasoning),
    )


class OpenAIModelAdapter:
    """对 OpenAI 兼容 chat.completions 接口的包装（文本协议）。

    仅依赖 `openai` 官方 SDK；通过 `api_base` 切换后端（本地 vLLM、代理网关等）。
    每次 `complete` 都新建 client，避免跨线程共享 HTTP 连接导致的状态问题。
    不传 `tools=` 参数，模型响应必须是纯文本 —— 文本 ReAct 协议。
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float | None = None,
        max_tokens: int = 0,
        enable_thinking: bool | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        # 去掉尾部斜杠：OpenAI SDK 会自行拼 `/chat/completions`，重复的 `/` 在部分
        # 反向代理下会被当成不同的 URL 命中 404
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        # 采样字段统一以 None = "不下发" 为约定；is not None 时按位置（顶层 vs extra_body）
        # 透传。temperature / top_p / presence_penalty / seed 走顶层（DashScope 文档支持），
        # top_k / min_p 走 extra_body 并与 enable_thinking 合并而非覆盖。
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.seed = seed
        # 0 = 不下发 max_tokens（服务端默认）；>0 = 显式上限
        self.max_tokens = max_tokens
        # None = 不下发 enable_thinking（服务端默认）；True/False = 经 extra_body 透传
        self.enable_thinking = enable_thinking

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        """发起一次 chat.completions 请求并把响应打包成 `ModelResponse`。"""
        # 显式检查 api_key，否则 SDK 会抛一条含糊的 401；这里给用户明确提示
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            # 把 ModelMessage 转成 SDK 期望的 dict 形式
            "messages": [_message_to_payload(message) for message in messages],
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            request_kwargs["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        if self.max_tokens > 0:
            # 下发显式上限；Qwen3 下封顶的是 **总输出**（含 reasoning tokens）
            request_kwargs["max_tokens"] = self.max_tokens
        # extra_body 累积构建：top_k / min_p 与 enable_thinking 合并下发，避免后来者覆盖前者
        extra_body: dict[str, Any] = {}
        if self.enable_thinking is not None:
            # Qwen3 特有字段；经 extra_body 透传进请求体（OpenAI SDK 原生字段表没有）
            extra_body["enable_thinking"] = self.enable_thinking
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        started = perf_counter()
        try:
            response = cast(ChatCompletion, client.chat.completions.create(**request_kwargs))
        except APIError as exc:
            # 把供应商错误统一包成 RuntimeError，便于 ReAct 循环的通用异常分支处理
            raise RuntimeError(f"Model request failed: {exc}") from exc
        latency_ms = int((perf_counter() - started) * 1000)

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        message = choices[0].message
        # 允许 content 为 None（偶尔模型返回空串或 reasoning-only）—— 归一化成空串
        content = message.content or ""
        # 防御性剥离 `<think>` —— 若 vLLM reasoning-parser 未启用，避免污染 JSON 解析
        cleaned = _strip_think_tags(content)
        return ModelResponse(
            content=cleaned,
            tool_calls=[],
            raw_response=cleaned,
            raw_tool_calls=[],
            reasoning_content=_extract_reasoning_content(message),
            usage=_extract_usage(response),
            latency_ms=latency_ms,
        )


class NativeToolsOpenAIAdapter:
    """对 OpenAI 兼容接口的包装（原生 function calling 协议）。

    与 `OpenAIModelAdapter` 的核心差异：
    - 请求端：传 `tools=[...]` + `tool_choice="auto"` + `parallel_tool_calls=True`，
      把工具定义下推给模型，让其原生以 tool_calls 形式回应
    - 响应端：解析 `choices[0].message.tool_calls`，映射为 `ModelToolCall` 列表，
      `content` 保留为 assistant 的自由文本（可能为空字符串）

    vLLM 侧需启用 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`。
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float | None = None,
        tools: ToolRegistry,
        allow_parallel_tool_calls: bool = True,
        max_tokens: int = 0,
        enable_thinking: bool | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        # 采样字段：与 OpenAIModelAdapter 对齐，None = 不下发；
        # top_p / presence_penalty / seed 走顶层；top_k / min_p 走 extra_body 与
        # enable_thinking 合并；详见 OpenAIModelAdapter 的注释。
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.tools = tools
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        # 预先把不可变 ToolRegistry 的 definitions 转成 OpenAI tools 数组并缓存：
        # registry 在构造后无法增删工具，因此该缓存在适配器生命周期内永远有效，
        # 也保证了重复请求的 tools 字段字节稳定（prompt cache 命中）。
        self._openai_tools = tools.to_openai_tools()

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        """发起一次 chat.completions 请求，返回解析好的 `ModelResponse`。"""
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_payload(message) for message in messages],
            "tools": self._openai_tools,
            "tool_choice": "auto",
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            request_kwargs["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        if self.allow_parallel_tool_calls:
            # 并非所有 OpenAI-兼容后端都接受此字段，但 vLLM/OpenAI 官方都支持；
            # 配置项允许部署时关掉以兼容个别实现
            request_kwargs["parallel_tool_calls"] = True
        if self.max_tokens > 0:
            # 下发显式上限；Qwen3 下封顶的是 **总输出**（含 reasoning tokens）
            request_kwargs["max_tokens"] = self.max_tokens
        # extra_body 累积构建：top_k / min_p 与 enable_thinking 合并下发，避免后来者覆盖前者
        extra_body: dict[str, Any] = {}
        if self.enable_thinking is not None:
            # Qwen3 特有字段；经 extra_body 透传进请求体（OpenAI SDK 原生字段表没有）
            extra_body["enable_thinking"] = self.enable_thinking
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        started = perf_counter()
        try:
            response = cast(ChatCompletion, client.chat.completions.create(**request_kwargs))
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc
        latency_ms = int((perf_counter() - started) * 1000)

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")

        message = choices[0].message
        raw_content = message.content or ""
        cleaned_content = _strip_think_tags(raw_content)

        raw_tool_calls_payload: list[dict[str, Any]] = []
        parsed_calls: list[ModelToolCall] = []

        sdk_tool_calls = message.tool_calls or []
        for tool_call in sdk_tool_calls:
            # SDK 对象 → dict：OpenAI SDK 的 pydantic 模型提供 model_dump；
            # 失败时兜底成手工构造的结构
            try:
                dumped: _ToolCallDump = cast(_ToolCallDump, tool_call.model_dump(exclude_none=True))
            except AttributeError:
                fn_obj = cast(Any, tool_call).function
                dumped = {
                    "id": str(getattr(tool_call, "id", "")),
                    "type": "function",
                    "function": {
                        "name": str(getattr(fn_obj, "name", "")),
                        "arguments": cast("str | dict[str, Any]", getattr(fn_obj, "arguments", "")),
                    },
                }
            raw_tool_calls_payload.append(cast(dict[str, Any], dumped))

            function_payload: _ToolCallFunctionDump = dumped.get("function") or {}
            name = str(function_payload.get("name", ""))
            arguments_raw = function_payload.get("arguments", "")
            arguments: dict[str, Any]
            if isinstance(arguments_raw, dict):
                # 个别实现会把 arguments 当 dict 返回；透传即可
                arguments = dict(arguments_raw)
            else:
                arg_str = str(arguments_raw or "").strip()
                if not arg_str:
                    # 空参数视为 {}，让工具层做默认值兜底；避免模型传空字符串炸解析
                    arguments = {}
                else:
                    try:
                        parsed = json.loads(arg_str)
                    except json.JSONDecodeError as exc:
                        # 解析失败抛错，让 Agent 层记录为一条 __error__ 步并继续
                        raise RuntimeError(
                            f"Failed to parse tool_call arguments for {name!r}: {exc}"
                        ) from exc
                    if not isinstance(parsed, dict):
                        raise RuntimeError(
                            f"tool_call arguments for {name!r} must decode to an object."
                        )
                    arguments = cast(dict[str, Any], parsed)

            call_id = str(dumped.get("id") or "")
            parsed_calls.append(ModelToolCall(id=call_id, name=name, arguments=arguments))

        return ModelResponse(
            content=cleaned_content,
            tool_calls=parsed_calls,
            raw_response=cleaned_content,
            raw_tool_calls=raw_tool_calls_payload,
            reasoning_content=_extract_reasoning_content(message),
            usage=_extract_usage(response),
            latency_ms=latency_ms,
        )
