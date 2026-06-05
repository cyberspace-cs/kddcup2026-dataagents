"""模型调用抽象层。

通过 `ModelAdapter` 协议解耦具体 LLM 供应商，方便在部署时切到不同的 OpenAI 兼容
端点（含本地 vLLM、Anthropic/Gemini 的兼容网关等）。测试用的预置响应适配器位于
`tests.helpers.scripted_adapters`。

本模块同时承载两种对模型"一次输出"的抽象：
- `ModelStep`：旧文本 ReAct 协议下，解析 ```json fence 的结果（见 react.parse_model_step）
- `ModelResponse` + `ModelToolCall`：统一返回格式，同时覆盖文本协议与原生 function calling 协议
"""

from __future__ import annotations

import abc
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast

from openai import APIError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

if TYPE_CHECKING:
    from data_agent_baseline.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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


# 后端类型字面量。决定 `enable_thinking` 在 extra_body 中的位置：
# - "vllm":      嵌套到 `chat_template_kwargs`（chat template 渲染参数）
# - "dashscope": 顶层 `enable_thinking`（DashScope OpenAI 兼容模式约定）
# 错置会被对端静默忽略 —— vLLM 路径下顶层 enable_thinking 不进 chat template，
# 模型只会按 chat_template 默认值（Qwen3 系列默认 thinking on）渲染，配置开关失灵。
BackendKind = Literal["vllm", "dashscope"]

# 推断 dashscope 用的 host 关键字。命中即判 dashscope，否则归 vllm。
# 选这两个：DashScope 兼容模式入口稳定为 `dashscope.aliyuncs.com`；vLLM 自部署
# 通常是 IP/localhost/内网域名。误判时调用方显式写 backend_kind 即可覆盖。
_DASHSCOPE_HOST_HINTS = ("aliyuncs.com", "dashscope")


def _resolve_backend_kind(backend_kind: BackendKind | None, api_base: str) -> BackendKind:
    """显式 `backend_kind` 优先；为 None 时按 `api_base` 主机名推断。

    `api_base` 已在 adapter 构造时 `rstrip("/")`，可直接 lower 后做子串匹配。
    """
    if backend_kind is not None:
        return backend_kind
    lowered = api_base.lower()
    return "dashscope" if any(hint in lowered for hint in _DASHSCOPE_HOST_HINTS) else "vllm"


def _thinking_extra_body(enable_thinking: bool, backend_kind: BackendKind) -> dict[str, Any]:
    """按 backend_kind 把 `enable_thinking` 包成正确的 extra_body 片段。

    返回值用于 dict 合并。两条路径产出**不同 key**，互斥：
    - vLLM:      `{"chat_template_kwargs": {"enable_thinking": <bool>}}`
    - DashScope: `{"enable_thinking": <bool>}`
    """
    if backend_kind == "vllm":
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return {"enable_thinking": enable_thinking}


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """chat.completions 风格的单条消息。

    为了让同一结构同时表达 text/native 两种协议，字段做了如下扩展：
    - tool_calls:    仅 role="assistant" + native 协议下使用，承载原生 OpenAI tool_calls payload
    - tool_call_id:  仅 role="tool" 使用，OpenAI 要求每个 tool 消息必须回指其 tool_call_id

    常规 text 协议下只用 role + content 两个字段。
    """

    role: str
    content: str | list[dict[str, Any]]
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

    content 可以是 str 或 list[dict]（含 video_url block）。video_url 是
    DashScope / Qwen-VL 扩展，非 OpenAI 标准；换后端时需适配。
    """
    if message.role == "tool":
        content = message.content
        if not isinstance(content, str):
            logger.warning("tool message content is not str, serializing to JSON")
            content = json.dumps(content, ensure_ascii=False)
        return {
            "role": "tool",
            "content": content,
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


class ToolCallParseError(RuntimeError):
    """tool_call arguments JSON 解析失败，携带部分响应供 trace 保留。"""

    def __init__(self, message: str, *, partial_response: ModelResponse) -> None:
        super().__init__(message)
        self.partial_response = partial_response


class _BaseOpenAIAdapter(abc.ABC):
    """Shared init / request-build / call-and-parse for OpenAI-compatible adapters."""

    @abc.abstractmethod
    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        raise NotImplementedError

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float | None = None,
        max_tokens: int = 0,
        enable_thinking: bool | None = None,
        backend_kind: BackendKind | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        max_retries: int = 2,
        timeout: float = 600.0,
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
        # 解析后的 backend_kind 决定 enable_thinking 走顶层还是 chat_template_kwargs。
        # 显式优先；未指定则按 self.api_base 自动推断（含 aliyuncs.com/dashscope → dashscope）。
        self.backend_kind: BackendKind = _resolve_backend_kind(backend_kind, self.api_base)
        self.max_retries = max_retries
        self.timeout = timeout

    def estimate_prompt_tokens(self, messages: list[ModelMessage]) -> int:
        """Estimate text-protocol request tokens; no tool schema is sent on this path."""
        from data_agent_baseline.agents.rate_limit import (
            estimate_prompt_tokens as estimate_message_prompt_tokens,
        )

        return estimate_message_prompt_tokens(messages)

    def _build_request_kwargs(self, messages: list[ModelMessage]) -> dict[str, Any]:
        """Assemble the common chat.completions request dict."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
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
            request_kwargs["max_tokens"] = self.max_tokens
        # extra_body 累积构建：top_k / min_p 与 enable_thinking 合并下发，避免后来者覆盖前者。
        # enable_thinking 的位置依赖 self.backend_kind：vLLM 必须嵌套到 chat_template_kwargs，
        # DashScope 兼容模式则放顶层；错置会被对端静默忽略，详见 _thinking_extra_body 注释。
        extra_body: dict[str, Any] = {}
        if self.enable_thinking is not None:
            extra_body.update(_thinking_extra_body(self.enable_thinking, self.backend_kind))
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _call_and_parse_base(
        self, request_kwargs: dict[str, Any]
    ) -> tuple[ModelResponse, ChatCompletionMessage]:
        """Execute the API call and return (base_response, raw_message).

        The raw_message is the SDK message object from choices[0], exposed so
        subclasses can extract tool_calls from it.
        """
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

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
        content = message.content or ""
        cleaned = _strip_think_tags(content)
        base_response = ModelResponse(
            content=cleaned,
            tool_calls=[],
            raw_response=cleaned,
            raw_tool_calls=[],
            reasoning_content=_extract_reasoning_content(message),
            usage=_extract_usage(response),
            latency_ms=latency_ms,
        )
        return base_response, message


class OpenAIModelAdapter(_BaseOpenAIAdapter):
    """对 OpenAI 兼容 chat.completions 接口的包装（文本协议）。

    仅依赖 `openai` 官方 SDK；通过 `api_base` 切换后端（本地 vLLM、代理网关等）。
    每次 `complete` 都新建 client，避免跨线程共享 HTTP 连接导致的状态问题。
    不传 `tools=` 参数，模型响应必须是纯文本 —— 文本 ReAct 协议。
    """

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        """发起一次 chat.completions 请求并把响应打包成 `ModelResponse`。"""
        request_kwargs = self._build_request_kwargs(messages)
        base_response, _message = self._call_and_parse_base(request_kwargs)
        return base_response


class NativeToolsOpenAIAdapter(_BaseOpenAIAdapter):
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
        backend_kind: BackendKind | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        max_retries: int = 2,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            backend_kind=backend_kind,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            seed=seed,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.tools = tools
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        # 预先把不可变 ToolRegistry 的 definitions 转成 OpenAI tools 数组并缓存：
        # registry 在构造后无法增删工具，因此该缓存在适配器生命周期内永远有效，
        # 也保证了重复请求的 tools 字段字节稳定（prompt cache 命中）。
        self._openai_tools = tools.to_openai_tools()
        self._tools_prompt_tokens: int | None = None

    def _estimate_tools_prompt_tokens(self) -> int:
        """Count the stable native tools schema prompt contribution lazily."""
        if self._tools_prompt_tokens is None:
            from data_agent_baseline.agents.tokenizer_loader import count_qwen_tokens

            tools_text = json.dumps(self._openai_tools, ensure_ascii=False)
            self._tools_prompt_tokens = count_qwen_tokens(tools_text)
        return self._tools_prompt_tokens

    def estimate_prompt_tokens(self, messages: list[ModelMessage]) -> int:
        """Estimate native request tokens, including vLLM-rendered tool definitions."""
        return super().estimate_prompt_tokens(messages) + self._estimate_tools_prompt_tokens()

    def _build_request_kwargs(self, messages: list[ModelMessage]) -> dict[str, Any]:
        request_kwargs = super()._build_request_kwargs(messages)
        request_kwargs["tools"] = self._openai_tools
        request_kwargs["tool_choice"] = "auto"
        if self.allow_parallel_tool_calls:
            request_kwargs["parallel_tool_calls"] = True
        return request_kwargs

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        """发起一次 chat.completions 请求，返回解析好的 `ModelResponse`。"""
        request_kwargs = self._build_request_kwargs(messages)
        base_response, message = self._call_and_parse_base(request_kwargs)

        raw_tool_calls_payload: list[dict[str, Any]] = []
        parsed_calls: list[ModelToolCall] = []

        sdk_tool_calls = message.tool_calls or []
        for tool_call in sdk_tool_calls:
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
                arguments = dict(arguments_raw)
            else:
                arg_str = str(arguments_raw or "").strip()
                if not arg_str:
                    arguments = {}
                else:
                    try:
                        parsed = json.loads(arg_str)
                    except json.JSONDecodeError as exc:
                        err = ToolCallParseError(
                            f"Failed to parse tool_call arguments for {name!r}: {exc}",
                            partial_response=ModelResponse(
                                content=base_response.content,
                                tool_calls=[],
                                raw_response=base_response.raw_response,
                                raw_tool_calls=raw_tool_calls_payload,
                                reasoning_content=base_response.reasoning_content,
                                usage=base_response.usage,
                                latency_ms=base_response.latency_ms,
                            ),
                        )
                        raise err from exc
                    if not isinstance(parsed, dict):
                        raise RuntimeError(
                            f"tool_call arguments for {name!r} must decode to an object."
                        )
                    arguments = cast(dict[str, Any], parsed)

            call_id = str(dumped.get("id") or "")
            parsed_calls.append(ModelToolCall(id=call_id, name=name, arguments=arguments))

        return ModelResponse(
            content=base_response.content,
            tool_calls=parsed_calls,
            raw_response=base_response.raw_response,
            raw_tool_calls=raw_tool_calls_payload,
            reasoning_content=base_response.reasoning_content,
            usage=base_response.usage,
            latency_ms=base_response.latency_ms,
        )
