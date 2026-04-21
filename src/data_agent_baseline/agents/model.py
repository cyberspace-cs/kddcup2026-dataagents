"""模型调用抽象层。

通过 `ModelAdapter` 协议解耦具体 LLM 供应商，方便在测试里替换成
`ScriptedModelAdapter`（预置响应）或在部署时切到不同的 OpenAI 兼容端点
（含本地 vLLM、Anthropic/Gemini 的兼容网关等）。

本模块同时承载两种对模型"一次输出"的抽象：
- `ModelStep`：旧文本 ReAct 协议下，解析 ```json fence 的结果（见 react.parse_model_step）
- `ModelResponse` + `ModelToolCall`：统一返回格式，同时覆盖文本协议与原生 function calling 协议
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from openai import APIError, OpenAI

if TYPE_CHECKING:
    from data_agent_baseline.tools.registry import ToolRegistry


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
class ModelResponse:
    """模型一次调用的统一返回结构。

    - content:          assistant 文本（text 模式 = 原始响应；native 模式 = thought 或空串）
    - tool_calls:       本轮模型请求执行的工具列表；text 模式下为空，由 ReAct 层从 content 里解析
    - raw_response:     原始 content（text 模式下用于回放成 assistant 消息）
    - raw_tool_calls:   原始 OpenAI tool_calls payload（SDK 对象的 dict 形式），
                        native 模式下用来把 assistant 消息重放回 messages 数组
    """

    content: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    raw_response: str = ""
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)


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
        temperature: float,
    ) -> None:
        self.model = model
        # 去掉尾部斜杠：OpenAI SDK 会自行拼 `/chat/completions`，重复的 `/` 在部分
        # 反向代理下会被当成不同的 URL 命中 404
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        """发起一次 chat.completions 请求并把响应打包成 `ModelResponse`。"""
        # 显式检查 api_key，否则 SDK 会抛一条含糊的 401；这里给用户明确提示
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                # 把 ModelMessage 转成 SDK 期望的 dict 形式
                messages=[_message_to_payload(message) for message in messages],
                temperature=self.temperature,
            )
        except APIError as exc:
            # 把供应商错误统一包成 RuntimeError，便于 ReAct 循环的通用异常分支处理
            raise RuntimeError(f"Model request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        # 允许 content 为 None（偶尔模型返回空串或 reasoning-only）—— 归一化成空串
        content = choices[0].message.content or ""
        if not isinstance(content, str):
            # 保护：极少数 SDK/代理会返回多模态块的 list，统一拒收
            raise RuntimeError("Model response text content is not a string.")
        # 防御性剥离 `<think>` —— 若 vLLM reasoning-parser 未启用，避免污染 JSON 解析
        cleaned = _strip_think_tags(content)
        return ModelResponse(
            content=cleaned,
            tool_calls=[],
            raw_response=cleaned,
            raw_tool_calls=[],
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
        temperature: float,
        tools: ToolRegistry,
        allow_parallel_tool_calls: bool = True,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.tools = tools
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        # 预先把 ToolRegistry 转成 OpenAI tools 数组；保持实例不可变方便 prompt cache
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
            "temperature": self.temperature,
            "tools": self._openai_tools,
            "tool_choice": "auto",
        }
        if self.allow_parallel_tool_calls:
            # 并非所有 OpenAI-兼容后端都接受此字段，但 vLLM/OpenAI 官方都支持；
            # 配置项允许部署时关掉以兼容个别实现
            request_kwargs["parallel_tool_calls"] = True

        try:
            response = client.chat.completions.create(**request_kwargs)
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")

        message = choices[0].message
        raw_content = message.content or ""
        if not isinstance(raw_content, str):
            raise RuntimeError("Model response text content is not a string.")
        cleaned_content = _strip_think_tags(raw_content)

        raw_tool_calls_payload: list[dict[str, Any]] = []
        parsed_calls: list[ModelToolCall] = []

        sdk_tool_calls = message.tool_calls or []
        for tool_call in sdk_tool_calls:
            # SDK 对象 → dict：OpenAI SDK 的 pydantic 模型提供 model_dump；
            # 失败时兜底成手工构造的结构
            try:
                dumped = tool_call.model_dump(exclude_none=True)
            except AttributeError:
                dumped = {
                    "id": getattr(tool_call, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(tool_call.function, "name", ""),
                        "arguments": getattr(tool_call.function, "arguments", ""),
                    },
                }
            raw_tool_calls_payload.append(dumped)

            function_payload = dumped.get("function", {}) or {}
            name = str(function_payload.get("name", ""))
            arguments_raw = function_payload.get("arguments", "")
            if isinstance(arguments_raw, dict):
                # 个别实现会把 arguments 当 dict 返回；透传即可
                arguments: dict[str, Any] = dict(arguments_raw)
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
                    arguments = parsed

            call_id = str(dumped.get("id") or "")
            parsed_calls.append(ModelToolCall(id=call_id, name=name, arguments=arguments))

        return ModelResponse(
            content=cleaned_content,
            tool_calls=parsed_calls,
            raw_response=cleaned_content,
            raw_tool_calls=raw_tool_calls_payload,
        )


class ScriptedModelAdapter:
    """测试/调试专用：按预置**文本**列表依次返回响应。

    真正的 OpenAI 调用在 CI 上不可靠（网络 + 额度），文本 ReAct 循环的单元测试
    用它注入确定性的 JSON 输出。签名保持与 `ModelAdapter` 协议一致（返回 `ModelResponse`）。
    """

    def __init__(self, responses: list[str]) -> None:
        # 拷贝一份：避免外部修改列表影响播放顺序
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        # messages 对预置响应无用，显式 del 提示读者（也能在启用严格 linter 时消除未用警告）
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        text = self._responses.pop(0)
        return ModelResponse(
            content=text,
            tool_calls=[],
            raw_response=text,
            raw_tool_calls=[],
        )


class ScriptedNativeModelAdapter:
    """测试/调试专用：按预置 `ModelResponse` 列表依次返回，覆盖 native 协议路径。

    与 `ScriptedModelAdapter` 的差异：测试可以直接塞入已解析好的 `tool_calls`
    与 `raw_tool_calls`，无需经过 JSON 字符串往返。用于断言 native 循环的
    分支行为（并行 tool_calls / answer 短路 / 空 tool_calls 兜底等）。
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
