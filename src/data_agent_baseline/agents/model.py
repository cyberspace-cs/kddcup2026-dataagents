"""模型调用抽象层。

通过 `ModelAdapter` 协议解耦具体 LLM 供应商，方便在测试里替换成
`ScriptedModelAdapter`（预置响应）或在部署时切到不同的 OpenAI 兼容端点
（含本地 vLLM、Anthropic/Gemini 的兼容网关等）。

`ModelStep` 是 ReAct 层对 "模型一次输出" 的结构化表示，解析逻辑在
`react.py:parse_model_step` 里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI


@dataclass(frozen=True, slots=True)
class ModelMessage:
    """chat.completions 风格的单条消息：role + content。

    跨适配器统一格式，避免在 ReAct 层直接依赖 OpenAI SDK 的类型。
    """

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ModelStep:
    """对模型一次原始输出的结构化抽象。

    由 `parse_model_step` 从 JSON fence 块里抽取而成；`raw_response` 保留
    原始文本，方便后续回放/调试。
    """

    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


class ModelAdapter(Protocol):
    """鸭子类型协议：任何暴露 `complete(messages) -> str` 的对象都可充当模型。"""

    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


class OpenAIModelAdapter:
    """对 OpenAI 兼容 chat.completions 接口的包装。

    仅依赖 `openai` 官方 SDK；通过 `api_base` 切换后端（本地 vLLM、代理网关等）。
    每次 `complete` 都新建 client，避免跨线程共享 HTTP 连接导致的状态问题。
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

    def complete(self, messages: list[ModelMessage]) -> str:
        """发起一次 chat.completions 请求并返回首条 choice 的文本内容。"""
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
                messages=[
                    {"role": message.role, "content": message.content} for message in messages
                ],
                temperature=self.temperature,
            )
        except APIError as exc:
            # 把供应商错误统一包成 RuntimeError，便于 ReAct 循环的通用异常分支处理
            raise RuntimeError(f"Model request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        # content 为 None/非 str 通常意味着上游返回了 tool_calls 或多模态块，
        # 本适配器只支持纯文本响应
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


class ScriptedModelAdapter:
    """测试/调试专用：按预置列表依次返回响应，用完即抛错。

    真正的 OpenAI 调用在 CI 上不可靠（网络 + 额度），ReAct 循环的单元测试
    用它注入确定性的 JSON 输出。
    """

    def __init__(self, responses: list[str]) -> None:
        # 拷贝一份：避免外部修改列表影响播放顺序
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        # messages 对预置响应无用，显式 del 提示读者（也能在启用严格 linter 时消除未用警告）
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
