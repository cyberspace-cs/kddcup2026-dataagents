"""工具分发层。

把"派发一次 tool_call"从 ReAct 主循环里抽出来：driver 不再同时关心
"调哪个工具 / 工具是否终止 / 工具异常如何记账"三件事。

`dispatch_tool_call` 的契约（D3）：
- 工具名经 `registry.execute` 分发；未知名直接 `KeyError`，由 driver 捕获记 `tool_error`
- handler 抛任何异常 → 原样上抛（driver 决定怎么记），**不**短路
- handler 返回成功 → 读取 `registry.definitions[name].is_terminal` 作为终止权威
- 终止条件：`definition.is_terminal AND result.ok`，二者必须同时满足
  （即"调用 answer 工具"且"answer payload 校验通过"才终结循环；
   answer 校验失败时 handler 抛 ValueError，走 tool_error 路径并继续）
"""

from __future__ import annotations

from agents.benchmark.schema import PublicTask
from agents.llm.types import ModelToolCall
from agents.tools.registry import ToolExecutionResult, ToolRegistry


def dispatch_tool_call(
    task: PublicTask,
    call: ModelToolCall,
    registry: ToolRegistry,
) -> tuple[ToolExecutionResult, bool]:
    """执行一次 tool_call 并返回 `(result, should_terminate)`。

    `should_terminate` 仅在 **同时** 满足以下两个条件时为真（D3）：
    - 注册表元数据 `definition.is_terminal=True`（构造期固定，仅 `answer` 工具）
    - handler 成功返回且 `result.ok=True`

    任何 handler 异常都会原样上抛——driver 应捕获并记 `tool_error` 事件，循环继续。
    """
    result = registry.execute(task, call.name, call.arguments)
    definition = registry.definitions[call.name]
    should_terminate = definition.is_terminal and result.ok
    return result, should_terminate
