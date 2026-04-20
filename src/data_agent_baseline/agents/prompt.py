"""ReAct prompt 模板与构造器。

本模块是 Agent 行为契约的单一真相源：
- `REACT_SYSTEM_PROMPT`:  告诉模型身份、访问权限与输出格式
- `RESPONSE_EXAMPLES`:    few-shot 示例（展示 JSON fence 的精确形态）
- `build_*`:              把工具描述/任务问题/观察结果拼接成实际消息内容

关键格式约束：响应必须是**单个** ```json 围栏块，且块内是**单个** JSON 对象，
包含 `thought / action / action_input` 三个键——`react.parse_model_step` 按此解析。
"""

from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


# ReAct 系统提示词基座：权限边界 + 输出契约
# 关键约束点：
#  - 只能通过工具读取 context/ 下文件（防止越权推理）
#  - 必须以 answer 工具结束（没有它，循环只能因 max_steps 被动终止）
#  - 响应格式严格，是为了让 parse_model_step 能稳定提取
REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()

# few-shot：展示首轮探索和最终 answer 两种形态
# 刻意分别给出一个 "未终止" 和一个 "终止" 的样例，避免模型只模仿其一
RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    """拼装系统消息：基础 prompt + 可用工具表 + 响应示例 + 格式硬约束。

    `tool_descriptions` 由 `ToolRegistry.describe_for_prompt()` 生成，
    其中每个工具含名字、用途和 input_schema，引导模型选对工具和入参。
    """
    # 允许外部注入自定义基础 prompt（eg. 针对评测集调优），缺省用 REACT_SYSTEM_PROMPT
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        # 最后再强调一次输出契约：多数失败都是模型漏了 fence 或加了多余解释
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    """任务起始的 user 消息。

    只注入问题本身与路径约定，其他上下文交由模型通过 `list_context` 等工具主动探索。
    """
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    """把工具执行结果以格式化 JSON 回传给模型作为下一轮的观察。

    ensure_ascii=False 保留中文等非 ASCII 字符原样可读；indent=2 让模型更易 parse。
    """
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
