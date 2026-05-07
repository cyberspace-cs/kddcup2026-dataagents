"""单任务调试前端的 Streamlit 主入口。

页面结构（top-down）：
1. Sidebar:    Run 选择 + Task 选择 + 累计 token / tool 频次 mini chart
2. 任务头卡:    question + verdict + (gold | prediction | side-by-side) tabs + system prompt 折叠
3. ReAct 时间线: 每步一个 expander，含 4 个 tab（思考 / Prompt / 工具 / Response）

启动方式：
    uv run dabench view <run_id> --task <task_id>

CLI 通过 `streamlit run streamlit_app.py -- --run-id <id> --task-id <id>` 透传参数；
本文件用 `argparse` 解析 `sys.argv` 末尾的额外参数。

`st.query_params` 用作"URL 状态同步"：刷新页面后保留 run/task 选择。
"""

# Streamlit 官方 stubs 在 `st.dataframe` / `st.line_chart` / `st.bar_chart` 等
# 重载签名里把 Snowpark/连接器类型写成 `Unknown`（参见调用处错误信息），strict
# 模式会逐个 call 报 reportUnknownMemberType。这是 streamlit 上游 stub 的限制，
# 不在我们代码控制内——文件级关掉这一规则；其它 strict 检查（参数类型/变量
# 类型/不必要 isinstance 等）继续生效。
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

from data_agent_baseline.ui.compare import CompareResult, compare
from data_agent_baseline.ui.loaders import (
    RunInfo,
    TaskInfo,
    list_runs,
    list_tasks_in_run,
    load_gold,
    load_prediction,
    load_trace,
)

# ----------------------------- CLI 参数 -----------------------------


def _parse_extra_args() -> argparse.Namespace:
    """从 `sys.argv` 解析 `streamlit run -- --run-id X --task-id Y` 后的额外参数。

    streamlit 会把 `--` 后的所有参数透传到 `sys.argv`，本函数容错解析。
    """
    parser = argparse.ArgumentParser(description="DABench single-task debug UI")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--public-data-dir", type=str, default=None)
    # parse_known_args 容错 streamlit 自身参数
    args, _ = parser.parse_known_args()
    return args


def _project_root() -> Path:
    """推断项目根：本文件位于 `<root>/src/data_agent_baseline/ui/`。"""
    return Path(__file__).resolve().parents[3]


# ----------------------------- 类型 helpers -----------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    """JSON 反序列化值若为 dict 则规约成 dict[str, Any]，否则返回 {}。

    pyright strict 把 isinstance(any, dict) narrowing 后视作 dict[Unknown, Unknown]，
    走 cast 一次性把 trace.json 解析出来的 dict 收敛到统一类型，避免上层每个 .get
    都触发 reportUnknown* 系列警告。
    """
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """同 `_as_dict`，但收敛到 list[Any]。"""
    return cast(list[Any], value) if isinstance(value, list) else []


# ----------------------------- 渲染 helpers -----------------------------


_VERDICT_BADGE = {
    "match": ":green[✅ MATCH]",
    # 列子集匹配：评测判定正确，但有多余列/缺失列触发惩罚——用橙色提示"过得去但不完美"
    "column_subset_match": ":orange[🟡 COLUMN SUBSET MATCH]",
    "value_mismatch": ":orange[⚠️ VALUE MISMATCH]",
    "shape_mismatch": ":red[❌ SHAPE MISMATCH]",
    "no_gold": ":gray[— NO GOLD]",
    "no_prediction": ":red[❌ NO PREDICTION]",
}


def _render_verdict(result: CompareResult) -> None:
    """把 CompareResult 渲染成一行短文案 + 形状 + 列匹配摘要。"""
    badge = _VERDICT_BADGE.get(result.status, result.status)
    parts: list[str] = [badge]
    if result.gold_shape is not None and result.pred_shape is not None:
        parts.append(
            f"gold={result.gold_shape[0]}×{result.gold_shape[1]}, "
            f"pred={result.pred_shape[0]}×{result.pred_shape[1]}"
        )
    elif result.gold_shape is not None:
        parts.append(f"gold={result.gold_shape[0]}×{result.gold_shape[1]}")
    elif result.pred_shape is not None:
        parts.append(f"pred={result.pred_shape[0]}×{result.pred_shape[1]}")
    if result.status == "value_mismatch" and result.cell_diff_count > 0:
        parts.append(f"diff_cells={result.cell_diff_count}")
    if result.status == "column_subset_match":
        # 关键摘要：matched/N gold 列 + extras（惩罚因子）
        n_gold = result.gold_shape[1] if result.gold_shape else 0
        parts.append(f"matched={result.matched_columns}/{n_gold} gold cols")
        if result.extra_pred_columns > 0:
            parts.append(f"extras={result.extra_pred_columns} pred cols (penalty)")
        if result.missing_gold_columns > 0:
            parts.append(f"missing={result.missing_gold_columns} gold cols")
    st.markdown(" · ".join(parts))


def _render_run_picker(runs: list[RunInfo], default_run_id: str | None) -> RunInfo | None:
    """侧栏 Run 选择器；返回选中的 RunInfo（无 run 时返回 None）。"""
    if not runs:
        st.sidebar.warning("No runs found under artifacts/runs/.")
        return None

    options = [info.run_id for info in runs]
    default_index = 0
    if default_run_id and default_run_id in options:
        default_index = options.index(default_run_id)

    selected_id = st.sidebar.selectbox("Run", options, index=default_index)
    selected = next(info for info in runs if info.run_id == selected_id)

    # 子标签：成败计数 + 总数
    if selected.task_count is not None and selected.succeeded_count is not None:
        st.sidebar.caption(f"{selected.succeeded_count}/{selected.task_count} succeeded")
    return selected


def _render_task_picker(tasks: list[TaskInfo], default_task_id: str | None) -> TaskInfo | None:
    """侧栏 Task 选择器；带 ✅/❌ 前缀。"""
    if not tasks:
        st.sidebar.warning("This run contains no tasks.")
        return None

    def _label(info: TaskInfo) -> str:
        prefix = "✅" if info.succeeded else "❌"
        return f"{prefix} {info.task_id}"

    options = [info.task_id for info in tasks]
    default_index = 0
    if default_task_id and default_task_id in options:
        default_index = options.index(default_task_id)

    selected_id = st.sidebar.selectbox(
        "Task",
        options,
        index=default_index,
        format_func=lambda tid: _label(next(t for t in tasks if t.task_id == tid)),
    )
    return next(t for t in tasks if t.task_id == selected_id)


def _render_token_chart(steps: list[dict[str, Any]]) -> None:
    """侧栏：累计 token 折线图 + tool 频次柱状图。"""
    if not steps:
        return
    cumulative_tokens: list[int] = []
    running = 0
    for step in steps:
        usage = _as_dict(step.get("usage"))
        running += int(usage.get("total_tokens", 0))
        cumulative_tokens.append(running)
    st.sidebar.markdown("**Cumulative tokens**")
    st.sidebar.line_chart(pd.DataFrame({"tokens": cumulative_tokens}))

    tool_counts: dict[str, int] = {}
    for step in steps:
        action = step.get("action")
        if isinstance(action, str):
            tool_counts[action] = tool_counts.get(action, 0) + 1
    if tool_counts:
        st.sidebar.markdown("**Tool calls**")
        st.sidebar.bar_chart(pd.DataFrame({"count": tool_counts}))


def _render_table_tabs(gold: pd.DataFrame | None, pred: pd.DataFrame | None) -> None:
    """三 Tab：gold / prediction / side-by-side。"""
    tab_gold, tab_pred, tab_side = st.tabs(["gold.csv", "prediction.csv", "side-by-side"])

    with tab_gold:
        if gold is not None:
            st.dataframe(gold, use_container_width=True)
        else:
            st.info("No gold.csv available for this task.")

    with tab_pred:
        if pred is not None:
            st.dataframe(pred, use_container_width=True)
        else:
            st.warning("No prediction.csv — Agent did not submit an answer.")

    with tab_side:
        if gold is None or pred is None:
            st.info("Side-by-side requires both gold and prediction.")
            return
        col_gold, col_pred = st.columns(2)
        with col_gold:
            st.markdown("**gold**")
            st.dataframe(gold, use_container_width=True)
        with col_pred:
            st.markdown("**prediction**")
            st.dataframe(pred, use_container_width=True)


def _render_step_expander(step: dict[str, Any]) -> None:
    """单步 expander：显示 4 个 Tab（思考 / Prompt / 工具 / Response）。"""
    step_index = step.get("step_index", "?")
    action = step.get("action", "?")
    ok = step.get("ok", False)
    usage = _as_dict(step.get("usage"))
    total_tokens = int(usage.get("total_tokens", 0))
    label = f"Step {step_index} · [{action}] · {total_tokens} tok · {'✓' if ok else '✗'}"

    with st.expander(label, expanded=False):
        tab_thought, tab_prompt, tab_tool, tab_response = st.tabs(
            ["🧠 思考", "📤 Prompt", "🔧 工具", "📥 Response"]
        )

        with tab_thought:
            reasoning = step.get("reasoning") or ""
            thought = step.get("thought") or ""
            if reasoning:
                st.markdown("**reasoning_content**")
                st.markdown(f"```\n{reasoning}\n```")
            if thought:
                st.markdown("**thought**")
                st.markdown(thought)
            if not reasoning and not thought:
                st.caption("(no thought captured for this step)")

        with tab_prompt:
            prompt_messages = _as_list(step.get("prompt_messages"))
            if prompt_messages:
                _render_prompt_messages(prompt_messages)
            else:
                st.caption(
                    "(no prompt_messages — likely a non-lead step in a parallel "
                    "tool_call turn, or trace produced before this feature shipped)"
                )

        with tab_tool:
            st.markdown(f"**action:** `{action}`")
            action_input = _as_dict(step.get("action_input"))
            observation = _as_dict(step.get("observation"))
            _render_tool_call(action, action_input, observation)

        with tab_response:
            response_message = step.get("response_message")
            if response_message:
                st.json(response_message)
            else:
                st.caption("(no response_message — non-lead step or model_error path)")


# 工具名 → 主代码字段 + 高亮语言。覆盖现有 registry 中所有产生"长文本"参数的工具，
# 让人类阅读时不必和 JSON 转义字符（\n / \t / \"）较劲。
_CODE_TOOL_LANGUAGES: dict[str, tuple[str, str]] = {
    "execute_python": ("code", "python"),
    "execute_context_sql": ("query", "sql"),
}


def _render_action_input(action: str, action_input: dict[str, Any]) -> None:
    """`action_input` 智能渲染：

    - 若工具属于 `_CODE_TOOL_LANGUAGES` 且对应字段是字符串，则用 `st.code()` 高亮代码，
      其余字段保持 `st.json()` 折叠
    - 否则统一用 `st.json()`
    """
    code_field_lang = _CODE_TOOL_LANGUAGES.get(action)
    if code_field_lang is None:
        st.json(action_input)
        return
    code_field, language = code_field_lang
    code_value = action_input.get(code_field)
    if not isinstance(code_value, str):
        # 该工具但没拿到字符串代码：fallback 到原始 JSON
        st.json(action_input)
        return

    st.markdown(f"**{code_field}** ({language})")
    st.code(code_value, language=language)
    # 把代码字段以外的其他参数仍以 JSON 形式展示，避免丢失 timeout / params 等
    other = {k: v for k, v in action_input.items() if k != code_field}
    if other:
        st.markdown("**其他参数:**")
        st.json(other)


def _render_observation(action: str, observation: dict[str, Any]) -> None:
    """`observation` 智能渲染：

    - 错误对象（observation.ok=False）：突出 error 文案 + 其他字段 JSON
    - `execute_python` 成功：把 stdout / stderr / traceback 展开成 plain text 块，
      其它字段（success/result_table 等）走 JSON
    - `execute_context_sql`：把 rows 走表格渲染（如果有），其它字段走 JSON
    - 其它工具：默认 `st.json()`
    """
    ok = observation.get("ok")
    if ok is False:
        # 错误：把 error 字段单独高亮成红色提示，便于一眼看到失败原因
        error_text = observation.get("error")
        if isinstance(error_text, str) and error_text:
            st.error(error_text)
        leftover = {k: v for k, v in observation.items() if k not in {"ok", "error"}}
        if leftover:
            st.markdown("**其他字段:**")
            st.json(leftover)
        return

    content_raw = observation.get("content")
    if not isinstance(content_raw, dict):
        st.json(observation)
        return
    content = cast(dict[str, Any], content_raw)

    if action == "execute_python":
        _render_python_observation(content)
    elif action == "execute_context_sql":
        _render_sql_observation(content)
    else:
        st.json(observation)


def _render_python_observation(content: dict[str, Any]) -> None:
    """`execute_python` 返回体：stdout / stderr / traceback 用 code block 展示。"""
    plain_fields = ("stdout", "stderr", "traceback")
    handled: set[str] = set()
    for field_name in plain_fields:
        value = content.get(field_name)
        if isinstance(value, str) and value:
            st.markdown(f"**{field_name}**")
            # stderr/traceback 用 text 语言；stdout 也走 text 避免误高亮成 python
            st.code(value, language="text")
            handled.add(field_name)
    leftover = {k: v for k, v in content.items() if k not in handled}
    if leftover:
        st.markdown("**其他字段:**")
        st.json(leftover)


def _render_sql_observation(content: dict[str, Any]) -> None:
    """`execute_context_sql` 返回体：把 rows 渲染成 dataframe（如果有 columns + rows）。"""
    columns_raw = content.get("columns")
    rows_raw = content.get("rows")
    if (
        isinstance(columns_raw, list)
        and isinstance(rows_raw, list)
        and all(isinstance(c, str) for c in cast(list[Any], columns_raw))
    ):
        columns = cast(list[str], columns_raw)
        rows = cast(list[Any], rows_raw)
        try:
            df = pd.DataFrame(rows, columns=columns)
            st.markdown("**rows**")
            st.dataframe(df, use_container_width=True)
        except (ValueError, TypeError):
            # 降级：rows 形状异常时回到 JSON
            pass
    leftover = {k: v for k, v in content.items() if k not in {"columns", "rows"}}
    if leftover:
        st.markdown("**其他字段:**")
        st.json(leftover)


def _render_tool_call(
    action: str, action_input: dict[str, Any], observation: dict[str, Any]
) -> None:
    """🔧 工具 Tab 的整体编排：先 action_input，再 observation，按工具名做格式化。"""
    st.markdown("**action_input:**")
    _render_action_input(action, action_input)
    st.markdown("**observation:**")
    _render_observation(action, observation)


def _render_prompt_messages(messages: list[Any]) -> None:
    """把一组 messages 渲染成 chat-bubble 风格。"""
    for idx, raw in enumerate(messages):
        if not isinstance(raw, dict):
            continue
        msg = cast(dict[str, Any], raw)
        role = str(msg.get("role", "?"))
        content = msg.get("content", "")
        # 用 streamlit 的 chat_message 控件，role 不能任意——映射到内置 4 角色
        bubble_role = {
            "system": "assistant",
            "user": "user",
            "assistant": "assistant",
            "tool": "assistant",
        }.get(role, "assistant")
        with st.chat_message(bubble_role):
            st.caption(f"#{idx} · role={role}")
            if isinstance(content, str) and content:
                # tool 消息体多为 JSON 字符串，尝试解析以便折叠展示
                if role == "tool":
                    try:
                        parsed = json.loads(content)
                        st.json(parsed)
                    except json.JSONDecodeError:
                        st.markdown(content)
                else:
                    st.markdown(content)
            tool_calls = _as_list(msg.get("tool_calls"))
            if tool_calls:
                st.markdown("**tool_calls**")
                _render_tool_calls_pretty(tool_calls)
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                st.caption(f"tool_call_id: {tool_call_id}")


def _render_tool_calls_pretty(tool_calls: list[Any]) -> None:
    """把 OpenAI 风格的 tool_calls 列表渲染得对人类更友好。

    OpenAI tool_calls 长这样：
        {"id": "...", "type": "function",
         "function": {"name": "execute_python", "arguments": "<JSON 字符串>"}}
    `arguments` 是 JSON 字符串，里面包了 `code` 等字段，原始展示会一堆 `\\n` 转义。

    本函数把每个 call 拆开，code 字段用 `st.code()` 高亮，其余 args 用 JSON 折叠。
    无法识别的 call 形态走 `st.json()` 兜底。
    """
    for idx, raw in enumerate(tool_calls):
        if not isinstance(raw, dict):
            st.json(raw)
            continue
        call = cast(dict[str, Any], raw)
        function = _as_dict(call.get("function"))
        name = function.get("name", "?")
        st.caption(f"tool_call #{idx} · {name} · id={call.get('id', '?')}")

        # arguments 可能是 JSON 字符串或已经解析好的 dict
        raw_args = function.get("arguments")
        parsed_args: dict[str, Any] | None = None
        if isinstance(raw_args, dict):
            parsed_args = cast(dict[str, Any], raw_args)
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                candidate = json.loads(raw_args)
                if isinstance(candidate, dict):
                    parsed_args = cast(dict[str, Any], candidate)
            except json.JSONDecodeError:
                parsed_args = None

        if parsed_args is None:
            # 解析失败：原样展示
            st.json(call)
            continue

        _render_action_input(name if isinstance(name, str) else "", parsed_args)


# ----------------------------- main -----------------------------


def main() -> None:
    """Streamlit 主流程（被 `streamlit run` 调用）。"""
    args = _parse_extra_args()
    project_root = _project_root()

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else project_root / "artifacts"
    public_data_dir = (
        Path(args.public_data_dir) if args.public_data_dir else project_root / "data" / "public"
    )

    st.set_page_config(page_title="DABench Trace Viewer", layout="wide")
    st.title("DABench · 单任务调试查看器")

    # 优先级：URL query > CLI 参数；这样刷新页面也保留状态
    query_params = st.query_params
    initial_run_id = query_params.get("run") or args.run_id
    initial_task_id = query_params.get("task") or args.task_id

    runs = list_runs(artifacts_dir)
    selected_run = _render_run_picker(runs, initial_run_id)
    if selected_run is None:
        st.stop()

    tasks = list_tasks_in_run(selected_run.run_dir)
    selected_task = _render_task_picker(tasks, initial_task_id)
    if selected_task is None:
        st.stop()

    # 同步 URL（便于 share 链接）
    st.query_params["run"] = selected_run.run_id
    st.query_params["task"] = selected_task.task_id

    trace = load_trace(selected_task.task_dir)
    if trace is None:
        st.error(f"trace.json not found under {selected_task.task_dir}")
        st.stop()

    gold = load_gold(selected_task.task_id, public_data_dir)
    pred = load_prediction(selected_task.task_dir)
    compare_result = compare(gold, pred)

    # ---- 任务头卡 ----
    question = _load_question(public_data_dir, selected_task.task_id)
    if question:
        st.markdown(f"**Question:** {question}")
    if selected_task.failure_reason:
        st.error(f"failure_reason: {selected_task.failure_reason}")
    _render_verdict(compare_result)

    _render_table_tabs(gold, pred)

    system_prompt = trace.get("system_prompt")
    if isinstance(system_prompt, str) and system_prompt:
        with st.expander("系统提示词 (system prompt)", expanded=False):
            st.markdown(f"```\n{system_prompt}\n```")

    st.divider()
    st.subheader("ReAct 时间线")

    steps = _as_list(trace.get("steps"))

    # 侧栏图表（依赖 steps 中的 usage）
    typed_steps: list[dict[str, Any]] = [
        cast(dict[str, Any], s) for s in steps if isinstance(s, dict)
    ]
    _render_token_chart(typed_steps)

    if not typed_steps:
        st.info("No steps recorded for this task.")
        return
    for step in typed_steps:
        _render_step_expander(step)


def _load_question(public_data_dir: Path, task_id: str) -> str | None:
    """读取 `<public_data_dir>/input/<task_id>/task.json` 里的 question 字段。"""
    task_json_path = public_data_dir / "input" / task_id / "task.json"
    if not task_json_path.exists():
        return None
    try:
        payload = json.loads(task_json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        question = cast(dict[str, Any], payload).get("question")
        if isinstance(question, str):
            return question
    return None


# Streamlit 通过 `streamlit run streamlit_app.py` 启动时会把本文件作为 `__main__` 执行；
# import 时不触发 main()，便于其他模块按需引用 helpers 而不副作用启动 UI。
if __name__ == "__main__":
    main()
