from __future__ import annotations

from pathlib import Path

from data_agent_baseline.agents.prompt import (
    build_native_system_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord


def _task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Find the requested value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_native_system_prompt_requires_tool_calls() -> None:
    """Behavior 段的核心契约：必须 tool_calls + 纯文本不送达。

    旧版 prompt 把这两点拆成 Rule 1 / Rule 2 同义复述；现已合并为一条。
    """
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    # 必须用 tool_calls 调用至少一个工具
    assert "must use tool_calls" in lower
    # 纯文本不会送达 runner（合并后的措辞）
    assert "plain text replies are not delivered to the runner" in lower
    assert "the answer is lost and the task fails" in lower

    # row-length 校验由 `AnswerInput` Pydantic schema 兜底，不再在 prompt 里复述
    assert "each row length must match" not in lower


def test_native_system_prompt_keeps_tool_calls_out_of_reasoning() -> None:
    """`<think>` 内的 tool_call draft 一律不被执行；prompt 必须把这一点讲透。

    commit 9e5a24f 把抽象的 "private scratchpad" 警告替换成了具体的失败示例 +
    Required flow + Self-check。本测试钉死新版 prompt 的关键短语，避免再次脱同步。
    """
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    # 核心契约：<think> 内的内容会被 runtime 丢弃
    assert "anything inside <think>...</think> is discarded by the runtime" in lower
    assert "never parsed, never extracted, and never executed" in lower

    # 显式列举常见失败形态（JSON / <tool_call> / <function=...>）
    assert "the single most common failure to avoid" in lower
    assert "<tool_call>...</tool_call> wrappers" in lower
    assert "<function=name><parameter=key>value</parameter></function>" in lower

    # 给出正向流程与自检
    assert "required flow on every turn" in lower
    assert "self-check" in lower

    # answer 工具是唯一的提交通道
    assert "the `answer` tool is your only way to submit" in lower


def test_native_system_prompt_includes_tool_decision_tree() -> None:
    prompt = build_native_system_prompt()

    for keyword in [
        "inspect_files",
        "read_csv",
        "read_json",
        "read_doc",
        "inspect_sqlite_schema",
        "execute_context_sql",
        "execute_python",
        "answer",
    ]:
        assert keyword in prompt


def test_native_system_prompt_custom_override_is_preserved() -> None:
    assert build_native_system_prompt("custom native prompt") == "custom native prompt"


def test_task_prompt_keeps_only_task_specific_context(tmp_path: Path) -> None:
    """user 消息只承载任务特定上下文：question 本身。

    跨任务的不变量（工具决策、tool_call 协议、`<think>` 内 draft 不执行、
    路径约定）全部由 system prompt + path 字段 schema description 承担。
    user 消息保持极简，最大化 prompt cache 命中率与 `Question:` 的注意力。
    """
    prompt = build_task_prompt(_task(tmp_path))
    lower = prompt.lower()

    # 仅承载 task-specific 的 question
    assert prompt == "Question: Find the requested value."

    # 不变量不应再在 user 消息里重复
    assert "make a tool call" not in lower
    assert "<think>" not in lower
    assert "inspect_files" not in lower
    assert "context/" not in lower  # 路径约定已上移至 system prompt


def test_system_prompts_carry_path_convention() -> None:
    """两套 system prompt 都必须显式声明 path 字段约定。

    路径约定是跨任务不变量；上移到 system prompt 后，user prompt 不再重复。
    `PATH_CONVENTION_NOTE`（tools/inputs.py）作为单一真相源，
    分别在 system prompt 与 5 个 path 字段 schema description 中复用。
    """
    from data_agent_baseline.tools.inputs import PATH_CONVENTION_NOTE

    for prompt in (
        build_native_system_prompt(),
        build_system_prompt("- inspect_files: Use first to inspect context."),
    ):
        assert "Path convention" in prompt
        assert PATH_CONVENTION_NOTE in prompt


def test_system_prompts_share_grouped_sections() -> None:
    """两套 system prompt 都用 `## <section>` 分组，模型 attention 可按段 routing。"""
    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    # 两协议共有的段落
    for section in ("## Behavior", "## Data conventions"):
        assert section in text_prompt
        assert section in native_prompt

    # text-only 段落
    assert "## Output format" in text_prompt
    # native-only 段落
    assert "## Decision tree" in native_prompt
    assert "## Output protocol" in native_prompt


def test_system_prompts_share_data_conventions_text() -> None:
    """`## Data conventions` 段在两个 prompt 必须字面一致——单一真相源。

    回归保护：commit 9e5a24f 等历史事故都源于"两个 prompt 各写一遍数据规则、
    改一处漏一处"。本测试通过共享常量 `_DATA_OUTPUT_RULES` 强制两份输出一致。
    """
    from data_agent_baseline.agents.prompt import _DATA_OUTPUT_RULES

    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    assert _DATA_OUTPUT_RULES in text_prompt
    assert _DATA_OUTPUT_RULES in native_prompt


def test_text_system_prompt_keeps_text_protocol_contract() -> None:
    prompt = build_system_prompt("- inspect_files: Use first to inspect context.")

    assert "single ```json fenced block" in prompt
    assert "tool_calls" not in prompt


def test_native_system_prompt_requires_source_schema_mirroring() -> None:
    """answer 必须沿用源数据列结构——拒绝把 first_name+last_name 合成 full_name。

    回归 task_19：question 字面是 "list the full name"，但 gold 期待两列
    `first_name, last_name`。prompt 必须显式告诉模型不要合成 derived 列。
    """
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "mirror the source schema" in lower
    assert "do not concatenate" in lower
    assert "full_name" in lower
    assert "first_name" in lower and "last_name" in lower


def test_native_system_prompt_requires_cell_type_preservation() -> None:
    """整数主键 ID 必须以 JSON int 提交——回归 prediction.csv 把 163109 写成 163109.0。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "preserve cell types" in lower
    assert "163109" in prompt and "163109.0" in prompt


def test_text_system_prompt_requires_source_schema_mirroring() -> None:
    """文本协议下同样要约束 schema mirroring + cell type preservation。"""
    prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    lower = prompt.lower()

    assert "mirror the source schema" in lower
    assert "do not concatenate" in lower
    assert "full_name" in lower
    assert "preserve cell types" in lower
    assert "163109" in prompt and "163109.0" in prompt


def test_native_system_prompt_distinguishes_result_from_filter_columns() -> None:
    """WH-target 才是 result column，过滤/排序用的列不许塞进 answer。

    回归 task_25：模型把 `total_cost` 这种"我用来比大小的列"塞进 answer，
    grader 按"无关列要罚"扣分。新规则要把"WH-target → result column"
    这条语义讲透，并显式提到 grader 会罚多余列。
    """
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "distinguish result columns from filter" in lower
    assert "wh-target" in lower
    assert "lowest cost" in lower
    assert "grader penalizes extra columns" in lower
    assert "when in doubt, output only the wh-target" in lower


def test_text_system_prompt_distinguishes_result_from_filter_columns() -> None:
    """文本协议下同样需要 result-vs-filter column 规则。"""
    prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    lower = prompt.lower()

    assert "distinguish result columns from filter" in lower
    assert "wh-target" in lower
    assert "grader penalizes extra columns" in lower
