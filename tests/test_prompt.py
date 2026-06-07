from __future__ import annotations

from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.prompts import (
    REACT_NATIVE_SYSTEM_PROMPT,
    build_native_system_prompt,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)


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


@pytest.mark.parametrize("enable_thinking", [None, True])
def test_native_system_prompt_keeps_tool_calls_out_of_reasoning(
    enable_thinking: bool | None,
) -> None:
    """`<think>` 内的 tool_call draft 一律不被执行；prompt 必须把这一点讲透。

    commit 9e5a24f 把抽象的 "private scratchpad" 警告替换成了具体的失败示例 +
    Required flow + Self-check。本测试钉死新版 prompt 的关键短语，避免再次脱同步。

    参数化锁住保守缺省：`enable_thinking` 为 None（未配置）或 True（thinking-on）
    时都必须注入警告段。仅 False 时才省略——见 `..._omits_think_warning_when_thinking_disabled`。
    """
    prompt = build_native_system_prompt(enable_thinking=enable_thinking)
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


def test_native_system_prompt_omits_think_warning_when_thinking_disabled() -> None:
    """`enable_thinking=False` 时省略 `<think>` 警告段（约 240 tokens）。

    动机：`_NATIVE_OUTPUT_PROTOCOL` 专治 Qwen3-thinking-on 的失败模式
    （commit 9e5a24f），但对显式关闭 thinking 的配置（GPT/Claude/非-thinking
    Qwen）来说，警告既冗余又占 token。`build_native_system_prompt` 在
    `enable_thinking is False` 时跳过这段。

    本测试断言警告段的关键短语**不**出现，并把 `## Output protocol` 段头一并
    检查（杜绝"段头还在但内容被剥"的中间状态）。
    """
    prompt = build_native_system_prompt(enable_thinking=False)
    lower = prompt.lower()

    # `## Output protocol` 段头与警告关键短语都不应出现
    assert "## output protocol" not in lower
    assert "anything inside <think>...</think> is discarded by the runtime" not in lower
    assert "the single most common failure to avoid" not in lower

    # 但其它段——Behavior / Decision tree / Path convention——必须仍在
    assert "## Behavior" in prompt
    assert "## Decision tree" in prompt
    assert "Path convention" in prompt


def test_react_native_system_prompt_constant_matches_default_builder() -> None:
    """模块级常量 `REACT_NATIVE_SYSTEM_PROMPT` 必须等于 `build_native_system_prompt()`。

    向后兼容钉子：早期 callers（notebooks、自定义 A/B 配置）直接 import 常量；
    重构后常量被 `build_native_system_prompt()`（默认 enable_thinking=None）
    动态生成，保守缺省下两者必须 byte-equal。
    """
    assert build_native_system_prompt() == REACT_NATIVE_SYSTEM_PROMPT
    assert build_native_system_prompt(enable_thinking=None) == REACT_NATIVE_SYSTEM_PROMPT


def test_native_system_prompt_includes_tool_decision_tree() -> None:
    prompt = build_native_system_prompt()

    for keyword in [
        "explore",
        "read_csv",
        "read_json",
        "read_doc",
        "inspect_sqlite_schema",
        "execute_context_sql",
        "execute_python",
        "answer",
    ]:
        assert keyword in prompt


def test_native_system_prompt_front_loads_knowledge_md() -> None:
    """`knowledge.md` must be consumed during the explore phase.

    The `explore` sub-agent now reads `knowledge.md` automatically, so the
    prompt tells the main agent to use the findings directly. The test
    verifies that knowledge.md handling is addressed in the decision tree.
    """
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "knowledge.md" in prompt
    assert "code↔label mappings" in prompt or "code-label mappings" in lower
    # explore already handles knowledge.md — prompt must say so
    assert "explore" in lower
    assert "knowledge" in lower


def test_native_system_prompt_warns_execute_python_is_stateless() -> None:
    """`execute_python` 子进程之间不共享状态——必须把读文件和计算合并到一次调用。

    回归同一个 trace：模型对 59 MB / 1M 行的 trans.csv 做了 5 次 `pd.read_csv`，
    在 5 个独立 execute_python step 里反复 I/O。新规则在 execute_python bullet 里
    显式声明 fresh subprocess + 合并 read/filter/projection/verification。
    """
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    # 子进程隔离的硬声明
    assert "fresh subprocess" in lower
    assert "not carried over" in lower
    # 行动指令：把读 + 过滤 + 投影 + 验证合并到一次调用
    assert "combine read + filter + projection + verification into one call" in lower


def test_execute_python_tool_description_pins_sqlite_in_python_idiom() -> None:
    """`.db` 文件的两条合法路径都在 `execute_python` 描述中列出。"""
    from agents.tools.registry import create_default_tool_registry

    desc = create_default_tool_registry().definitions["execute_python"].description
    lower = desc.lower()

    assert "sqlite3.connect(path)" in desc
    assert "pd.read_sql_query(query, conn)" in desc
    assert "sqlite:///" in lower
    assert "sqlalchemy" in lower


def test_native_system_prompt_enforces_format_tolerance() -> None:
    """题/数格式不一致时按最粗共有形截断/归一匹配。

    回归 task_13 早期失效模式（run 20260430-008）：模型把 `0:01:54` 与数据
    `1:54.455` 死磕字面相等，找不到匹配后又把"closest numeric match"
    当兜底，错选 number=44。

    2026-04-30 调整：规则从 `## Decision tree` 段（routing 信号）搬到
    `DATA_OUTPUT_RULES`（数据规约）。Decision tree 里其余 8 条都是
    "If X → call Y" 的工具路由，混入值比较规则会稀释路由 attention；
    迁到 `DATA_OUTPUT_RULES` 后还顺带让文本协议首次拿到这条防护
    （Decision tree 是 `_NATIVE_HEADER` 独享段，文本协议看不到）。

    曾经一并写过的 "singular tolerance" 实测（run 008/010/011/012 共四轮）
    无效——模型每次都能围绕"the X 单数 → 挑较好的一个"合理化绕过任何禁令
    措辞，所以删除以减少 prompt 噪声。Singular 引发的多行答案目前作为
    已知 model-capability 边界。
    """
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    # 格式不一致时归一/截断到共有形
    assert "format tolerance" in lower
    assert "normalize or truncate" in lower
    assert "coarsest shared form" in lower
    # 不允许"closest numeric match"作为字面不匹配的兜底
    assert 'do not fall back to "closest numeric match"' in lower


def test_text_system_prompt_enforces_format_tolerance() -> None:
    """文本协议同样需要 format tolerance 防护。

    迁移前文本协议完全没有这条规则（Decision tree 是 `_NATIVE_HEADER`
    独享段）。规则现在落在 `DATA_OUTPUT_RULES`，两协议共享，
    `test_system_prompts_share_data_conventions_text` 已锁定字面一致；
    本测试是面向规则本身的语义钉子，删常量也能立刻定位回归点。
    """
    prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    lower = prompt.lower()

    assert "format tolerance" in lower
    assert "normalize or truncate" in lower
    assert "coarsest shared form" in lower
    assert 'do not fall back to "closest numeric match"' in lower


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
    from agents.prompts.rules import PATH_CONVENTION_NOTE

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
    改一处漏一处"。本测试通过共享常量 `DATA_OUTPUT_RULES` 强制两份输出一致。
    """
    from agents.prompts import DATA_OUTPUT_RULES

    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    assert DATA_OUTPUT_RULES in text_prompt
    assert DATA_OUTPUT_RULES in native_prompt


def test_system_prompts_share_task_profile_and_grader_note() -> None:
    """`TASK_PROFILE` + `GRADER_NOTE` 在两套 prompt 中都必须出现，且字面一致。

    取代旧版仅讲身份的 `_INTRO`，把 success criterion（通过 `answer` 提交）
    + 访问边界（仅 `context/`）+ grader 罚多余列/近似值，三条都前置到 prompt
    起首。两套 prompt 共享同一段文本，避免"改 native 漏改 text"的双写事故。
    """
    from agents.prompts import GRADER_NOTE, TASK_PROFILE

    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    assert TASK_PROFILE in text_prompt
    assert TASK_PROFILE in native_prompt
    assert GRADER_NOTE in text_prompt
    assert GRADER_NOTE in native_prompt


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


def test_system_prompts_use_runtime_clock_for_current_year() -> None:
    """`current year` 语义必须来自 execute_python 运行时，而不是数据最大日期。"""
    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    for prompt in (text_prompt, native_prompt):
        lower = prompt.lower()
        assert "runtime date" in lower
        assert "current year" in lower
        assert "execute_python" in prompt
        assert "date.today().year" in prompt
        assert "do not infer the current year from the latest date in the dataset" in lower


def test_native_system_prompt_avoids_hardcoded_medical_ranges() -> None:
    """abnormal lab filters must be evidence-driven, not prompt-provided lab constants."""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "clinical / domain" in lower
    assert "local evidence" in lower
    assert "last-resort fallback" in lower
    assert "never assume both low and high values are abnormal" in lower

    forbidden = [
        "creatinine normal",
        "wbc normal",
        "fibrinogen/fg normal",
        "150–400",
        "150-400",
        "0.7–1.3",
        "0.7-1.3",
        "3.5–9.0",
        "3.5-9.0",
    ]
    for phrase in forbidden:
        assert phrase not in lower


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


def test_native_system_prompt_requires_full_numeric_precision() -> None:
    """计算得到的浮点必须保留全精度——回归 task_10 把 60.77956989247312 压成 60.78。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "preserve full numeric precision" in lower
    assert "60.77956989247312" in prompt and "60.78" in prompt
    assert "round only when the question explicitly asks" in lower


def test_text_system_prompt_requires_full_numeric_precision() -> None:
    """文本协议下同样需要"计算值不许擅自舍入"规则。"""
    prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    lower = prompt.lower()

    assert "preserve full numeric precision" in lower
    assert "60.77956989247312" in prompt and "60.78" in prompt
    assert "round only when the question explicitly asks" in lower


def test_system_prompts_keep_distinct_separate_from_group_by() -> None:
    """DISTINCT depends on entity vs record; GROUP BY is for aggregates only."""
    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    for prompt in (text_prompt, native_prompt):
        lower = prompt.lower()
        assert "distinct depends on what the question asks about" in lower
        assert "deduplicate by entity identity" in lower
        assert "each log entry" in lower
        assert "do not add distinct" in lower
        assert "group by is for grouped aggregates, not deduplication" in lower
        assert '"by", "per", "for each"' in lower
        assert "do not use group by solely to remove duplicate rows" in lower
        assert '"tally" rule below' in lower


def test_system_prompts_prefer_null_safe_anti_join_patterns() -> None:
    """不存在记录的过滤应避开裸 NOT IN 的 NULL 陷阱。"""
    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    for prompt in (text_prompt, native_prompt):
        lower = prompt.lower()
        assert '"does not have x" / "no x" filter semantics' in lower
        assert "prefer `not exists` or `left join ... where right.key is null`" in lower
        assert "use `not in` only after filtering null keys from the subquery" in lower


def test_native_system_prompt_clarifies_knowledge_md_conflict_precedence() -> None:
    """knowledge.md 的公式/映射优先，但必须能套到观测到的 schema/data。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "`knowledge.md` is authoritative where it explicitly defines a field" in lower
    assert "observed data wins on conflict" in lower
    assert "use that exact formula/mapping if it applies to the observed schema/data" in lower
    assert "if sampling proves the mapping cannot apply or conflicts with actual values" in lower


def test_system_prompts_advertise_artifact_handoff() -> None:
    """两套 system prompt 都必须告知模型 from_csv artifact 协议 + DABENCH_ANSWER_DIR。

    `DATA_OUTPUT_RULES` 是单一真相源，所以两套 prompt 拿到的措辞必须字面一致；
    本断言锁住关键短语，避免 prompt 改动只覆盖一份的回归。
    """
    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    for prompt in (text_prompt, native_prompt):
        # artifact 工作流必须出现两个关键关键词
        assert "from_csv" in prompt
        assert "DABENCH_ANSWER_DIR" in prompt
        # 阈值"30 行 / 200 cell"是模型何时切 artifact 的判定线
        assert "30+ rows" in prompt and "200+ cells" in prompt
        # verification print 是 prompt 的另一条新规
        lower = prompt.lower()
        assert "print a verification snapshot" in lower
        assert "before calling `answer`" in lower


def test_system_prompts_force_wh_target_column_projection() -> None:
    """artifact bullet 必须强制按 WH-target 决策列集 + pandas-first 习惯。

    回归 run 20260429T210729531308Z task_8：from_csv 性能优化生效（87s→24s），
    但模型用 `csv.DictWriter` 手动列出 trans.csv 全部 10 列写进 artifact，
    prediction.csv 多出 9 列（其中 3 列全为空：k_symbol/bank/account）。
    上一版 prompt 同时给 pandas / csv.DictWriter / 手写 dict 三套 API 写指引，
    不仅占 token，csv/手写两条路也是 task_8 那种"枚举源列"故障的温床。新规则改为：
    (1)"先按 WH-target 决定列集，库无关"；(2) **强制 pandas only**，三段式
    （`filtered = ...` → `answer_df = filtered[[...]]` → `answer_df.to_csv(...)`），
    让列投影独占一条赋值语句，与模型自然书写习惯对齐；(3) `csv.DictWriter` 与
    手写 dict rows 直接被禁令封禁，prompt 里不再给它们的"如何用"指引。
    """
    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    for prompt in (text_prompt, native_prompt):
        lower = prompt.lower()
        # 核心契约：列集来自 WH-target，不来自源 schema
        assert "decide the column set first from the question's wh-target alone" in lower
        assert "not from the source schema" in lower
        # pandas only：三段式中间变量（贴合模型自然书写习惯，
        # 让 `answer_df = filtered[[...]]` 这一行独占列投影的视觉位置）
        assert "use pandas" in lower
        assert "filtered = df[df['col'] == ...]" in prompt
        assert "answer_df = filtered[['target_col_a', 'target_col_b']]" in prompt
        assert "answer_df.to_csv(out_path, index=False)" in prompt
        # csv.DictWriter / 手写 dict 必须被显式禁令封禁（不只是不推荐，是"do NOT use"）
        assert "do not use `csv.dictwriter` or hand-build dict rows" in lower
        # 唯一保留的 NEVER：pandas 整表写出（这是 pandas 路径自身的失败模式）
        assert "NEVER `df.to_csv(...)` on the whole frame" in prompt


def test_system_prompts_mandate_two_verification_self_checks() -> None:
    """verification bullet 必须给出强制 self-check：all-None 列 + 列数 vs 题面计数 + content-entity。

    回归同一个 run：模型实际 PRINT 了 verification snapshot（k_symbol/bank/account
    在 sample rows 都是 ''），但仍然提交了那 10 列。说明"仅打印"不够——必须把
    "看到 → 删掉 → 重跑"写成强制流程，并加一条"列数 vs 题面计数"反向校验。
    """
    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    for prompt in (text_prompt, native_prompt):
        lower = prompt.lower()
        # 触发关键词：'' / None / NaN 全空列必须删
        assert "mandatory self-checks" in lower
        assert "across all sampled rows" in lower
        assert "passthrough source column" in lower
        # 给出具体例子（k_symbol/bank/account），不只是抽象描述
        assert "k_symbol" in prompt or "withdrawal" in lower
        # 反向计数 self-check（b）
        assert "count the things it explicitly names you should output" in lower
        # 反对"以防万一多带几列"
        assert 'do not "include extra columns just in case"' in lower


# ---------- build_observation_prompt: ok parameter & recovery hints ----------


def test_observation_prompt_ok_none_no_hint() -> None:
    """ok=None (default) does not append recovery hint."""
    obs: dict[str, object] = {"error": "no such column: foo"}
    result = build_observation_prompt(obs)
    assert "Recovery hint" not in result


def test_observation_prompt_ok_true_no_hint() -> None:
    """ok=True does not append recovery hint."""
    obs: dict[str, object] = {"error": "no such column: foo"}
    result = build_observation_prompt(obs, ok=True)
    assert "Recovery hint" not in result


def test_observation_prompt_ok_false_no_such_column() -> None:
    """ok=False + 'no such column' -> schema recovery hint."""
    obs: dict[str, object] = {"error": "no such column: amount"}
    result = build_observation_prompt(obs, ok=False)
    assert "Recovery hint:" in result
    assert "inspect_sqlite_schema" in result


def test_observation_prompt_ok_false_no_such_table() -> None:
    obs: dict[str, object] = {"error": "no such table: orders"}
    result = build_observation_prompt(obs, ok=False)
    assert "Recovery hint:" in result
    assert "inspect_sqlite_schema" in result


def test_observation_prompt_ok_false_syntax_error() -> None:
    obs: dict[str, object] = {"error": 'near "SELEC": syntax error'}
    result = build_observation_prompt(obs, ok=False)
    assert "Recovery hint:" in result
    assert "Fix the syntax" in result


def test_observation_prompt_ok_false_timeout() -> None:
    obs: dict[str, object] = {"error": "execution timed out after 30s"}
    result = build_observation_prompt(obs, ok=False)
    assert "Recovery hint:" in result
    assert "Simplify" in result


def test_observation_prompt_ok_false_answer_tool_call_json_parse_error() -> None:
    obs: dict[str, object] = {
        "error": "model.complete failed: Failed to parse tool_call arguments for 'answer': bad json"
    }
    result = build_observation_prompt(obs, ok=False)
    assert "Recovery hint:" in result
    assert "too large or malformed" in result
    assert "DABENCH_ANSWER_DIR" in result
    assert "from_csv" in result


def test_observation_prompt_ok_false_generic_error() -> None:
    obs: dict[str, object] = {"error": "some unexpected failure"}
    result = build_observation_prompt(obs, ok=False)
    assert "Recovery hint:" in result
    assert "Review the error" in result


# ---------- Disambiguation rules ----------


def test_system_prompts_share_disambiguation_rules() -> None:
    """`## Disambiguation` 段在两个 prompt 必须字面一致——单一真相源。

    歧义检测与消解规则是跨协议的不变量，`DISAMBIGUATION_RULES` 常量
    被两套 prompt 引用，避免双写脱同步。
    """
    from agents.prompts import DISAMBIGUATION_RULES

    text_prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    native_prompt = build_native_system_prompt()

    assert DISAMBIGUATION_RULES in text_prompt
    assert DISAMBIGUATION_RULES in native_prompt


def test_disambiguation_covers_four_ambiguity_types() -> None:
    """Disambiguation 段必须覆盖四类歧义及其消解策略。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "## disambiguation" in lower
    assert "schema ambiguity" in lower
    assert "value ambiguity" in lower
    assert "vague intent" in lower
    assert "numeric reference ambiguity" in lower


def test_disambiguation_requires_column_first_resolution() -> None:
    """Column-first 策略：先识别列再反推表——对属性驱动查询更高效。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "column-first resolution" in lower
    assert "identify the target column first" in lower
    assert "join path" in lower


def test_disambiguation_anchors_on_knowledge_md() -> None:
    """领域知识从 knowledge.md 消解歧义，是首选来源。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "domain knowledge from `knowledge.md` is the primary disambiguation source" in lower
    assert "read it before attempting any filter" in lower


def test_disambiguation_prefers_non_empty_result() -> None:
    """歧义耗尽后偏好非空结果集——空结果几乎总是消解错误。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "prefer the interpretation that returns a non-empty result set" in lower
    assert "re-examine your column/value choice before submitting" in lower


def test_text_prompt_includes_disambiguation_section() -> None:
    """文本协议同样包含 ## Disambiguation 段。"""
    prompt = build_system_prompt("- inspect_files: Use first to inspect context.")
    lower = prompt.lower()

    assert "## disambiguation" in lower
    assert "schema ambiguity" in lower
    assert "column-first resolution" in lower


def test_decision_tree_phase2_includes_disambiguation_check() -> None:
    """Phase 2 VERIFY 必须包含歧义检查步骤。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "disambiguation check" in lower
    assert "sample all candidates before choosing" in lower


# ---------- Decision tree phase keywords ----------


def test_native_system_prompt_has_phase_keywords() -> None:
    """Decision tree three-phase protocol keywords."""
    prompt = build_native_system_prompt()
    lower = prompt.lower()
    assert "phase 1" in lower
    assert "phase 2" in lower
    assert "phase 3" in lower
    assert "discover" in lower
    assert "verify" in lower


def test_verifier_distinguishes_output_from_ranking_columns() -> None:
    """Verifier Check 2 must not flag ranking/filter criteria as missing columns."""
    from agents.verification.answer import _VERIFIER_SYSTEM

    lower = _VERIFIER_SYSTEM.lower()
    assert "ranking criterion, not a requested output" in lower
    assert "filter/ranking criterion" in lower
    assert "which event has the highest cost" in lower
    assert "only count attributes the question asks\nyou to return" in lower


def test_prompts_scope_superlative_row_count_to_global_questions() -> None:
    """Global superlative row-count rules must not reject grouped extrema."""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "global singular superlative" in lower
    assert "one row per group" in lower
    assert "for each" in lower
    assert "per group/category/region" in lower
    assert "by category/group" in lower

    from agents.verification.answer import _VERIFIER_SYSTEM

    verifier_lower = _VERIFIER_SYSTEM.lower()
    assert "global singular superlative" in verifier_lower
    assert "one row per" in verifier_lower
    assert "for each" in verifier_lower


def test_native_prompt_allows_single_call_subquery_verification() -> None:
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "subquery / intermediate aggregation verification" in lower
    assert "doing both inside the same `execute_python` call is fine" in lower
