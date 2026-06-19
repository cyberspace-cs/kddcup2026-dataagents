"""Video sub-agent tool registry: a single terminal `report` tool.

校验经 `FunctionTool.input_model` 在 registry 边界强制——不依赖 vLLM
constrained decoding（`adapt_model_response` 的 recovered 调用路径会绕过它）。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.benchmark.schema import PublicTask
from agents.tools.registry import FunctionTool, ToolExecutionResult, ToolRegistry
from agents.tools.schema_normalize import normalize_for_vllm

# 报告序列化总量兜底（≈ schema 密集型 explorer findings 实测体量的 2 倍）：
# observation 会被逐字回放进 explorer 后续每一轮消息且消息层无截断，必须在执行边界拦截。
VIDEO_REPORT_MAX_BYTES = 64 * 1024


class VideoExtractedData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Configuration the video defines: field/column names, thresholds with "
            "operator and value, group-by / sort-by / top-N directives, scheme or "
            "batch IDs, statistical periods, and table names."
        ),
    )
    displayed_samples: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Numbers, distribution snapshots, TOP-N panels, preview rows, and other "
            "values the screen merely shows. These are not direct answer values."
        ),
    )


class VideoReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        max_length=4000,
        description="What the video shows overall: content type, scenes, purpose.",
    )
    timeline: list[dict[str, Any]] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description=(
            'Timestamped events, e.g. [{"time": "00:42", "event": "bar chart of Q3 '
            'revenue by category"}]. Cover the whole video. MUST be an array; use an '
            "empty array when there are no events. FORBIDDEN: null or one plain string."
        ),
    )
    extracted_data: VideoExtractedData = Field(
        default_factory=VideoExtractedData,
        description=(
            "Two top-level keys REQUIRED, even if one is empty: "
            "`rules` = the configuration the video defines (field/column names, "
            "thresholds with operator and value, group-by / sort-by / top-N, scheme or "
            "batch IDs, statistical periods, table names) — the QUERY to apply against "
            "real data; "
            "`displayed_samples` = numbers, distribution snapshots, TOP-N panels, "
            "preview rows the screen merely SHOWS — NOT the answer. "
            "MUST be a structured object. FORBIDDEN: quoted JSON text. "
            "Transcribe all numbers exactly as shown. Same value may appear in BOTH "
            "keys with a `warnings` note if you cannot tell which it is."
        ),
    )
    coverage: list[str] = Field(
        default_factory=list,
        description=(
            "Content you saw but did NOT fully transcribe (one item per entry), so the "
            "caller can request a targeted follow-up. MUST be an array of strings; use "
            "an empty array when there is no omitted content. FORBIDDEN: null or one "
            "plain string."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Blurry / truncated / uncertain readings, one atomic fact each. MUST be an "
            "array of strings; use an empty array when there are no warnings. FORBIDDEN: "
            "null or one plain string."
        ),
    )

    @field_validator("extracted_data", mode="before")
    @classmethod
    def _parse_stringified_extracted_data(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
        return value

    @field_validator("timeline", mode="before")
    @classmethod
    def _none_timeline_to_empty(cls, value: Any) -> Any:
        if value is None:
            return []
        return value

    @field_validator("coverage", "warnings", mode="before")
    @classmethod
    def _normalize_text_list_fields(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return value


VIDEO_REPORT_SCHEMA = normalize_for_vllm(VideoReportInput.model_json_schema())


def handle_video_report(_task: PublicTask, args: VideoReportInput) -> ToolExecutionResult:
    findings = json.dumps(args.model_dump(), ensure_ascii=False)
    if len(findings.encode("utf-8")) > VIDEO_REPORT_MAX_BYTES:
        return ToolExecutionResult(
            ok=False,
            content={
                "error": (
                    f"report payload exceeds {VIDEO_REPORT_MAX_BYTES} bytes; resubmit "
                    "with only question-relevant data and move omitted content into "
                    "`coverage` entries"
                )
            },
        )
    return ToolExecutionResult(
        ok=True,
        content={"status": "reported", "findings": findings},
        is_terminal=True,
    )


def create_video_tool_registry() -> ToolRegistry:
    report = FunctionTool(
        name="report",
        description=(
            "Submit your video analysis findings. This is your ONLY tool and the only "
            "way to complete the task — call it exactly once with complete findings. "
            "Nested fields MUST stay structured. FORBIDDEN: stringified objects or "
            "stringified arrays."
        ),
        json_schema=VIDEO_REPORT_SCHEMA,
        handler=handle_video_report,
        is_terminal=True,
        input_model=VideoReportInput,
    )
    return ToolRegistry(definitions={report.name: report})
