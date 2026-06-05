"""Public API for the tracing subsystem."""

from data_agent_baseline.tracing.create import (
    agent_span,
    error_span,
    function_span,
    generation_span,
    get_current_span,
    get_current_trace,
    trace,
    turn_span,
)
from data_agent_baseline.tracing.processor_interface import TracingExporter, TracingProcessor
from data_agent_baseline.tracing.setup import get_trace_provider, set_trace_provider
from data_agent_baseline.tracing.span_data import (
    AgentSpanData,
    ErrorSpanData,
    FunctionSpanData,
    GenerationSpanData,
    SpanData,
    TurnSpanData,
)
from data_agent_baseline.tracing.spans import NoOpSpan, Span, SpanError, SpanImpl
from data_agent_baseline.tracing.traces import NoOpTrace, Trace, TraceImpl


def add_trace_processor(processor: TracingProcessor) -> None:
    get_trace_provider().register_processor(processor)


def set_tracing_disabled(disabled: bool = True) -> None:
    get_trace_provider().set_disabled(disabled)


def flush_traces() -> None:
    get_trace_provider().force_flush()


__all__ = [
    "AgentSpanData",
    "ErrorSpanData",
    "FunctionSpanData",
    "GenerationSpanData",
    "NoOpSpan",
    "NoOpTrace",
    "Span",
    "SpanData",
    "SpanError",
    "SpanImpl",
    "Trace",
    "TraceImpl",
    "TracingExporter",
    "TracingProcessor",
    "TurnSpanData",
    "add_trace_processor",
    "agent_span",
    "error_span",
    "flush_traces",
    "function_span",
    "generation_span",
    "get_current_span",
    "get_current_trace",
    "get_trace_provider",
    "set_trace_provider",
    "set_tracing_disabled",
    "trace",
    "turn_span",
]
