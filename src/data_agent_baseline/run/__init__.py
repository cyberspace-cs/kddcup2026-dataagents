"""Run 子包：任务运行调度、超时隔离与产物落盘。

对外暴露 `run_single_task` / `run_benchmark` 及产物句柄类型；内部细节
（子进程封装、CSV/JSON 写入辅助）不对外可见。
"""

from data_agent_baseline.run.runner import TaskRunArtifacts, create_run_id, run_benchmark, run_single_task

__all__ = [
    "TaskRunArtifacts",
    "create_run_id",
    "run_benchmark",
    "run_single_task",
]
