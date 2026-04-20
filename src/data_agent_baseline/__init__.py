"""Data Agent Baseline 包根。

仅暴露版本号；子模块（`agents` / `benchmark` / `tools` / `run`）按需单独 import。
保持根 __init__ 精简是为了让 CLI 启动更快（Typer 应用初始化时不拖入 OpenAI SDK）。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
