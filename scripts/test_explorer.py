#!/usr/bin/env python3
"""Standalone test for the Explorer sub-agent.

Usage:
    uv run python scripts/test_explorer.py --config configs/react.local.yaml --task task_1
    uv run python scripts/test_explorer.py --config configs/react.local.yaml --task task_1 --max-steps 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.application import build_model_adapter  # noqa: E402
from agents.benchmark.dataset import DABenchPublicDataset  # noqa: E402
from agents.config import load_app_config  # noqa: E402
from agents.explorer.registry import create_explorer_tool_registry  # noqa: E402
from agents.explorer.runner import EXPLORER_MAX_STEPS, run_explorer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_explorer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Explorer sub-agent on a single task")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--task", required=True, help="Task ID (e.g. task_1)")
    parser.add_argument(
        "--max-steps", type=int, default=EXPLORER_MAX_STEPS, help="Max steps for Explorer"
    )
    parser.add_argument(
        "--protocol", default="native", choices=["native", "text"], help="Tool protocol"
    )
    args = parser.parse_args()

    config = load_app_config(Path(args.config))

    # Disable rate limiter for standalone test (it requires state_dir from runner)
    if config.agent.rate_limit is not None:
        from dataclasses import replace as dc_replace

        config = dc_replace(config, agent=dc_replace(config.agent, rate_limit=None))

    dataset = DABenchPublicDataset(config.dataset.root_path)
    task = dataset.get_task(args.task)

    print(f"\n{'=' * 70}")
    print(f"  Explorer Test: {task.task_id}")
    print(f"  Question: {task.question}")
    print(f"  Context dir: {task.context_dir}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'=' * 70}\n")

    # Build model adapter — Explorer reuses the same model as the main agent.
    # We need to pass tools for native protocol.
    registry = create_explorer_tool_registry()
    model = build_model_adapter(config, tools=registry)

    def step_callback(state):
        if not state.steps:
            return
        step = state.steps[-1]
        action = step.action
        ok_marker = "✓" if step.ok else "✗"
        obs_preview = json.dumps(step.observation, ensure_ascii=False)
        if len(obs_preview) > 400:
            obs_preview = obs_preview[:400] + "..."
        print(f"  [{ok_marker}] Step {step.step_index} (turn {step.turn_index}): {action}")
        if step.action_input:
            input_preview = json.dumps(step.action_input, ensure_ascii=False)
            if len(input_preview) > 200:
                input_preview = input_preview[:200] + "..."
            print(f"      input: {input_preview}")
        print(f"      obs: {obs_preview}")
        if step.reasoning:
            reasoning_preview = step.reasoning[:300].replace("\n", " ")
            if len(step.reasoning) > 300:
                reasoning_preview += "..."
            print(f"      reasoning: {reasoning_preview}")
        u = step.usage
        if u.total_tokens > 0:
            print(
                f"      tokens: prompt={u.prompt_tokens} completion={u.completion_tokens} "
                f"cached={u.cached_tokens} reasoning={u.reasoning_tokens}"
            )
        sys.stdout.flush()

    result = run_explorer(
        task=task,
        question=task.question,
        model=model,
        protocol=args.protocol,
        max_steps=args.max_steps,
        step_callback=step_callback,
    )

    print(f"\n{'=' * 70}")
    print("  Explorer Result")
    print(f"  Success: {result.success}")
    print(f"  Steps used: {result.steps_used}")
    if result.failure_reason:
        print(f"  Failure reason: {result.failure_reason}")
    print(f"{'=' * 70}")

    if result.success and result.findings:
        print("\n--- FINDINGS ---\n")
        # Try to pretty-print as JSON; fall back to raw string
        try:
            parsed = json.loads(result.findings)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(result.findings)
    elif not result.success:
        print("\n--- EXPLORER FAILED ---\n")
        print(f"Reason: {result.failure_reason}")


if __name__ == "__main__":
    main()
