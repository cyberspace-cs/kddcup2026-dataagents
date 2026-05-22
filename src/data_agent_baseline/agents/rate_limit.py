"""跨进程 Key 池限速适配器（PR-2）。

把 `OpenAIModelAdapter` / `NativeToolsOpenAIAdapter` 装饰成带 "RPM+TPM 双桶、429/5xx 重试、
预扣-调用-对账" 的限速壳。核心设计：

- **文件令牌桶**：实现拆到 `agents/token_bucket.py`（`FileTokenBucket`）。每个 Key 一份
  状态文件 + 一份 `filelock` 锁，路径在
  `RateLimitConfig.state_dir/bucket_<sha1(key)[:10]>.{json,lock}`。
- **预扣与对账**：调用前按 `(1 RPM, estimate_prompt + reserve_output)` 预扣；收到真正的
  `response.usage.total_tokens` 后按差值补扣或退款（本模块唯一能摊销估算误差的位置）。
- **429/5xx 重试**：解析内层 `RuntimeError.__cause__` 的 `APIError`，优先遵循 `Retry-After`
  header，否则按指数退避；超过 `max_retries` 抛 `RateLimitExhausted`，由 ReAct 层按现有
  `__error__` step 逻辑落盘。

注意事项：
- 状态文件和锁文件必须落在**本地文件系统**（ext4/tmpfs），NFS 下 `flock` 语义未定义。
- runner 必须在外层用 `dataclasses.replace` 把 `RateLimitConfig.state_dir` 从 None 解成
  具体路径后再构造本适配器；这里不再回退到 `None` 默认路径，避免隐式行为。
- 测试通过注入 fake 的 `FakeBucket` / Scripted 内层适配器覆盖分支；外部注入 model 的
  in-process 路径（即测试 fixture 中的 Scripted 适配器）**不会**被本壳包裹。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from typing import TYPE_CHECKING, Any, cast

from openai import APIError

from data_agent_baseline.agents.token_bucket import FileTokenBucket

if TYPE_CHECKING:
    from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelResponse
    from data_agent_baseline.config import RateLimitConfig

__all__ = [
    "FileTokenBucket",
    "RateLimitExhausted",
    "RateLimitedAdapter",
    "estimate_prompt_tokens",
]


class RateLimitExhausted(RuntimeError):
    """重试耗尽时抛出；`react.py` 的 `except Exception` 会把它记录为 `__error__` step。

    使用独立子类而非裸 `RuntimeError`，测试能精确断言 `pytest.raises(RateLimitExhausted)`，
    运维在日志里 grep 也能和普通 SDK 错误区分开。
    """


def estimate_prompt_tokens(messages: list[ModelMessage]) -> int:
    """粗估 prompt tokens = bytes / 2.5。

    选 2.5：Chinese 1.5 / English 4 / Code 3 的混合。偏高（估出的 token 数比真实多）比
    偏低更安全——估小了会击穿 TPM 引起 429；估大了只是多占点桶容量，对账时会被退还。

    覆盖 ModelMessage 的所有有效负载字段：
    - role（每条 ~4 bytes 的 "system"/"user"/"assistant"/"tool" 开销 + 格式符）
    - content（主要 payload）
    - tool_call_id（native 下 tool 响应消息必填，~36 bytes UUID）
    - tool_calls[].function.{name, arguments}（assistant 发起工具调用时的原生 payload）
    """
    total_bytes = 0
    for message in messages:
        total_bytes += len(message.role.encode("utf-8")) + 4  # role + 小量格式符开销
        if message.content:
            total_bytes += len(message.content.encode("utf-8"))
        if message.tool_call_id:
            total_bytes += len(message.tool_call_id.encode("utf-8"))
        for tool_call in message.tool_calls or []:
            function_payload: Any = tool_call.get("function") or {}
            if isinstance(function_payload, dict):
                fn_payload = cast(dict[str, Any], function_payload)
                total_bytes += len(str(fn_payload.get("name", "")).encode("utf-8"))
                # arguments 在 OpenAI SDK 侧已经是 JSON 字符串；其他实现兜底成 dict→json
                args = fn_payload.get("arguments", "")
                if isinstance(args, str):
                    total_bytes += len(args.encode("utf-8"))
                else:
                    total_bytes += len(json.dumps(args, ensure_ascii=False).encode("utf-8"))
    return max(1, int(total_bytes / 2.5))


# ------- 错误分类辅助：走 __cause__ 链拿到最底层的 openai.APIError -----------------


def _walk_cause(exc: BaseException) -> APIError | None:
    """在 exc.__cause__ 链里找第一个 `openai.APIError`；内层适配器 `raise RuntimeError(...) from exc` 会把它挂在 __cause__ 上。"""
    cur: BaseException | None = exc
    while cur is not None:
        if isinstance(cur, APIError):
            return cur
        cur = cur.__cause__
    return None


def _is_retriable(exc: BaseException) -> bool:
    """判定是否 429/5xx/连接错误，值得退避重试。

    优先 APIError.status_code；缺状态码（连接/超时错误）时看类名；完全没有 APIError 挂载时
    回退到消息字串匹配（"429" / "rate"）—— 尽量宽容，避免因 SDK 版本差异错过重试。
    """
    cause = _walk_cause(exc)
    if cause is not None:
        status = getattr(cause, "status_code", None)
        if status == 429:
            return True
        if isinstance(status, int) and 500 <= status < 600:
            return True
        # APIConnectionError / APITimeoutError 通常没 status_code
        name = type(cause).__name__.lower()
        return "connection" in name or "timeout" in name
    # 没挂 APIError（完全包裹进 str 的情况）：消息启发式
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg


def _parse_retry_after(exc: BaseException) -> float | None:
    """从底层 APIError.response.headers 里抠出 `Retry-After`（秒）；缺失返回 None。"""
    cause = _walk_cause(exc)
    if cause is None:
        return None
    response = getattr(cause, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        return None
    value: Any = None
    # httpx.Headers 与普通 dict 都支持 .get；但个别 mock 可能不支持
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        # RFC 7231 里 Retry-After 也可以是 HTTP-date 格式；为简化不解析，视作缺失
        return None


# ------- 装饰器本体 ----------------------------------------------------------------


class RateLimitedAdapter:
    """把任意 `ModelAdapter` 包一层 "预扣/调用/对账/重试" 的限速壳。

    - Key 已在上游按 `task_index % len(api_keys)` 固定到 `inner.api_key`；本类只需按
      `(api_key, state_dir)` 派生桶文件路径，不感知 "池"。
    - 桶容量 = `cap * safety_factor`，留 15% buffer 兜住估算误差 + 突发。
    - 预扣只做一次（整个 complete 调用期间），即便 429/5xx 触发了内部重试也不再预扣；
      理由见模块顶注。

    使用方在 runner 侧通过 `build_model_adapter` 决定是否包壳；外部注入 model 的路径
    （测试 fixture）绕过本壳。
    """

    def __init__(
        self,
        *,
        inner: ModelAdapter,
        api_key: str,
        rate_limit: RateLimitConfig,
    ) -> None:
        if rate_limit.state_dir is None:
            # 严格约束：由 runner 在 dataclasses.replace 时填充 state_dir
            raise ValueError(
                "RateLimitConfig.state_dir must be resolved before constructing "
                "RateLimitedAdapter; the runner is responsible for filling it in."
            )
        if not api_key:
            raise ValueError("RateLimitedAdapter requires a non-empty api_key.")

        self.inner = inner
        self.api_key = api_key
        self.rate_limit_config = rate_limit
        # 透传 inner 的 `enable_thinking` 元数据。本壳不参与 thinking 行为本身——
        # `complete` 只把 messages 转交给 inner——但 `ReActAgent._system_message`
        # 会 introspect `model.enable_thinking` 决定是否注入 `<think>` 警告段
        # （见 `prompt.build_native_system_prompt`）。不转发会导致 wrapper 路径下
        # 永远走"未指定→保守缺省→注入"分支，即使 config 显式配了 `False`。
        self.enable_thinking = getattr(inner, "enable_thinking", None)
        # Key 哈希取前 10 位：2^40 空间，2-4 个 Key 碰撞概率可忽略；文件名也短些
        key_hash = hashlib.sha1(api_key.encode("utf-8")).hexdigest()[:10]
        state_path = rate_limit.state_dir / f"bucket_{key_hash}.json"
        lock_path = rate_limit.state_dir / f"bucket_{key_hash}.lock"
        # 实际容量 = 名义上限 * safety_factor，留 buffer 吸收估算误差和突发
        rpm_cap = max(1, int(rate_limit.rpm_per_key * rate_limit.safety_factor))
        tpm_cap = max(1, int(rate_limit.tpm_per_key * rate_limit.safety_factor))
        self.bucket = FileTokenBucket(
            state_path=state_path,
            lock_path=lock_path,
            rpm_capacity=rpm_cap,
            tpm_capacity=tpm_cap,
        )

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        """预扣 → 调 inner → 对账 → 返回；429/5xx 走内部重试，其他异常原样上抛。"""
        reserved_tpm = (
            estimate_prompt_tokens(messages) + self.rate_limit_config.reserve_output_tokens
        )

        # Step 1: 预扣 (1 RPM, reserved_tpm)；循环直到拿到令牌
        self._acquire_or_sleep(need_rpm=1, need_tpm=reserved_tpm)

        # Step 2: 调用内层；遇 429/5xx 退避重试，其他异常直抛
        backoff = self.rate_limit_config.backoff_initial_seconds
        # 循环 max_retries+1 次（初次 + 最多 max_retries 次重试）
        for attempt in range(self.rate_limit_config.max_retries + 1):
            try:
                response = self.inner.complete(messages)
            except RuntimeError as exc:
                if not _is_retriable(exc):
                    # 不可重试错误：退回预扣的 TPM（没真正生成 token），RPM 不退（请求已打到服务端）
                    self.bucket.refund(rpm=0, tpm=reserved_tpm)
                    raise
                if attempt == self.rate_limit_config.max_retries:
                    self.bucket.refund(rpm=0, tpm=reserved_tpm)
                    raise RateLimitExhausted(
                        f"Rate limit retries exhausted after {attempt + 1} attempts: {exc}"
                    ) from exc
                retry_after = _parse_retry_after(exc)
                if retry_after is not None:
                    sleep_s = retry_after
                else:
                    sleep_s = min(backoff, self.rate_limit_config.backoff_max_seconds)
                # 抖动：避免多 worker 同步重试造成雪崩
                time.sleep(sleep_s + random.uniform(0.0, 0.5))
                backoff *= 2
                continue

            # Step 3: 成功——用真实 usage 做 TPM 对账
            actual_tpm = response.usage.total_tokens
            delta = actual_tpm - reserved_tpm
            if delta > 0:
                # 实际比预扣多——补扣（不再需要 RPM，只动 TPM）
                self._acquire_or_sleep(need_rpm=0, need_tpm=delta)
            elif delta < 0:
                # 实际比预扣少——把富余的 TPM 退还
                self.bucket.refund(rpm=0, tpm=-delta)
            return response

        # 理论上不可达：循环必定走 return 或 raise；保险起见再兜一层
        raise RateLimitExhausted("Rate limit retry loop exited without return.")

    def _acquire_or_sleep(self, *, need_rpm: int, need_tpm: int) -> None:
        """阻塞直到 acquire 返 0.0；每次失败按 `wait + 小抖动` 休眠后重试。"""
        while True:
            wait = self.bucket.acquire(need_rpm=need_rpm, need_tpm=need_tpm)
            if wait <= 0.0:
                return
            # 抖动防并发 worker 在刷新边界同时唤醒
            time.sleep(wait + random.uniform(0.0, 0.05))
