"""跨进程 Key 池限速适配器（PR-2）。

把 `OpenAIModelAdapter` / `NativeToolsOpenAIAdapter` 装饰成带 "RPM+TPM 双桶、429/5xx 重试、
预扣-调用-对账" 的限速壳。核心设计：

- **文件令牌桶（FileTokenBucket）**：每个 Key 一份状态文件 + 一份 `filelock` 锁，放在
  `RateLimitConfig.state_dir/bucket_<sha1(key)[:10]>.{json,lock}`。桶按线性刷新：
  `tokens = min(cap, tokens + elapsed * cap / 60)`，保证多子进程 / 多线程共享配额而不超限。
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

import contextlib
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from filelock import FileLock
from openai import APIError

if TYPE_CHECKING:
    from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelResponse
    from data_agent_baseline.config import RateLimitConfig


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


@dataclass(slots=True)
class _BucketState:
    """桶的内存表示；序列化为 state 文件里的 JSON。

    - rpm / tpm: 当前剩余令牌数（float 便于线性刷新）
    - last: 上次刷新的壁钟（time.time()）；跨进程用壁钟而不是 monotonic，因为子进程的
      monotonic 基准与父进程不一致
    """

    rpm: float
    tpm: float
    last: float


class FileTokenBucket:
    """基于 filelock + JSON 文件的跨进程令牌桶。

    单次 acquire/refund 的临界区内耗约为 "读 1KB + 写 1KB + fsync"，在本地 SSD 上通常 <1ms。
    即便 16 workers 挤一把锁，P99 也只是增加个位 ms；不会成为瓶颈。

    状态文件不存在时视为满桶（首次启动/新 run_id）；文件损坏时按满桶重建，避免
    "上次 crash 留下半截 JSON" 拖垮整次运行。
    """

    def __init__(
        self,
        *,
        state_path: Path,
        lock_path: Path,
        rpm_capacity: int,
        tpm_capacity: int,
    ) -> None:
        if rpm_capacity <= 0 or tpm_capacity <= 0:
            raise ValueError("rpm_capacity and tpm_capacity must be positive.")
        self._state_path = state_path
        # filelock 接受 str 路径
        self._lock = FileLock(str(lock_path))
        self._rpm_cap = int(rpm_capacity)
        self._tpm_cap = int(tpm_capacity)
        # 父目录可能还不存在——acquire/refund 都会走这个路径，提前建好
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    # -- 私有工具 -----------------------------------------------------------------

    def _load_or_init(self) -> _BucketState:
        """读磁盘状态；不存在或损坏时按满桶初始化。"""
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text())
                return _BucketState(
                    rpm=float(raw.get("rpm", self._rpm_cap)),
                    tpm=float(raw.get("tpm", self._tpm_cap)),
                    last=float(raw.get("last", time.time())),
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                # 损坏的状态文件当满桶处理；下面的 _save 会覆盖
                pass
        return _BucketState(rpm=float(self._rpm_cap), tpm=float(self._tpm_cap), last=time.time())

    def _save(self, state: _BucketState) -> None:
        # 写临时文件 + os.replace 实现原子替换：并发读者只会看到旧值或新值，
        # 永远不会读到撕裂的半截 JSON。tempfile.mkstemp 用 O_CREAT|O_EXCL 拿到唯一文件名，
        # 防止外部预放置 symlink 把写入引向 state_dir 之外。
        payload = {"rpm": state.rpm, "tpm": state.tpm, "last": state.last}
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self._state_path.name}.", suffix=".tmp", dir=self._state_path.parent
        )
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp_name, self._state_path)
        except BaseException:
            # rename 失败时清理临时文件，避免目录里留垃圾
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise

    def _refill(self, state: _BucketState) -> _BucketState:
        """按壁钟线性补充令牌；负 elapsed（时钟回拨）视作 0。"""
        now = time.time()
        elapsed = max(0.0, now - state.last)
        state.rpm = min(float(self._rpm_cap), state.rpm + elapsed * self._rpm_cap / 60.0)
        state.tpm = min(float(self._tpm_cap), state.tpm + elapsed * self._tpm_cap / 60.0)
        state.last = now
        return state

    # -- 公共 API -----------------------------------------------------------------

    def acquire(self, *, need_rpm: int, need_tpm: int) -> float:
        """尝试消费 `(need_rpm, need_tpm)`；成功返 0.0，失败返应等待秒数。

        失败时不消费、仅把刷新后的状态落盘（让其他等待者也能读到最新时间戳）。调用方
        收到 >0 的返回值时应 `time.sleep(wait + jitter)` 后再试。
        """
        with self._lock:
            state = self._refill(self._load_or_init())
            if state.rpm >= need_rpm and state.tpm >= need_tpm:
                state.rpm -= need_rpm
                state.tpm -= need_tpm
                self._save(state)
                return 0.0
            # 不够：计算还差多少令牌、转成秒数（桶的刷新速率是 cap/60 令牌/秒）
            wait_r = (need_rpm - state.rpm) * 60.0 / self._rpm_cap if need_rpm > 0 else 0.0
            wait_t = (need_tpm - state.tpm) * 60.0 / self._tpm_cap if need_tpm > 0 else 0.0
            self._save(state)
            return max(0.0, wait_r, wait_t)

    def refund(self, *, rpm: int, tpm: int) -> None:
        """把预扣过但实际未消耗的令牌退回桶里；封顶在 capacity，不会溢出。"""
        if rpm <= 0 and tpm <= 0:
            return
        with self._lock:
            state = self._refill(self._load_or_init())
            state.rpm = min(float(self._rpm_cap), state.rpm + max(0, rpm))
            state.tpm = min(float(self._tpm_cap), state.tpm + max(0, tpm))
            self._save(state)


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
        self._rl = rate_limit
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
        reserved_tpm = estimate_prompt_tokens(messages) + self._rl.reserve_output_tokens

        # Step 1: 预扣 (1 RPM, reserved_tpm)；循环直到拿到令牌
        self._acquire_or_sleep(need_rpm=1, need_tpm=reserved_tpm)

        # Step 2: 调用内层；遇 429/5xx 退避重试，其他异常直抛
        backoff = self._rl.backoff_initial_seconds
        # 循环 max_retries+1 次（初次 + 最多 max_retries 次重试）
        for attempt in range(self._rl.max_retries + 1):
            try:
                response = self.inner.complete(messages)
            except RuntimeError as exc:
                if not _is_retriable(exc):
                    # 不可重试错误：退回预扣的 TPM（没真正生成 token），RPM 不退（请求已打到服务端）
                    self.bucket.refund(rpm=0, tpm=reserved_tpm)
                    raise
                if attempt == self._rl.max_retries:
                    self.bucket.refund(rpm=0, tpm=reserved_tpm)
                    raise RateLimitExhausted(
                        f"Rate limit retries exhausted after {attempt + 1} attempts: {exc}"
                    ) from exc
                retry_after = _parse_retry_after(exc)
                if retry_after is not None:
                    sleep_s = retry_after
                else:
                    sleep_s = min(backoff, self._rl.backoff_max_seconds)
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
