"""`FileTokenBucket._save` 原子性测试。

`_save` 必须用 "写临时文件 + os.replace" 保证并发读者要么读到旧的完整 JSON，
要么读到新的完整 JSON；绝不能读到截断或半写的字节。

测试方法：多个 writer 线程持续 acquire/refund（间接触发 _save），同时一个 reader
线程绕过锁直接读状态文件并 `json.loads`，断言所有非空读取都是合法 JSON。
旧实现（直接 write_text）会被本测试稳定捕获——open(W) 截断与后续 write 之间存在
撕裂窗口。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from data_agent_baseline.agents.rate_limit import FileTokenBucket


def test_concurrent_reads_observe_only_valid_json(tmp_path: Path) -> None:
    bucket = FileTokenBucket(
        state_path=tmp_path / "bucket.json",
        lock_path=tmp_path / "bucket.lock",
        rpm_capacity=600,
        tpm_capacity=10_000,
    )
    # 触发首次落盘，让 reader 不会一直读到空文件
    bucket.acquire(need_rpm=1, need_tpm=10)

    state_path = tmp_path / "bucket.json"
    stop = threading.Event()
    parse_failures: list[str] = []
    successful_reads = 0
    reads_lock = threading.Lock()

    def writer() -> None:
        # 反复 refund + acquire：每次都触发一次 _save，制造高频写
        while not stop.is_set():
            bucket.refund(rpm=1, tpm=10)
            bucket.acquire(need_rpm=1, need_tpm=10)

    def reader() -> None:
        nonlocal successful_reads
        while not stop.is_set():
            try:
                raw = state_path.read_text()
            except FileNotFoundError:
                # os.replace 期间窗口理论上不存在（rename 是原子的），
                # 但 read_text 有可能在文件刚被替换的瞬间命中——容忍
                continue
            if not raw:
                # 撕裂的旧实现会出现这种情况（write_text 先 truncate 后写）
                parse_failures.append("empty file observed")
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                parse_failures.append(f"json error: {exc!r} on raw={raw!r}")
                continue
            # schema 也得对（防止以后 _save 写错字段）
            if not {"rpm", "tpm", "last"}.issubset(payload):
                parse_failures.append(f"missing keys in {payload!r}")
                continue
            with reads_lock:
                successful_reads += 1

    writers = [threading.Thread(target=writer) for _ in range(4)]
    readers = [threading.Thread(target=reader) for _ in range(2)]
    for t in writers + readers:
        t.start()
    # 短而密的窗口：足够触发上千次写/读，又不至于让测试跑太慢
    time.sleep(0.5)
    stop.set()
    for t in writers + readers:
        t.join(timeout=5.0)
        assert not t.is_alive(), "thread failed to exit"

    assert parse_failures == [], f"observed torn writes: {parse_failures[:5]}"
    # 没读到任何东西意味着调度太偏，测试本身失效——给个下限断言
    assert successful_reads > 50, f"only {successful_reads} reads—too sparse to be meaningful"
