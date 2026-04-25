"""`create_run_id` 微秒级精度测试。

run_id 用作单级输出目录名；秒级精度会让连发 run（脚本/测试/CI）撞同一目录。
微秒精度（`%f` = 6 位）把窗口缩小到 1µs，足以避开任何同进程连发场景。
"""

from __future__ import annotations

from data_agent_baseline.run.runner import create_run_id


def test_create_run_id_distinguishes_adjacent_calls() -> None:
    # 100 次连发：实际跨度通常 >10µs，必然能产生至少 2 个不同字符串。
    # 旧实现（秒级）会把全部 100 个塌缩成 1 个；新实现（微秒级）应至少有 2 个。
    ids = {create_run_id() for _ in range(100)}
    assert len(ids) >= 2, f"create_run_id() collapsed 100 calls into {ids!r}; need microsecond precision"


def test_create_run_id_format_is_microsecond_utc() -> None:
    # 22 chars: YYYYMMDD(8) + T(1) + HHMMSS(6) + ffffff(6) + Z(1) = 22
    rid = create_run_id()
    assert len(rid) == 22, f"unexpected length {len(rid)}: {rid!r}"
    assert rid[8] == "T" and rid[-1] == "Z"
    # 微秒位（最后一个 Z 前的 6 位）必须全是数字
    micros = rid[15:21]
    assert micros.isdigit(), f"microsecond segment not numeric: {micros!r} in {rid!r}"
