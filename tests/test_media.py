"""media.py：视频不再内联主对话；按需提示行；video agent 专用内容构造。"""

from __future__ import annotations

import base64
from pathlib import Path

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.runtime.media import (
    build_initial_user_content,
    build_video_user_content,
    find_videos,
)


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_no_video_returns_plain_text(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    content = build_initial_user_content(task)
    assert isinstance(content, str)
    assert "video" not in content.lower()


def test_video_present_adds_notice_without_inlining(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "demo.mp4").write_bytes(b"\x00\x01")
    content = build_initial_user_content(task)
    assert isinstance(content, str)  # 不再是多模态 list
    assert "demo.mp4" in content
    assert "base64" not in content


def test_multiple_videos_all_listed_sorted(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "b.mp4").write_bytes(b"\x00")
    sub = task.context_dir / "sub"
    sub.mkdir()
    (sub / "a.mov").write_bytes(b"\x00")
    videos = find_videos(task.context_dir)
    assert [v.name for v in videos] == ["b.mp4", "a.mov"] or [v.name for v in videos] == [
        "a.mov",
        "b.mp4",
    ]
    content = build_initial_user_content(task)
    assert "b.mp4" in content and "a.mov" in content


def test_build_video_user_content_attaches_data_uri(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    payload = b"\x00\x01\x02"
    video.write_bytes(payload)
    blocks = build_video_user_content("HELLO BRIEF", video)
    assert blocks[0] == {"type": "text", "text": "HELLO BRIEF"}
    url = blocks[1]["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.standard_b64encode(payload).decode("ascii") in url
