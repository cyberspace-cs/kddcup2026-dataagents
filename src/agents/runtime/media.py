"""Task media helpers: video discovery, notice lines, and video-agent content."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.prompts import build_task_prompt

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
VIDEO_MAX_BYTES = 100 * 1024 * 1024


def find_videos(context_dir: Path) -> list[Path]:
    """Recurse into ``context/`` and return all video files, sorted for determinism."""
    return sorted(
        p for p in context_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_initial_user_content(task: PublicTask, *, preact: bool = False) -> str:
    """Build the initial user message content.

    视频不再内联（原始帧只进 video 子代理的上下文，见 agents/video/）；
    存在视频时仅追加一行文字提示，列出全部相对路径。
    """
    text = build_task_prompt(task, preact=preact)
    videos = find_videos(task.context_dir)
    if not videos:
        return text
    rels: list[str] = []
    for video_path in videos:
        try:
            rels.append(str(video_path.relative_to(task.context_dir)))
        except ValueError:
            rels.append(video_path.name)
    return (
        f"{text}\n\nThe task context contains video file(s): {', '.join(rels)}. "
        "Raw video is not attached to this conversation; use your available tools "
        "to have the video content analyzed."
    )


def build_video_user_content(text: str, video_path: Path) -> list[dict[str, Any]]:
    """Build multimodal user content for the video sub-agent: text + base64 data URI."""
    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    video_b64 = base64.standard_b64encode(video_path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": text},
        {"type": "video_url", "video_url": {"url": f"data:{mime_type};base64,{video_b64}"}},
    ]
