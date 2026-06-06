"""Task media attachment helpers for the ReAct runtime."""

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


def find_first_video(context_dir: Path) -> Path | None:
    """Recurse into ``context/`` and return the first video file, sorted for determinism."""
    candidates = sorted(
        p for p in context_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(candidates) > 1:
        logger.warning(
            "Multiple videos in %s; using %s, ignoring %s",
            context_dir.name,
            candidates[0].name,
            [c.name for c in candidates[1:]],
        )
    return candidates[0] if candidates else None


def build_initial_user_content(
    task: PublicTask, *, preact: bool = False
) -> str | list[dict[str, Any]]:
    """Build the initial user message content, attaching a video if one exists."""
    text = build_task_prompt(task, preact=preact)
    video_path = find_first_video(task.context_dir)
    if video_path is None:
        return text
    size = video_path.stat().st_size
    if size > VIDEO_MAX_BYTES:
        logger.warning(
            "Video %s skipped: %d bytes exceeds %d byte limit",
            video_path.name,
            size,
            VIDEO_MAX_BYTES,
        )
        return text
    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    video_b64 = base64.standard_b64encode(video_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime_type};base64,{video_b64}"
    try:
        relative_video_path = video_path.relative_to(task.context_dir)
    except ValueError:
        relative_video_path = video_path.name
    return [
        {
            "type": "text",
            "text": f"{text}\n\nA video file from the task context is attached: {relative_video_path}.",
        },
        {"type": "video_url", "video_url": {"url": data_url}},
    ]
