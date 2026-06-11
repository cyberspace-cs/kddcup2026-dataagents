"""Task media helpers: video discovery and video-agent content."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
VIDEO_MAX_BYTES = 100 * 1024 * 1024


def find_videos(context_dir: Path) -> list[Path]:
    """Recurse into ``context/`` and return all video files, sorted for determinism."""
    return sorted(
        p for p in context_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_video_user_content(text: str, video_path: Path) -> list[dict[str, Any]]:
    """Build multimodal user content for the video sub-agent: text + base64 data URI."""
    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    video_b64 = base64.standard_b64encode(video_path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": text},
        {"type": "video_url", "video_url": {"url": f"data:{mime_type};base64,{video_b64}"}},
    ]
