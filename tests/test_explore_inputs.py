"""ExploreInput 与 ExploreVideoInput 的契约。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.tools.inputs import EXPLORE_VIDEO_SCHEMA, ExploreInput, ExploreVideoInput


def test_explore_input_takes_no_params() -> None:
    ExploreInput.model_validate({})
    with pytest.raises(ValidationError):
        ExploreInput.model_validate({"unknown": 1})


def test_explore_video_input_requires_path_and_instructions() -> None:
    ok = ExploreVideoInput.model_validate(
        {"path": "demo.mp4", "instructions": "transcribe the bar chart"}
    )
    assert ok.path == "demo.mp4"
    with pytest.raises(ValidationError):
        ExploreVideoInput.model_validate({"path": "demo.mp4"})
    with pytest.raises(ValidationError):
        ExploreVideoInput.model_validate({"instructions": "x"})


def test_explore_video_schema_forbids_extras() -> None:
    assert EXPLORE_VIDEO_SCHEMA["additionalProperties"] is False
    assert set(EXPLORE_VIDEO_SCHEMA["required"]) == {"path", "instructions"}
