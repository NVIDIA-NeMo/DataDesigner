# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for multimodal media context values."""

from __future__ import annotations

import json
import re
from typing import Any

from data_designer.config.utils.type_helpers import StrEnum


class AudioFormat(StrEnum):
    """Supported audio formats for audio context."""

    MP3 = "mp3"
    WAV = "wav"


class VideoFormat(StrEnum):
    """Supported video formats for video context."""

    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"


SUPPORTED_AUDIO_EXTENSIONS = tuple(f".{fmt.value.lower()}" for fmt in AudioFormat)
SUPPORTED_VIDEO_EXTENSIONS = tuple(f".{fmt.value.lower()}" for fmt in VideoFormat)

_DATA_URI_RE = re.compile(r"^data:(?P<media_type>[^;]+);base64,(?P<data>.+)$")

_AUDIO_FORMAT_TO_MIME_TYPE: dict[AudioFormat, str] = {
    AudioFormat.MP3: "audio/mpeg",
    AudioFormat.WAV: "audio/wav",
}
_VIDEO_FORMAT_TO_MIME_TYPE: dict[VideoFormat, str] = {
    VideoFormat.MP4: "video/mp4",
    VideoFormat.MOV: "video/quicktime",
    VideoFormat.WEBM: "video/webm",
}
_AUDIO_MIME_TYPE_TO_FORMAT: dict[str, AudioFormat] = {
    "audio/mpeg": AudioFormat.MP3,
    "audio/mp3": AudioFormat.MP3,
    "audio/wav": AudioFormat.WAV,
    "audio/wave": AudioFormat.WAV,
    "audio/x-wav": AudioFormat.WAV,
    "audio/vnd.wave": AudioFormat.WAV,
}
_VIDEO_MIME_TYPE_TO_FORMAT: dict[str, VideoFormat] = {
    "video/mp4": VideoFormat.MP4,
    "video/quicktime": VideoFormat.MOV,
    "video/webm": VideoFormat.WEBM,
}


def normalize_media_context_values(raw_value: Any) -> list[Any]:
    """Normalize scalar, JSON-list, list, and array-like media values."""
    if isinstance(raw_value, str):
        try:
            parsed_value = json.loads(raw_value)
            if isinstance(parsed_value, list):
                return parsed_value
        except (json.JSONDecodeError, TypeError):
            pass
        return [raw_value]

    if isinstance(raw_value, list):
        return raw_value

    if hasattr(raw_value, "__iter__") and not isinstance(raw_value, (str, bytes, dict)):
        return list(raw_value)

    return [raw_value]


def parse_base64_data_uri(value: str) -> tuple[str, str] | None:
    """Return ``(media_type, data)`` for a base64 data URI."""
    if not isinstance(value, str):
        return None
    match = _DATA_URI_RE.match(value)
    if match is None:
        return None
    return match.group("media_type"), match.group("data")


def is_media_url(value: str) -> bool:
    """Return whether a value is an HTTP(S) media URL."""
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def is_audio_url(value: str) -> bool:
    """Return whether a value looks like an audio URL."""
    return is_media_url(value) and _has_media_extension(value, SUPPORTED_AUDIO_EXTENSIONS)


def is_video_url(value: str) -> bool:
    """Return whether a value looks like a video URL."""
    return is_media_url(value) and _has_media_extension(value, SUPPORTED_VIDEO_EXTENSIONS)


def is_audio_path(value: str) -> bool:
    """Return whether a value looks like a local audio path."""
    return _has_path_extension(value, SUPPORTED_AUDIO_EXTENSIONS)


def is_video_path(value: str) -> bool:
    """Return whether a value looks like a local video path."""
    return _has_path_extension(value, SUPPORTED_VIDEO_EXTENSIONS)


def audio_mime_type(audio_format: AudioFormat) -> str:
    """Return the MIME type for an audio format."""
    return _AUDIO_FORMAT_TO_MIME_TYPE[audio_format]


def video_mime_type(video_format: VideoFormat) -> str:
    """Return the MIME type for a video format."""
    return _VIDEO_FORMAT_TO_MIME_TYPE[video_format]


def audio_format_from_mime_type(media_type: str) -> AudioFormat | None:
    """Infer an audio format from a MIME type."""
    return _AUDIO_MIME_TYPE_TO_FORMAT.get(media_type.lower())


def video_format_from_mime_type(media_type: str) -> VideoFormat | None:
    """Infer a video format from a MIME type."""
    return _VIDEO_MIME_TYPE_TO_FORMAT.get(media_type.lower())


def _has_media_extension(value: str, supported_extensions: tuple[str, ...]) -> bool:
    if not isinstance(value, str):
        return False
    return any(ext in value.lower() for ext in supported_extensions)


def _has_path_extension(value: str, supported_extensions: tuple[str, ...]) -> bool:
    if not isinstance(value, str):
        return False
    return not is_media_url(value) and value.lower().endswith(supported_extensions)
