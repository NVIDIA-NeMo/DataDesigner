# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.utils.media_helpers import (
    AudioFormat,
    VideoFormat,
    audio_format_from_mime_type,
    audio_mime_type,
    is_audio_url,
    is_video_url,
    normalize_media_context_values,
    parse_base64_data_uri,
    video_format_from_mime_type,
    video_mime_type,
)


def test_normalize_media_context_values() -> None:
    assert normalize_media_context_values("single") == ["single"]
    assert normalize_media_context_values(["one", "two"]) == ["one", "two"]
    assert normalize_media_context_values(json.dumps(["one", "two"])) == ["one", "two"]
    assert normalize_media_context_values(json.dumps({"nested": "value"})) == [json.dumps({"nested": "value"})]
    assert normalize_media_context_values(lazy.np.array(["one", "two"])) == ["one", "two"]


def test_parse_base64_data_uri() -> None:
    assert parse_base64_data_uri("data:audio/mpeg;base64,abc123") == ("audio/mpeg", "abc123")
    assert parse_base64_data_uri("abc123") is None


def test_audio_url_detection() -> None:
    assert is_audio_url("https://example.com/audio.mp3") is True
    assert is_audio_url("https://example.com/audio.wav?download=1") is True
    assert is_audio_url("https://example.com/image.png") is False
    assert is_audio_url(123) is False  # type: ignore[arg-type]


def test_video_url_detection() -> None:
    assert is_video_url("https://example.com/video.mp4") is True
    assert is_video_url("https://example.com/video.webm?download=1") is True
    assert is_video_url("https://example.com/audio.mp3") is False
    assert is_video_url(123) is False  # type: ignore[arg-type]


def test_media_format_mime_helpers() -> None:
    assert audio_mime_type(AudioFormat.MP3) == "audio/mpeg"
    assert audio_format_from_mime_type("audio/mpeg") == AudioFormat.MP3
    assert video_mime_type(VideoFormat.MP4) == "video/mp4"
    assert video_format_from_mime_type("video/mp4") == VideoFormat.MP4
