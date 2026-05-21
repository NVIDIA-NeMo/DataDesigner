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
    get_media_base64_context,
    get_media_context,
    get_media_url_context,
    is_audio_path,
    is_media_url,
    is_video_path,
    normalize_media_context_values,
    parse_base64_data_uri,
    video_format_from_mime_type,
    video_mime_type,
)


def test_media_context_builders() -> None:
    assert get_media_context("image", {"type": "url", "url": "https://example.com/image.png"}) == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/image.png"},
    }
    assert get_media_url_context("audio", "https://example.com/audio.mp3") == {
        "type": "audio",
        "source": {"type": "url", "url": "https://example.com/audio.mp3"},
    }
    assert get_media_base64_context("video", "video/mp4", "abc123") == {
        "type": "video",
        "source": {"type": "base64", "media_type": "video/mp4", "data": "abc123"},
    }


def test_normalize_media_context_values() -> None:
    assert normalize_media_context_values("single") == ["single"]
    assert normalize_media_context_values(["one", "two"]) == ["one", "two"]
    assert normalize_media_context_values(json.dumps(["one", "two"])) == ["one", "two"]
    assert normalize_media_context_values(json.dumps({"nested": "value"})) == [json.dumps({"nested": "value"})]
    assert normalize_media_context_values(lazy.np.array(["one", "two"])) == ["one", "two"]


def test_parse_base64_data_uri() -> None:
    assert parse_base64_data_uri("data:audio/mpeg;base64,abc123") == ("audio/mpeg", "abc123")
    assert parse_base64_data_uri("abc123") is None


def test_media_url_detection() -> None:
    assert is_media_url("https://example.com/download?id=123") is True
    assert is_media_url("http://example.com/media") is True
    assert is_media_url("ftp://example.com/media") is False
    assert is_media_url(123) is False  # type: ignore[arg-type]


def test_local_media_path_detection() -> None:
    assert is_audio_path("screen_recording.mp3") is True
    assert is_audio_path("nested/screen_recording.wav") is True
    assert is_audio_path("https://example.com/audio.mp3") is False
    assert is_video_path("screen_recording.mp4") is True
    assert is_video_path("nested/screen_recording.webm") is True
    assert is_video_path("https://example.com/video.mp4") is False


def test_media_format_mime_helpers() -> None:
    assert audio_mime_type(AudioFormat.MP3) == "audio/mpeg"
    assert audio_format_from_mime_type("audio/mpeg") == AudioFormat.MP3
    assert audio_format_from_mime_type("audio/mp3") == AudioFormat.MP3
    assert audio_format_from_mime_type("audio/x-wav") == AudioFormat.WAV
    assert video_mime_type(VideoFormat.MP4) == "video/mp4"
    assert video_format_from_mime_type("video/mp4") == VideoFormat.MP4
    assert video_format_from_mime_type("VIDEO/MP4") == VideoFormat.MP4
