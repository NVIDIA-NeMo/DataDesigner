# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Audio and Video Context Smoke Test

# %% [markdown]
# This notebook verifies the audio and video context flow without calling a real model endpoint.
#
# It checks that:
#
# - `AudioContext` and `VideoContext` produce canonical media blocks.
# - HTTP(S) URLs stay as URLs, including URLs without file extensions.
# - Base64 media values keep the required media metadata.
# - Local path-looking values are rejected instead of being resolved in the config layer.
# - The OpenAI-compatible adapter forwards URL media as URL blocks in the endpoint payload.

# %%
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import data_designer.config as dd
from data_designer.engine.models.clients.adapters.anthropic_translation import (
    UnsupportedAnthropicMediaBlockError,
    translate_content_blocks,
)
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.types import ChatCompletionRequest

# %%
AUDIO_URL = "https://example.com/download?id=audio-123"
VIDEO_URL = "https://example.com/download?id=video-456"
AUDIO_DATA_URI = "data:audio/mpeg;base64,YXVkaW8="
VIDEO_DATA_URI = "data:video/mp4;base64,dmlkZW8="

record = {
    "audio_url": AUDIO_URL,
    "video_url": VIDEO_URL,
    "audio_data_uri": AUDIO_DATA_URI,
    "video_data_uri": VIDEO_DATA_URI,
}

# %% [markdown]
# ## Config blocks keep URLs and base64 media distinct

# %%
url_context_blocks = [
    *dd.AudioContext(column_name="audio_url").get_contexts(record),
    *dd.VideoContext(column_name="video_url").get_contexts(record),
]

assert url_context_blocks == [
    {"type": "audio", "source": {"type": "url", "url": AUDIO_URL}},
    {"type": "video", "source": {"type": "url", "url": VIDEO_URL}},
]

url_context_blocks

# %%
base64_context_blocks = [
    *dd.AudioContext(column_name="audio_data_uri").get_contexts(record),
    *dd.VideoContext(column_name="video_data_uri").get_contexts(record),
]

assert base64_context_blocks == [
    {
        "type": "audio",
        "source": {
            "type": "base64",
            "media_type": "audio/mpeg",
            "data": "YXVkaW8=",
            "format": "mp3",
        },
    },
    {"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "dmlkZW8="}},
]

base64_context_blocks

# %% [markdown]
# ## Local file names are not resolved by audio/video context


# %%
def assert_raises_message(callback: Callable[[], Any], expected_message: str) -> str:
    try:
        callback()
    except ValueError as exc:
        message = str(exc)
        assert expected_message in message
        return message
    raise AssertionError(f"Expected ValueError containing {expected_message!r}")


audio_path_error = assert_raises_message(
    lambda: dd.AudioContext(column_name="audio_path", audio_format=dd.AudioFormat.MP3).get_contexts(
        {"audio_path": "screen_recording.mp3"}
    ),
    "Local audio paths are not supported",
)
video_path_error = assert_raises_message(
    lambda: dd.VideoContext(column_name="video_path", video_format=dd.VideoFormat.MP4).get_contexts(
        {"video_path": "screen_recording.mp4"}
    ),
    "Local video paths are not supported",
)

audio_path_error, video_path_error

# %% [markdown]
# ## Column config round-trips mixed media context

# %%
column_config = dd.LLMTextColumnConfig(
    name="media_summary",
    prompt="Summarize the audio and video context.",
    model_alias="mock-multimodal-model",
    multi_modal_context=[
        dd.AudioContext(column_name="audio_url"),
        dd.VideoContext(column_name="video_url"),
    ],
)

round_tripped = dd.LLMTextColumnConfig(**column_config.model_dump())

assert [type(context).__name__ for context in round_tripped.multi_modal_context or []] == [
    "AudioContext",
    "VideoContext",
]
assert set(round_tripped.required_columns) == {"audio_url", "video_url"}

round_tripped

# %% [markdown]
# ## OpenAI-compatible payloads send URL media as URLs


# %%
def mock_httpx_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = json.dumps(json_data)
    response.headers = {}
    return response


def make_mock_sync_client(response_json: dict[str, Any]) -> MagicMock:
    client = MagicMock()
    client.post = MagicMock(return_value=mock_httpx_response(response_json))
    return client


def chat_response(content: str = "ok") -> dict[str, Any]:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


sync_client = make_mock_sync_client(chat_response())
client = OpenAICompatibleClient(
    provider_name="smoke-provider",
    endpoint="https://api.example.com/v1",
    api_key="not-used",
    concurrency_mode=ClientConcurrencyMode.SYNC,
    sync_client=sync_client,
)

client.completion(
    ChatCompletionRequest(
        model="mock-multimodal-model",
        messages=[
            {
                "role": "user",
                "content": [*url_context_blocks, {"type": "text", "text": "Summarize the media."}],
            }
        ],
    )
)

url_payload_blocks = sync_client.post.call_args.kwargs["json"]["messages"][0]["content"]

assert url_payload_blocks[:2] == [
    {"type": "audio_url", "audio_url": {"url": AUDIO_URL}},
    {"type": "video_url", "video_url": {"url": VIDEO_URL}},
]

url_payload_blocks

# %%
sync_client = make_mock_sync_client(chat_response())
client = OpenAICompatibleClient(
    provider_name="smoke-provider",
    endpoint="https://api.example.com/v1",
    api_key="not-used",
    concurrency_mode=ClientConcurrencyMode.SYNC,
    sync_client=sync_client,
)

client.completion(
    ChatCompletionRequest(
        model="mock-multimodal-model",
        messages=[
            {
                "role": "user",
                "content": [*base64_context_blocks, {"type": "text", "text": "Summarize the media."}],
            }
        ],
    )
)

base64_payload_blocks = sync_client.post.call_args.kwargs["json"]["messages"][0]["content"]

assert base64_payload_blocks[:2] == [
    {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "mp3"}},
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,dmlkZW8="}},
]

base64_payload_blocks

# %% [markdown]
# ## Anthropic rejects unsupported audio/video context before an HTTP call

# %%
for block in url_context_blocks:
    try:
        translate_content_blocks([block])
    except UnsupportedAnthropicMediaBlockError as exc:
        assert exc.modality in {"audio", "video"}
    else:
        raise AssertionError(f"Expected Anthropic to reject {block['type']} context")

"All audio/video context smoke checks passed."
