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
# # Audio and Video Context End-to-End Smoke Test

# %% [markdown]
# This notebook verifies audio and video context through a full Data Designer preview pipeline.
#
# It starts a tiny local OpenAI-compatible HTTP server, configures Data Designer to use that server as a model
# provider, builds a seeded dataset with audio/video context columns, runs `DataDesigner.preview(...)`, and asserts on
# the payload received by the endpoint.
#
# No external model API key is required.

# %%
from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pandas as pd

import data_designer.config as dd
from data_designer.interface import DataDesigner

# %% [markdown]
# ## Seed media values

# %%
AUDIO_URL = "https://example.com/download?id=audio-123"
VIDEO_URL = "https://example.com/download?id=video-456"
AUDIO_DATA_URI = "data:audio/mpeg;base64,YXVkaW8="
VIDEO_DATA_URI = "data:video/mp4;base64,dmlkZW8="

seed_df = pd.DataFrame(
    [
        {
            "record_id": "row-1",
            "audio_url": AUDIO_URL,
            "video_url": VIDEO_URL,
            "audio_data_uri": AUDIO_DATA_URI,
            "video_data_uri": VIDEO_DATA_URI,
        }
    ]
)

seed_df

# %% [markdown]
# ## Local OpenAI-compatible test endpoint


# %%
class RecordingOpenAIHandler(BaseHTTPRequestHandler):
    captured_requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))

        self.captured_requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "json": payload,
            }
        )

        content_blocks = payload["messages"][0]["content"]
        media_blocks = [
            block
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") in {"audio_url", "input_audio", "video_url"}
        ]
        response_json = {
            "id": "chatcmpl-audio-video-smoke",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"mock summary received {len(media_blocks)} media blocks",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        response_body = json.dumps(response_json).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class LocalOpenAIServer:
    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingOpenAIHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def captured_requests(self) -> list[dict[str, Any]]:
        return RecordingOpenAIHandler.captured_requests

    def __enter__(self) -> LocalOpenAIServer:
        RecordingOpenAIHandler.captured_requests = []
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


# %% [markdown]
# ## Build and run a full Data Designer preview


# %%
def build_audio_video_config(model_alias: str, provider_name: str) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias=model_alias,
                model="local-audio-video-model",
                provider=provider_name,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.0,
                    max_tokens=64,
                    max_parallel_requests=1,
                    timeout=10,
                ),
                skip_health_check=True,
            )
        ]
    )
    config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))
    config_builder.add_column(
        dd.LLMTextColumnConfig(
            name="media_summary",
            model_alias=model_alias,
            prompt="Summarize the audio and video context for {{ record_id }}.",
            multi_modal_context=[
                dd.AudioContext(column_name="audio_url"),
                dd.VideoContext(column_name="video_url"),
                dd.AudioContext(column_name="audio_data_uri"),
                dd.VideoContext(column_name="video_data_uri"),
            ],
        )
    )
    return config_builder


with LocalOpenAIServer() as local_server, tempfile.TemporaryDirectory() as artifact_dir:
    provider = dd.ModelProvider(
        name="local-openai",
        endpoint=local_server.endpoint,
        provider_type="openai",
        api_key="local-test-key",
    )
    data_designer = DataDesigner(artifact_path=artifact_dir, model_providers=[provider])
    config_builder = build_audio_video_config(model_alias="local-multimodal", provider_name=provider.name)

    preview = data_designer.preview(config_builder, num_records=1)
    captured_requests = local_server.captured_requests

preview.dataset

# %% [markdown]
# ## Verify the preview output and endpoint payload

# %%
assert len(preview.dataset) == 1
assert preview.dataset.loc[0, "media_summary"] == "mock summary received 4 media blocks"
assert len(captured_requests) == 1

request = captured_requests[0]
payload = request["json"]
content_blocks = payload["messages"][0]["content"]

expected_media_blocks = [
    {"type": "audio_url", "audio_url": {"url": AUDIO_URL}},
    {"type": "video_url", "video_url": {"url": VIDEO_URL}},
    {"type": "input_audio", "input_audio": {"data": "YXVkaW8=", "format": "mp3"}},
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,dmlkZW8="}},
]

assert request["path"] == "/v1/chat/completions"
assert payload["model"] == "local-audio-video-model"
assert content_blocks[:4] == expected_media_blocks
assert content_blocks[4] == {"type": "text", "text": "Summarize the audio and video context for row-1."}

content_blocks

# %% [markdown]
# ## Verify local path rejection through the pipeline


# %%
def assert_raises_message(callback: Callable[[], Any], expected_message: str) -> str:
    try:
        callback()
    except Exception as exc:
        message = str(exc)
        assert expected_message in message
        return message
    raise AssertionError(f"Expected exception containing {expected_message!r}")


bad_seed_df = pd.DataFrame([{"record_id": "bad-row", "video_path": "screen_recording.mp4"}])


def run_bad_path_preview() -> None:
    with LocalOpenAIServer() as local_server, tempfile.TemporaryDirectory() as artifact_dir:
        provider = dd.ModelProvider(
            name="local-openai",
            endpoint=local_server.endpoint,
            provider_type="openai",
            api_key="local-test-key",
        )
        data_designer = DataDesigner(artifact_path=artifact_dir, model_providers=[provider])
        config_builder = dd.DataDesignerConfigBuilder(
            model_configs=[
                dd.ModelConfig(
                    alias="local-multimodal",
                    model="local-audio-video-model",
                    provider=provider.name,
                    inference_parameters=dd.ChatCompletionInferenceParams(max_parallel_requests=1, timeout=10),
                    skip_health_check=True,
                )
            ]
        )
        config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=bad_seed_df))
        config_builder.add_column(
            dd.LLMTextColumnConfig(
                name="media_summary",
                model_alias="local-multimodal",
                prompt="Summarize the video for {{ record_id }}.",
                multi_modal_context=[
                    dd.VideoContext(column_name="video_path", video_format=dd.VideoFormat.MP4),
                ],
            )
        )
        data_designer.preview(config_builder, num_records=1)


path_error = assert_raises_message(run_bad_path_preview, "Local video paths are not supported")

path_error

# %%
"Full Data Designer audio/video context smoke test passed."
