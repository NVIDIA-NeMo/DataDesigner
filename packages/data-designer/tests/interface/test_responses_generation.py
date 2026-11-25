# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
from pytest_httpx import HTTPXMock

from data_designer.config.column_configs import LLMCodeColumnConfig, LLMJudgeColumnConfig, LLMTextColumnConfig, Score
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ChatCompletionInferenceParams, ImageContext, ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.interface.data_designer import DataDesigner


def respond_to_generation(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert request.url.path == "/v1/responses"
    assert body["store"] is False
    assert body["max_output_tokens"] == 1024
    content = body["input"][-1]["content"]
    prompt = "\n".join(block["text"] for block in content if block["type"] == "input_text")
    if "Judge" in prompt:
        assert "print(1)" in prompt
        assert "quality" in str(body["input"])
        text = '{"quality":{"score":1,"reasoning":"Correct"}}'
    elif "CODE" in prompt:
        assert "A blue square" in prompt
        text = "```python\nprint(1)\n```"
    elif "Describe the image" in prompt:
        assert next(block for block in content if block["type"] == "input_image") == {
            "type": "input_image",
            "image_url": "https://example.com/square.png",
            "detail": "auto",
        }
        text = "A blue square"
    else:
        text = "Hello"
    return httpx.Response(
        200,
        json={
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
    )


def test_responses_dependent_columns_judge_and_image(
    httpx_mock: HTTPXMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("data_designer.engine.models.telemetry.TELEMETRY_ENABLED", False)
    httpx_mock.add_callback(respond_to_generation, url="https://api.example.com/v1/responses", is_reusable=True)
    score = Score(name="quality", description="Correctness", options={1: "Correct", 0: "Incorrect"})
    managed = tmp_path / "managed"
    managed.mkdir()
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        managed_assets_path=managed,
        secret_resolver=PlaintextResolver(),
        model_providers=[
            ModelProvider(name="responses", provider_type="openai", endpoint="https://api.example.com/v1")
        ],
    )
    designer.set_run_config(RunConfig(otel_metrics_port=None))
    config = DataDesignerConfigBuilder(
        model_configs=[
            ModelConfig(
                alias="test",
                api="responses",
                model="gpt-test",
                provider="responses",
                inference_parameters=ChatCompletionInferenceParams(
                    max_tokens=1024, extra_body={"reasoning_effort": "medium"}
                ),
            )
        ]
    )
    config.with_seed_dataset(DataFrameSeedSource(df=pd.DataFrame({"image": ["https://example.com/square.png"]})))
    config.add_column(
        LLMTextColumnConfig(
            name="description",
            model_alias="test",
            prompt="Describe the image",
            multi_modal_context=[ImageContext(column_name="image")],
        )
    )
    config.add_column(
        LLMCodeColumnConfig(name="code", model_alias="test", prompt="CODE for {{ description }}", code_lang="python")
    )
    config.add_column(
        LLMJudgeColumnConfig(
            name="judgment",
            model_alias="test",
            prompt="Judge {{ code }}",
            scores=[score],
        )
    )
    dataset = designer.create(config, num_records=1, dataset_name="responses").load_dataset()
    assert dataset.iloc[0]["description"] == "A blue square"
    assert dataset.iloc[0]["code"].strip() == "print(1)"
    assert dataset.iloc[0]["judgment"]["quality"]["score"] == 1
    assert len(httpx_mock.get_requests()) == 4
