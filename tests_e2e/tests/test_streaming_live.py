# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real DeepSeek streaming checks for both native HTTP adapter protocols."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

import data_designer.config as dd
from data_designer.config import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.streaming import ChatStream
from data_designer.engine.models.clients.types import ChatCompletionResponse
from data_designer.engine.models.factory import create_model_registry
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.secret_resolver import EnvironmentResolver
from data_designer.engine.testing import StubMCPFacade, StubMCPRegistry
from data_designer.interface import DataDesigner

pytestmark = pytest.mark.skipif(
    os.environ.get("DATA_DESIGNER_LIVE_STREAMING") != "1",
    reason="Set DATA_DESIGNER_LIVE_STREAMING=1 to call the real DeepSeek API using API_KEY from .env",
)

_PROVIDERS = {
    "openai": "https://api.deepseek.com",
    "anthropic": "https://api.deepseek.com/anthropic",
}


@pytest.fixture(params=list(_PROVIDERS))
def live_model(request: pytest.FixtureRequest) -> tuple[ModelProvider, ModelConfig]:
    """Build shared DeepSeek configuration for each supported API protocol.

    Args:
        request: Supplies the parametrized provider type (openai or anthropic).

    Returns:
        Provider and per-test model configuration with streaming enabled and thinking
        disabled; tests can override inference parameters or enable the health check.
    """
    provider = ModelProvider(
        name=f"deepseek-{request.param}",
        endpoint=_PROVIDERS[request.param],
        provider_type=request.param,
        api_key="API_KEY",
    )
    config = ModelConfig(
        alias="stream-test",
        model="deepseek-v4-flash",
        provider=provider.name,
        skip_health_check=True,
        inference_parameters=ChatCompletionInferenceParams(
            stream=True,
            temperature=0,
            max_tokens=512,
            timeout=120,
            max_parallel_requests=2,
            extra_body={"thinking": {"type": "disabled"}},
        ),
    )
    return provider, config


@pytest.fixture
def live_streams(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Load only API_KEY and observe actual SSE decoding without replacing network responses.

    Args:
        monkeypatch: Restores the environment and decoder observer after each test.

    Returns:
        Per-request event counts and terminal-marker status, excluding credentials/content.
    """
    key = dotenv_values(Path(__file__).resolve().parents[2] / ".env").get("API_KEY") or os.environ.get("API_KEY")
    assert key, "API_KEY must be available in .env or the environment"
    monkeypatch.setenv("API_KEY", key)
    observed: dict[ChatStream, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    original = ChatStream.feed_line

    def observe(stream: ChatStream, line: str) -> bool:
        """Count actual SSE data lines for stream and return the original decoder result."""
        if stream not in observed:
            observed[stream] = {"provider": stream.provider_name, "data_events": 0, "complete": False}
            records.append(observed[stream])
        if line.startswith("data:"):
            observed[stream]["data_events"] += 1
        complete = original(stream, line)
        observed[stream]["complete"] = complete
        return complete

    monkeypatch.setattr(ChatStream, "feed_line", observe)
    return records


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("scenario", ["json", "thinking", "tools"])
async def test_live_streaming(
    live_model: tuple[ModelProvider, ModelConfig], is_async: bool, scenario: str, live_streams: list[dict[str, Any]]
) -> None:
    """Exercise real model generation through config, admission, SSE, parsing, and usage.

    Args:
        live_model: DeepSeek provider and model configuration for the API protocol under test.
        is_async: Selects native synchronous or asynchronous model execution.
        scenario: JSON parsing, reasoning capture, or a complete model/tool/model exchange.
        live_streams: Observer of real streamed responses; no model outputs are mocked.
    """
    provider, config = live_model
    if scenario == "thinking":
        config.inference_parameters.max_tokens = 2048
        config.inference_parameters.extra_body = {"thinking": {"type": "enabled"}}
    calls: list[dict[str, Any]] = []

    def execute_sum(response: ChatCompletionResponse) -> list[ChatMessage]:
        """Execute the fully assembled sum tool request and return assistant/tool trace messages."""
        assert len(response.message.tool_calls) == 1
        call = response.message.tool_calls[0]
        assert call.name == "sum_numbers"
        args = json.loads(call.arguments_json)
        assert args == {"a": 17, "b": 25}
        calls.append(args)
        return [
            ChatMessage.as_assistant(
                content=response.message.content or "",
                tool_calls=[
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments_json},
                    }
                ],
            ),
            ChatMessage.as_tool(content=str(args["a"] + args["b"]), tool_call_id=call.id),
        ]

    tools = StubMCPRegistry(
        StubMCPFacade(
            process_fn=execute_sum,
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "sum_numbers",
                        "description": "Add two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                            "required": ["a", "b"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        )
    )
    registry = create_model_registry(
        model_configs=[config],
        secret_resolver=EnvironmentResolver(),
        model_provider_registry=ModelProviderRegistry(providers=[provider]),
        client_concurrency_mode=ClientConcurrencyMode.ASYNC if is_async else ClientConcurrencyMode.SYNC,
        mcp_registry=tools if scenario == "tools" else None,
    )
    model = registry.get_model(model_alias=config.alias)
    kwargs: dict[str, Any] = {}
    if scenario == "tools":
        prompt = "You must call sum_numbers with a=17 and b=25 before answering. Then return only its numeric result."
        kwargs["tool_alias"] = "tools"
    elif scenario == "json":
        prompt = 'Return exactly the JSON object {"answer":42,"label":"你好"}, without Markdown or explanation.'
        kwargs["parser"] = json.loads
    else:
        prompt = "Calculate 17+25. Think briefly, then return only the numeric answer."
    try:
        if is_async:
            output, trace = await model.agenerate(prompt, **kwargs)
        else:
            output, trace = await asyncio.to_thread(model.generate, prompt, **kwargs)
        assert output == {"answer": 42, "label": "你好"} if scenario == "json" else output.strip() == "42"
        if scenario == "thinking":
            assert any(message.reasoning_content for message in trace if message.role == "assistant")
        if scenario == "tools":
            assert calls == [{"a": 17, "b": 25}]
            assert any(message.role == "tool" for message in trace)
        expected_requests = 2 if scenario == "tools" else 1
        assert len(live_streams) == expected_requests
        assert all(record["complete"] and record["data_events"] > 2 for record in live_streams)
        stats = model.usage_stats.model_dump(mode="json")
        assert stats["request_usage"] == {
            "successful_requests": expected_requests,
            "failed_requests": 0,
            "total_requests": expected_requests,
        }
        assert stats["token_usage"]["input_tokens"] > 0
        assert stats["token_usage"]["output_tokens"] > 0
        print(
            json.dumps(
                {
                    "protocol": provider.provider_type,
                    "async": is_async,
                    "scenario": scenario,
                    "streams": live_streams,
                    "usage": stats,
                },
                ensure_ascii=False,
            )
        )
    finally:
        if is_async:
            await registry.aclose()
        else:
            registry.close()


def test_live_streaming_dataset(
    live_model: tuple[ModelProvider, ModelConfig], live_streams: list[dict[str, Any]], tmp_path: Path
) -> None:
    """Create and reload a real dataset with streamed text, code, structured, and judge columns.

    Args:
        live_model: DeepSeek provider and model configuration used for every generated column.
        live_streams: Observer proving all generated responses were complete SSE streams.
        tmp_path: Isolated directory for generated dataset artifacts.
    """
    provider, config = live_model
    config.skip_health_check = False
    builder = dd.DataDesignerConfigBuilder(model_configs=[config])
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="answer",
            model_alias=config.alias,
            prompt="Return only the number 42.",
            with_trace=dd.TraceType.ALL_MESSAGES,
        )
    )
    builder.add_column(
        dd.LLMCodeColumnConfig(
            name="code",
            model_alias=config.alias,
            prompt="Write the single Python statement answer = 42.",
            code_lang=dd.CodeLang.PYTHON,
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="structured",
            model_alias=config.alias,
            prompt="Return answer as the integer {{ answer }}.",
            output_format={"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]},
        )
    )
    builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="judgment",
            model_alias=config.alias,
            prompt="The expected answer is 42. Evaluate this answer: {{ answer }}.",
            scores=[
                dd.Score(
                    name="correctness", description="Whether the answer is 42.", options={0: "Incorrect", 1: "Correct"}
                )
            ],
        )
    )
    designer = DataDesigner(model_providers=[provider], secret_resolver=EnvironmentResolver(), artifact_path=tmp_path)
    results = designer.create(builder, num_records=1, dataset_name=f"stream-{provider.provider_type}")
    frame = results.load_dataset()
    assert len(frame) == 1
    assert frame.iloc[0]["answer"].strip() == "42"
    assert "42" in frame.iloc[0]["code"]
    assert frame.iloc[0]["structured"] == {"answer": 42}
    assert frame.iloc[0]["answer__trace"][-1]["content"][0]["text"].strip() == "42"
    assert frame.iloc[0]["judgment"]["correctness"]["score"] == 1
    assert len(live_streams) == 5  # One model health check and four generated columns.
    assert all(record["complete"] and record["data_events"] > 2 for record in live_streams)
    print(json.dumps({"protocol": provider.provider_type, "dataset_rows": len(frame), "streams": live_streams}))
