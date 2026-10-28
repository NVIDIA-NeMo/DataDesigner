# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError
from pytest_httpx import HTTPXMock

import data_designer.interface.data_designer as dd_mod
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RequestAdmissionTuningConfig, RunConfig
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.utils import ChatMessage
from data_designer.engine.secret_resolver import EnvironmentResolver
from data_designer.interface.data_designer import DataDesigner


@pytest.fixture
def model_designer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_model_providers: list[ModelProvider],
) -> Iterator[DataDesigner]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROVIDER_1_API_KEY", "resolved-test-key")
    with patch.object(dd_mod, "create_person_reader", side_effect=AssertionError("Person reader is not needed")):
        yield DataDesigner(
            model_providers=stub_model_providers,
            secret_resolver=EnvironmentResolver(),
            managed_assets_path=tmp_path / "missing-assets",
        )
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "missing-assets").exists()


@pytest.fixture
def custom_model_configs(stub_model_configs: list[ModelConfig]) -> list[ModelConfig]:
    return [stub_model_configs[0].model_copy(update={"alias": alias}) for alias in ("judge", "patient")]


@pytest.fixture
def completion_payload() -> dict[str, Any]:
    return {
        "id": "test-completion",
        "model": "stub-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


@pytest.mark.parametrize("config_source", ["defaults", "list", "yaml", "path"])
def test_get_models_sync(
    model_designer: DataDesigner,
    custom_model_configs: list[ModelConfig],
    stub_model_configs: list[ModelConfig],
    completion_payload: dict[str, Any],
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_source: str,
) -> None:
    monkeypatch.setattr(dd_mod, "get_default_model_configs", lambda: stub_model_configs)
    configs: list[ModelConfig] | str | Path | None = custom_model_configs
    if config_source == "defaults":
        configs = None
    elif config_source in {"yaml", "path"}:
        configs = json.dumps({"model_configs": [config.model_dump(mode="json") for config in custom_model_configs]})
        if config_source == "path":
            config_path = tmp_path / "models.yaml"
            config_path.write_text(configs)
            configs = config_path

    aliases = ["stub-model"] if configs is None else ["patient", "judge", "patient"]
    models = model_designer.get_models(aliases, model_configs=configs)
    assert list(models) == list(dict.fromkeys(aliases))
    httpx_mock.add_response(json=completion_payload, is_reusable=True)
    try:
        for alias, model in models.items():
            assert model.model_alias == alias
            assert model.model_name == "stub-model"
            assert model.completion([ChatMessage.as_user("test")]).message.content == "hello"
            assert model.usage_stats.token_usage.total_tokens == 5
            assert model.usage_stats.request_usage.successful_requests == 1
    finally:
        for model in models.values():
            model.close()

    for request in httpx_mock.get_requests():
        assert str(request.url) == "https://api.provider-1.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer resolved-test-key"
        assert json.loads(request.content)["model"] == "stub-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_defaults", [False, True])
async def test_get_models_async_shares_admission_and_preserves_lifetime(
    model_designer: DataDesigner,
    custom_model_configs: list[ModelConfig],
    completion_payload: dict[str, Any],
    httpx_mock: HTTPXMock,
    monkeypatch: pytest.MonkeyPatch,
    use_defaults: bool,
) -> None:
    for config in custom_model_configs:
        config.inference_parameters.max_parallel_requests = 1
    model_designer.set_run_config(RunConfig(request_admission=RequestAdmissionTuningConfig(startup_ramp_seconds=0)))
    monkeypatch.setattr(dd_mod, "get_default_model_configs", lambda: custom_model_configs)
    configs = None if use_defaults else custom_model_configs
    judge = model_designer.get_models(
        ["judge"], model_configs=configs, client_concurrency_mode=ClientConcurrencyMode.ASYNC
    )["judge"]
    patient = model_designer.get_models(
        ["patient"], model_configs=configs, client_concurrency_mode=ClientConcurrencyMode.ASYNC
    )["patient"]
    in_flight = peak = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        assert request.headers["Authorization"] == "Bearer resolved-test-key"
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json=completion_payload)

    httpx_mock.add_callback(respond, is_reusable=True)
    messages = [ChatMessage.as_user("test")]
    try:
        responses = await asyncio.gather(judge.acompletion(messages), patient.acompletion(messages))
        assert [response.message.content for response in responses] == ["hello", "hello"]
        assert peak == 1
        await judge.aclose()
        assert (await patient.acompletion(messages)).message.content == "hello"
        assert patient.usage_stats.token_usage.total_tokens == 10
        assert patient.usage_stats.request_usage.successful_requests == 2
    finally:
        await judge.aclose()
        await patient.aclose()


def test_get_models_retries(
    model_designer: DataDesigner,
    custom_model_configs: list[ModelConfig],
    completion_payload: dict[str, Any],
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(status_code=503, headers={"Retry-After": "0"}, json={"error": {"message": "retry"}})
    httpx_mock.add_response(json=completion_payload)
    model = model_designer.get_models(["judge"], model_configs=custom_model_configs)["judge"]
    try:
        assert model.completion([ChatMessage.as_user("test")]).message.content == "hello"
        assert len(httpx_mock.get_requests()) == 2
        assert model.usage_stats.request_usage.successful_requests == 1
    finally:
        model.close()


@pytest.mark.parametrize(
    "config_source,aliases",
    [
        ("defaults", ["stub-model", "missing"]),
        ("custom", ["judge", "missing"]),
        ("custom", ["judge", "stub-model"]),
        ("empty", ["stub-model"]),
    ],
)
def test_get_models_validates_aliases_before_creating_clients(
    model_designer: DataDesigner,
    custom_model_configs: list[ModelConfig],
    stub_model_configs: list[ModelConfig],
    monkeypatch: pytest.MonkeyPatch,
    config_source: str,
    aliases: list[str],
) -> None:
    monkeypatch.setattr(dd_mod, "get_default_model_configs", lambda: stub_model_configs)
    configs = {"defaults": None, "custom": custom_model_configs, "empty": []}[config_source]
    with patch("data_designer.engine.models.clients.factory.create_model_client") as create_client:
        with pytest.raises(ValueError, match="No model config with alias"):
            model_designer.get_models(aliases, model_configs=configs)
        create_client.assert_not_called()


@pytest.mark.parametrize(
    "configs,error,match",
    [
        ("{}", InvalidConfigError, "model_configs"),
        ('{"model_configs": [{}]}', ValidationError, "alias"),
    ],
)
def test_get_models_validates_config(
    model_designer: DataDesigner, configs: str, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        model_designer.get_models(["judge"], model_configs=configs)


def test_get_models_rejects_duplicate_configs(
    model_designer: DataDesigner, custom_model_configs: list[ModelConfig]
) -> None:
    with pytest.raises(ValueError, match="aliases must be unique"):
        model_designer.get_models(["judge"], model_configs=custom_model_configs * 2)


def test_get_models_rejects_invalid_concurrency_mode(model_designer: DataDesigner) -> None:
    with pytest.raises(ValueError, match="not a valid ClientConcurrencyMode"):
        model_designer.get_models([], client_concurrency_mode="invalid")


def test_get_models_empty_aliases(model_designer: DataDesigner) -> None:
    assert model_designer.get_models([], model_configs=[]) == {}
