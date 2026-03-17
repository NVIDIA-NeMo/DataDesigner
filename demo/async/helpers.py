# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for async engine test scripts."""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeMessage:
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    reasoning_content: str | None = None


@dataclass(frozen=True)
class FakeChoice:
    message: FakeMessage


@dataclass(frozen=True)
class FakeResponse:
    choices: list[FakeChoice]
    usage: Any | None = None
    model: str | None = None


def _stable_seed(model: str, messages: list[dict[str, Any]]) -> int:
    payload = json.dumps(
        {"model": model, "messages": messages},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _mock_response_text(model: str, messages: list[dict[str, Any]]) -> str:
    rng = random.Random(_stable_seed(model, messages))
    score = rng.uniform(0, 10)
    category = rng.choice(["low", "mid", "high"])
    return f"mock:{model}|cat={category}|score={score:.3f}"


def _fake_response(model: str, messages: list[dict[str, Any]], **_kwargs: Any) -> FakeResponse:
    text = _mock_response_text(model, messages)
    return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=text))], model=model)


@contextlib.contextmanager
def patch_llm_responses(*, fail_pattern: str | None = None, fail_rate: float = 1.0) -> Iterator[None]:
    """Patch LLM completion to return deterministic mock responses.

    Args:
        fail_pattern: If set, raise RuntimeError when this string appears in the
            serialized messages. Used to simulate LLM failures for specific columns.
        fail_rate: Probability of failure when fail_pattern matches (0.0-1.0).
    """
    from data_designer.engine.models.litellm_overrides import CustomRouter

    original_completion = CustomRouter.completion
    original_acompletion = getattr(CustomRouter, "acompletion", None)

    def _should_fail(messages: list[dict[str, Any]]) -> bool:
        if fail_pattern is None:
            return False
        serialized = json.dumps(messages, default=str)
        if fail_pattern not in serialized:
            return False
        return random.random() < fail_rate

    def fake_completion(self: Any, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> FakeResponse:
        if _should_fail(messages):
            raise RuntimeError(f"Simulated LLM failure for {model}")
        return _fake_response(model, messages, **kwargs)

    async def fake_acompletion(self: Any, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> FakeResponse:
        if _should_fail(messages):
            raise RuntimeError(f"Simulated LLM failure for {model}")
        return _fake_response(model, messages, **kwargs)

    CustomRouter.completion = fake_completion
    CustomRouter.acompletion = fake_acompletion
    try:
        yield
    finally:
        CustomRouter.completion = original_completion
        if original_acompletion is not None:
            CustomRouter.acompletion = original_acompletion
        else:
            with contextlib.suppress(AttributeError):
                delattr(CustomRouter, "acompletion")


# ---------------------------------------------------------------------------
# Dataset fingerprinting
# ---------------------------------------------------------------------------


def dataset_fingerprint(df: Any) -> str:
    import numpy as np
    import pandas as pd

    def _default(v: Any) -> Any:
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (pd.Timestamp, pd.Timedelta)):
            return v.isoformat()
        if isinstance(v, set):
            return sorted(v)
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return str(v)

    normalized = df.reset_index(drop=True)
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    records = normalized.to_dict(orient="records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_default)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Model / DataDesigner factory helpers
# ---------------------------------------------------------------------------


def create_mock_model_config() -> tuple:
    """Return (ModelConfig, ModelProvider) for openai-text with mock backend."""
    from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider

    provider = ModelProvider(
        name="mock-provider",
        endpoint="https://mock.local",
        provider_type="openai",
        api_key="mock-key",
    )
    model_config = ModelConfig(
        alias="openai-text",
        model="gpt-4.1",
        provider="mock-provider",
        inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=8),
        skip_health_check=True,
    )
    return model_config, provider


def create_data_designer(*, artifact_path: str, async_trace: bool = True) -> Any:
    from data_designer.config.run_config import RunConfig
    from data_designer.interface import DataDesigner

    _, provider = create_mock_model_config()
    dd = DataDesigner(artifact_path=artifact_path, model_providers=[provider])
    dd.set_run_config(
        RunConfig(
            buffer_size=1000,
            disable_early_shutdown=True,
            async_trace=async_trace,
        )
    )
    return dd


def create_base_config() -> Any:
    """Create a DataDesignerConfigBuilder with openai-text model and a seed column."""
    from data_designer.config.column_configs import SamplerColumnConfig
    from data_designer.config.config_builder import DataDesignerConfigBuilder
    from data_designer.config.sampler_params import SamplerType, UniformSamplerParams

    model_config, _ = create_mock_model_config()
    config = DataDesignerConfigBuilder(model_configs=[model_config])
    config.add_column(
        SamplerColumnConfig(
            name="seed_value",
            sampler_type=SamplerType.UNIFORM,
            params=UniformSamplerParams(low=0.0, high=100.0, decimal_places=2),
        )
    )
    return config


def seed_rng(seed: int = 42) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
