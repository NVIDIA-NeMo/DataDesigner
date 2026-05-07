# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CustomColumnGenerator with decorator-based API."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

import data_designer.lazy_heavy_imports as lazy

if TYPE_CHECKING:
    import pandas as pd

from data_designer.config.column_configs import CustomColumnConfig, GenerationStrategy
from data_designer.config.custom_column import custom_column_generator
from data_designer.engine.column_generators.generators.custom import (
    CustomColumnGenerator,
    _AsyncBridgedModelFacade,
    _compute_bridge_timeout,
)
from data_designer.engine.column_generators.utils.errors import CustomColumnGenerationError
from data_designer.engine.models.clients.errors import SyncClientUnavailableError
from data_designer.engine.models.errors import RETRYABLE_MODEL_ERRORS, ModelTimeoutError
from data_designer.engine.resources.resource_provider import ResourceProvider


class SampleParams(BaseModel):
    """Sample params class for tests."""

    multiplier: int = 1
    prefix: str = ""
    suffix: str = "_processed"


# Test fixtures


@custom_column_generator(required_columns=["input"])
def generator_with_required_columns(row: dict) -> dict:
    """Generator that requires input column."""
    row["result"] = row["input"].upper()
    return row


@custom_column_generator(required_columns=["input"], side_effect_columns=["secondary"])
def generator_with_side_effects(row: dict) -> dict:
    """Generator that creates additional columns."""
    row["primary"] = row["input"] * 2
    row["secondary"] = row["input"] * 3
    return row


def _create_test_generator(
    name: str = "test_column",
    generator_function: Any = None,
    generator_params: BaseModel | None = None,
    resource_provider: ResourceProvider | None = None,
    generation_strategy: GenerationStrategy = GenerationStrategy.CELL_BY_CELL,
) -> CustomColumnGenerator:
    """Helper function to create test generator."""
    if generator_function is None:

        @custom_column_generator()
        def simple_generator(row: dict) -> dict:
            row[name] = "test_value"
            return row

        generator_function = simple_generator

    config = CustomColumnConfig(
        name=name,
        generator_function=generator_function,
        generator_params=generator_params,
        generation_strategy=generation_strategy,
    )
    if resource_provider is None:
        resource_provider = Mock(spec=ResourceProvider)
    return CustomColumnGenerator(config=config, resource_provider=resource_provider)


# Config and creation tests


def test_config_and_decorator_integration() -> None:
    """Test config reads decorator metadata, serializes correctly, and creates generator."""

    @custom_column_generator(
        required_columns=["col1", "col2"],
        side_effect_columns=["extra"],
        model_aliases=["model-a"],
    )
    def decorated_generator(row: dict) -> dict:
        return row

    config = CustomColumnConfig(name="test", generator_function=decorated_generator)

    # Decorator metadata is read
    assert config.required_columns == ["col1", "col2"]
    assert config.side_effect_columns == ["extra"]
    assert config.model_aliases == ["model-a"]

    # Serialization works
    assert config.model_dump()["generator_function"] == "decorated_generator"

    # Generator creation works with defaults
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))
    assert generator.config.column_type == "custom"
    assert generator.get_generation_strategy() == GenerationStrategy.CELL_BY_CELL


def test_config_validation_non_callable() -> None:
    """Test that non-callable generator_function raises an error."""
    with pytest.raises(ValidationError, match="must be callable"):
        CustomColumnConfig(name="test", generator_function="not_a_function")


def test_config_validation_allow_resize_allows_full_column_and_cell_by_cell() -> None:
    """allow_resize=True is valid with full_column or cell_by_cell."""

    @custom_column_generator()
    def dummy_fn(row: dict) -> dict:
        return row

    for strategy in (GenerationStrategy.FULL_COLUMN, GenerationStrategy.CELL_BY_CELL):
        config = CustomColumnConfig(
            name="test",
            generator_function=dummy_fn,
            allow_resize=True,
            generation_strategy=strategy,
        )
        assert config.allow_resize is True


# Cell-by-cell generation tests


def test_cell_by_cell_generation() -> None:
    """Test basic cell-by-cell generation with 1-arg function."""
    generator = _create_test_generator(name="result", generator_function=generator_with_required_columns)
    result = generator.generate({"input": "hello"})
    assert result["result"] == "HELLO"


def test_cell_by_cell_with_params_and_models(stub_resource_provider, stub_model_facade) -> None:
    """Test 3-arg function with generator_params and models dict for LLM access."""

    @custom_column_generator(required_columns=["input"], model_aliases=["test-model"])
    def llm_generator(row: dict, generator_params: SampleParams, models: dict) -> dict:
        response, _ = models["test-model"].generate(
            prompt=f"{generator_params.prefix}{row['input']}",
            system_prompt="You are helpful.",
        )
        row["result"] = response
        return row

    generator = _create_test_generator(
        name="result",
        generator_function=llm_generator,
        generator_params=SampleParams(prefix="Process: "),
        resource_provider=stub_resource_provider,
    )

    result = generator.generate({"input": "test"})

    # Model was called with correct params
    stub_model_facade.generate.assert_called_once()
    call_kwargs = stub_model_facade.generate.call_args[1]
    assert "Process: test" in call_kwargs["prompt"]
    assert result["result"] == "Generated summary text"


def test_side_effect_columns() -> None:
    """Test that declared side_effect_columns are created and kept."""
    generator = _create_test_generator(name="primary", generator_function=generator_with_side_effects)
    result = generator.generate({"input": 5})

    assert result["primary"] == 10
    assert result["secondary"] == 15


# cell_by_cell allow_resize: dict | list[dict]


def test_cell_by_cell_allow_resize_return_dict() -> None:
    """With allow_resize, returning a single dict (1:1) works like normal cell-by-cell."""
    config = CustomColumnConfig(
        name="result",
        generator_function=generator_with_required_columns,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
        allow_resize=True,
    )
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))
    result = generator.generate({"input": "hi"})
    assert isinstance(result, dict)
    assert result["result"] == "HI"


def test_cell_by_cell_allow_resize_return_list_expand() -> None:
    """With allow_resize, returning list[dict] expands one row into multiple."""

    @custom_column_generator(required_columns=["x"])
    def expand(row: dict) -> list[dict]:
        return [
            {**row, "out": row["x"] * 1},
            {**row, "out": row["x"] * 2},
        ]

    config = CustomColumnConfig(
        name="out",
        generator_function=expand,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
        allow_resize=True,
    )
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))
    result = generator.generate({"x": 10})
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"x": 10, "out": 10}
    assert result[1] == {"x": 10, "out": 20}


def test_cell_by_cell_allow_resize_return_list_single() -> None:
    """With allow_resize, returning [dict] (1:1 via list) is valid."""

    @custom_column_generator(required_columns=["x"])
    def one_row(row: dict) -> list[dict]:
        return [{**row, "out": row["x"]}]

    config = CustomColumnConfig(
        name="out",
        generator_function=one_row,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
        allow_resize=True,
    )
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))
    result = generator.generate({"x": 42})
    assert result == [{"x": 42, "out": 42}]


def test_cell_by_cell_allow_resize_return_empty_list() -> None:
    """With allow_resize, returning [] drops that row (0 rows)."""

    @custom_column_generator(required_columns=["x"])
    def drop(row: dict) -> list[dict]:
        return []

    config = CustomColumnConfig(
        name="out",
        generator_function=drop,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
        allow_resize=True,
    )
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))
    result = generator.generate({"x": 1})
    assert result == []


def test_cell_by_cell_allow_resize_invalid_return_type() -> None:
    """With allow_resize, return must be dict or list[dict]."""

    @custom_column_generator(required_columns=["x"])
    def bad_return(row: dict):
        return [1, 2]

    config = CustomColumnConfig(
        name="out",
        generator_function=bad_return,
        generation_strategy=GenerationStrategy.CELL_BY_CELL,
        allow_resize=True,
    )
    generator = CustomColumnGenerator(config=config, resource_provider=Mock(spec=ResourceProvider))
    with pytest.raises(CustomColumnGenerationError, match="list elements must be dicts"):
        generator.generate({"x": 1})


# Error handling tests


@pytest.mark.parametrize(
    "generator_fn,input_row,error_match",
    [
        # Missing required column
        (generator_with_required_columns, {"other": 1}, "Missing required columns"),
        # Function raises error
        (
            custom_column_generator()(lambda row: (_ for _ in ()).throw(ValueError("fail"))),
            {"input": 1},
            "Custom generator function failed",
        ),
    ],
    ids=["missing_required", "function_raises"],
)
def test_generation_errors(generator_fn, input_row, error_match) -> None:
    """Test various error conditions during generation."""
    generator = _create_test_generator(name="result", generator_function=generator_fn)
    with pytest.raises(CustomColumnGenerationError, match=error_match):
        generator.generate(input_row)


def test_output_validation_errors() -> None:
    """Test output validation: wrong return type, missing column, missing side effects."""

    # Wrong return type
    @custom_column_generator()
    def returns_list(row: dict) -> list:
        return [1, 2, 3]

    generator = _create_test_generator(name="result", generator_function=returns_list)
    with pytest.raises(CustomColumnGenerationError, match="must return a dict"):
        generator.generate({"input": 1})

    # Missing expected column
    @custom_column_generator()
    def wrong_column(row: dict) -> dict:
        row["wrong"] = "value"
        return row

    generator = _create_test_generator(name="expected", generator_function=wrong_column)
    with pytest.raises(CustomColumnGenerationError, match="did not create the expected column"):
        generator.generate({"input": 1})

    # Missing declared side effect
    @custom_column_generator(side_effect_columns=["secondary"])
    def missing_side_effect(row: dict) -> dict:
        row["primary"] = 1
        return row

    generator = _create_test_generator(name="primary", generator_function=missing_side_effect)
    with pytest.raises(CustomColumnGenerationError, match="did not create declared side_effect_columns"):
        generator.generate({"input": 1})


def test_function_error_logs_warning_cell_by_cell(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a warning is logged when the user's generator function raises in cell-by-cell mode."""
    import logging

    @custom_column_generator()
    def failing_generator(row: dict) -> dict:
        raise ValueError("something broke")

    generator = _create_test_generator(name="result", generator_function=failing_generator)

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(CustomColumnGenerationError, match="Custom generator function failed"),
    ):
        generator.generate({"input": 1})

    assert "failing_generator" in caplog.text
    assert "This record will be skipped" in caplog.text
    assert "something broke" in caplog.text


@pytest.mark.parametrize("exc_cls", RETRYABLE_MODEL_ERRORS, ids=lambda c: c.__name__)
def test_retryable_model_errors_pass_through_sync_wrap(exc_cls: type[Exception]) -> None:
    """Retryable model errors raised inside a sync generator must NOT be wrapped.

    Without this, the scheduler classifies the wrapped error as non-retryable and
    counts it toward the early-shutdown gate (regression seen in #575 follow-up).
    """

    @custom_column_generator()
    def raising_gen(row: dict) -> dict:
        raise exc_cls("boom")

    generator = _create_test_generator(name="result", generator_function=raising_gen)
    with pytest.raises(exc_cls):
        generator.generate({"input": 1})


@pytest.mark.parametrize("exc_cls", RETRYABLE_MODEL_ERRORS, ids=lambda c: c.__name__)
@pytest.mark.asyncio
async def test_retryable_model_errors_pass_through_async_wrap(exc_cls: type[Exception]) -> None:
    """Retryable errors raised inside an async user generator must propagate unchanged."""

    @custom_column_generator()
    async def raising_gen(row: dict) -> dict:
        raise exc_cls("boom")

    generator = _create_test_generator(name="result", generator_function=raising_gen)
    with pytest.raises(exc_cls):
        await generator.agenerate({"input": 1})


def test_undeclared_columns_removed_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that undeclared columns are removed with a warning."""
    import logging

    @custom_column_generator()
    def creates_undeclared(row: dict) -> dict:
        row["result"] = "value"
        row["undeclared"] = "should be removed"
        return row

    generator = _create_test_generator(name="result", generator_function=creates_undeclared)

    with caplog.at_level(logging.WARNING):
        result = generator.generate({"input": 1})

    assert "result" in result
    assert "undeclared" not in result
    assert "undeclared columns" in caplog.text


# Full column strategy tests


def test_full_column_strategy() -> None:
    """Test full_column strategy processes DataFrame."""

    @custom_column_generator(required_columns=["input"])
    def batch_processor(df: pd.DataFrame) -> pd.DataFrame:
        df["result"] = df["input"] * 2
        return df

    generator = _create_test_generator(
        name="result", generator_function=batch_processor, generation_strategy=GenerationStrategy.FULL_COLUMN
    )

    assert generator.get_generation_strategy() == GenerationStrategy.FULL_COLUMN

    result = generator.generate(lazy.pd.DataFrame({"input": [1, 2, 3]}))
    assert list(result["result"]) == [2, 4, 6]


def test_full_column_with_params() -> None:
    """Test full_column with generator_params."""

    @custom_column_generator(required_columns=["input"])
    def batch_with_params(df: pd.DataFrame, generator_params: SampleParams) -> pd.DataFrame:
        df["result"] = df["input"] * generator_params.multiplier
        return df

    generator = _create_test_generator(
        name="result",
        generator_function=batch_with_params,
        generator_params=SampleParams(multiplier=3),
        generation_strategy=GenerationStrategy.FULL_COLUMN,
    )

    result = generator.generate(lazy.pd.DataFrame({"input": [1, 2, 3]}))
    assert list(result["result"]) == [3, 6, 9]


# Parameter name validation tests


def test_invalid_param_names_at_decoration_time() -> None:
    """Test parameter name validation at decoration time."""

    # Wrong first param
    with pytest.raises(TypeError, match="param 1 must be 'df' or 'row'"):

        @custom_column_generator()
        def bad_first(data: dict) -> dict:
            return data

    # Wrong second param
    with pytest.raises(TypeError, match="param 2 must be 'generator_params'"):

        @custom_column_generator()
        def bad_second(row: dict, params: None) -> dict:
            return row

    # Wrong third param
    with pytest.raises(TypeError, match="param 3 must be 'models'"):

        @custom_column_generator()
        def bad_third(row: dict, generator_params: None, llm: dict) -> dict:
            return row

    # Too many params
    with pytest.raises(TypeError, match="must have 1-3 parameters, got 4"):

        @custom_column_generator()
        def too_many(row: dict, generator_params: None, models: dict, extra: str) -> dict:
            return row


def test_strategy_mismatch_at_runtime() -> None:
    """Test that first param must match generation strategy at runtime."""

    # row function used with full_column strategy
    @custom_column_generator()
    def row_func(row: dict) -> dict:
        return row

    gen = _create_test_generator(
        name="result", generator_function=row_func, generation_strategy=GenerationStrategy.FULL_COLUMN
    )
    with pytest.raises(CustomColumnGenerationError, match="first parameter must be 'df', got 'row'"):
        gen.generate(lazy.pd.DataFrame({"input": [1]}))

    # df function used with cell_by_cell strategy
    @custom_column_generator()
    def df_func(df: pd.DataFrame) -> pd.DataFrame:
        return df

    gen = _create_test_generator(name="result", generator_function=df_func)
    with pytest.raises(CustomColumnGenerationError, match="first parameter must be 'row', got 'df'"):
        gen.generate({"input": 1})


# Async model bridge tests for _AsyncBridgedModelFacade


def test_async_bridge_proxy_transparent_in_sync_mode(stub_resource_provider, stub_model_facade) -> None:
    """Proxy passes through generate(), forwards attributes; _build_models_dict returns raw facades."""

    @custom_column_generator(required_columns=["input"], model_aliases=["test-model"])
    def gen_with_model(row: dict, generator_params: SampleParams, models: dict) -> dict:
        row["result"] = "ok"
        return row

    generator = _create_test_generator(
        name="result",
        generator_function=gen_with_model,
        generator_params=SampleParams(),
        resource_provider=stub_resource_provider,
    )

    # _build_models_dict returns raw facades (wrapping happens at the call site)
    models = generator._build_models_dict()
    assert not isinstance(models["test-model"], _AsyncBridgedModelFacade)

    # Proxy itself passes through generate() and forwards attributes
    proxy = _AsyncBridgedModelFacade(stub_model_facade)
    result, _ = proxy.generate("test", parser=str)
    assert result == "Generated summary text"
    stub_model_facade.generate.assert_called_once_with("test", parser=str)
    assert proxy.model_alias == "test_model"


def test_async_bridge_falls_back_to_agenerate_on_sync_client_error() -> None:
    """When sync generate() fails with an async/sync error, falls back to agenerate()."""
    facade = Mock()
    facade.generate.side_effect = SyncClientUnavailableError(
        "Sync methods are not available on an async-mode HttpModelClient."
    )
    facade.request_timeout = 60.0

    async def fake_agenerate(*args: Any, **kwargs: Any) -> tuple:
        return ("async_result", list(args), kwargs)

    facade.agenerate = fake_agenerate
    proxy = _AsyncBridgedModelFacade(facade)

    engine_loop = asyncio.new_event_loop()
    engine_thread = threading.Thread(target=engine_loop.run_forever, daemon=True)
    engine_thread.start()

    try:
        with patch(
            "data_designer.engine.dataset_builders.utils.async_concurrency.ensure_async_engine_loop",
            return_value=engine_loop,
        ):
            result = proxy.generate("hello", parser=str)
        assert result == ("async_result", ["hello"], {"parser": str})
    finally:
        engine_loop.call_soon_threadsafe(engine_loop.stop)
        engine_thread.join(timeout=5)


def test_async_bridge_non_client_mode_errors_propagate() -> None:
    """Only SyncClientUnavailableError triggers bridging; other errors propagate."""
    # ValueError - different type entirely
    facade = Mock()
    facade.generate.side_effect = ValueError("invalid prompt format")
    proxy = _AsyncBridgedModelFacade(facade)
    with pytest.raises(ValueError, match="invalid prompt format"):
        proxy.generate(prompt="hello")

    # RuntimeError - same base type as SyncClientUnavailableError, but not caught
    facade = Mock()
    facade.generate.side_effect = RuntimeError("connection timed out for async request")
    proxy = _AsyncBridgedModelFacade(facade)
    with pytest.raises(RuntimeError, match="connection timed out"):
        proxy.generate(prompt="hello")


def test_async_bridge_timeout_raises_model_timeout_error() -> None:
    """A bridge timeout must surface as ModelTimeoutError so the scheduler sees it as retryable."""
    facade = Mock()
    facade.generate.side_effect = SyncClientUnavailableError(
        "Sync methods are not available on an async-mode HttpModelClient."
    )
    # Bridge derives timeout from facade.request_timeout × max_correction_steps
    # (clamped to _BRIDGE_TIMEOUT_FLOOR_S). Patch the floor down so this test
    # finishes in milliseconds rather than the production default of 60s.
    facade.request_timeout = 0.01

    async def hangs_forever(*args: Any, **kwargs: Any) -> tuple:
        await asyncio.sleep(60)
        return ("never", [], {})

    facade.agenerate = hangs_forever
    proxy = _AsyncBridgedModelFacade(facade)

    engine_loop = asyncio.new_event_loop()
    engine_thread = threading.Thread(target=engine_loop.run_forever, daemon=True)
    engine_thread.start()

    try:
        with (
            patch(
                "data_designer.engine.dataset_builders.utils.async_concurrency.ensure_async_engine_loop",
                return_value=engine_loop,
            ),
            patch("data_designer.engine.column_generators.generators.custom._BRIDGE_TIMEOUT_FLOOR_S", 0.05),
            pytest.raises(ModelTimeoutError, match="bridge timed out"),
        ):
            proxy.generate("hello")
    finally:
        engine_loop.call_soon_threadsafe(engine_loop.stop)
        engine_thread.join(timeout=5)


def test_async_bridge_deadlock_guard_on_event_loop() -> None:
    """Raises a clear error instead of deadlocking when called from the event loop."""
    facade = Mock()
    facade.generate.side_effect = SyncClientUnavailableError(
        "Sync methods are not available on an async-mode HttpModelClient."
    )
    proxy = _AsyncBridgedModelFacade(facade)

    async def call_from_loop() -> None:
        proxy.generate(prompt="hello")

    with pytest.raises(RuntimeError, match="Use 'await model.agenerate\\(\\)'"):
        asyncio.run(call_from_loop())


@pytest.mark.parametrize(
    "request_timeout,correction_steps,conversation_restarts,expected",
    [
        (60.0, 0, 0, 90.0),  # 1 * 1 * 60 * 1.5 = 90, above floor
        (60.0, 2, 0, 270.0),  # 3 * 1 * 60 * 1.5 = 270
        (60.0, 0, 2, 270.0),  # 1 * 3 * 60 * 1.5 = 270 — restarts contribute too
        (60.0, 1, 1, 360.0),  # 2 * 2 * 60 * 1.5 = 360 — corrections × restarts compound
        (10.0, 0, 0, 60.0),  # 1 * 1 * 10 * 1.5 = 15, clamped to 60s floor
    ],
    ids=[
        "no-corrections-no-restarts",
        "corrections-only",
        "restarts-only",
        "corrections-and-restarts-compound",
        "small-clamped-to-floor",
    ],
)
def test_compute_bridge_timeout(
    request_timeout: float, correction_steps: int, conversation_restarts: int, expected: float
) -> None:
    """Bridge deadline = max(floor, (1+restarts) * (1+corrections) * request_timeout * 1.5)."""
    assert _compute_bridge_timeout(request_timeout, correction_steps, conversation_restarts) == expected


@pytest.mark.parametrize(
    "kwargs,expected_per_request",
    [
        ({}, 60.0),  # No override; bridge uses facade.request_timeout
        ({"timeout": 600.0}, 600.0),  # Per-call timeout overrides the model default
    ],
    ids=["no-override-uses-facade-default", "override-uses-per-call-value"],
)
def test_async_bridge_honors_per_call_timeout(kwargs: dict[str, object], expected_per_request: float) -> None:
    """``model.generate(timeout=...)`` must drive the bridge deadline, not just the facade default."""
    facade = Mock()
    facade.generate.side_effect = SyncClientUnavailableError("sync unavailable")
    facade.request_timeout = 60.0  # would be used if no override

    captured: dict[str, float] = {}

    async def fake_agenerate(*_args: object, **_kwargs: object) -> tuple:
        return ("ok", [], {})

    facade.agenerate = fake_agenerate
    proxy = _AsyncBridgedModelFacade(facade)

    real_compute = _compute_bridge_timeout

    def capture_compute(per_request: float, *args: object, **inner: object) -> float:
        captured["per_request"] = per_request
        return real_compute(per_request, *args, **inner)

    engine_loop = asyncio.new_event_loop()
    engine_thread = threading.Thread(target=engine_loop.run_forever, daemon=True)
    engine_thread.start()

    try:
        with (
            patch(
                "data_designer.engine.dataset_builders.utils.async_concurrency.ensure_async_engine_loop",
                return_value=engine_loop,
            ),
            patch(
                "data_designer.engine.column_generators.generators.custom._compute_bridge_timeout",
                capture_compute,
            ),
        ):
            proxy.generate("hello", **kwargs)
    finally:
        engine_loop.call_soon_threadsafe(engine_loop.stop)
        engine_thread.join(timeout=5)

    assert captured["per_request"] == expected_per_request
