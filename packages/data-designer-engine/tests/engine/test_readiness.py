# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pytest

import data_designer.engine.readiness as readiness_mod
from data_designer.config.column_configs import LLMTextColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.sampler_params import SamplerType, UUIDSamplerParams
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.mcp.registry import MCPRegistry
from data_designer.engine.readiness import run_readiness_check


@pytest.fixture(autouse=True)
def _force_sync_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin readiness tests to the sync engine.

    The async path branches on ``readiness_mod.DATA_DESIGNER_ASYNC_ENGINE``;
    pinning to sync lets us assert against ``run_health_check`` directly
    without standing up an event loop.
    """
    monkeypatch.setattr(readiness_mod, "DATA_DESIGNER_ASYNC_ENGINE", False)


def _build_columns(*, model_configs, llm_columns: list[tuple[str, str]] = (), include_sampler: bool = True):
    """Build a ``DataDesignerConfig`` and return its (already-flat) column configs.

    ``llm_columns`` is a list of ``(name, model_alias)`` pairs. ``include_sampler``
    adds a UUID sampler column at the start so configs that use a seed-only fast
    path are still well-formed.
    """
    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    if include_sampler:
        builder.add_column(
            SamplerColumnConfig(name="seed_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams())
        )
    for name, model_alias in llm_columns:
        builder.add_column(LLMTextColumnConfig(name=name, prompt="x", model_alias=model_alias))
    return builder.build().columns


# ---------------------------------------------------------------------------
# Model health check
# ---------------------------------------------------------------------------


def test_run_readiness_check_collects_aliases_from_get_model_aliases(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """The model probe pings every alias returned by each config's ``get_model_aliases()``.

    Regression coverage for #606 — secondary aliases on multi-model plugin configs
    (returned via ``get_model_aliases()``) must be passed to ``run_health_check()``,
    not just the primary ``model_alias`` field.
    """
    stub_resource_provider.model_registry.run_health_check = Mock()
    stub_resource_provider.mcp_registry = None

    @custom_column_generator(model_aliases=["custom-model-a", "custom-model-b"])
    def _gen_with_two_models(row, generator_params, models):
        del generator_params, models
        return row

    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(SamplerColumnConfig(name="seed_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams()))
    builder.add_column(LLMTextColumnConfig(name="builtin", prompt="x", model_alias="builtin-model"))
    from data_designer.config.column_configs import CustomColumnConfig

    builder.add_column(CustomColumnConfig(name="custom_col", generator_function=_gen_with_two_models))

    run_readiness_check(builder.build().columns, stub_resource_provider)

    stub_resource_provider.model_registry.run_health_check.assert_called_once()
    (called_aliases,), _ = stub_resource_provider.model_registry.run_health_check.call_args
    assert set(called_aliases) == {"builtin-model", "custom-model-a", "custom-model-b"}


def test_run_readiness_check_skips_model_probe_when_no_aliases(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """Configs with no model aliases (samplers only) skip the model health check entirely."""
    stub_resource_provider.model_registry.run_health_check = Mock()
    stub_resource_provider.mcp_registry = None

    columns = _build_columns(model_configs=stub_model_configs, llm_columns=[])

    run_readiness_check(columns, stub_resource_provider)

    stub_resource_provider.model_registry.run_health_check.assert_not_called()


def test_run_readiness_check_propagates_model_probe_error(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """Exceptions from ``run_health_check`` bubble up unchanged for caller branching."""
    from data_designer.engine.models.errors import ModelAuthenticationError

    stub_resource_provider.model_registry.run_health_check = Mock(side_effect=ModelAuthenticationError("bad creds"))
    stub_resource_provider.mcp_registry = None

    columns = _build_columns(model_configs=stub_model_configs, llm_columns=[("col", "stub-text")])

    with pytest.raises(ModelAuthenticationError, match="bad creds"):
        run_readiness_check(columns, stub_resource_provider)


# ---------------------------------------------------------------------------
# MCP tool health check
# ---------------------------------------------------------------------------


def test_run_readiness_check_collects_unique_sorted_tool_aliases(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """Tool probes are called once per unique alias, sorted, after the model probe."""
    stub_resource_provider.model_registry.run_health_check = Mock()
    mock_mcp_registry = Mock(spec=MCPRegistry)
    stub_resource_provider.mcp_registry = mock_mcp_registry

    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(SamplerColumnConfig(name="seed_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams()))
    builder.add_column(LLMTextColumnConfig(name="a", prompt="x", model_alias="stub-text", tool_alias="zebra"))
    builder.add_column(LLMTextColumnConfig(name="b", prompt="x", model_alias="stub-text", tool_alias="alpha"))
    builder.add_column(
        LLMTextColumnConfig(name="c", prompt="x", model_alias="stub-text", tool_alias="alpha")  # duplicate
    )

    run_readiness_check(builder.build().columns, stub_resource_provider)

    mock_mcp_registry.run_health_check.assert_called_once_with(["alpha", "zebra"])


def test_run_readiness_check_skips_tool_probe_when_no_tool_aliases(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """Configs with no tool aliases never touch the MCP registry."""
    stub_resource_provider.model_registry.run_health_check = Mock()
    mock_mcp_registry = Mock(spec=MCPRegistry)
    stub_resource_provider.mcp_registry = mock_mcp_registry

    columns = _build_columns(model_configs=stub_model_configs, llm_columns=[("col", "stub-text")])

    run_readiness_check(columns, stub_resource_provider)

    mock_mcp_registry.run_health_check.assert_not_called()


def test_run_readiness_check_raises_when_tools_referenced_but_no_mcp_registry(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """Tool aliases are referenced but ``mcp_registry`` is ``None`` — must fail loudly."""
    stub_resource_provider.model_registry.run_health_check = Mock()
    stub_resource_provider.mcp_registry = None

    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(SamplerColumnConfig(name="seed_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams()))
    builder.add_column(LLMTextColumnConfig(name="col", prompt="x", model_alias="stub-text", tool_alias="missing-tools"))

    with pytest.raises(DatasetGenerationError, match="missing-tools"):
        run_readiness_check(builder.build().columns, stub_resource_provider)


def test_run_readiness_check_propagates_tool_probe_error(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """Exceptions from MCP ``run_health_check`` bubble up unchanged."""
    stub_resource_provider.model_registry.run_health_check = Mock()
    mock_mcp_registry = Mock(spec=MCPRegistry)
    mock_mcp_registry.run_health_check = Mock(side_effect=RuntimeError("mcp down"))
    stub_resource_provider.mcp_registry = mock_mcp_registry

    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(SamplerColumnConfig(name="seed_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams()))
    builder.add_column(LLMTextColumnConfig(name="col", prompt="x", model_alias="stub-text", tool_alias="tools"))

    with pytest.raises(RuntimeError, match="mcp down"):
        run_readiness_check(builder.build().columns, stub_resource_provider)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_run_readiness_check_runs_models_before_tools(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """The model probe must run first; an MCP failure is irrelevant if models fail first."""
    from data_designer.engine.models.errors import ModelAuthenticationError

    stub_resource_provider.model_registry.run_health_check = Mock(side_effect=ModelAuthenticationError("bad creds"))
    mock_mcp_registry = Mock(spec=MCPRegistry)
    stub_resource_provider.mcp_registry = mock_mcp_registry

    builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    builder.add_column(SamplerColumnConfig(name="seed_id", sampler_type=SamplerType.UUID, params=UUIDSamplerParams()))
    builder.add_column(LLMTextColumnConfig(name="col", prompt="x", model_alias="stub-text", tool_alias="tools"))

    with pytest.raises(ModelAuthenticationError):
        run_readiness_check(builder.build().columns, stub_resource_provider)

    # The MCP probe must not have been reached.
    mock_mcp_registry.run_health_check.assert_not_called()


def test_run_readiness_check_no_models_no_tools_is_noop(
    stub_resource_provider,
    stub_model_configs,
) -> None:
    """A pure-sampler config touches neither registry."""
    stub_resource_provider.model_registry.run_health_check = Mock()
    mock_mcp_registry = Mock(spec=MCPRegistry)
    stub_resource_provider.mcp_registry = mock_mcp_registry

    columns = _build_columns(model_configs=stub_model_configs, llm_columns=[])

    run_readiness_check(columns, stub_resource_provider)

    stub_resource_provider.model_registry.run_health_check.assert_not_called()
    mock_mcp_registry.run_health_check.assert_not_called()
