# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External-readiness checks for a DataDesigner workload.

A "readiness" check is a pre-flight probe of every external resource a
configuration depends on: each referenced model alias is sent a tiny
generation request, and every referenced MCP tool alias is contacted to
confirm its server is reachable.

This module hosts the shared logic invoked from two places:

- ``DatasetBuilder.build`` / ``DatasetBuilder.build_preview`` — at the start
  of a workload, to fail fast before any expensive work begins.
- ``DataDesigner.check_models`` — exposed publicly so users can verify
  external dependencies are responsive without triggering a workload.

The two callers must use the same code path here so the standalone method
cannot drift from the workload-startup gate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from data_designer.engine.column_generators.utils.generator_classification import column_type_is_model_generated
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode

if TYPE_CHECKING:
    from data_designer.config.column_types import ColumnConfigT
    from data_designer.engine.resources.resource_provider import ResourceProvider

logger = logging.getLogger(__name__)

# Match the timeout the dataset builder's startup gate has always used.
_MODEL_HEALTH_CHECK_TIMEOUT_SECONDS = 180


def run_readiness_check(
    column_configs: Sequence[ColumnConfigT],
    resource_provider: ResourceProvider,
    *,
    client_concurrency_mode: ClientConcurrencyMode,
) -> None:
    """Probe every model and MCP tool referenced by ``column_configs``.

    For each unique model alias collected from the column configs,
    ``ModelRegistry.run_health_check`` (or ``arun_health_check`` when async
    mode is selected) sends a tiny ``"Hello!"`` generation. Models whose ``ModelConfig``
    has ``skip_health_check=True`` are skipped by the registry. After the
    model pass, every unique MCP tool alias is probed via
    ``MCPRegistry.run_health_check``.

    Args:
        column_configs: The column configs whose ``get_model_aliases()`` and
            ``tool_alias`` fields determine which aliases are probed.
        resource_provider: Provides access to the model registry and MCP
            registry. ``mcp_registry`` may be ``None`` only if no tool
            aliases are referenced.
        client_concurrency_mode: Resolved client mode for this run.

    Raises:
        Typed model errors from ``data_designer.engine.models.errors`` for
            any failing model probe.
        DatasetGenerationError: If a tool alias is referenced but no MCP
            registry is configured on the resource provider.
        TimeoutError: If async health-check execution exceeds
            ``_MODEL_HEALTH_CHECK_TIMEOUT_SECONDS``.
    """
    _run_model_health_check(column_configs, resource_provider, client_concurrency_mode=client_concurrency_mode)
    _run_mcp_tool_health_check(column_configs, resource_provider)


def _run_model_health_check(
    column_configs: Sequence[ColumnConfigT],
    resource_provider: ResourceProvider,
    *,
    client_concurrency_mode: ClientConcurrencyMode,
) -> None:
    model_aliases: set[str] = set()
    for config in column_configs:
        model_aliases.update(config.get_model_aliases())

    if not model_aliases:
        return

    if client_concurrency_mode == ClientConcurrencyMode.ASYNC:
        # Defer the async-engine imports to here so users on the legacy sync
        # engine never pay the import cost.
        import asyncio

        from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

        loop = ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(
            resource_provider.model_registry.arun_health_check(list(model_aliases)),
            loop,
        )
        try:
            future.result(timeout=_MODEL_HEALTH_CHECK_TIMEOUT_SECONDS)
        except TimeoutError:
            future.cancel()
            raise
    else:
        resource_provider.model_registry.run_health_check(list(model_aliases))


def _run_mcp_tool_health_check(
    column_configs: Sequence[ColumnConfigT],
    resource_provider: ResourceProvider,
) -> None:
    # Tool aliases are only meaningful on model-generated column configs.
    tool_aliases = sorted(
        {
            config.tool_alias
            for config in column_configs
            if column_type_is_model_generated(config.column_type) and getattr(config, "tool_alias", None)
        }
    )
    if not tool_aliases:
        return
    if resource_provider.mcp_registry is None:
        raise DatasetGenerationError(f"Tool alias(es) {tool_aliases!r} specified but no MCPRegistry configured.")
    resource_provider.mcp_registry.run_health_check(tool_aliases)
