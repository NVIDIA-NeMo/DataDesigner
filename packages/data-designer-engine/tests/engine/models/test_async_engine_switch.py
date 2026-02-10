# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.dataset_builders.column_wise_builder import ColumnWiseDatasetBuilder
from data_designer.engine.models.facade import ModelFacade


def test_model_facade_has_async_methods() -> None:
    """ModelFacade exposes async variants of its core methods."""
    assert hasattr(ModelFacade, "acompletion")
    assert hasattr(ModelFacade, "agenerate")
    assert hasattr(ModelFacade, "agenerate_text_embeddings")


def test_model_facade_has_sync_methods() -> None:
    """ModelFacade exposes synchronous core methods."""
    assert hasattr(ModelFacade, "completion")
    assert hasattr(ModelFacade, "generate")
    assert hasattr(ModelFacade, "generate_text_embeddings")


def test_async_engine_env_controls_builder_execution_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """When DATA_DESIGNER_ASYNC_ENGINE is set, _run_cell_by_cell_generator dispatches to async fan-out."""
    import data_designer.engine.dataset_builders.column_wise_builder as cwb_module

    mock_generator = MagicMock()
    mock_generator.get_generation_strategy.return_value = GenerationStrategy.CELL_BY_CELL
    mock_generator.inference_parameters.max_parallel_requests = 4

    builder = MagicMock()
    builder._resource_provider.run_config.non_inference_max_parallel_workers = 4

    # Test with async enabled
    with patch.object(cwb_module, "DATA_DESIGNER_ASYNC_ENGINE", True):
        ColumnWiseDatasetBuilder._run_cell_by_cell_generator(builder, mock_generator)
        builder._fan_out_with_async.assert_called_once()
        builder._fan_out_with_threads.assert_not_called()

    builder.reset_mock()

    # Test with async disabled
    with patch.object(cwb_module, "DATA_DESIGNER_ASYNC_ENGINE", False):
        ColumnWiseDatasetBuilder._run_cell_by_cell_generator(builder, mock_generator)
        builder._fan_out_with_threads.assert_called_once()
        builder._fan_out_with_async.assert_not_called()
