# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture
def stub_processor_config():
    return DropColumnsProcessorConfig(name="drop_columns_processor", column_names=["col1", "col2"])


@pytest.fixture
def stub_processor(stub_processor_config):
    mock_resource_provider = Mock()
    mock_resource_provider.artifact_storage = Mock()
    mock_resource_provider.artifact_storage.create_batch_file_path = Mock()
    mock_resource_provider.artifact_storage.create_batch_file_path.return_value.name = "dropped.parquet"
    processor = DropColumnsProcessor(
        config=stub_processor_config,
        resource_provider=mock_resource_provider,
    )
    return processor


@pytest.fixture
def stub_empty_dataframe():
    return pd.DataFrame()


@pytest.mark.parametrize(
    "test_case,column_names,expected_result,expected_warning",
    [
        (
            "drop_existing_columns",
            ["col1", "col2"],
            {"col3": [True, False, True, False], "category": ["A", "B", "A", "B"], "other_col": [1, 2, 3, 4]},
            None,
        ),
        (
            "drop_nonexistent_column",
            ["nonexistent"],
            {
                "col1": [1, 2, 3, 4],
                "col2": ["a", "b", "c", "d"],
                "col3": [True, False, True, False],
                "category": ["A", "B", "A", "B"],
                "other_col": [1, 2, 3, 4],
            },
            "⚠️ Cannot drop column: `nonexistent` not found in the dataset.",
        ),
        (
            "drop_mixed_columns",
            ["col1", "nonexistent", "col2"],
            {"col3": [True, False, True, False], "category": ["A", "B", "A", "B"], "other_col": [1, 2, 3, 4]},
            "⚠️ Cannot drop column: `nonexistent` not found in the dataset.",
        ),
        (
            "empty_column_list",
            [],
            {
                "col1": [1, 2, 3, 4],
                "col2": ["a", "b", "c", "d"],
                "col3": [True, False, True, False],
                "category": ["A", "B", "A", "B"],
                "other_col": [1, 2, 3, 4],
            },
            None,
        ),
    ],
)
def test_process_after_batch_scenarios(
    stub_processor, stub_sample_dataframe, test_case, column_names, expected_result, expected_warning
):
    stub_processor.config.column_names = column_names

    if expected_warning:
        with patch("data_designer.engine.processing.processors.drop_columns.logger") as mock_logger:
            result = stub_processor.process_after_batch(stub_sample_dataframe.copy(), current_batch_number=0)

            pd.testing.assert_frame_equal(result, pd.DataFrame(expected_result))
            mock_logger.warning.assert_called_once_with(expected_warning)
    else:
        result = stub_processor.process_after_batch(stub_sample_dataframe.copy(), current_batch_number=0)
        pd.testing.assert_frame_equal(result, pd.DataFrame(expected_result))


def test_process_after_batch_logging(stub_processor, stub_sample_dataframe):
    with patch("data_designer.engine.processing.processors.drop_columns.logger") as mock_logger:
        stub_processor.process_after_batch(stub_sample_dataframe.copy(), current_batch_number=0)

        mock_logger.info.assert_called_once_with("🙈 Dropping columns: ['col1', 'col2']")


def test_save_dropped_columns(stub_processor, stub_sample_dataframe):
    stub_processor.config.column_names = ["col1", "col2"]

    with patch("data_designer.engine.processing.processors.drop_columns.logger") as mock_logger:
        stub_processor.process_after_batch(stub_sample_dataframe.copy(), current_batch_number=0)

        stub_processor.artifact_storage.write_parquet_file.assert_called_once()
        call_args = stub_processor.artifact_storage.write_parquet_file.call_args

        assert call_args.kwargs["parquet_file_name"] == "dropped.parquet"
        assert call_args.kwargs["batch_stage"] == BatchStage.DROPPED_COLUMNS

        saved_df = call_args.kwargs["dataframe"]
        assert list(saved_df.columns) == ["col1", "col2"]
        assert len(saved_df) == 4

        mock_logger.debug.assert_called_once_with("📦 Saving dropped columns to dropped-columns directory")


def test_save_dropped_columns_with_nonexistent_columns(stub_processor, stub_sample_dataframe):
    """When columns don't exist, no file is written but warnings are logged."""
    stub_processor.config.column_names = ["nonexistent1", "nonexistent2"]

    with patch("data_designer.engine.processing.processors.drop_columns.logger"):
        stub_processor.process_after_batch(stub_sample_dataframe.copy(), current_batch_number=0)
        # No file is written for nonexistent columns
        stub_processor.artifact_storage.write_parquet_file.assert_not_called()


def test_process_after_batch_inplace_modification(stub_processor, stub_sample_dataframe):
    original_df = stub_sample_dataframe.copy()
    result = stub_processor.process_after_batch(original_df, current_batch_number=0)

    assert result is original_df

    assert "col1" not in result.columns
    assert "col2" not in result.columns
    assert "col3" in result.columns


def test_process_after_batch_empty_dataframe(stub_processor, stub_empty_dataframe):
    stub_processor.config.column_names = ["col1"]

    with patch("data_designer.engine.processing.processors.drop_columns.logger") as mock_logger:
        result = stub_processor.process_after_batch(stub_empty_dataframe, current_batch_number=0)

        pd.testing.assert_frame_equal(result, stub_empty_dataframe)
        mock_logger.warning.assert_called_once_with("⚠️ Cannot drop column: `col1` not found in the dataset.")


def test_process_after_batch_preview_mode_does_not_save(stub_processor, stub_sample_dataframe):
    """In preview mode (current_batch_number=None), columns are dropped but not saved to disk."""
    stub_processor.config.column_names = ["col1", "col2"]

    result = stub_processor.process_after_batch(stub_sample_dataframe.copy(), current_batch_number=None)

    # Columns should still be dropped
    assert "col1" not in result.columns
    assert "col2" not in result.columns
    assert "col3" in result.columns

    # But no file should be written
    stub_processor.artifact_storage.write_parquet_file.assert_not_called()
