# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.engine.dataset_builders.artifact_storage import BatchStage
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor


@pytest.fixture
def stub_processor_config():
    return DropColumnsProcessorConfig(
        name="drop_columns_processor", build_stage=BuildStage.POST_BATCH, column_names=["col1", "col2"]
    )


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


def test_metadata():
    metadata = DropColumnsProcessor.metadata()

    assert metadata.name == "drop_columns_processor"
    assert metadata.description == "Drop columns from the input dataset."
    assert metadata.required_resources is None


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
def test_process_scenarios(
    stub_processor, stub_sample_dataframe, test_case, column_names, expected_result, expected_warning
):
    stub_processor.config.column_names = column_names

    if expected_warning:
        with patch("data_designer.engine.processing.processors.drop_columns.logger") as mock_logger:
            result = stub_processor.process(stub_sample_dataframe.copy())

            pd.testing.assert_frame_equal(result, pd.DataFrame(expected_result))
            mock_logger.warning.assert_called_once_with(expected_warning)
    else:
        result = stub_processor.process(stub_sample_dataframe.copy())
        pd.testing.assert_frame_equal(result, pd.DataFrame(expected_result))


def test_process_logging(stub_processor, stub_sample_dataframe):
    with patch("data_designer.engine.processing.processors.drop_columns.logger") as mock_logger:
        stub_processor.process(stub_sample_dataframe.copy())

        mock_logger.info.assert_called_once_with("🙈 Dropping columns: ['col1', 'col2']")


def test_save_dropped_columns_without_preview(stub_processor, stub_sample_dataframe):
    stub_processor.config.column_names = ["col1", "col2"]

    with patch("data_designer.engine.processing.processors.drop_columns.logger") as mock_logger:
        stub_processor.process(stub_sample_dataframe.copy(), current_batch_number=0)

        stub_processor.artifact_storage.write_parquet_file.assert_called_once()
        call_args = stub_processor.artifact_storage.write_parquet_file.call_args

        assert call_args.kwargs["parquet_file_name"] == "dropped.parquet"
        assert call_args.kwargs["batch_stage"] == BatchStage.DROPPED_COLUMNS

        saved_df = call_args.kwargs["dataframe"]
        assert list(saved_df.columns) == ["col1", "col2"]
        assert len(saved_df) == 4

        mock_logger.debug.assert_called_once_with("📦 Saving dropped columns to dropped-columns directory")


def test_save_dropped_columns_with_preview(stub_processor, stub_sample_dataframe):
    stub_processor.config.column_names = ["col1", "col2"]

    stub_processor.process(stub_sample_dataframe.copy())
    stub_processor.artifact_storage.write_parquet_file.assert_not_called()


def test_save_dropped_columns_with_nonexistent_columns(stub_processor, stub_sample_dataframe):
    stub_processor.config.column_names = ["nonexistent1", "nonexistent2"]

    with patch("data_designer.engine.processing.processors.drop_columns.logger"):
        with pytest.raises(KeyError):
            stub_processor.process(stub_sample_dataframe.copy(), current_batch_number=0)


def test_process_inplace_modification(stub_processor, stub_sample_dataframe):
    original_df = stub_sample_dataframe.copy()
    result = stub_processor.process(original_df)

    assert result is original_df

    assert "col1" not in result.columns
    assert "col2" not in result.columns
    assert "col3" in result.columns


def test_process_empty_dataframe(stub_processor, stub_empty_dataframe):
    stub_processor.config.column_names = ["col1"]

    with patch("data_designer.engine.processing.processors.drop_columns.logger") as mock_logger:
        result = stub_processor.process(stub_empty_dataframe)

        pd.testing.assert_frame_equal(result, stub_empty_dataframe)
        mock_logger.warning.assert_called_once_with("⚠️ Cannot drop column: `col1` not found in the dataset.")
